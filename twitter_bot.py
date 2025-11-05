#!/usr/bin/env python3
"""
twitter_bot.py

Updates:
- Multi-candidate generation + novelty selection (embeddings)
- Style buckets / weighted rotation via env var STYLE_WEIGHTS
- Env-configurable CANDIDATE_COUNT, SIMILARITY_THRESHOLD, EMBED_MODEL, GEN_TEMPERATURE, GEN_TOP_P
- Candidate & selection logging to same tweet log file
"""

import os
import time
import logging
import random
import re
import json
from datetime import datetime
from typing import Optional, List, Dict

import pytz
import regex
import tweepy
import numpy as np
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# --- OpenAI ---
try:
    from openai import OpenAI
    _OPENAI_SDK = "v1"
except Exception:
    import openai as OpenAI
    _OPENAI_SDK = "legacy"

# ------------- Logging -------------
loglevel = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, loglevel, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("twitter-bot")

def log_raw_reply(raw: str, topic: str):
    preview = (raw or "").replace("\n", "\\n")
    if len(preview) > 200:
        preview = preview[:200] + "..."
    log.debug(f"Raw model reply for topic '{topic}': {preview}")

# ------------- Config -------------
IST = pytz.timezone("Asia/Kolkata")

DRY_RUN = os.getenv("DRY_RUN", "false").strip().lower() in {"1", "true", "yes", "y"}
RUN_ONCE = os.getenv("RUN_ONCE", "false").strip().lower() in {"1", "true", "yes", "y"}

OPENAI_MODEL = "gpt-4o-mini"  # ✅ Hardwired
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "6"))
USE_MODERATION = os.getenv("OPENAI_USE_MODERATION", "true").strip().lower() in {"1", "true", "yes", "y"}

DEFAULT_TOPICS = "productivity,leadership,learning,writing,craftsmanship,critical thinking,calm focus,habits"
TOPIC_STR = os.getenv("TWEET_TOPICS", DEFAULT_TOPICS)
TOPICS = [t.strip() for t in TOPIC_STR.split(",") if t.strip()]

TWEET_TIMES_IST = os.getenv("TWEET_TIMES_IST", "09:15,17:15")
TIMES = [t.strip() for t in TWEET_TIMES_IST.split(",") if t.strip()]

# Logging / persistence
LOG_PATH = os.getenv("TWEET_LOG_PATH", "tweets_log.jsonl")

# Novelty / generation params (env-driven)
CANDIDATE_COUNT = int(os.getenv("CANDIDATE_COUNT", "6"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.78"))  # higher => allow more similarity; lower => stricter novelty
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.85"))
GEN_TOP_P = float(os.getenv("GEN_TOP_P", "0.95"))
OPENAI_MAX_COMPLETION_TOKENS = int(os.getenv("OPENAI_MAX_COMPLETION_TOKENS", "120"))

# ---------- Style cooldown / anti-repetition across runs ----------
STYLE_COOLDOWN_ENABLED = os.getenv("STYLE_COOLDOWN_ENABLED", "true").strip().lower() in {"1", "true", "yes", "y"}
STYLE_COOLDOWN_WINDOW = int(os.getenv("STYLE_COOLDOWN_WINDOW", "1"))  # avoid this many last styles
STYLE_PICK_RETRIES = int(os.getenv("STYLE_PICK_RETRIES", "6"))        # attempts to sample a non-repeating style

# Style weights (env var format):
# STYLE_WEIGHTS="observational:0.4,micro-story:0.2,contrarian:0.15,question:0.15,tip:0.1"
STYLE_WEIGHTS_RAW = os.getenv(
    "STYLE_WEIGHTS",
    "observational:0.4,micro-story:0.2,contrarian:0.15,question:0.15,tip:0.1"
)

# ------------- Twitter Client -------------
def twitter_client() -> tweepy.Client:
    return tweepy.Client(
        consumer_key=os.environ["TWITTER_API_KEY"],
        consumer_secret=os.environ["TWITTER_API_SECRET"],
        access_token=os.environ["TWITTER_ACCESS_TOKEN"],
        access_token_secret=os.environ["TWITTER_ACCESS_TOKEN_SECRET"]
    )

# ------------- OpenAI Client -------------
def openai_client():
    key = os.environ["OPENAI_API_KEY"]
    if _OPENAI_SDK == "v1":
        return OpenAI(api_key=key)
    else:
        OpenAI.api_key = key
        return OpenAI

# ------------- Validation -------------
PROFANITY_PATTERNS = [
    r"\b(?i)(shit|fuck|bitch|bastard|asshole|dick|cunt|slut|whore|motherf\w+|bullshit)\b",
    r"(?i)\b(nigga|nigger|chink|spic|kike|pussy|retard|faggot)\b",
]

PII_PATTERNS = {
    "email": r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
    "phone_generic": r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{4}(?!\d)",
    "aadhaar_india": r"(?<!\d)(\d{4}\s?\d{4}\s?\d{4})(?!\d)",
    "pan_india": r"(?<![A-Z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Z0-9])",
    "ssn_like": r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)",
    "ip_address": r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)",
    "credit_card": r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)",
}

def contains_profanity(text: str) -> Optional[str]:
    for pat in PROFANITY_PATTERNS:
        if regex.search(pat, text):
            return "profanity"
    return None

def contains_pii(text: str) -> Optional[str]:
    for label, pat in PII_PATTERNS.items():
        if re.search(pat, text, flags=re.IGNORECASE):
            return f"PII:{label}"
    return None

def length_ok(text: str) -> bool:
    return len(text) <= 280

def is_meaningful_text(text: str) -> bool:
    try:
        return bool(regex.search(r"[\p{L}\p{N}]", text))
    except Exception:
        return bool(re.search(r"[A-Za-z0-9]", text or ""))

# ------------- Utility: style parsing -------------
def parse_style_weights(raw: str) -> List[Dict]:
    """
    Parse STYLE_WEIGHTS env var into a list of dicts: [{"style":name, "weight":float}, ...]
    Example input: "observational:0.4,micro-story:0.2,contrarian:0.15"
    """
    items = []
    for piece in raw.split(","):
        if not piece.strip():
            continue
        try:
            name, w = piece.split(":")
            items.append({"style": name.strip(), "weight": float(w)})
        except Exception:
            # fallback: treat as equal weight single style name if missing weight
            items.append({"style": piece.strip(), "weight": 1.0})
    # normalize weights
    total = sum(i["weight"] for i in items) or 1.0
    for i in items:
        i["weight"] = i["weight"] / total
    return items

STYLE_BUCKETS = parse_style_weights(STYLE_WEIGHTS_RAW)

STYLE_PROMPTS = {
    "observational": "Observational humor: short, wry, single twist. Use a tiny, concrete sensory detail (smell, sound). Avoid groan puns.",
    "micro-story": "A tiny story in one sentence with a mini-arc and a surprising end. Create a character detail.",
    "contrarian": "Contrarian insight: start with a common belief, then flip it with a concise unexpected twist.",
    "question": "Start with a hook question then add a short concrete image or insight.",
    "tip": "A crisp, unexpected practical tip related to the topic, short and actionable."
}

def _load_recent_style_prompts(limit: int = 1) -> List[str]:
    """
    Read the last `limit` style_prompt values from the tweets_log.jsonl file (most recent first).
    Returns a list of style keys (the keys used in STYLE_PROMPTS or the raw prompt string).
    """
    if not os.path.exists(LOG_PATH):
        return []
    found: List[str] = []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()[-(limit * 5 + 10):]
            for line in reversed(lines):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                sp = obj.get("style_prompt") or obj.get("style") or obj.get("style_name")
                if sp:
                    for key, prompt_text in STYLE_PROMPTS.items():
                        if sp == key or sp == prompt_text or sp.startswith(key):
                            sp = key
                            break
                    if sp not in found:
                        found.append(sp)
                if len(found) >= limit:
                    break
    except Exception:
        log.warning("Failed to read recent style prompts from log file")
    return found

def pick_style_prompt() -> str:
    """
    Choose a style prompt while avoiding recent styles if STYLE_COOLDOWN_ENABLED.
    Will try up to STYLE_PICK_RETRIES times to sample a style not in the recent set.
    """
    choices = [b["style"] for b in STYLE_BUCKETS]
    weights = [b["weight"] for b in STYLE_BUCKETS]

    avoid: List[str] = []
    if STYLE_COOLDOWN_ENABLED and STYLE_COOLDOWN_WINDOW > 0:
        avoid = _load_recent_style_prompts(limit=STYLE_COOLDOWN_WINDOW)

    chosen = None
    for attempt in range(max(1, STYLE_PICK_RETRIES)):
        chosen = random.choices(choices, weights=weights, k=1)[0]
        if chosen not in avoid:
            return STYLE_PROMPTS.get(chosen, chosen)

    log.info(f"Style cooldown couldn't find a new style after {STYLE_PICK_RETRIES} tries; using '{chosen}' anyway.")
    return STYLE_PROMPTS.get(chosen, chosen)

# ------------- Prompt builder -------------
USER_PROMPT_TEMPLATE = """GOAL: Generate a single original tweet that feels authentic, casual, and engaging while staying under Twitter’s 280-character limit.
Topic: "{topic}"
SUCCESS CRITERIA:
- Output must be exactly one tweet, no preamble or extra text.
- Tone: conversational, human, and natural, not corporate, essay-like, or overly polished.
- Humor or emojis may be included if it flows naturally.
- Absolutely avoid clichés (e.g., "resilience is bouncing back", "learning is a journey").
- Must include a concrete sensory detail or a brief image.
CONSTRAINTS:
- Strictly under 280 characters.
- No negative parallelisms, no hashtags unless explicitly requested, no bullet lists.
- Do not return multiple options—only one tweet.
OUTPUT FORMAT:
Plain text containing only the tweet. No quotes, labels, or additional commentary."""

def build_user_prompt(topic: str, style_prompt: str) -> str:
    safe_topic = topic.replace('"', "'")
    return f"{USER_PROMPT_TEMPLATE}\nTone/style: {style_prompt}\nTopic: {safe_topic}\nTweet:"

# ------------- Persistence helpers -------------
def load_recent_texts(limit=50) -> List[str]:
    if not os.path.exists(LOG_PATH):
        return []
    texts = []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            # read last N lines efficiently
            lines = f.readlines()[-limit:]
            for line in reversed(lines):
                try:
                    obj = json.loads(line)
                    if "text" in obj:
                        texts.append(obj["text"])
                except Exception:
                    continue
    except Exception as e:
        log.warning(f"Failed to load recent texts: {e}")
    return texts

def log_tweet_entry(entry: Dict):
    entry.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning(f"Failed to write tweet log: {e}")

# ------------- OpenAI helpers (Responses + Embeddings wrappers) -------------
def responses_create(client, **kwargs):
    """
    Wrapper that supports both SDK shapes:
    - v1: client.responses.create(...)
    - legacy: client.responses.create(...) too but response access differs
    We return the raw resp object.
    """
    # assume both variants expose .responses.create
    return client.responses.create(**kwargs)

def get_text_from_response(resp) -> Optional[str]:
    # try common shapes
    try:
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text
    except Exception:
        pass
    try:
        # common dict-like shape
        return resp["output"][0]["content"][0]["text"]
    except Exception:
        pass
    try:
        # some SDKs return content blocks
        blocks = getattr(resp, "output", None)
        if blocks:
            # try to stringify text contents
            texts = []
            for b in blocks:
                for c in getattr(b, "content", []):
                    if isinstance(c, dict) and "text" in c:
                        texts.append(c["text"])
                    elif hasattr(c, "text"):
                        texts.append(c.text)
            if texts:
                return "\n".join(texts)
    except Exception:
        pass
    # final fallback: str(resp)
    try:
        return str(resp)
    except Exception:
        return None

def embeddings_create(client, input_texts: List[str]) -> List[List[float]]:
    """
    Wrapper to call embeddings for both SDK variants and return list of embeddings.
    """
    if not input_texts:
        return []
    try:
        if _OPENAI_SDK == "v1":
            resp = client.embeddings.create(model=EMBED_MODEL, input=input_texts)
            return [d.embedding for d in resp.data]
        else:
            resp = client.Embeddings.create(model=EMBED_MODEL, input=input_texts)
            # legacy shape: resp["data"]
            return [item["embedding"] for item in resp["data"]]
    except Exception as e:
        log.warning(f"Embeddings call failed: {e}")
        return []

def moderation_check(client, text: str) -> bool:
    """
    Reuse your fail-open moderation strategy. Return True if passes (or fail-open).
    """
    if not USE_MODERATION:
        return True
    try:
        if _OPENAI_SDK == "v1":
            resp = client.moderations.create(model="omni-moderation-latest", input=text)
            return not any(r.flagged for r in resp.results)
        else:
            resp = client.Moderation.create(model="omni-moderation-latest", input=text)
            # legacy: resp["results"]
            return not any(r.get("flagged", False) for r in resp.get("results", []))
    except Exception as e:
        log.warning(f"Moderation API failed (fail-open): {e}")
        return True

# ------------- Similarity helpers -------------
def cosine_sim(a: List[float], b: List[float]) -> float:
    a_np = np.array(a, dtype=float)
    b_np = np.array(b, dtype=float)
    if np.linalg.norm(a_np) == 0 or np.linalg.norm(b_np) == 0:
        return 0.0
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))

def max_similarity_to_set(emb: List[float], set_embs: List[List[float]]) -> float:
    if not set_embs:
        return 0.0
    return max(cosine_sim(emb, e) for e in set_embs)

# ------------- Candidate generation & selection -------------
def generate_candidate_once(client, prompt: str, temperature: float, top_p: float) -> Optional[str]:
    """
    Call Responses API once and return a single text candidate (stripped).
    """
    try:
        resp = responses_create(client, model=OPENAI_MODEL, input=prompt, temperature=temperature, top_p=top_p, max_output_tokens=OPENAI_MAX_COMPLETION_TOKENS)
        text = get_text_from_response(resp)
        if text:
            return text.strip()
    except Exception as e:
        log.warning(f"Generation call failed: {e}")
    return None

def generate_candidates(client, topic: str, style_prompt: str, n: int = CANDIDATE_COUNT) -> List[Dict]:
    base_prompt = build_user_prompt(topic, style_prompt)
    candidates = []
    for _ in range(n):
        temp = max(0.0, min(1.2, GEN_TEMPERATURE + random.uniform(-0.15, 0.15)))
        top_p = max(0.3, min(1.0, GEN_TOP_P + random.uniform(-0.1, 0.05)))
        text = generate_candidate_once(client, base_prompt, temperature=temp, top_p=top_p)
        if text:
            # compact single-line
            text = " ".join(text.splitlines()).strip()
            candidates.append({"text": text, "temp": temp, "top_p": top_p})
    return candidates

def pick_most_novel(candidates: List[Dict], recent_texts: List[str]) -> Optional[Dict]:
    # basic safety/length filter first
    filtered = []
    for c in candidates:
        t = c["text"].strip()
        if not t:
            continue
        if len(t) > 280:
            t = t[:279].rsplit(" ", 1)[0]
        if contains_profanity(t):
            continue
        if contains_pii(t):
            continue
        if not is_meaningful_text(t):
            continue
        filtered.append({**c, "text": t})
    if not filtered:
        return None

    # embed recent texts and candidates
    client = openai_client()
    recent_embs = embeddings_create(client, recent_texts) if recent_texts else []
    cand_texts = [c["text"] for c in filtered]
    cand_embs = embeddings_create(client, cand_texts)

    # compute novelty score
    scored = []
    for i, emb in enumerate(cand_embs):
        max_sim_recent = max_similarity_to_set(emb, recent_embs) if recent_embs else 0.0
        # penalize similarity to other candidates (to favor diverse)
        pair_sims = [cosine_sim(emb, other) for j, other in enumerate(cand_embs) if j != i] if len(cand_embs) > 1 else [0.0]
        pair_penalty = max(pair_sims) if pair_sims else 0.0
        # novelty: lower when similar to recent or similar to other candidates
        novelty = 1.0 - max_sim_recent - 0.5 * pair_penalty
        scored.append({**filtered[i], "novelty": novelty, "max_sim_recent": max_sim_recent})
    scored.sort(key=lambda x: x["novelty"], reverse=True)

    # require best candidate to be sufficiently novel relative to threshold
    best = scored[0]
    # interpret threshold: we want max_sim_recent to be < SIMILARITY_THRESHOLD
    if best["max_sim_recent"] >= SIMILARITY_THRESHOLD:
        return None
    return best

# ------------- High-level generation API -------------
def generate_best_tweet_for_topic(client, topic: str) -> Optional[Dict]:
    style_prompt = pick_style_prompt()
    recent_texts = load_recent_texts(limit=50)
    candidates = generate_candidates(client, topic, style_prompt, n=CANDIDATE_COUNT)
    log.debug(f"Generated {len(candidates)} candidates for topic '{topic}' (style='{style_prompt}')")
    chosen = pick_most_novel(candidates, recent_texts)
    if chosen:
        log_tweet_entry({
            "text": chosen["text"],
            "topic": topic,
            "style_prompt": style_prompt,
            "candidates": [c["text"] for c in candidates],
            "novelty": chosen.get("novelty"),
            "max_sim_recent": chosen.get("max_sim_recent")
        })
        return chosen
    # fallback: one more forced attempt with stronger creativity (higher temp)
    log.info("No sufficiently novel candidate found, forcing higher creativity fallback.")
    fallback_temp = min(1.2, GEN_TEMPERATURE + 0.25)
    fallback_candidates = []
    base_prompt = build_user_prompt(topic, style_prompt + " (be more surprising)")
    for _ in range(max(2, CANDIDATE_COUNT // 2)):
        text = generate_candidate_once(client, base_prompt, temperature=fallback_temp, top_p=GEN_TOP_P)
        if text:
            fallback_candidates.append({"text": " ".join(text.splitlines()).strip(), "temp": fallback_temp, "top_p": GEN_TOP_P})
    chosen = pick_most_novel(fallback_candidates, recent_texts)
    if chosen:
        log_tweet_entry({
            "text": chosen["text"],
            "topic": topic,
            "style_prompt": style_prompt + " (fallback)",
            "candidates": [c["text"] for c in fallback_candidates],
            "novelty": chosen.get("novelty"),
            "max_sim_recent": chosen.get("max_sim_recent")
        })
    return chosen

# ------------- Moderation (reuse fail-open) -------------
def passes_moderation(client, text: str) -> bool:
    return moderation_check(client, text)

# ------------- Tweet Posting -------------
def generate_and_post():
    client = openai_client()
    twitter = twitter_client()

    topic = random.choice(TOPICS)
    log.info(f"Selected topic: {topic}")

    for attempt in range(MAX_RETRIES):
        try:
            chosen = generate_best_tweet_for_topic(client, topic)
            if not chosen:
                raise ValueError("No valid candidate found")

            text = chosen["text"]

            # final validation
            if not length_ok(text):
                raise ValueError("Tweet too long after trimming")
            if contains_profanity(text):
                raise ValueError("Contains profanity")
            if contains_pii(text):
                raise ValueError("Contains PII")
            if not is_meaningful_text(text):
                raise ValueError("Not meaningful")
            if not passes_moderation(client, text):
                raise ValueError("Failed moderation")

            if DRY_RUN:
                log.info(f"DRY_RUN enabled, would have tweeted: {text}")
            else:
                resp = twitter.create_tweet(text=text)
                # resp.data may be dict-like or object
                try:
                    tweet_id = resp.data["id"]
                except Exception:
                    try:
                        tweet_id = resp.data.id
                    except Exception:
                        tweet_id = None
                if tweet_id:
                    log.info(f"Tweet posted: https://twitter.com/i/web/status/{tweet_id}")
                else:
                    log.info(f"Tweet posted (id not extracted), resp: {resp}")
            return
        except Exception as e:
            log.warning(f"Validation/post attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)

    log.error("All retries failed for this run")

# ------------- Scheduler -------------
def main():
    if RUN_ONCE:
        generate_and_post()
        return

    scheduler = BlockingScheduler(timezone=IST)
    for t in TIMES:
        hour, minute = map(int, t.split(":"))
        scheduler.add_job(
            generate_and_post,
            CronTrigger(hour=hour, minute=minute),
            id=f"tweet-{t}",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=600,
        )
    log.info(f"Scheduler started for times: {TIMES}")
    scheduler.start()

if __name__ == "__main__":
    main()

