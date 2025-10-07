#!/usr/bin/env python3
"""
twitter_bot.py

Features:
- Posts twice daily (default 09:15 and 17:15 Asia/Kolkata) using APScheduler
- Fetches tweet text from OpenAI Responses API (gpt-4o-mini only)
- Enforces: length < 280, no profanity, no PII
- Optional OpenAI moderation (fail-open if moderation API fails)
- Retries fetching a fresh tweet if validation fails (exponential backoff)
- DRY_RUN mode to print instead of post
- Robust logging
"""

import os
import time
import logging
import random
import re
from datetime import datetime
from typing import Optional

import pytz
import regex
import tweepy
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

# ------------- User Prompt -------------
USER_PROMPT_TEMPLATE = """GOAL: Generate a single original tweet that feels authentic, casual, and engaging while staying under Twitter’s 280-character limit.
Topic: "{topic}"
SUCCESS CRITERIA:  
- Output must be exactly one tweet, no preamble or extra text.  
- Tone: conversational, human, and natural, not corporate, essay-like, or overly polished.  
- Wit, or emojis may be included if it flows naturally.  
- Absolutely avoid clichés (e.g., “resilience is bouncing back,” “learning is a journey”).
- Strictly no em dashes or en dashes. 
- Must feel like something a real person would post, not marketing copy or promotional language.  
- Ideal tweets should sound spontaneous, relatable, and easy to read in one go.  
CONSTRAINTS:  
- Strictly under 280 characters.
- No negative parallelisms, hashtags, or bullet points.  
- No multi-tweet threads or follow-up explanations.
- Do not start the tweet with "ever notice" or "just"  
- Do not return multiple options—only one tweet.  
OUTPUT FORMAT:  
Plain text containing only the tweet. No quotes, labels, or additional commentary."""

def build_user_prompt(topic: str) -> str:
    safe_topic = topic.replace('"', "'")
    return USER_PROMPT_TEMPLATE.format(topic=safe_topic)

# ------------- OpenAI fetch -------------
def fetch_tweet_from_openai(client, topic: str) -> str:
    """Fetch a tweet using only gpt-4o-mini."""
    max_tok = int(os.getenv("OPENAI_MAX_COMPLETION_TOKENS", "120"))
    prompt = build_user_prompt(topic)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=prompt,
                max_output_tokens=max_tok,
            )
            raw = resp.output_text
            log_raw_reply(raw, topic)
            if raw and raw.strip():
                return raw.strip()
        except Exception as e:
            log.warning(f"Attempt {attempt+1} failed: {e}")
            last_error = e
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Model failed after {MAX_RETRIES} retries: {last_error}")

# ------------- Moderation -------------
def passes_moderation(client, text: str) -> bool:
    if not USE_MODERATION:
        return True
    try:
        resp = client.moderations.create(model="omni-moderation-latest", input=text)
        return not any(r.flagged for r in resp.results)
    except Exception as e:
        log.warning(f"Moderation API failed (fail-open): {e}")
        return True  # ✅ Fail-open

# ------------- Tweet Posting -------------
def generate_and_post():
    client = openai_client()
    twitter = twitter_client()

    topic = random.choice(TOPICS)
    log.info(f"Selected topic: {topic}")

    for attempt in range(MAX_RETRIES):
        try:
            text = fetch_tweet_from_openai(client, topic)
            if not text:
                raise ValueError("Empty tweet from model")

            if not length_ok(text):
                raise ValueError("Tweet too long")

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
                log.info(f"Tweet posted: https://twitter.com/i/web/status/{resp.data['id']}")
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
