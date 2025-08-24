#!/usr/bin/env python3
"""
twitter_bot.py

Features:
- Posts twice daily (default 09:15 and 17:15 Asia/Kolkata) using APScheduler
- Fetches tweet text from OpenAI Chat Completions
- Enforces: length < 280, no profanity, no PII (emails, phones, Aadhaar, PAN, SSN-like), optional OpenAI moderation
- Retries fetching a fresh tweet if validation fails (exponential backoff)
- DRY_RUN mode to print instead of post
- Robust logging

Env Vars required:
  OPENAI_API_KEY=...
  TWITTER_API_KEY=...
  TWITTER_API_SECRET=...
  TWITTER_ACCESS_TOKEN=...
  TWITTER_ACCESS_TOKEN_SECRET=...

Optional:
  DRY_RUN=true
  TWEET_TOPICS="productivity,leadership,learning,writing"
  TWEET_TIMES_IST="09:15,17:15"
  OPENAI_MODEL="gpt-4o-mini"
  MAX_RETRIES="6"
  OPENAI_USE_MODERATION="true"

Install:
  pip install tweepy openai apscheduler pytz regex

Run:
  python twitter_bot.py
"""

import os
import time
import logging
import random
import re
from datetime import datetime
from typing import Optional, Tuple, List

import pytz
import regex  # better unicode word boundaries than re for profanity
import tweepy
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# --- OpenAI (new-style SDK import is "from openai import OpenAI") ---
try:
    from openai import OpenAI
    _OPENAI_SDK = "v1"
except Exception:
    import openai as OpenAI  # fallback if older package is installed
    _OPENAI_SDK = "legacy"

# ------------- Logging -------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("twitter-bot")

# ------------- Config -------------
IST = pytz.timezone("Asia/Kolkata")

DRY_RUN = os.getenv("DRY_RUN", "false").strip().lower() in {"1", "true", "yes", "y"}
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "6"))
USE_MODERATION = os.getenv("OPENAI_USE_MODERATION", "true").strip().lower() in {"1", "true", "yes", "y"}

DEFAULT_TOPICS = "productivity,leadership,learning,writing,craftsmanship,critical thinking,calm focus,habits"
TOPIC_STR = os.getenv("TWEET_TOPICS", DEFAULT_TOPICS)
TOPICS = [t.strip() for t in TOPIC_STR.split(",") if t.strip()]

TWEET_TIMES_IST = os.getenv("TWEET_TIMES_IST", "09:15,17:15")
TIMES = [t.strip() for t in TWEET_TIMES_IST.split(",") if t.strip()]

# ------------- Twitter Client -------------
def twitter_client() -> tweepy.Client:
    api_key = os.environ["TWITTER_API_KEY"]
    api_secret = os.environ["TWITTER_API_SECRET"]
    access_token = os.environ["TWITTER_ACCESS_TOKEN"]
    access_secret = os.environ["TWITTER_ACCESS_TOKEN_SECRET"]

    # Tweepy Client supports OAuth 1.0a user context for posting
    client = tweepy.Client(
        consumer_key=api_key,
        consumer_secret=api_secret,
        access_token=access_token,
        access_token_secret=access_secret
    )
    return client

# ------------- OpenAI Client -------------
def openai_client():
    key = os.environ["OPENAI_API_KEY"]
    if _OPENAI_SDK == "v1":
        return OpenAI(api_key=key)
    else:
        # legacy fallback
        OpenAI.api_key = key
        return OpenAI

# ------------- Validation: profanity + PII -------------
# Minimal-but-solid filters; tune/extend as needed.
PROFANITY_PATTERNS = [
    r"\b(?i)(shit|fuck|bitch|bastard|asshole|dick|cunt|slut|whore|motherf\w+|bullshit)\b",
    r"(?i)\b(nigga|nigger|chink|spic|kike|pussy|retard|faggot)\b",  # hateful slurs (reject outright)
]

# PII regexes (international-ish + India specifics)
PII_PATTERNS = {
    "email": r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
    "phone_generic": r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{4}(?!\d)",
    "aadhaar_india": r"(?<!\d)(\d{4}\s?\d{4}\s?\d{4})(?!\d)",  # 12 digits with or without spaces
    "pan_india": r"(?<![A-Z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Z0-9])",
    "ssn_like": r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)",
    "ip_address": r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)",
    "credit_card": r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)",  # very permissive; will over-catch numbers
}

def contains_profanity(text: str) -> Optional[str]:
    for pat in PROFANITY_PATTERNS:
        if regex.search(pat, text):
            return "profanity"
    return None

def contains_pii(text: str) -> Optional[str]:
    t = text
    for label, pat in PII_PATTERNS.items():
        if re.search(pat, t, flags=re.IGNORECASE):
            return f"PII:{label}"
    return None

def length_ok(text: str) -> bool:
    # Twitter limit 280 characters (codepoints). We'll enforce .__len__().
    return len(text) <= 280

# ------------- OpenAI moderation (optional) -------------
def openai_moderation_flagged(client, text: str) -> bool:
    """
    Returns True if OpenAI moderation flags the text, False otherwise.
    Works with both new SDK (>=1.x) and legacy (<1.x).
    """
    try:
        if _OPENAI_SDK == "v1":
            # New SDK: client.moderations.create(...)
            resp = client.moderations.create(
                model="omni-moderation-latest",
                input=text
            )
            # resp.results is a list of objects; each has .flagged (bool)
            results = getattr(resp, "results", []) or []
            return any(bool(getattr(r, "flagged", False)) for r in results)

        else:
            # Legacy SDK: openai.Moderation.create(...)
            resp = client.Moderation.create(
                model="omni-moderation-latest",
                input=text
            )
            # resp is a dict; resp["results"][0]["flagged"] is a bool
            results = resp.get("results", []) if isinstance(resp, dict) else []
            return any(bool(r.get("flagged", False)) for r in results)

    except Exception as e:
        # Fail-open to avoid blocking posts due to transient API issues
        log.warning(f"Moderation check failed (continuing without): {e}")
        return False

# ------------- Tweet generation -------------
SYSTEM_PROMPT = (
    "You write high-signal, clean, **tweet-length** insights (<=240 chars) "
    "for a broad professional audience. Avoid emojis, hashtags, links, m dashes, n dashes and numbers that look like IDs. "
    "No personal info, no commands to the reader, and no profanity. Keep it crisp."
)

USER_PROMPT_TEMPLATE = (
    "Write one original tweet-length insight (<=240 chars) on the topic: '{topic}'. "
    "Offer a specific idea, not a list. Avoid jargon; keep it useful, upbeat and quotable. "
    "Do not include links, @mentions, hashtags, or quotes."
)

def fetch_tweet_from_openai(client, topic: str) -> str:
    if _OPENAI_SDK == "v1":
        # New OpenAI SDK (>=1.x) requires max_completion_tokens
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(topic=topic)},
            ],
            temperature=0.7,
            max_completion_tokens=120   # <-- patched here
        )
        return resp.choices[0].message.content.strip()
    else:
        # Legacy SDK (<1.x) still uses max_tokens
        resp = client.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(topic=topic)},
            ],
            temperature=0.7,
            max_tokens=120              # <-- keep this for legacy
        )
        return resp["choices"][0]["message"]["content"].strip()

def generate_valid_tweet(max_retries: int = MAX_RETRIES) -> Tuple[str, List[str]]:
    client = openai_client()
    errors: List[str] = []

    delay = 1.5
    for attempt in range(1, max_retries + 1):
        topic = random.choice(TOPICS)
        text = fetch_tweet_from_openai(client, topic)

        # Basic trim/cleanup
        text = text.strip().strip('"').strip()

        # Validation
        if not length_ok(text):
            errors.append(f"Attempt {attempt}: length {len(text)} > 280")
            log.info(errors[-1])
            time.sleep(delay)
            delay = min(delay * 1.8, 12)
            continue

        p = contains_profanity(text)
        if p:
            errors.append(f"Attempt {attempt}: {p}")
            log.info(errors[-1])
            time.sleep(delay)
            delay = min(delay * 1.8, 12)
            continue

        pii = contains_pii(text)
        if pii:
            errors.append(f"Attempt {attempt}: {pii}")
            log.info(errors[-1])
            time.sleep(delay)
            delay = min(delay * 1.8, 12)
            continue

        if USE_MODERATION and openai_moderation_flagged(client, text):
            errors.append(f"Attempt {attempt}: OpenAI moderation flagged")
            log.info(errors[-1])
            time.sleep(delay)
            delay = min(delay * 1.8, 12)
            continue

        return text, errors

    raise RuntimeError(f"Failed to produce a valid tweet after {max_retries} attempts. Issues: {errors}")

# ------------- Posting -------------
def post_tweet(text: str) -> Optional[int]:
    if DRY_RUN:
        log.info(f"[DRY_RUN] Would tweet ({len(text)} chars): {text}")
        return None

    client = twitter_client()
    resp = client.create_tweet(text=text)
    tweet_id = getattr(resp, "data", {}).get("id")
    log.info(f"Tweet posted: https://twitter.com/i/web/status/{tweet_id}")
    return tweet_id

def run_once():
    try:
        text, errs = generate_valid_tweet()
        if errs:
            log.info(f"Previous attempts encountered: {errs}")
        post_tweet(text)
    except Exception as e:
        log.error(f"Run failed: {e}", exc_info=True)

# ------------- Scheduler (twice daily IST) -------------
def schedule_jobs():
    sched = BlockingScheduler(timezone=IST)
    # Add slight random jitter (0–300s) on each run to avoid clock-pattern posting
    for t in TIMES:
        hh, mm = t.split(":")
        trigger = CronTrigger(hour=int(hh), minute=int(mm), second=0)
        def job_wrapper():
            jitter = random.randint(0, 300)
            log.info(f"Trigger fired; sleeping {jitter}s jitter before posting.")
            time.sleep(jitter)
            run_once()
        sched.add_job(job_wrapper, trigger, name=f"tweet_at_{t}_IST")

    log.info(f"Scheduling daily posts at (IST): {', '.join(TIMES)} | DRY_RUN={DRY_RUN}")
    sched.start()

# ------------- CLI -------------
if __name__ == "__main__":
    # If you prefer running once via cron/GitHub Actions, set RUN_ONCE=true in env.
    if os.getenv("RUN_ONCE", "").lower() in {"1", "true", "yes"}:
        run_once()
    else:
        schedule_jobs()
