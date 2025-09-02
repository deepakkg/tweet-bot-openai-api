#!/usr/bin/env python3
"""
twitter_bot.py

Features:
- Posts twice daily (default 09:15 and 17:15 Asia/Kolkata) using APScheduler
- Fetches tweet text from OpenAI Responses API
- Enforces: length < 280, no profanity, no PII (emails, phones, Aadhaar, PAN, SSN-like), optional OpenAI moderation
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

# ------------- Prompt (tightened) -------------
SYSTEM_PROMPT = (
    "Write like a real Twitter user: casual, short, human, sometimes witty. "
    "Avoid corporate or essay tone. No hashtags or emojis unless asked. "
    "Vary style: some tweets funny, some thoughtful."
)

def build_user_prompt(topic: str) -> str:
    safe_topic = topic.replace('"', "'")
    return f"Topic: {safe_topic}\nWrite one tweet under 280 characters:"

# ------------- OpenAI fetch -------------
def fetch_tweet_from_openai(client, topic: str) -> str:
    max_tok = int(os.getenv("OPENAI_MAX_COMPLETION_TOKENS", "120"))
    full_prompt = f"{SYSTEM_PROMPT}\n\n{build_user_prompt(topic)}"

    models = [os.getenv("OPENAI_MODEL", "gpt-5-nano").strip()]
    fb = os.getenv("FALLBACK_MODEL", "").strip()
    if fb and fb not in models:
        models.append(fb)

    last_error = None
    for m in models:
        try:
            resp = client.responses.create(
                model=m,
                input=full_prompt,
                max_output_tokens=max_tok,
            )
            raw = resp.output_text
            log_raw_reply(raw, topic)
            if raw and raw.strip():
                return raw.strip()
        except Exception as e:
            log.warning(f"Model {m} failed: {e}")
            last_error = e
            time.sleep(2)

    raise RuntimeError(f"All models failed: {last_error}")

# ---------------- Scheduler / Main logic would continue unchanged ----------------
