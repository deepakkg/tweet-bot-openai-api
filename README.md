# Tweet Bot

Python bot that generates tweets with OpenAI, validates them, and posts to X/Twitter.

## What It Does
- Runs on a GitHub Actions schedule (twice daily by default).
- Generates multiple tweet candidates with the OpenAI Responses API.
- Chooses the most novel candidate using embeddings against recent tweet history.
- Applies validation checks (length, profanity, PII, meaningful text, moderation).
- Posts via the Twitter/X API using Tweepy.
- Persists run metadata and candidates to `tweets_log.jsonl`.

## Project Files
- `twitter_bot.py`: bot logic (generation, filtering, novelty selection, posting, scheduler mode).
- `.github/workflows/tweet.yml`: scheduled CI workflow and runtime environment wiring.
- `tweet_state.json`: compact persisted state (last N tweet texts/styles) used for cross-run novelty/style cooldown.
- `tweets_log.jsonl`: per-run JSONL log (uploaded as GitHub Actions artifact).
- `requirements.txt`: Python dependencies.

## How Scheduling Works
This repo supports two run modes:
- GitHub Actions mode: workflow sets `RUN_ONCE=true`, so each workflow run posts once.
- Local scheduler mode: if `RUN_ONCE=false`, `twitter_bot.py` starts APScheduler and uses `TWEET_TIMES_IST`.

Default GitHub Action cron schedule:
- `45 3 * * *` (09:15 IST)
- `45 11 * * *` (17:15 IST)

## Required Secrets
Add these in **Settings -> Secrets and variables -> Actions -> Secrets**:
- `OPENAI_API_KEY`
- `TWITTER_API_KEY`
- `TWITTER_API_SECRET`
- `TWITTER_ACCESS_TOKEN`
- `TWITTER_ACCESS_TOKEN_SECRET`

## Common Variables
Add these in **Settings -> Secrets and variables -> Actions -> Variables**.

Core:
- `OPENAI_MODEL` (default `gpt-4o-mini`)
- `OPENAI_MAX_COMPLETION_TOKENS` (default `120`)
- `OPENAI_USE_MODERATION` (`true`/`false`, default `true`)
- `DRY_RUN` (`true`/`false`, default `false`)
- `MAX_RETRIES` (default `6`)
- `LOGLEVEL` (default `INFO`)

Topics/schedule:
- `TWEET_TOPICS` (comma-separated topics)
- `TWEET_TIMES_IST` (used only when local scheduler mode is enabled)

Novelty/generation:
- `CANDIDATE_COUNT` (default `6`)
- `SIMILARITY_THRESHOLD` (default `0.78`)
- `EMBED_MODEL` (default `text-embedding-3-small`)
- `GEN_TEMPERATURE` (default `0.85`)
- `GEN_TOP_P` (default `0.95`)

Style rotation:
- `STYLE_WEIGHTS` (for example `observational:0.4,micro-story:0.2,contrarian:0.15,question:0.15,tip:0.1`)
- `STYLE_COOLDOWN_ENABLED` (default `true`)
- `STYLE_COOLDOWN_WINDOW` (default `1`)
- `STYLE_PICK_RETRIES` (default `6`)

Logging:
- `TWEET_LOG_PATH` (default `tweets_log.jsonl`)
- `TWEET_STATE_PATH` (default `tweet_state.json`)
- `STATE_RECENT_LIMIT` (default `30`)

## Local Run
```bash
git clone <your-fork-url>
cd twitter-bot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` (example):
```env
OPENAI_API_KEY=sk-your-openai-key
TWITTER_API_KEY=your-twitter-key
TWITTER_API_SECRET=your-twitter-secret
TWITTER_ACCESS_TOKEN=your-access-token
TWITTER_ACCESS_TOKEN_SECRET=your-access-secret
OPENAI_MODEL=gpt-4o-mini
DRY_RUN=true
RUN_ONCE=true
```

Run once:
```bash
python twitter_bot.py
```

Run local scheduler:
```bash
RUN_ONCE=false TWEET_TIMES_IST=09:15,17:15 python twitter_bot.py
```

## Notes
- The bot currently appends `#botWrites` when posting.
- `tweets_log.jsonl` is no longer committed to git; it is uploaded as a GitHub Actions artifact (30-day retention in workflow).
