# Tweet Bot 🤖🐦

A simple Python bot that posts **two tweets a day** (default: 09:15 and 17:15 IST).  
It uses [OpenAI’s Responses API](https://platform.openai.com/docs/guides/responses) to generate tweet text, runs validation (length, profanity, PII, moderation), and posts via the Twitter/X API.

## Features
- Posts **2x daily** via GitHub Actions scheduler  
- Uses **OpenAI model** (e.g. `gpt-5-nano`) with an optional **fallback model** (e.g. `gpt-4o-mini`)  
- Validates tweets:  
  - ≤280 characters  
  - No profanity, PII, or unsafe text  
  - Optional OpenAI moderation check  
- **DRY_RUN mode** for safe local testing  
- Configurable topics, posting times, and logging level via GitHub Variables  

---

## Setup (your own bot)

### 1. Fork this repo
Click **Fork** on GitHub to make your own copy.

### 2. Get API keys
- **Twitter/X API**: Create a developer app at [developer.x.com](https://developer.x.com/).  
  - Enable **OAuth 1.0a** with **Read and Write** permissions.  
  - Generate:
    - API Key (`TWITTER_API_KEY`)  
    - API Secret (`TWITTER_API_SECRET`)  
    - Access Token (`TWITTER_ACCESS_TOKEN`)  
    - Access Secret (`TWITTER_ACCESS_TOKEN_SECRET`)  

- **OpenAI API**: Create an API key at [platform.openai.com](https://platform.openai.com/api-keys).  
  - Set as `OPENAI_API_KEY`.  

### 3. Add Secrets
In your repo:  
**Settings → Secrets and variables → Actions → New repository secret**

Add these (required):  
- `OPENAI_API_KEY`  
- `TWITTER_API_KEY`  
- `TWITTER_API_SECRET`  
- `TWITTER_ACCESS_TOKEN`  
- `TWITTER_ACCESS_TOKEN_SECRET`  

### 4. Add Variables
In your repo:  
**Settings → Secrets and variables → Actions → Variables**

Add these (non-secret config):  
- `OPENAI_MODEL = gpt-5-nano`  
- `FALLBACK_MODEL = gpt-4o-mini`  
- `OPENAI_MAX_COMPLETION_TOKENS = 120`  
- `TWEET_TOPICS = productivity,leadership,learning,writing`  
- `TWEET_TIMES_IST = 09:15,17:15`  
- `DRY_RUN = false` (set `true` if you want to test without posting)  
- `LOGLEVEL = INFO`  

### 5. Enable GitHub Actions
- Go to the **Actions** tab of your repo.  
- Enable workflows.  
- The provided workflow (`.github/workflows/tweet.yml`) will trigger twice daily.  

---

## Run locally (optional)
### For testing before deploying:
Create a .env file in the repo root:
`OPENAI_API_KEY=sk-your-openai-key` 
`TWITTER_API_KEY=your-twitter-key` 
`TWITTER_API_SECRET=your-twitter-secret` 
`TWITTER_ACCESS_TOKEN=your-access-token`
`TWITTER_ACCESS_TOKEN_SECRET=your-access-secret`
`DRY_RUN=true`
`RUN_ONCE=true`
`OPENAI_MODEL=gpt-5-nano`
`FALLBACK_MODEL=gpt-4o-mini`


```bash
git clone <your-fork-url>
cd tweet-bot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python twitter_bot.py
