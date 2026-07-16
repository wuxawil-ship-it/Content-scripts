# IG → Notion Pipeline

Zero-retention pipeline: monitors a list of Instagram creators via Apify
(residential proxies — no Instagram cookies or login needed), downloads each
creator's latest video directly from the CDN `videoUrl` Apify returns,
transcribes the audio to Urdu (Groq Whisper), rewrites it as a condensed
Roman Urdu script (Gemini — video link on top, 4-line hook, then script
body), and syncs the result to a Notion database. Media files are deleted
immediately after a successful Notion sync.

## Setup

1. Add Instagram usernames to `creators.txt` (one per line, no `@`).
2. In your Notion database, create these properties:
   - **Creator** — Title (holds the creator's username)
   - **Instagram URL** — URL (holds the video link)
3. Add these repository secrets (Settings → Secrets and variables → Actions):
   `GEMINI_API_KEY`, `GROQ_API_KEY`, `NOTION_API_KEY`, `NOTION_DATABASE_ID`,
   `APIFY_API_TOKEN`
4. The workflow runs once daily at 6:00 PM Pakistan time (13:00 UTC), or
   trigger it manually from the Actions tab.

## Local testing

```bash
cp .env.example .env   # fill in your keys
pip install -r requirements.txt
python main.py
```

## Notes

- Currently processes only the **1 latest video** per creator per run
  (`LATEST_VIDEOS_PER_CREATOR` in `main.py`).
- Runs are idempotent: videos already present in Notion (matched by URL) are
  skipped, so only genuinely new uploads are processed each run.
- Logs rotate daily with a 1-day history in `logs/` (git-ignored).
- Apify's free plan includes $5/month of platform credit; the Instagram
  scraper bills per dataset result.
