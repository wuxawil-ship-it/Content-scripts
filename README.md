# IG → Notion Pipeline

Zero-cost, zero-retention pipeline: downloads audio from a predefined list of
Instagram Reels, transcribes it to Urdu (Groq Whisper), converts it to Roman
Urdu (Gemini), and syncs the result to a Notion database. Media files are
deleted immediately after a successful Notion sync.

## Setup

1. Add reel URLs to `reels.txt` (one per line).
2. In your Notion database, create these properties:
   - **Creator** — Title
   - **Instagram URL** — URL
3. Add these repository secrets (Settings → Secrets and variables → Actions):
   `GEMINI_API_KEY`, `GROQ_API_KEY`, `NOTION_API_KEY`, `NOTION_DATABASE_ID`
4. The workflow runs every 6 hours, or trigger it manually from the Actions tab.

## Local testing

```bash
cp .env.example .env   # fill in your keys
pip install -r requirements.txt
python main.py
```

Requires `ffmpeg` on PATH for yt-dlp audio extraction.

## Notes

- Runs are idempotent: reels already present in Notion (matched by URL) are skipped.
- Logs rotate daily with a 1-day history in `logs/` (git-ignored).
- Instagram may require authentication for some downloads; if runs fail with
  login errors, you may need to supply a cookies file to yt-dlp.
