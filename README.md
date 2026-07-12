# IG → Notion Pipeline

Zero-cost, zero-retention pipeline: monitors a list of Instagram creators,
pulls each creator's latest 3 videos, transcribes the audio to Urdu (Groq
Whisper), rewrites it as a condensed Roman Urdu script (Gemini — video link on
top, 4-line hook, then script body), and syncs the result to a Notion
database. Media files are deleted immediately after a successful Notion sync.

## Setup

1. Add Instagram usernames to `creators.txt` (one per line, no `@`).
2. In your Notion database, create these properties:
   - **Creator** — Title (holds the creator's username)
   - **Instagram URL** — URL (holds the video link)
3. Add these repository secrets (Settings → Secrets and variables → Actions):
   `GEMINI_API_KEY`, `GROQ_API_KEY`, `NOTION_API_KEY`, `NOTION_DATABASE_ID`
4. The workflow runs every 4 hours, or trigger it manually from the Actions tab.

## Local testing

```bash
cp .env.example .env   # fill in your keys
pip install -r requirements.txt
python main.py
```

Requires `ffmpeg` on PATH for yt-dlp audio extraction.

## Notes

- Runs are idempotent: videos already present in Notion (matched by URL) are
  skipped, so only genuinely new uploads are processed each run.
- Logs rotate daily with a 1-day history in `logs/` (git-ignored).
- Instagram may require authentication for profile listing or downloads; if
  runs fail with login errors, you may need to supply a cookies file to yt-dlp.
