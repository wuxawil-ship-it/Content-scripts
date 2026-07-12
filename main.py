"""
Automated Instagram -> Notion content pipeline (profile monitoring).

Flow (per creator in creators.txt):
    1. Apify (instagram-scraper actor) lists the latest 3 videos on the profile
    2. Videos already present in Notion are skipped (duplicate check by URL)
    3. yt-dlp downloads each new video's audio into ./tmp_media
    4. Groq (Whisper large v3) transcribes the audio in Urdu
    5. Gemini rewrites it as a condensed Roman Urdu script
       (link on top + 4-line hook + script body)
    6. Notion page is created (Creator username, video URL, full script)
    7. Media file is permanently deleted (zero retention)

Secrets come exclusively from environment variables:
    GEMINI_API_KEY, GROQ_API_KEY, NOTION_DATABASE_ID, NOTION_API_KEY
"""

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from apify_client import ApifyClient
from dotenv import load_dotenv
from google import genai
from groq import Groq
from notion_client import Client as NotionClient
import yt_dlp

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()  # no-op on GitHub Actions; enables local .env testing

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
IG_COOKIES = os.getenv("IG_COOKIES")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")

BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BASE_DIR / "tmp_media"
LOG_DIR = BASE_DIR / "logs"
CREATORS_FILE = BASE_DIR / "creators.txt"
COOKIES_FILE = BASE_DIR / "cookies.txt"

GROQ_WHISPER_MODEL = "whisper-large-v3"
GEMINI_MODEL = "gemini-2.5-flash"
APIFY_INSTAGRAM_ACTOR = "apify/instagram-scraper"

LATEST_VIDEOS_PER_CREATOR = 3

# Notion caps a single rich_text element at 2000 characters.
NOTION_TEXT_CHUNK = 2000

SCRIPT_PROMPT = """You are an expert scriptwriter who converts raw Urdu transcripts into polished, ready-to-read Roman Urdu scripts.

Video link: {video_url}

Rewrite the transcript below into a final script, following EVERY rule strictly:

FORMAT STRUCTURE — the output must be exactly three parts, in this order:
1. The original IG video link ({video_url}) alone at the very top.
2. A brief, catchy 4-line introductory hook summarizing the core value of the video.
3. The main script body.
Output ONLY these three parts — no headings, no labels, no explanations, no notes.

LANGUAGE & TONE:
- Write exclusively in Roman Urdu. Do NOT mix in Hindi words.
- Maintain a friendly yet authoritative tone.

VOCABULARY RULES:
- Strictly use informal pronouns: 'tum', 'tumharay', 'tumhay', 'tumhari', 'tumhara'.
- The formal pronouns 'aap', 'apkay', 'apki', 'apka' are absolutely FORBIDDEN.
  Rewrite any formal address into the informal 'tum' form, adjusting verbs to
  match (e.g. 'aap karein' -> 'tum karo').

CONTENT CONSTRAINTS:
- Remove standard regional filler phrases and generic terms; replace them with
  high-conversion alternative hooks.
- Edit and condense the raw transcript so the final script takes a maximum of
  1 to 2 minutes to read aloud (approximately 150-250 words). Do NOT exceed
  this limit.

FORMATTING RESTRICTIONS:
- No timestamps. No parentheses.
- Ensure clear readability: add a newline after every sentence and a blank
  line between paragraphs.

END CONSTRAINT:
- Completely exclude any automated captions, emojis, or hashtags from the end
  of the script.

Urdu transcript:
{urdu_text}
"""


# ---------------------------------------------------------------------------
# Logging: 1-day rotating history
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("ig_pipeline")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = TimedRotatingFileHandler(
        LOG_DIR / "pipeline.log",
        when="D",
        interval=1,
        backupCount=1,  # keep exactly one day of history
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


log = setup_logging()


# ---------------------------------------------------------------------------
# Instagram authentication cookies
# ---------------------------------------------------------------------------

def write_cookies_file() -> bool:
    """Write the IG_COOKIES secret to cookies.txt for yt-dlp authentication."""
    if not IG_COOKIES:
        log.warning(
            "IG_COOKIES is not set — yt-dlp will run unauthenticated and may "
            "hit 429 errors."
        )
        return False
    COOKIES_FILE.write_text(IG_COOKIES, encoding="utf-8")
    log.info("Wrote Instagram cookies to %s", COOKIES_FILE.name)
    return True


def delete_cookies_file() -> None:
    """Remove cookies.txt so sensitive cookies never linger on the runner."""
    try:
        COOKIES_FILE.unlink(missing_ok=True)
        log.info("Deleted cookies file.")
    except OSError as exc:
        log.error("Failed to delete cookies file: %s", exc)


def _ydl_opts(**extra) -> dict:
    """Base yt-dlp options, including the cookie file when available."""
    opts = {"quiet": True, **extra}
    if COOKIES_FILE.exists():
        opts["cookiefile"] = str(COOKIES_FILE)
    return opts


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def load_creators() -> list[str]:
    """Read the list of Instagram usernames from creators.txt."""
    if not CREATORS_FILE.exists():
        log.error("creators.txt not found — nothing to process.")
        return []
    creators = [
        line.strip().lstrip("@")
        for line in CREATORS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    log.info("Loaded %d creator(s) from creators.txt", len(creators))
    return creators


def get_latest_video_urls(username: str, limit: int = LATEST_VIDEOS_PER_CREATOR) -> list[str]:
    """List the latest `limit` video URLs from a creator's profile via Apify.

    GitHub Actions IPs are blocked by Instagram (403 even with cookies), so
    profile listing runs through Apify's instagram-scraper actor, which uses
    residential proxies. yt-dlp is still used to download the individual
    video URLs returned here.
    """
    client = ApifyClient(APIFY_API_TOKEN)
    run = client.actor(APIFY_INSTAGRAM_ACTOR).call(
        run_input={
            "directUrls": [f"https://www.instagram.com/{username}/"],
            "resultsType": "posts",
            "resultsLimit": limit,  # only the absolute latest posts — conserves Apify credits
            "addParentData": False,
        }
    )

    urls: list[str] = []
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        # Actor marks post types as "Video" / "Image" / "Sidecar".
        if item.get("type") != "Video":
            continue
        url = item.get("url")
        if not url and item.get("shortCode"):
            url = f"https://www.instagram.com/p/{item['shortCode']}/"
        if url:
            urls.append(url)
            if len(urls) >= limit:
                break

    log.info("Found %d latest video(s) for @%s", len(urls), username)
    return urls


def download_audio(url: str) -> Path:
    """Download the video's audio into tmp_media. Returns the file path."""
    MEDIA_DIR.mkdir(exist_ok=True)
    ydl_opts = _ydl_opts(
        format="bestaudio/best",
        outtmpl=str(MEDIA_DIR / "%(id)s.%(ext)s"),
        postprocessors=[
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
        noplaylist=True,
    )
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    audio_path = MEDIA_DIR / f"{info['id']}.mp3"
    if not audio_path.exists():
        raise FileNotFoundError(f"Expected audio file missing: {audio_path}")
    log.info("Downloaded audio for %s", url)
    return audio_path


def transcribe_urdu(audio_path: Path) -> str:
    """Transcribe the audio to Urdu text via Groq's Whisper large model."""
    client = Groq(api_key=GROQ_API_KEY)
    with open(audio_path, "rb") as fh:
        result = client.audio.transcriptions.create(
            file=(audio_path.name, fh.read()),
            model=GROQ_WHISPER_MODEL,
            language="ur",
            response_format="text",
        )
    text = result if isinstance(result, str) else result.text
    text = text.strip()
    if not text:
        raise ValueError("Groq returned an empty transcription.")
    log.info("Transcription complete (%d chars)", len(text))
    return text


def generate_roman_urdu_script(urdu_text: str, video_url: str) -> str:
    """Rewrite the Urdu transcript as a structured Roman Urdu script via Gemini."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=SCRIPT_PROMPT.format(video_url=video_url, urdu_text=urdu_text),
    )
    script = (response.text or "").strip()
    if not script:
        raise ValueError("Gemini returned an empty script.")
    log.info("Script generated (%d chars)", len(script))
    return script


def _chunk_text(text: str, size: int = NOTION_TEXT_CHUNK) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def already_in_notion(notion: NotionClient, url: str) -> bool:
    """Skip videos that already have a page (keeps scheduled runs idempotent)."""
    result = notion.databases.query(
        database_id=NOTION_DATABASE_ID,
        filter={"property": "Instagram URL", "url": {"equals": url}},
        page_size=1,
    )
    return len(result.get("results", [])) > 0


def sync_to_notion(notion: NotionClient, username: str, url: str, script: str) -> None:
    """Create the Notion page — the single source of truth for this pipeline.

    Title = creator username, URL property = video link, and the entire Gemini
    response (link + 4-line hook + script body) becomes the page body.
    """
    paragraphs = []
    for block in script.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        for chunk in _chunk_text(block):
            paragraphs.append(
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": chunk}}]
                    },
                }
            )

    notion.pages.create(
        parent={"database_id": NOTION_DATABASE_ID},
        properties={
            "Creator": {"title": [{"text": {"content": username}}]},
            "Instagram URL": {"url": url},
        },
        children=paragraphs,
    )
    log.info("Notion page created for %s (@%s)", url, username)


def delete_media(audio_path: Path) -> None:
    """Zero-retention cleanup: permanently remove the media file."""
    try:
        audio_path.unlink(missing_ok=True)
        log.info("Deleted media file: %s", audio_path.name)
    except OSError as exc:
        log.error("Failed to delete %s: %s", audio_path, exc)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_video(notion: NotionClient, username: str, url: str) -> bool:
    audio_path: Path | None = None
    try:
        if already_in_notion(notion, url):
            log.info("Skipping (already in Notion): %s", url)
            return True

        audio_path = download_audio(url)
        urdu_text = transcribe_urdu(audio_path)
        script = generate_roman_urdu_script(urdu_text, url)
        sync_to_notion(notion, username, url, script)

        # Delete ONLY after a successful Notion response — zero retention.
        delete_media(audio_path)
        return True
    except Exception as exc:  # noqa: BLE001 — one bad video must not kill the run
        log.error("Failed to process %s (@%s): %s", url, username, exc)
        # Even on failure, never leave media behind on a shared runner.
        if audio_path is not None:
            delete_media(audio_path)
        return False


def process_creator(notion: NotionClient, username: str) -> tuple[int, int]:
    """Process a creator's latest videos. Returns (succeeded, failed)."""
    try:
        urls = get_latest_video_urls(username)
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to list videos for @%s: %s", username, exc)
        return 0, 1

    ok = sum(process_video(notion, username, url) for url in urls)
    return ok, len(urls) - ok


def validate_environment() -> bool:
    missing = [
        name
        for name, value in {
            "GEMINI_API_KEY": GEMINI_API_KEY,
            "GROQ_API_KEY": GROQ_API_KEY,
            "NOTION_DATABASE_ID": NOTION_DATABASE_ID,
            "NOTION_API_KEY": NOTION_API_KEY,
            "APIFY_API_TOKEN": APIFY_API_TOKEN,
        }.items()
        if not value
    ]
    if missing:
        log.error("Missing required environment variables: %s", ", ".join(missing))
        return False
    return True


def main() -> int:
    log.info("=== Instagram -> Notion pipeline starting ===")
    if not validate_environment():
        return 1

    creators = load_creators()
    if not creators:
        log.warning("No creators to process. Exiting.")
        return 0

    write_cookies_file()
    try:
        notion = NotionClient(auth=NOTION_API_KEY)
        total_ok = total_failed = 0
        for username in creators:
            ok, failed = process_creator(notion, username)
            total_ok += ok
            total_failed += failed

        log.info(
            "=== Run finished: %d succeeded, %d failed ===", total_ok, total_failed
        )
        return 1 if total_failed else 0
    finally:
        # Sensitive cookies must never linger on the runner.
        delete_cookies_file()


if __name__ == "__main__":
    sys.exit(main())
