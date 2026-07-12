"""
Automated Instagram -> Notion content pipeline.

Flow (per reel URL):
    1. yt-dlp downloads audio into ./tmp_media
    2. Groq (Whisper large v3) transcribes the audio in Urdu
    3. Gemini transliterates the Urdu into strictly formatted Roman Urdu
    4. Notion page is created (Creator, URL, Roman Urdu transcript)
    5. Media file is permanently deleted (zero retention)

Secrets come exclusively from environment variables:
    GEMINI_API_KEY, GROQ_API_KEY, NOTION_DATABASE_ID, NOTION_API_KEY
"""

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

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

BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BASE_DIR / "tmp_media"
LOG_DIR = BASE_DIR / "logs"
REELS_FILE = BASE_DIR / "reels.txt"

GROQ_WHISPER_MODEL = "whisper-large-v3"
GEMINI_MODEL = "gemini-2.5-flash"

# Notion caps a single rich_text element at 2000 characters.
NOTION_TEXT_CHUNK = 2000

TRANSLITERATION_PROMPT = """You are an expert Urdu-to-Roman-Urdu transliterator.

Convert the following Urdu text into Roman Urdu. Follow EVERY rule below strictly:

1. Output ONLY the Roman Urdu text. No timestamps, no parentheses, no brackets,
   no headings, no explanations, no notes of any kind.
2. Break the text into short paragraphs: add a newline after each sentence and a
   blank line between paragraphs so the result is easy to read.
3. Always use informal second-person pronouns: 'tum', 'tumharay', 'tumhay',
   'tumhari', 'tumhara'.
4. NEVER use formal pronouns or their derivatives: 'aap', 'apkay', 'apki',
   'apka' are strictly forbidden. Rewrite any formal address into the informal
   'tum' form, adjusting verbs to match (e.g. 'aap karein' -> 'tum karo').

Urdu text:
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
# Pipeline steps
# ---------------------------------------------------------------------------

def load_reel_urls() -> list[str]:
    """Read the predefined list of Instagram Reel URLs from reels.txt."""
    if not REELS_FILE.exists():
        log.error("reels.txt not found — nothing to process.")
        return []
    urls = [
        line.strip()
        for line in REELS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    log.info("Loaded %d reel URL(s) from reels.txt", len(urls))
    return urls


def download_audio(url: str) -> tuple[Path, str]:
    """Download the reel's audio into tmp_media. Returns (file_path, creator)."""
    MEDIA_DIR.mkdir(exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(MEDIA_DIR / "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    creator = (
        info.get("uploader")
        or info.get("channel")
        or info.get("uploader_id")
        or "Unknown Creator"
    )
    audio_path = MEDIA_DIR / f"{info['id']}.mp3"
    if not audio_path.exists():
        raise FileNotFoundError(f"Expected audio file missing: {audio_path}")
    log.info("Downloaded audio for %s (creator: %s)", url, creator)
    return audio_path, creator


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


def transliterate_to_roman_urdu(urdu_text: str) -> str:
    """Convert Urdu text to strictly formatted Roman Urdu via Gemini."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=TRANSLITERATION_PROMPT.format(urdu_text=urdu_text),
    )
    roman = (response.text or "").strip()
    if not roman:
        raise ValueError("Gemini returned an empty transliteration.")
    log.info("Transliteration complete (%d chars)", len(roman))
    return roman


def _chunk_text(text: str, size: int = NOTION_TEXT_CHUNK) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def already_in_notion(notion: NotionClient, url: str) -> bool:
    """Skip reels that already have a page (keeps scheduled runs idempotent)."""
    result = notion.databases.query(
        database_id=NOTION_DATABASE_ID,
        filter={"property": "Instagram URL", "url": {"equals": url}},
        page_size=1,
    )
    return len(result.get("results", [])) > 0


def sync_to_notion(creator: str, url: str, roman_transcript: str) -> None:
    """Create the Notion page — the single source of truth for this pipeline."""
    notion = NotionClient(auth=NOTION_API_KEY)

    paragraphs = []
    for block in roman_transcript.split("\n\n"):
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
            "Creator": {"title": [{"text": {"content": creator}}]},
            "Instagram URL": {"url": url},
        },
        children=paragraphs,
    )
    log.info("Notion page created for %s", url)


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

def process_reel(url: str, notion: NotionClient) -> bool:
    audio_path: Path | None = None
    try:
        if already_in_notion(notion, url):
            log.info("Skipping (already in Notion): %s", url)
            return True

        audio_path, creator = download_audio(url)
        urdu_text = transcribe_urdu(audio_path)
        roman_text = transliterate_to_roman_urdu(urdu_text)
        sync_to_notion(creator, url, roman_text)

        # Delete ONLY after a successful Notion response — zero retention.
        delete_media(audio_path)
        return True
    except Exception as exc:  # noqa: BLE001 — one bad reel must not kill the run
        log.error("Failed to process %s: %s", url, exc)
        # Even on failure, never leave media behind on a shared runner.
        if audio_path is not None:
            delete_media(audio_path)
        return False


def validate_environment() -> bool:
    missing = [
        name
        for name, value in {
            "GEMINI_API_KEY": GEMINI_API_KEY,
            "GROQ_API_KEY": GROQ_API_KEY,
            "NOTION_DATABASE_ID": NOTION_DATABASE_ID,
            "NOTION_API_KEY": NOTION_API_KEY,
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

    urls = load_reel_urls()
    if not urls:
        log.warning("No reel URLs to process. Exiting.")
        return 0

    notion = NotionClient(auth=NOTION_API_KEY)
    ok = sum(process_reel(url, notion) for url in urls)
    failed = len(urls) - ok
    log.info("=== Run finished: %d succeeded, %d failed ===", ok, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
