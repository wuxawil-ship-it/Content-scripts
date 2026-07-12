"""
Automated Instagram -> Notion content pipeline (profile monitoring).

Flow (per creator in creators.txt):
    1. Apify (instagram-scraper actor) returns the latest video post,
       including a direct CDN videoUrl (no Instagram frontend involved)
    2. Videos already present in Notion are skipped (duplicate check by URL)
    3. The video is downloaded straight from the Apify videoUrl into
       ./tmp_media via plain HTTP (requests)
    4. Groq (Whisper large v3) transcribes the audio in Urdu
    5. Gemini rewrites it as a condensed Roman Urdu script
       (link on top + 4-line hook + script body)
    6. Notion page is created (Creator username, video URL, full script)
    7. Media file is permanently deleted (zero retention)

Secrets come exclusively from environment variables:
    GEMINI_API_KEY, GROQ_API_KEY, NOTION_DATABASE_ID, NOTION_API_KEY,
    APIFY_API_TOKEN
"""

import logging
import os
import re
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import requests
from apify_client import ApifyClient
from dotenv import load_dotenv
from google import genai
from groq import Groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()  # no-op on GitHub Actions; enables local .env testing

def _env(name: str) -> str:
    """Read an env var, stripping stray whitespace/newlines that break URLs
    and auth headers (e.g. a trailing newline pasted into a GitHub secret)."""
    return (os.getenv(name) or "").strip()


def _clean_database_id(raw: str) -> str:
    """Extract a bare Notion database ID from whatever was pasted.

    Accepts a plain 32-hex ID, a hyphenated UUID, or a full Notion URL like
    https://www.notion.so/Workspace/Name-<id>?v=<view>. Notion rejects
    anything else with "Invalid request URL".
    """
    match = re.search(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
        r"|[0-9a-fA-F]{32}",
        raw,
    )
    return match.group(0) if match else raw


GEMINI_API_KEY = _env("GEMINI_API_KEY")
GROQ_API_KEY = _env("GROQ_API_KEY")
NOTION_DATABASE_ID = _clean_database_id(_env("NOTION_DATABASE_ID"))
NOTION_API_KEY = _env("NOTION_API_KEY")
APIFY_API_TOKEN = _env("APIFY_API_TOKEN")

BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BASE_DIR / "tmp_media"
LOG_DIR = BASE_DIR / "logs"
CREATORS_FILE = BASE_DIR / "creators.txt"

GROQ_WHISPER_MODEL = "whisper-large-v3"
GEMINI_MODEL = "gemini-2.5-flash"
APIFY_INSTAGRAM_ACTOR = "apify/instagram-scraper"

# Notion is called directly over HTTP with an explicit URL and pinned API
# version — SDK versions kept changing endpoint paths/behavior under us.
NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_API_VERSION = "2022-06-28"

# Process only the single latest video per creator for now (may increase later).
LATEST_VIDEOS_PER_CREATOR = 1

DOWNLOAD_TIMEOUT_SECONDS = 120

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


def get_latest_videos(username: str, limit: int = LATEST_VIDEOS_PER_CREATOR) -> list[dict]:
    """List the latest `limit` videos from a creator's profile via Apify.

    GitHub Actions IPs are blocked by Instagram, so scraping runs through
    Apify's instagram-scraper actor (residential proxies). Each returned dict
    has: url (post page), video_url (direct CDN media), shortcode.
    """
    client = ApifyClient(APIFY_API_TOKEN)
    run = client.actor(APIFY_INSTAGRAM_ACTOR).call(
        run_input={
            "directUrls": [f"https://www.instagram.com/{username}/"],
            "resultsType": "posts",
            "resultsLimit": limit,  # only the absolute latest posts — conserves Apify credits
            "addParentData": False,
        },
        logger=None,  # don't stream actor logs — keeps CI output readable
    )

    if run is None:
        raise RuntimeError(f"Apify actor run failed for @{username}")

    videos: list[dict] = []
    # apify-client returns a Run object — access fields as attributes.
    for item in client.dataset(run.default_dataset_id).iterate_items():
        # Actor marks post types as "Video" / "Image" / "Sidecar".
        if item.get("type") != "Video":
            continue
        shortcode = item.get("shortCode", "")
        url = item.get("url") or (
            f"https://www.instagram.com/p/{shortcode}/" if shortcode else None
        )
        video_url = item.get("videoUrl")
        if not url:
            continue
        if not video_url:
            log.warning("No videoUrl in Apify result for %s — skipping.", url)
            continue
        videos.append({"url": url, "video_url": video_url, "shortcode": shortcode})
        if len(videos) >= limit:
            break

    log.info("Found %d latest video(s) for @%s", len(videos), username)
    return videos


def download_video(video_url: str, shortcode: str) -> Path:
    """Download the video straight from the Apify-provided CDN URL."""
    MEDIA_DIR.mkdir(exist_ok=True)
    media_path = MEDIA_DIR / f"{shortcode or 'video'}.mp4"
    with requests.get(
        video_url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS
    ) as response:
        response.raise_for_status()
        with open(media_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                fh.write(chunk)

    if media_path.stat().st_size == 0:
        raise ValueError(f"Downloaded file is empty: {media_path}")
    log.info(
        "Downloaded video %s (%.1f MB)",
        media_path.name,
        media_path.stat().st_size / 1_048_576,
    )
    return media_path


def transcribe_urdu(media_path: Path) -> str:
    """Transcribe the media's audio to Urdu text via Groq's Whisper large model.

    Groq accepts mp4 directly, so the raw video file is uploaded as-is.
    """
    client = Groq(api_key=GROQ_API_KEY)
    with open(media_path, "rb") as fh:
        result = client.audio.transcriptions.create(
            file=(media_path.name, fh.read()),
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


def notion_request(method: str, path: str, payload: dict | None = None) -> dict:
    """Call the Notion REST API directly with a full, explicit URL."""
    response = requests.request(
        method,
        f"{NOTION_API_BASE}/{path}",
        headers={
            "Authorization": f"Bearer {NOTION_API_KEY}",
            "Notion-Version": NOTION_API_VERSION,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(
            f"Notion API {method} /{path.split('/')[0]} failed "
            f"({response.status_code}): {response.text}"
        )
    return response.json()


def prepare_database() -> str | None:
    """Verify the Notion database is reachable and fit for the pipeline.

    Returns the name of the database's title property (schemas differ — e.g.
    "Name" vs "Creator"), and creates the "Instagram URL" url property if the
    database doesn't have one yet. Returns None if the database can't be
    accessed, logging Notion's exact error once instead of per-video.
    """
    try:
        db = notion_request("GET", f"databases/{NOTION_DATABASE_ID}")
    except Exception as exc:  # noqa: BLE001
        log.error(
            "Cannot access Notion database (ID starts %r): %s — check that "
            "NOTION_DATABASE_ID is the database ID and the integration is "
            "connected to it (database ... menu -> Connections).",
            NOTION_DATABASE_ID[:8],
            exc,
        )
        return None

    props = db.get("properties", {})
    title_prop = next(
        (name for name, p in props.items() if p.get("type") == "title"), "Name"
    )

    if "Instagram URL" not in props:
        notion_request(
            "PATCH",
            f"databases/{NOTION_DATABASE_ID}",
            {"properties": {"Instagram URL": {"url": {}}}},
        )
        log.info('Added missing "Instagram URL" property to the database.')

    log.info("Notion database OK (title property: %r).", title_prop)
    return title_prop


def already_in_notion(url: str) -> bool:
    """Skip videos that already have a page (keeps scheduled runs idempotent)."""
    response = notion_request(
        "POST",
        f"databases/{NOTION_DATABASE_ID}/query",
        {
            "filter": {"property": "Instagram URL", "url": {"equals": url}},
            "page_size": 1,
        },
    )
    return len(response.get("results", [])) > 0


def sync_to_notion(title_prop: str, username: str, url: str, script: str) -> None:
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

    notion_request(
        "POST",
        "pages",
        {
            "parent": {"database_id": NOTION_DATABASE_ID},
            "properties": {
                title_prop: {"title": [{"text": {"content": username}}]},
                "Instagram URL": {"url": url},
            },
            "children": paragraphs,
        },
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

def process_video(title_prop: str, username: str, video: dict) -> bool:
    url = video["url"]
    media_path: Path | None = None
    try:
        if already_in_notion(url):
            log.info("Skipping (already in Notion): %s", url)
            return True

        media_path = download_video(video["video_url"], video["shortcode"])
        urdu_text = transcribe_urdu(media_path)
        script = generate_roman_urdu_script(urdu_text, url)
        sync_to_notion(title_prop, username, url, script)

        # Delete ONLY after a successful Notion response — zero retention.
        delete_media(media_path)
        return True
    except Exception as exc:  # noqa: BLE001 — one bad video must not kill the run
        log.error("Failed to process %s (@%s): %s", url, username, exc)
        # Even on failure, never leave media behind on a shared runner.
        if media_path is not None:
            delete_media(media_path)
        return False


def process_creator(title_prop: str, username: str) -> tuple[int, int]:
    """Process a creator's latest videos. Returns (succeeded, failed)."""
    try:
        videos = get_latest_videos(username)
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to list videos for @%s: %s", username, exc)
        return 0, 1

    ok = sum(process_video(title_prop, username, video) for video in videos)
    return ok, len(videos) - ok


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

    if not re.fullmatch(r"[0-9a-fA-F]{32}|[0-9a-fA-F-]{36}", NOTION_DATABASE_ID):
        log.error(
            "NOTION_DATABASE_ID does not look like a Notion database ID "
            "(got %d chars starting with %r). Paste the 32-character ID from "
            "the database URL: notion.so/<workspace>/<Name>-<ID>?v=...",
            len(NOTION_DATABASE_ID),
            NOTION_DATABASE_ID[:4],
        )
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

    title_prop = prepare_database()
    if title_prop is None:
        return 1

    total_ok = total_failed = 0
    for username in creators:
        ok, failed = process_creator(title_prop, username)
        total_ok += ok
        total_failed += failed

    log.info(
        "=== Run finished: %d succeeded, %d failed ===", total_ok, total_failed
    )
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
