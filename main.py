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
import subprocess
import sys
import time
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
# Tried in order; Google retires model names (gemini-2.5-flash 404s for new
# users), so prefer the rolling "-latest" alias with dated fallbacks.
# GEMINI_MODEL env var, if set, is tried first.
GEMINI_MODELS = [
    m
    for m in (
        os.getenv("GEMINI_MODEL", "").strip(),
        "gemini-flash-latest",
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
    )
    if m
]
APIFY_INSTAGRAM_ACTOR = "apify/instagram-scraper"
# Actor used to find topic images; override with APIFY_IMAGE_ACTOR env var.
# Default is Google Images scraper (input: queries[] + maxResultsPerQuery).
APIFY_IMAGE_ACTOR = os.getenv(
    "APIFY_IMAGE_ACTOR", "hooli/google-images-scraper"
).strip()
MAX_TOPIC_IMAGES = 3

# Below this many transcript characters the video has no usable speech
# (music-only reels etc.) — skip it instead of failing.
MIN_TRANSCRIPT_CHARS = 40

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

SEARCH QUERY (metadata — stripped before publishing):
- After the script, add ONE final line in exactly this format:
  SEARCH_QUERY: <concise English image-search phrase for the video's topic,
  3-6 words, e.g. "crypto market crash chart" or "bitcoin bullish trend">
- This line is machine-read and removed from the script automatically, so it
  does not violate the three-part structure above.

Urdu transcript:
{urdu_text}
"""

ABOUT_PROMPT = """In ONE short line of Roman Urdu (maximum 12 words), state what
this video is about. Output only that single line — no quotes, no emojis, no
hashtags, no trailing punctuation.

Urdu transcript:
{urdu_text}
"""

PAGE_ICON = "🟡"


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
        videos.append(
            {
                "url": url,
                "video_url": video_url,
                "shortcode": shortcode,
                "posted_at": item.get("timestamp"),  # ISO 8601 post time
            }
        )
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


def extract_audio(video_path: Path) -> Path:
    """Convert the video to a small mp3 so uploads stay under Groq's 25 MB cap.

    Falls back to the original video if ffmpeg is unavailable or fails.
    """
    audio_path = video_path.with_suffix(".mp3")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-acodec", "libmp3lame", "-b:a", "96k",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
            timeout=300,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("ffmpeg audio extraction failed (%s) — uploading video as-is.", exc)
        audio_path.unlink(missing_ok=True)
        return video_path

    video_path.unlink(missing_ok=True)
    log.info(
        "Extracted audio %s (%.1f MB)",
        audio_path.name,
        audio_path.stat().st_size / 1_048_576,
    )
    return audio_path


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


def _gemini_generate(prompt: str) -> str:
    """Run a Gemini prompt with resilience.

    Falls through the model list on 404 (retired model). Retries with backoff
    on 503/429 (overload) before moving to the next model, since a different
    model often has spare capacity when one is saturated.
    """
    client = genai.Client(api_key=GEMINI_API_KEY)
    last_error: Exception | None = None
    for model in GEMINI_MODELS:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model, contents=prompt
                )
            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                last_error = exc
                if "404" in message or "NOT_FOUND" in message:
                    log.warning("Gemini model %r unavailable — trying next.", model)
                    break  # next model
                if any(
                    marker in message
                    for marker in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")
                ):
                    wait = 10 * (attempt + 1)
                    log.warning(
                        "Gemini %r overloaded (attempt %d/3) — waiting %ds.",
                        model,
                        attempt + 1,
                        wait,
                    )
                    time.sleep(wait)
                    continue  # retry same model
                raise
            text = (response.text or "").strip()
            if not text:
                raise ValueError("Gemini returned an empty response.")
            return text

    raise RuntimeError("All Gemini models failed.") from last_error


def generate_roman_urdu_script(urdu_text: str, video_url: str) -> tuple[str, str]:
    """Rewrite the Urdu transcript as a structured Roman Urdu script via Gemini.

    Returns (script, search_query). The SEARCH_QUERY metadata line is parsed
    out and stripped so it never appears in the published script.
    """
    raw = _gemini_generate(
        SCRIPT_PROMPT.format(video_url=video_url, urdu_text=urdu_text)
    )

    search_query = ""
    script_lines = []
    for line in raw.splitlines():
        match = re.match(r"\s*SEARCH_QUERY\s*:\s*(.+)", line, re.IGNORECASE)
        if match:
            search_query = match.group(1).strip().strip('"')
        else:
            script_lines.append(line)
    script = "\n".join(script_lines).strip()

    if not script:
        raise ValueError("Gemini returned an empty script.")
    log.info(
        "Script generated (%d chars, search query: %r)", len(script), search_query
    )
    return script, search_query


def generate_about_line(urdu_text: str) -> str:
    """One-line Roman Urdu summary for the About column. Optional — never fatal."""
    try:
        about = _gemini_generate(ABOUT_PROMPT.format(urdu_text=urdu_text))
        return about.splitlines()[0].strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not generate About line: %s", exc)
        return ""


_IMAGE_URL_RE = re.compile(
    r"^https?://\S+\.(?:jpg|jpeg|png|webp|gif)(?:\?\S*)?$", re.IGNORECASE
)
_IMAGE_KEYS = {"imageUrl", "originalImageUrl", "thumbnailUrl", "image"}


def _collect_image_urls(node, found: list[str]) -> None:
    """Recursively pull image-looking URLs out of an Apify dataset item."""
    if isinstance(node, dict):
        for key, value in node.items():
            if (
                isinstance(value, str)
                and value.startswith("http")
                and (key in _IMAGE_KEYS or _IMAGE_URL_RE.match(value))
            ):
                found.append(value)
            else:
                _collect_image_urls(value, found)
    elif isinstance(node, list):
        for value in node:
            _collect_image_urls(value, found)
    elif isinstance(node, str) and _IMAGE_URL_RE.match(node):
        found.append(node)


def fetch_topic_images(search_query: str) -> list[str]:
    """Find up to MAX_TOPIC_IMAGES image URLs for the topic via an Apify actor.

    Best-effort: any failure logs a warning and returns [] so the page is
    still created without visual assets.
    """
    if not search_query:
        return []
    try:
        client = ApifyClient(APIFY_API_TOKEN)
        run = client.actor(APIFY_IMAGE_ACTOR).call(
            run_input={
                "queries": [search_query],
                "maxResultsPerQuery": 10,
            },
            logger=None,
        )
        if run is None:
            raise RuntimeError("image search actor run failed")

        found: list[str] = []
        for item in client.dataset(run.default_dataset_id).iterate_items():
            _collect_image_urls(item, found)
            if len(found) >= MAX_TOPIC_IMAGES * 3:  # enough candidates
                break

        unique = list(dict.fromkeys(found))[:MAX_TOPIC_IMAGES]
        log.info("Found %d topic image(s) for %r", len(unique), search_query)
        return unique
    except Exception as exc:  # noqa: BLE001
        log.warning("Image search failed for %r: %s", search_query, exc)
        return []


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


def prepare_database() -> tuple[str, set[str]] | None:
    """Verify the Notion database is reachable and fit for the pipeline.

    Returns (title property name, set of all property names) — schemas differ,
    and optional columns like About/Date are only filled if they exist.
    Creates the "Instagram URL" url property if missing. Returns None if the
    database can't be accessed, logging Notion's exact error once.
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
        props["Instagram URL"] = {"type": "url"}
        log.info('Added missing "Instagram URL" property to the database.')

    log.info("Notion database OK (title property: %r).", title_prop)
    return title_prop, set(props)


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


def sync_to_notion(
    title_prop: str,
    db_props: set[str],
    username: str,
    video: dict,
    script: str,
    about: str,
    images: list[str],
) -> None:
    """Create the Notion page — the single source of truth for this pipeline.

    Title = creator username, URL property = video link, and the entire Gemini
    response (link + 4-line hook + script body) becomes the page body,
    followed by embedded topic images for editing. About and Date columns are
    filled when the database has them; every page gets the yellow-circle icon.
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

    if images:
        paragraphs.append({"object": "block", "type": "divider", "divider": {}})
        paragraphs.append(
            {
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": "Visual Assets"}}]
                },
            }
        )
        for image_url in images:
            paragraphs.append(
                {
                    "object": "block",
                    "type": "image",
                    "image": {"type": "external", "external": {"url": image_url}},
                }
            )

    properties = {
        title_prop: {"title": [{"text": {"content": username}}]},
        "Instagram URL": {"url": video["url"]},
    }
    if about and "About" in db_props:
        properties["About"] = {
            "rich_text": [{"type": "text", "text": {"content": about[:2000]}}]
        }
    if video.get("posted_at") and "Date" in db_props:
        properties["Date"] = {"date": {"start": video["posted_at"]}}

    notion_request(
        "POST",
        "pages",
        {
            "parent": {"database_id": NOTION_DATABASE_ID},
            "icon": {"type": "emoji", "emoji": PAGE_ICON},
            "properties": properties,
            "children": paragraphs,
        },
    )
    log.info("Notion page created for %s (@%s)", video["url"], username)


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

def process_video(
    title_prop: str, db_props: set[str], username: str, video: dict
) -> bool:
    url = video["url"]
    media_path: Path | None = None
    try:
        if already_in_notion(url):
            log.info("Skipping (already in Notion): %s", url)
            return True

        media_path = download_video(video["video_url"], video["shortcode"])
        media_path = extract_audio(media_path)
        urdu_text = transcribe_urdu(media_path)
        if len(urdu_text) < MIN_TRANSCRIPT_CHARS:
            log.warning(
                "Transcript only %d chars for %s — no usable speech, skipping.",
                len(urdu_text),
                url,
            )
            delete_media(media_path)
            return True

        script, search_query = generate_roman_urdu_script(urdu_text, url)
        about = generate_about_line(urdu_text)
        images = fetch_topic_images(search_query)
        sync_to_notion(title_prop, db_props, username, video, script, about, images)

        # Delete ONLY after a successful Notion response — zero retention.
        delete_media(media_path)
        return True
    except Exception as exc:  # noqa: BLE001 — one bad video must not kill the run
        if media_path is not None:
            delete_media(media_path)
        if "no audio track" in str(exc).lower():
            log.warning("No audio track in %s (@%s) — skipping.", url, username)
            return True
        log.error("Failed to process %s (@%s): %s", url, username, exc)
        return False


def process_creator(
    title_prop: str, db_props: set[str], username: str
) -> tuple[int, int]:
    """Process a creator's latest videos. Returns (succeeded, failed)."""
    try:
        videos = get_latest_videos(username)
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to list videos for @%s: %s", username, exc)
        return 0, 1

    ok = sum(process_video(title_prop, db_props, username, video) for video in videos)
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

    db_info = prepare_database()
    if db_info is None:
        return 1
    title_prop, db_props = db_info

    total_ok = total_failed = 0
    for username in creators:
        ok, failed = process_creator(title_prop, db_props, username)
        total_ok += ok
        total_failed += failed

    log.info(
        "=== Run finished: %d succeeded, %d failed ===", total_ok, total_failed
    )
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
