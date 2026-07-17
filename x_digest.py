"""
Daily X (Twitter) news digest -> Notion.

Runs early morning (Pakistan time) and builds topic-wise news entries:
    1. Apify tweet scraper pulls the last 24h of tweets from the accounts
       listed in x_accounts.txt (crypto + geopolitics news channels)
    2. Gemini groups the tweets into developing-story topics and assigns a
       category tag (e.g. "Geopolitics Update", "Crypto Market")
    3. Each topic becomes ONE Notion entry: Creator = "X — <topic>",
       Tags column = category, About = one-line Roman Urdu summary,
       body = every update chronologically with its images and tweet link
    4. If the same story already has an entry today (developing news), the
       new updates are APPENDED to that entry instead of creating a new one

Reuses the Notion/Gemini helpers from main.py. Extra env var:
    APIFY_X_ACTOR (optional) — tweet scraper actor, default apidojo/tweet-scraper
"""

import json
import re
import sys
from datetime import datetime, timedelta, timezone

from apify_client import ApifyClient

from main import (
    APIFY_API_TOKEN,
    BASE_DIR,
    GEMINI_API_KEY,
    NOTION_API_KEY,
    NOTION_DATABASE_ID,
    _env,
    _gemini_generate,
    log,
    notion_request,
)

X_ACCOUNTS_FILE = BASE_DIR / "x_accounts.txt"
# apidojo actors cap free-plan users at 5 runs/month with 10 items — the
# kaito actor bills per result ($0.18/1k) against normal platform credit.
APIFY_X_ACTOR = (
    _env("APIFY_X_ACTOR")
    or "kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest"
)
MAX_TWEETS = 80
PKT = timezone(timedelta(hours=5))
PAGE_ICON = "🟡"

_TWEET_IMAGE_RE = re.compile(r"https://pbs\.twimg\.com/media/\S+")

GROUPING_PROMPT = """You are a news editor for a crypto content creator.
Below are the last 24 hours of tweets from crypto/geopolitics news accounts,
each with an index number.

Group them into distinct news topics (developing stories). Rules:
- Tweets about the SAME story belong to ONE topic (e.g. a strike on Iran and
  the resulting Strait of Hormuz closure are one topic).
- Skip pure engagement bait, giveaways, ads, or contentless hype — not every
  tweet must be used.
- "title": short English topic title, max 8 words.
- "tag": short category, e.g. "Geopolitics Update", "Crypto Market",
  "Bitcoin", "Altcoins", "Regulation", "Macro Economy".
- "summary": ONE line in Roman Urdu describing the story so far.
- "existing_title": if the story continues one of the existing topics listed
  below, copy that title EXACTLY; otherwise use "".
- "tweet_indices": indices of the tweets in this topic, oldest first.

Existing topics already in today's digest:
{existing}

Tweets:
{tweets}

Output ONLY a valid JSON array, no markdown fences, in this shape:
[{{"title": "...", "tag": "...", "summary": "...", "existing_title": "", "tweet_indices": [0, 3]}}]
"""


# ---------------------------------------------------------------------------
# Fetch tweets
# ---------------------------------------------------------------------------

def load_accounts() -> list[str]:
    if not X_ACCOUNTS_FILE.exists():
        log.error("x_accounts.txt not found — nothing to digest.")
        return []
    handles = [
        line.strip().lstrip("@")
        for line in X_ACCOUNTS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    log.info("Loaded %d X account(s)", len(handles))
    return handles


def _parse_created_at(raw: str) -> datetime | None:
    for fmt in ("%a %b %d %H:%M:%S %z %Y",):
        try:
            return datetime.strptime(raw, fmt)
        except (ValueError, TypeError):
            pass
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _tweet_images(item: dict) -> list[str]:
    urls = _TWEET_IMAGE_RE.findall(json.dumps(item))
    cleaned = [u.rstrip('\\"').rstrip('"') for u in urls]
    return list(dict.fromkeys(cleaned))[:4]


def _actor_input(handles: list[str]) -> dict:
    """Build the actor input for the configured scraper's schema."""
    if "kaitoeasyapi" in APIFY_X_ACTOR:
        query = " OR ".join(f"from:{h}" for h in handles)
        return {
            "twitterContent": f"({query})",
            "queryType": "Latest",
            "maxItems": MAX_TWEETS,
        }
    return {"twitterHandles": handles, "maxItems": MAX_TWEETS, "sort": "Latest"}


def fetch_tweets(handles: list[str]) -> list[dict]:
    """Last 24h of tweets from the given handles via Apify."""
    client = ApifyClient(APIFY_API_TOKEN)
    run = client.actor(APIFY_X_ACTOR).call(
        run_input=_actor_input(handles), logger=None
    )
    if run is None:
        raise RuntimeError("tweet scraper actor run failed to start")
    status = getattr(run, "status", "UNKNOWN")
    if status != "SUCCEEDED":
        raise RuntimeError(f"tweet scraper run ended with status {status}")

    raw_items = list(client.dataset(run.default_dataset_id).iterate_items())
    log.info("Actor returned %d raw item(s)", len(raw_items))

    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    tweets: list[dict] = []
    for item in raw_items:
        text = (item.get("text") or item.get("fullText") or "").strip()
        if not text or text.startswith("RT @"):
            continue
        created = _parse_created_at(item.get("createdAt"))
        if created and created < cutoff:
            continue
        author = (item.get("author") or {}).get("userName") or item.get(
            "username", "unknown"
        )
        tweets.append(
            {
                "author": author,
                "text": text,
                "url": item.get("url") or item.get("twitterUrl") or "",
                "created": created,
                "images": _tweet_images(item),
            }
        )

    tweets.sort(key=lambda t: t["created"] or datetime.now(timezone.utc))
    log.info("Fetched %d usable tweet(s) from last 24h", len(tweets))
    if not tweets and raw_items:
        # Shape mismatch or actor notice items — surface what came back.
        log.warning(
            "No usable tweets despite raw items; first item keys: %s",
            sorted(raw_items[0])[:20],
        )
    return tweets


# ---------------------------------------------------------------------------
# Group into topics
# ---------------------------------------------------------------------------

def todays_x_pages() -> dict[str, str]:
    """Existing 'X — ...' pages created today, as {title: page_id}."""
    today = datetime.now(PKT).date().isoformat()
    response = notion_request(
        "POST",
        f"databases/{NOTION_DATABASE_ID}/query",
        {
            "filter": {
                "and": [
                    {"property": "Creator", "title": {"starts_with": "X — "}},
                    {"property": "Date", "date": {"equals": today}},
                ]
            },
            "page_size": 50,
        },
    )
    pages = {}
    for page in response.get("results", []):
        title_parts = page["properties"]["Creator"]["title"]
        title = "".join(part["plain_text"] for part in title_parts)
        pages[title.removeprefix("X — ")] = page["id"]
    return pages


def group_topics(tweets: list[dict], existing_titles: list[str]) -> list[dict]:
    tweet_lines = "\n".join(
        f"[{i}] @{t['author']}: {t['text'][:300]}" for i, t in enumerate(tweets)
    )
    existing = "\n".join(f"- {t}" for t in existing_titles) or "(none)"
    raw = _gemini_generate(
        GROUPING_PROMPT.format(existing=existing, tweets=tweet_lines)
    )
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        topics = json.loads(raw)
        assert isinstance(topics, list)
    except (json.JSONDecodeError, AssertionError):
        log.warning("Could not parse Gemini topic JSON — using one bucket.")
        topics = [
            {
                "title": "X Updates",
                "tag": "Crypto Market",
                "summary": "Aaj ke X updates",
                "existing_title": "",
                "tweet_indices": list(range(len(tweets))),
            }
        ]
    log.info("Grouped into %d topic(s)", len(topics))
    return topics


# ---------------------------------------------------------------------------
# Notion output
# ---------------------------------------------------------------------------

def ensure_tags_property() -> None:
    db = notion_request("GET", f"databases/{NOTION_DATABASE_ID}")
    if "Tags" not in db.get("properties", {}):
        notion_request(
            "PATCH",
            f"databases/{NOTION_DATABASE_ID}",
            {"properties": {"Tags": {"multi_select": {}}}},
        )
        log.info('Added "Tags" property to the database.')


def _tweet_blocks(tweet: dict) -> list[dict]:
    when = ""
    if tweet["created"]:
        when = tweet["created"].astimezone(PKT).strftime("%d %b %I:%M %p PKT")
    header = f"@{tweet['author']}" + (f" — {when}" if when else "")

    rich_header = [{"type": "text", "text": {"content": header}}]
    if tweet["url"]:
        rich_header.append({"type": "text", "text": {"content": "  "}})
        rich_header.append(
            {"type": "text", "text": {"content": "(tweet)", "link": {"url": tweet["url"]}}}
        )

    blocks: list[dict] = [
        {"object": "block", "type": "paragraph", "paragraph": {"rich_text": rich_header}},
        {
            "object": "block",
            "type": "quote",
            "quote": {
                "rich_text": [
                    {"type": "text", "text": {"content": tweet["text"][:2000]}}
                ]
            },
        },
    ]
    for image_url in tweet["images"]:
        blocks.append(
            {
                "object": "block",
                "type": "image",
                "image": {"type": "external", "external": {"url": image_url}},
            }
        )
    return blocks


def write_topic(topic: dict, tweets: list[dict], existing_pages: dict[str, str]) -> None:
    indices = [i for i in topic.get("tweet_indices", []) if 0 <= i < len(tweets)]
    if not indices:
        return
    blocks: list[dict] = []
    for i in indices:
        blocks.extend(_tweet_blocks(tweets[i]))

    existing_title = (topic.get("existing_title") or "").removeprefix("X — ").strip()
    page_id = existing_pages.get(existing_title) or existing_pages.get(
        topic.get("title", "")
    )

    if page_id:
        notion_request("PATCH", f"blocks/{page_id}/children", {"children": blocks})
        log.info("Appended %d tweet(s) to existing topic %r", len(indices), existing_title or topic.get("title"))
        return

    title = f"X — {topic.get('title', 'Updates')}"
    properties = {
        "Creator": {"title": [{"text": {"content": title}}]},
        "Date": {"date": {"start": datetime.now(PKT).isoformat()}},
        "Tags": {"multi_select": [{"name": (topic.get("tag") or "News")[:100]}]},
    }
    if topic.get("summary"):
        properties["About"] = {
            "rich_text": [{"type": "text", "text": {"content": topic["summary"][:2000]}}]
        }
    page = notion_request(
        "POST",
        "pages",
        {
            "parent": {"database_id": NOTION_DATABASE_ID},
            "icon": {"type": "emoji", "emoji": PAGE_ICON},
            "properties": properties,
            "children": blocks[:100],
        },
    )
    existing_pages[topic.get("title", "")] = page["id"]
    log.info("Created topic page %r with %d tweet(s)", title, len(indices))


# ---------------------------------------------------------------------------

def main() -> int:
    log.info("=== X daily digest starting ===")
    missing = [
        n
        for n, v in {
            "APIFY_API_TOKEN": APIFY_API_TOKEN,
            "NOTION_API_KEY": NOTION_API_KEY,
            "NOTION_DATABASE_ID": NOTION_DATABASE_ID,
            "GEMINI_API_KEY": GEMINI_API_KEY,
        }.items()
        if not v
    ]
    if missing:
        log.error("Missing required environment variables: %s", ", ".join(missing))
        return 1

    handles = load_accounts()
    if not handles:
        return 0

    try:
        tweets = fetch_tweets(handles)
    except Exception as exc:  # noqa: BLE001
        log.error("Tweet fetch failed: %s", exc)
        return 1
    if not tweets:
        log.info("No tweets in the last 24h — nothing to do.")
        return 0

    ensure_tags_property()
    existing_pages = todays_x_pages()
    topics = group_topics(tweets, list(existing_pages))

    failed = 0
    for topic in topics:
        try:
            write_topic(topic, tweets, existing_pages)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            log.error("Failed to write topic %r: %s", topic.get("title"), exc)

    log.info("=== X digest finished: %d topic(s), %d failed ===", len(topics), failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
