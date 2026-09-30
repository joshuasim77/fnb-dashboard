"""5W1H: unbiased local F&B sentiment (FastAPI on Vercel).

Sections
  1. Config        brands, subreddits, cache time, word lists
  2. Ingestion     PRAW fetch from Reddit + sample-data fallback
  3. Cache         in-memory, refreshed every few hours
  4. Processing    raw text -> uniform JSON schema
  5. Routes        /, /api/mentions, /api/process, /api/health
"""
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# --------------------------------------------------------------------------
# 1. Config
# --------------------------------------------------------------------------
SUBREDDITS = os.getenv("SUBREDDITS", "singapore+askSingapore")  # "+" joins subreddits
CACHE_TTL_SECONDS = int(float(os.getenv("CACHE_TTL_HOURS", "3")) * 3600)
SAMPLE_RETRY_SECONDS = 300      # if we fell back to sample data, try Reddit again after 5 min
FETCH_BUDGET_SECONDS = 20       # stop collecting after this long so the page never hangs
POSTS_PER_QUERY = 15
COMMENT_POSTS_PER_BRAND = 3     # scan comments on this many top posts per brand
COMMENTS_PER_POST = 40

# search: what we type into Reddit search. pattern: what must appear in the text.
BRANDS = {
    "KOI": {
        "search": ["KOI", "KOI Thé"],
        "pattern": r"\bkoi\b(?!\s*(?:fish|pond|carp))|koi\s?th[eé]",
    },
    "Chagee": {"search": ["Chagee"], "pattern": r"\bchagee\b|\bchaji\b|霸王茶姬"},
    "LiHO": {"search": ["LiHO"], "pattern": r"\bli\s?ho\b"},
    "Gong Cha": {"search": ["Gong Cha"], "pattern": r"\bgong\s?cha\b"},
    "Playmade": {"search": ["Playmade"], "pattern": r"\bplaymade\b"},
    "Each A Cup": {"search": ["Each A Cup"], "pattern": r"\beach\s?a\s?cup\b"},
}
BRAND_PATTERNS = {b: re.compile(c["pattern"], re.I) for b, c in BRANDS.items()}

THEMES = {
    "Taste": ["taste", "flavour", "flavor", "tasty", "delicious", "sedap", "shiok", "bland", "watery"],
    "Sweetness": ["sweet", "sugar", "cloying"],
    "Price": ["price", "expensive", "cheap", "overpriced", "$", "cost", "pricey"],
    "Value": ["worth", "value", "rip off", "ripoff"],
    "Queue & wait": ["queue", "wait", "waiting", "line", "slow"],
    "Service": ["service", "staff", "rude", "friendly", "mistake"],
    "Toppings": ["pearl", "boba", "topping", "cheese foam", "jelly", "cream"],
    "Tea quality": ["tea", "oolong", "jasmine", "aroma", "fragrant"],
    "Portion": ["cup size", "portion", "small", "tiny", "big cup"],
}

POSITIVE = {
    "good", "great", "love", "loved", "best", "nice", "shiok", "sedap", "solid", "power",
    "steady", "delicious", "tasty", "fragrant", "friendly", "recommend", "fresh", "yummy",
    "favourite", "favorite", "smooth", "awesome", "amazing", "cheap", "fast",
}
NEGATIVE = {
    "bad", "worst", "hate", "hated", "bland", "watery", "sian", "jialat", "overpriced",
    "expensive", "pricey", "rude", "slow", "disappointing", "disappointed", "meh", "sucks",
    "sweet", "cloying", "terrible", "awful", "queue", "small", "tiny",
}
NEGATIONS = {"not", "no", "never", "isn't", "wasn't", "don't", "didn't", "hardly", "nothing"}

# Wording that usually signals sponsored or promotional posts
PROMO_PATTERN = re.compile(
    r"sponsored|#ad\b|\bad:|promo code|use my code|referral|affiliate|giveaway|discount code|"
    r"paid partnership|\bcollab\b|gifted|invited to try|free drinks? in exchange",
    re.I,
)

# --------------------------------------------------------------------------
# 2. Ingestion
# --------------------------------------------------------------------------
def _reddit_client():
    cid, secret = os.getenv("REDDIT_CLIENT_ID"), os.getenv("REDDIT_CLIENT_SECRET")
    if not (cid and secret):
        return None
    import praw  # imported here so sample-data mode starts fast

    return praw.Reddit(
        client_id=cid,
        client_secret=secret,
        user_agent=os.getenv("REDDIT_USER_AGENT", "5W1H/1.0"),
        check_for_async=False,
        timeout=8,  # seconds to wait on each Reddit request
    )


def fetch_reddit() -> list[dict]:
    """Collect raw posts/comments. Raises if credentials are missing or Reddit fails."""
    reddit = _reddit_client()
    if reddit is None:
        raise RuntimeError("Reddit credentials not set")

    deadline = time.monotonic() + FETCH_BUDGET_SECONDS
    sub = reddit.subreddit(SUBREDDITS)
    seen: set[str] = set()
    records: list[dict] = []

    for brand, cfg in BRANDS.items():
        if time.monotonic() > deadline:
            break
        pattern = BRAND_PATTERNS[brand]
        posts, post_ids = [], set()
        for query in cfg["search"]:
            for post in sub.search(query, sort="new", time_filter="month", limit=POSTS_PER_QUERY):
                if post.id not in post_ids:
                    post_ids.add(post.id)
                    posts.append(post)

        for post in posts:
            if post.id in seen:
                continue
            seen.add(post.id)
            records.append({
                "id": post.id,
                "text": f"{post.title}. {post.selftext or ''}".strip(),
                "created_utc": post.created_utc,
                "url": f"https://reddit.com{post.permalink}",
            })

        busiest = sorted(posts, key=lambda p: p.num_comments, reverse=True)[:COMMENT_POSTS_PER_BRAND]
        for post in busiest:
            if time.monotonic() > deadline:
                break
            post.comments.replace_more(limit=0)
            for c in list(post.comments)[:COMMENTS_PER_POST]:
                if c.id not in seen and pattern.search(c.body or ""):
                    seen.add(c.id)
                    records.append({
                        "id": c.id,
                        "text": c.body,
                        "created_utc": c.created_utc,
                        "url": f"https://reddit.com{c.permalink}",
                    })
    return records


def sample_records() -> list[dict]:
    now = datetime.now(timezone.utc)
    samples = [
        "Chagee's Boba Oolong is honestly shiok. Tea is fragrant, not too sweet. Queue was ok on a weekday.",
        "KOI is so overpriced now. $6 for a small cup and the sweetness is cloying. Not worth it.",
        "LiHO cheese foam series is solid. Staff were friendly and the wait was short.",
        "Queued 40 mins for Chagee. Tea was nice but the wait was jialat. Won't do again.",
        "KOI Golden Bubble is my go-to. Pearls are chewy and fresh, service fast.",
        "LiHO drink was watery today. Disappointing for the price.",
        "Sponsored: Chagee is my fave, use my code for 10% off! Best bubble tea in Singapore, amazing #ad",
        "Gong Cha's brown sugar milk tea is meh, too sweet and tiny portion.",
        "Playmade pearls are the best in town. Recommend, really good.",
        "KOI staff got my order wrong again and were rude about it. Sian.",
        "LiHO is cheap and good for office runs. Great value.",
        "Chagee price went up again, but the tea aroma is still amazing.",
    ]
    return [
        {"id": f"sample{i}", "text": t, "created_utc": (now - timedelta(hours=3 * i + 1)).timestamp(), "url": ""}
        for i, t in enumerate(samples)
    ]


# --------------------------------------------------------------------------
# 3. Cache (in memory; also see Cache-Control header on the route)
# --------------------------------------------------------------------------
_lock = threading.Lock()
_cache: dict = {"items": None, "source": None, "updated_at": None, "fetched": 0.0, "ttl": 0}


def get_data(force_sample: bool = False) -> dict:
    """Return {items, source, updated_at, stale}. Never raises."""
    if force_sample:
        return _pack(process_records(sample_records()), "sample", time.time(), False)

    with _lock:
        fresh = _cache["items"] is not None and (time.time() - _cache["fetched"] < _cache["ttl"])
        if not fresh:
            try:
                items = process_records(fetch_reddit())
                if not items:
                    raise RuntimeError("Reddit returned no matching posts")
                _cache.update(items=items, source="reddit", updated_at=time.time(),
                              fetched=time.time(), ttl=CACHE_TTL_SECONDS)
            except Exception as exc:  # missing credentials, timeout, rate limit, etc.
                print(f"[5W1H] live fetch failed: {exc}")
                if _cache["source"] == "reddit":
                    # keep serving the last good data, retry soon
                    _cache.update(fetched=time.time(), ttl=SAMPLE_RETRY_SECONDS)
                    return _pack(_cache["items"], "reddit", _cache["updated_at"], True)
                _cache.update(items=process_records(sample_records()), source="sample",
                              updated_at=time.time(), fetched=time.time(), ttl=SAMPLE_RETRY_SECONDS)
        return _pack(_cache["items"], _cache["source"], _cache["updated_at"], False)


def _pack(items, source, updated_at, stale):
    return {
        "items": items,
        "source": source,
        "updated_at": datetime.fromtimestamp(updated_at, tz=timezone.utc).isoformat(),
        "stale": stale,
    }


# --------------------------------------------------------------------------
# 4. Processing
# --------------------------------------------------------------------------
def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z$']+", text.lower())


def score_text(text: str) -> float:
    """Keyword score from -1.0 to 1.0, with simple negation handling."""
    lowered = text.lower()
    pos = neg = 0
    for phrase, is_pos in (("not worth", False), ("rip off", False), ("bo liao", False),
                           ("worth it", True), ("must try", True)):
        if phrase in lowered:
            pos, neg = (pos + 1, neg) if is_pos else (pos, neg + 1)
            lowered = lowered.replace(phrase, " ")
    toks = _tokens(lowered)
    for i, tok in enumerate(toks):
        flipped = any(t in NEGATIONS for t in toks[max(0, i - 2):i])
        if tok in POSITIVE:
            pos, neg = (pos, neg + 1) if flipped else (pos + 1, neg)
        elif tok in NEGATIVE:
            pos, neg = (pos + 1, neg) if flipped else (pos, neg + 1)
    if pos + neg == 0:
        return 0.0
    return round((pos - neg) / (pos + neg + 1), 3)


def label_for(score: float) -> str:
    if score >= 0.2:
        return "Positive"
    if score <= -0.2:
        return "Negative"
    return "Neutral"


def extract_themes(text: str) -> tuple[list[str], list[str], list[str]]:
    """Return (themes, pros, cons) using sentence-level polarity."""
    themes, pros, cons = [], [], []
    for sentence in re.split(r"(?<=[.!?\n])\s+", text):
        low = sentence.lower()
        hits = [t for t, kws in THEMES.items() if any(k in low for k in kws)]
        if not hits:
            continue
        s = score_text(sentence)
        for t in hits:
            if t not in themes:
                themes.append(t)
            if s > 0.1 and t not in pros:
                pros.append(t)
            elif s < -0.1 and t not in cons:
                cons.append(t)
    return themes, pros, cons


def detect_brand(text: str) -> Optional[str]:
    for brand, pattern in BRAND_PATTERNS.items():
        if pattern.search(text):
            return brand
    return None


def to_schema(item_id: str, text: str, brand: str, created_utc: float, url: str = "") -> dict:
    score = score_text(text)
    themes, pros, cons = extract_themes(text)
    return {
        "item_id": item_id,
        "category": "fnb",
        "brand_name": brand,
        "platform": "reddit",
        "raw_text": text.strip()[:1000],
        "sentiment_score": score,
        "sentiment_label": label_for(score),
        "key_themes": themes,
        "timestamp": datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat(),
        # extras used by the dashboard
        "pros": pros,
        "cons": cons,
        "promo_flag": bool(PROMO_PATTERN.search(text)),
        "url": url,
    }


def process_records(records: list[dict]) -> list[dict]:
    items = []
    for r in records:
        text = (r.get("text") or "").strip()
        if len(text) < 15:
            continue
        for brand, pattern in BRAND_PATTERNS.items():
            if pattern.search(text):
                items.append(to_schema(f"{r['id']}:{brand}", text, brand, r["created_utc"], r.get("url", "")))
    items.sort(key=lambda i: i["timestamp"], reverse=True)
    return items


def summarize(items: list[dict]) -> dict:
    """Index and counts ignore posts flagged as possible promotion."""
    real = [i for i in items if not i["promo_flag"]]
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for i in real:
        counts[i["sentiment_label"]] += 1
    total = len(real)
    nsi = round((counts["Positive"] - counts["Negative"]) / total * 100, 1) if total else 0.0
    return {
        "net_sentiment_index": nsi,  # -100 to +100
        "total_mentions": total,
        "promo_flagged": len(items) - total,
        "counts": counts,
    }


def brand_counts(items: list[dict]) -> list[dict]:
    tally: dict[str, int] = {}
    for i in items:
        if not i["promo_flag"]:
            tally[i["brand_name"]] = tally.get(i["brand_name"], 0) + 1
    return [{"brand_name": b, "mentions": n} for b, n in sorted(tally.items(), key=lambda x: -x[1])]


# --------------------------------------------------------------------------
# 5. Routes
# --------------------------------------------------------------------------
app = FastAPI(title="5W1H API")
INDEX_HTML = Path(__file__).resolve().parent.parent / "static" / "index.html"


class ProcessRequest(BaseModel):
    text: str
    brand_name: Optional[str] = None
    item_id: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/api/health")
def health(response: Response):
    response.headers["Cache-Control"] = "no-store"
    return {
        "status": "ok",
        "reddit_configured": bool(os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET")),
        "subreddits": SUBREDDITS,
        "cache_hours": CACHE_TTL_SECONDS / 3600,
    }


@app.get("/api/mentions")
def mentions(
    response: Response,
    category: str = Query("fnb"),
    brand: Optional[str] = None,
    limit: int = Query(30, ge=1, le=100),
    sample: bool = False,
):
    if category != "fnb":
        response.headers["Cache-Control"] = "public, s-maxage=3600"
        return {"category": category, "source": "none", "updated_at": None, "stale": False,
                "summary": summarize([]), "brands": [], "items": [],
                "message": "This category is coming soon."}

    data = get_data(force_sample=sample)
    all_items = data["items"]
    items = [i for i in all_items if not brand or i["brand_name"].lower() == brand.lower()]

    # Vercel's edge also caches this response, so the cache survives across server restarts.
    ttl = CACHE_TTL_SECONDS if data["source"] == "reddit" and not data["stale"] else SAMPLE_RETRY_SECONDS
    response.headers["Cache-Control"] = f"public, s-maxage={ttl}, stale-while-revalidate=600"

    return {
        "category": "fnb",
        "source": data["source"],  # "reddit" or "sample"
        "stale": data["stale"],
        "updated_at": data["updated_at"],
        "summary": summarize(items),
        "brands": brand_counts(all_items),
        "items": items[:limit],
    }


@app.post("/api/process")
def process(req: ProcessRequest):
    """Structure one piece of raw text into the uniform schema."""
    brand = req.brand_name or detect_brand(req.text) or "Unknown"
    return to_schema(req.item_id or f"adhoc-{int(time.time())}", req.text, brand, time.time())
