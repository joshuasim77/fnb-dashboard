"""F&B Consumer Sentiment API (FastAPI, Vercel Python runtime).

Layout:
  1. Config         brands, theme keywords, sentiment lexicon
  2. Ingestion      PRAW fetch from r/singapore, with mock fallback
  3. Processing     raw text -> uniform JSON schema
  4. Routes         /, /api/mentions, /api/process, /api/health
"""
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# --------------------------------------------------------------------------
# 1. Config
# --------------------------------------------------------------------------
SUBREDDIT = "singapore"
CACHE_TTL_SECONDS = 300

BRANDS = {
    "KOI": r"\bkoi\b",
    "Chagee": r"\bchagee\b|\bchaji\b|霸王茶姬",
    "LiHO": r"\bli\s?ho\b",
    "Gong Cha": r"\bgong\s?cha\b",
    "Playmade": r"\bplaymade\b",
    "Each A Cup": r"\beach\s?a\s?cup\b",
}
BRAND_PATTERNS = {b: re.compile(p, re.I) for b, p in BRANDS.items()}

# theme -> trigger keywords
THEMES = {
    "Taste": ["taste", "flavour", "flavor", "tasty", "delicious", "sedap", "shiok", "bland", "watery"],
    "Sweetness": ["sweet", "sugar", "less sweet", "too sweet", "cloying"],
    "Price": ["price", "expensive", "cheap", "overpriced", "$", "cost", "pricey"],
    "Value": ["worth", "value", "not worth", "rip off", "ripoff"],
    "Queue & wait": ["queue", "wait", "waiting", "line", "slow"],
    "Service": ["service", "staff", "rude", "friendly", "mistake"],
    "Toppings": ["pearl", "boba", "topping", "cheese foam", "jelly", "cream"],
    "Tea quality": ["tea", "oolong", "jasmine", "aroma", "fragrant"],
    "Portion": ["cup size", "portion", "small", "tiny", "big cup"],
}

POSITIVE = {
    "good", "great", "love", "loved", "best", "nice", "shiok", "sedap", "solid", "power",
    "steady", "worth", "delicious", "tasty", "fragrant", "friendly", "recommend", "fresh",
    "yummy", "favourite", "favorite", "smooth", "awesome", "amazing", "cheap", "fast",
    "worth it", "must try",
}
NEGATIVE = {
    "bad", "worst", "hate", "hated", "bland", "watery", "sian", "jialat", "overpriced",
    "expensive", "pricey", "rude", "slow", "disappointing", "disappointed", "meh", "sucks",
    "sweet", "cloying", "terrible", "awful", "queue", "rip off", "ripoff", "small", "tiny",
    "not worth", "bo liao",
}
NEGATIONS = {"not", "no", "never", "isn't", "wasn't", "don't", "didn't", "hardly", "nothing"}

# --------------------------------------------------------------------------
# 2. Ingestion
# --------------------------------------------------------------------------
_cache: dict = {"ts": 0.0, "key": None, "data": None}


def _reddit_client():
    cid, secret = os.getenv("REDDIT_CLIENT_ID"), os.getenv("REDDIT_CLIENT_SECRET")
    if not (cid and secret):
        return None
    import praw  # imported lazily so mock mode has no cold-start cost

    return praw.Reddit(
        client_id=cid,
        client_secret=secret,
        user_agent=os.getenv("REDDIT_USER_AGENT", "fnb-sentiment-mvp/0.1"),
        check_for_async=False,
    )


def fetch_reddit(limit_per_brand: int = 10, comments_per_post: int = 30) -> list[dict]:
    """Return raw records: {id, text, created_utc, url}. Empty list if no creds."""
    reddit = _reddit_client()
    if reddit is None:
        return []

    sub = reddit.subreddit(SUBREDDIT)
    seen: set[str] = set()
    records: list[dict] = []

    for brand, pattern in BRAND_PATTERNS.items():
        query = BRANDS[brand].replace(r"\b", "").replace(r"\s?", " ").split("|")[0]
        for post in sub.search(query, sort="new", time_filter="month", limit=limit_per_brand):
            if post.id not in seen:
                seen.add(post.id)
                records.append({
                    "id": post.id,
                    "text": f"{post.title}. {post.selftext or ''}".strip(),
                    "created_utc": post.created_utc,
                    "url": f"https://reddit.com{post.permalink}",
                })
            # scan a few comments on the top posts for direct brand mentions
            post.comments.replace_more(limit=0)
            for c in list(post.comments)[:comments_per_post]:
                if c.id not in seen and pattern.search(c.body or ""):
                    seen.add(c.id)
                    records.append({
                        "id": c.id,
                        "text": c.body,
                        "created_utc": c.created_utc,
                        "url": f"https://reddit.com{c.permalink}",
                    })
    return records


def mock_records() -> list[dict]:
    now = datetime.now(timezone.utc)
    samples = [
        "Chagee's Boba Oolong is honestly shiok. Tea is fragrant, not too sweet. Queue was ok on a weekday.",
        "KOI is so overpriced now. $6 for a small cup and the sweetness is cloying. Not worth it.",
        "LiHO cheese foam series is solid. Staff were friendly and the wait was short.",
        "Queued 40 mins for Chagee. Tea was nice but the wait was jialat. Won't do again.",
        "KOI Golden Bubble is my go-to. Pearls are chewy and fresh, service fast.",
        "LiHO drink was watery today. Disappointing for the price.",
        "Anyone tried the new Chagee outlet at Orchard? Thinking of going tomorrow.",
        "Gong Cha's brown sugar milk tea is meh, too sweet and tiny portion.",
        "Playmade pearls are the best in town. Worth the trip, recommend.",
        "KOI staff got my order wrong again and were rude about it. Sian.",
        "LiHO is cheap and good for office runs. Value is great.",
        "Chagee price went up again, but the tea aroma is still amazing.",
    ]
    return [
        {"id": f"mock{i}", "text": t, "created_utc": (now - timedelta(hours=3 * i + 1)).timestamp(), "url": ""}
        for i, t in enumerate(samples)
    ]


def ingest(use_mock: bool = False) -> tuple[list[dict], str]:
    if not use_mock:
        try:
            records = fetch_reddit()
            if records:
                return records, "reddit"
        except Exception as exc:  # network/auth errors should not take the API down
            print(f"[ingest] Reddit fetch failed, using mock data: {exc}")
    return mock_records(), "mock"


# --------------------------------------------------------------------------
# 3. Processing
# --------------------------------------------------------------------------
def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z$']+", text.lower())


def score_text(text: str) -> float:
    """Lexicon score in [-1, 1] with simple negation handling."""
    lowered = text.lower()
    pos = neg = 0
    for phrase in ("not worth", "worth it", "must try", "rip off", "bo liao"):
        if phrase in lowered:
            if phrase in ("worth it", "must try"):
                pos += 1
            else:
                neg += 1
            lowered = lowered.replace(phrase, " ")
    toks = _tokens(lowered)
    for i, tok in enumerate(toks):
        flipped = any(t in NEGATIONS for t in toks[max(0, i - 2):i])
        if tok in POSITIVE:
            neg, pos = (neg + 1, pos) if flipped else (neg, pos + 1)
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
    """Return (all themes, pros, cons) using sentence-level polarity."""
    themes, pros, cons = [], [], []
    for sentence in re.split(r"(?<=[.!?\n])\s+", text):
        low = sentence.lower()
        hit = [t for t, kws in THEMES.items() if any(k in low for k in kws)]
        if not hit:
            continue
        s = score_text(sentence)
        for t in hit:
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
        "url": url,
    }


def process_records(records: list[dict]) -> list[dict]:
    items = []
    for r in records:
        for brand, pattern in BRAND_PATTERNS.items():
            if pattern.search(r["text"]):
                items.append(to_schema(f"{r['id']}:{brand}", r["text"], brand, r["created_utc"], r.get("url", "")))
    items.sort(key=lambda i: i["timestamp"], reverse=True)
    return items


def summarize(items: list[dict]) -> dict:
    total = len(items)
    counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    by_brand: dict[str, dict] = {}
    for i in items:
        counts[i["sentiment_label"]] += 1
        b = by_brand.setdefault(i["brand_name"], {"mentions": 0, "score_sum": 0.0})
        b["mentions"] += 1
        b["score_sum"] += i["sentiment_score"]
    nsi = round((counts["Positive"] - counts["Negative"]) / total * 100, 1) if total else 0.0
    return {
        "net_sentiment_index": nsi,  # -100..100
        "total_mentions": total,
        "counts": counts,
        "brands": sorted(
            [{"brand_name": k, "mentions": v["mentions"], "avg_score": round(v["score_sum"] / v["mentions"], 2)}
             for k, v in by_brand.items()],
            key=lambda x: x["mentions"], reverse=True,
        ),
    }


# --------------------------------------------------------------------------
# 4. Routes
# --------------------------------------------------------------------------
app = FastAPI(title="F&B Consumer Sentiment API")
INDEX_HTML = Path(__file__).resolve().parent.parent / "static" / "index.html"


class ProcessRequest(BaseModel):
    text: str
    brand_name: Optional[str] = None
    item_id: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/api/health")
def health():
    return {"status": "ok", "reddit_configured": bool(os.getenv("REDDIT_CLIENT_ID"))}


@app.get("/api/mentions")
def mentions(
    category: str = Query("fnb"),
    brand: Optional[str] = None,
    limit: int = Query(30, ge=1, le=100),
    mock: bool = False,
):
    if category != "fnb":
        return {"category": category, "source": "none", "summary": summarize([]), "items": [],
                "message": "This category is coming soon."}

    key = "mock" if mock else "live"
    if _cache["key"] != key or time.time() - _cache["ts"] > CACHE_TTL_SECONDS:
        records, source = ingest(use_mock=mock)
        _cache.update(ts=time.time(), key=key, data=(process_records(records), source))
    items, source = _cache["data"]

    if brand:
        items = [i for i in items if i["brand_name"].lower() == brand.lower()]
    return {"category": "fnb", "source": source, "summary": summarize(items), "items": items[:limit]}


@app.post("/api/process")
def process(req: ProcessRequest):
    """Structure one piece of raw text into the uniform schema."""
    brand = req.brand_name or detect_brand(req.text) or "Unknown"
    item_id = req.item_id or f"adhoc-{int(time.time())}"
    return to_schema(item_id, req.text, brand, time.time())
