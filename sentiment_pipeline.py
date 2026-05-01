"""
================================================================================
SENTIMENT PIPELINE MODULE — AI Stock Trading Decision Support System
File: sentiment_pipeline.py
================================================================================

ARCHITECTURE:
    fetch_news()               → Raw RSS headlines (with per-feed timeout)
    clean_news()               → Deduplicated, time-filtered articles
    score_sentiment()          → FinBERT (optional) + VADER per headline
    aggregate_sentiment()      → Overall signal + recency-weighted stats
    adjust_prediction_with_sentiment() → ML output + sentiment reconciliation
    run_sentiment_pipeline()   → End-to-end cached pipeline
    render_sentiment_section() → Drop-in Streamlit UI block

DEPENDENCY STRATEGY:
    vaderSentiment : always attempted — lightweight, reliable
    transformers   : OPTIONAL — imported lazily via _safe_import_transformers(),
                     NEVER imported at module level
    torch          : OPTIONAL — pulled in by transformers only when available

CRITICAL ARCHITECTURE NOTES:
    ▸ _load_finbert() is decorated with @st.cache_resource so it is called ONCE
      per session. It must NEVER be invoked from inside a @st.cache_data function
      (Streamlit raises StreamlitAPIException if cache_resource is nested inside
      cache_data). run_sentiment_pipeline() therefore calls _load_finbert() at its
      own call-site and passes the result into score_sentiment() / aggregate_sentiment().
    ▸ Every public function returns a safe default on failure — the app never crashes.
================================================================================
"""

import re
import socket
import hashlib
import warnings
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# SAFE LAZY IMPORT — never at module level, always inside a function
# ─────────────────────────────────────────────────────────────────────────────

def _safe_import_transformers():
    """
    Attempt to import transformers.pipeline and torch at call-time.

    Environment variables are set BEFORE the import so HuggingFace never
    attempts a network download (TRANSFORMERS_OFFLINE=1) or phones home.

    Returns:
        (hf_pipeline_fn, torch_module)   on success
        (None, None)                     on any ImportError / Exception
    """
    try:
        import os
        os.environ.setdefault("TRANSFORMERS_OFFLINE",      "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM",    "false")

        from transformers import pipeline as hf_pipeline  # noqa: PLC0415
        import torch                                       # noqa: PLC0415
        return hf_pipeline, torch

    except ImportError:
        return None, None
    except Exception:
        return None, None


# ─── VADER — attempted once at module load ───────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_OK = True
except Exception:
    VADER_OK = False

# ─── feedparser — needed for RSS ─────────────────────────────────────────────
try:
    import feedparser
    FEEDPARSER_OK = True
except Exception:
    FEEDPARSER_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

RSS_FEEDS = [
    "https://news.google.com/rss/search?q=Reliance+Industries&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=RELIANCE+NSE+stock&hl=en-IN&gl=IN&ceid=IN:en",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=RELIANCE.NS&region=US&lang=en-US",
]

FINBERT_MODEL    = "ProsusAI/finbert"
FINBERT_WEIGHT   = 0.70
VADER_WEIGHT     = 0.30
MAX_ARTICLES     = 20
HOURS_LOOKBACK   = 24
RSS_TIMEOUT_SEC  = 10    # per-feed socket timeout

# Sentiment label thresholds
_POS_THRESH = 0.05
_NEG_THRESH = -0.05

# Fallback epoch used in place of datetime.min (avoids OverflowError)
_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — NEWS FETCH
# ─────────────────────────────────────────────────────────────────────────────

def fetch_news(max_articles: int = MAX_ARTICLES) -> list:
    """
    Fetch raw headlines from RSS feeds.

    Each feed is fetched with a RSS_TIMEOUT_SEC socket timeout so a single
    slow or dead feed never blocks the app.

    Returns list of dicts: { title, published_raw, source, url }
    Returns [] on any failure — never raises.
    """
    if not FEEDPARSER_OK:
        return []

    articles: list = []

    for feed_url in RSS_FEEDS:
        try:
            # Temporarily narrow the socket timeout for this feed
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(RSS_TIMEOUT_SEC)
            try:
                feed = feedparser.parse(feed_url)
            finally:
                socket.setdefaulttimeout(old_timeout)

            for entry in getattr(feed, "entries", []):
                title = (entry.get("title") or "").strip()
                if not title:
                    continue
                articles.append({
                    "title":         title,
                    "published_raw": entry.get("published", entry.get("updated", "")),
                    "source":        (entry.get("source") or {}).get("title", "Unknown"),
                    "url":           entry.get("link", ""),
                })
        except Exception:
            continue   # one bad feed must not crash the rest

    return articles[:max_articles * 2]   # trim before dedup


def _parse_published(raw: str) -> Optional[datetime]:
    """
    Parse an RSS date string into a UTC-aware datetime.
    Returns None on any failure — never raises.
    """
    if not raw:
        return None

    fmts = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(raw, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    try:
        from email.utils import parsedate_to_datetime  # stdlib — always safe
        dt = parsedate_to_datetime(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _fingerprint(title: str) -> str:
    """MD5 of normalised title — used for deduplication."""
    normalised = re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()
    return hashlib.md5(normalised.encode()).hexdigest()


def clean_news(
    raw_articles:   list,
    hours_lookback: int = HOURS_LOOKBACK,
    max_articles:   int = MAX_ARTICLES,
) -> list:
    """
    Deduplicate and time-filter raw articles.
    Falls back to all articles if none pass the time gate.
    Returns up to max_articles sorted newest-first.
    """
    if not raw_articles:
        return []

    now_utc = datetime.now(timezone.utc)
    cutoff  = now_utc - timedelta(hours=hours_lookback)

    seen: set    = set()
    cleaned: list = []
    for art in raw_articles:
        fp = _fingerprint(art["title"])
        if fp in seen:
            continue
        seen.add(fp)
        art["published_dt"] = _parse_published(art["published_raw"])
        cleaned.append(art)

    recent = [
        a for a in cleaned
        if a["published_dt"] is not None and a["published_dt"] >= cutoff
    ]

    # Graceful fallback: show something even if dates are stale / missing
    pool = recent if recent else cleaned
    pool.sort(
        # Use _EPOCH_UTC instead of datetime.min to avoid OverflowError
        key=lambda a: a["published_dt"] if a["published_dt"] is not None else _EPOCH_UTC,
        reverse=True,
    )
    return pool[:max_articles]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — MODEL LOADERS (session-cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_finbert():
    """
    Load FinBERT text-classification pipeline (CPU-only, single load per session).

    IMPORTANT: This function is @st.cache_resource. It must NEVER be called from
    inside a @st.cache_data function — Streamlit prohibits nesting cache types.
    Callers are responsible for obtaining the clf object at their own call-site
    and passing it into helper functions that need it.

    Returns:
        HuggingFace pipeline object   on success
        None                          on any failure (silent — VADER takes over)
    """
    hf_pipeline, _torch = _safe_import_transformers()
    if hf_pipeline is None:
        return None

    try:
        clf = hf_pipeline(
            "text-classification",
            model=FINBERT_MODEL,
            top_k=None,       # return all 3 class scores per input
            truncation=True,
            max_length=512,
            device=-1,        # CPU only
        )
        return clf
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_vader():
    """
    Load VADER SentimentIntensityAnalyzer (single load per session).

    Returns:
        SentimentIntensityAnalyzer   on success
        None                         if vaderSentiment is not installed
    """
    if not VADER_OK:
        return None
    try:
        return SentimentIntensityAnalyzer()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _finbert_batch_score(clf, texts: list) -> list:
    """
    Run FinBERT inference on a batch of texts.

    Maps FinBERT labels to scalars:
        positive → +score
        negative → -score
        neutral  →  0.0

    Returns:
        list[float | None] — exactly len(texts) elements.
        float : valid score in [-1, +1]
        None  : inference failed (treated as "no FinBERT" for this item)

    Never raises.
    """
    n_texts = len(texts)
    if clf is None or n_texts == 0:
        return [None] * n_texts

    try:
        raw_results = clf(texts)
    except Exception:
        return [None] * n_texts

    if not isinstance(raw_results, list):
        return [None] * n_texts

    scores: list = []
    for result in raw_results[:n_texts]:   # guard against over-length returns
        try:
            if isinstance(result, list):
                total = 0.0
                for item in result:
                    if not isinstance(item, dict):
                        continue
                    lbl = (item.get("label") or "").lower()
                    s   = float(item.get("score", 0.0))
                    if "positive" in lbl:
                        total += s
                    elif "negative" in lbl:
                        total -= s
                scores.append(float(np.clip(total, -1.0, 1.0)))

            elif isinstance(result, dict):
                lbl = (result.get("label") or "").lower()
                s   = float(result.get("score", 0.0))
                if "positive" in lbl:
                    scores.append(min(s, 1.0))
                elif "negative" in lbl:
                    scores.append(max(-s, -1.0))
                else:
                    scores.append(0.0)
            else:
                scores.append(None)

        except Exception:
            scores.append(None)

    # Pad to exact length in case raw_results was shorter than texts
    while len(scores) < n_texts:
        scores.append(None)

    return scores[:n_texts]


def _vader_score_single(analyzer, text: str) -> float:
    """
    Return VADER compound score in [-1, +1].
    Returns 0.0 on any failure — never raises.
    """
    if analyzer is None:
        return 0.0
    try:
        return float(analyzer.polarity_scores(text)["compound"])
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — SENTIMENT SCORING
# ─────────────────────────────────────────────────────────────────────────────

def score_sentiment(articles: list, clf=None) -> list:
    """
    Score each article dict with VADER (always) and FinBERT (when provided).

    IMPORTANT: clf is now an explicit parameter — it must be obtained by the
    caller via _load_finbert() OUTSIDE any @st.cache_data context, then passed
    in here. This avoids the Streamlit cache nesting restriction.

    Blending:
        FinBERT available → combined = 0.70 × finbert + 0.30 × vader
        FinBERT missing   → combined = vader only

    Adds to each article dict (in-place copy):
        vader_score    : float  always in [-1, +1]
        finbert_score  : float | None
        combined_score : float  always in [-1, +1]
        sentiment_label: str    "Positive" | "Negative" | "Neutral"

    Returns:
        scored_articles: list   (empty list if articles is empty)
    Never raises.
    """
    vader = _load_vader()

    if not articles:
        return []

    titles         = [art["title"] for art in articles]
    finbert_scores = _finbert_batch_score(clf, titles)

    scored: list = []
    for art, fb in zip(articles, finbert_scores):
        vs = _vader_score_single(vader, art["title"])

        if fb is not None:
            combined = float(np.clip(FINBERT_WEIGHT * fb + VADER_WEIGHT * vs, -1.0, 1.0))
        else:
            combined = float(np.clip(vs, -1.0, 1.0))

        if combined > _POS_THRESH:
            label = "Positive"
        elif combined < _NEG_THRESH:
            label = "Negative"
        else:
            label = "Neutral"

        scored.append({
            **art,
            "vader_score":     round(vs,              4),
            "finbert_score":   round(float(fb), 4) if fb is not None else None,
            "combined_score":  round(combined,        4),
            "sentiment_label": label,
        })

    return scored


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def _active_model_label(clf) -> str:
    """Return human-readable NLP backend label given the loaded clf object."""
    if clf is not None:
        return "FinBERT + VADER"
    if VADER_OK:
        return "VADER only (FinBERT unavailable)"
    return "No NLP model loaded"


def aggregate_sentiment(scored_articles: list, clf=None) -> dict:
    """
    Compute overall sentiment using recency weighting.

    Weights:
        Age ≤  6 h → 2.0
        Age ≤ 12 h → 1.5
        Older      → 1.0

    Args:
        scored_articles: output of score_sentiment()
        clf:             the finbert pipeline object (or None) — used only for
                         the model_used label; never triggers a new model load.

    Returns a dict with every key always present — never raises.
    """
    model_used = _active_model_label(clf)

    _base: dict = {
        "total_articles":  0,
        "avg_score":       0.0,
        "weighted_score":  0.0,
        "pct_positive":    0.0,
        "pct_negative":    0.0,
        "pct_neutral":     100.0,
        "overall_label":   "Neutral",
        "signal_strength": "Weak",
        "articles":        [],
        "model_used":      model_used,
    }

    if not scored_articles:
        return _base

    now_utc              = datetime.now(timezone.utc)
    scores:  list[float] = []
    weights: list[float] = []
    pos = neg = neu = 0

    for art in scored_articles:
        s = float(art.get("combined_score", 0.0))
        scores.append(s)

        dt = art.get("published_dt")
        if dt is not None:
            try:
                age_h = (now_utc - dt).total_seconds() / 3600
                w = 2.0 if age_h <= 6 else (1.5 if age_h <= 12 else 1.0)
            except Exception:
                w = 1.0
        else:
            w = 1.0
        weights.append(w)

        lbl = art.get("sentiment_label", "Neutral")
        if lbl == "Positive":
            pos += 1
        elif lbl == "Negative":
            neg += 1
        else:
            neu += 1

    n              = len(scores)
    avg_score      = float(np.mean(scores))
    weighted_score = float(np.average(scores, weights=weights))

    if weighted_score >= 0.10:
        overall_label = "Positive"
    elif weighted_score <= -0.10:
        overall_label = "Negative"
    else:
        overall_label = "Neutral"

    abs_ws = abs(weighted_score)
    signal_strength = (
        "Strong"   if abs_ws >= 0.40 else
        "Moderate" if abs_ws >= 0.20 else
        "Weak"
    )

    return {
        "total_articles":  n,
        "avg_score":       round(avg_score,      4),
        "weighted_score":  round(weighted_score, 4),
        "pct_positive":    round(pos / n * 100,  1),
        "pct_negative":    round(neg / n * 100,  1),
        "pct_neutral":     round(neu / n * 100,  1),
        "overall_label":   overall_label,
        "signal_strength": signal_strength,
        "articles":        scored_articles,
        "model_used":      model_used,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — PREDICTION ADJUSTMENT
# ─────────────────────────────────────────────────────────────────────────────

def adjust_prediction_with_sentiment(
    model_signal:     str,
    model_confidence: float,
    agg:              dict,
) -> dict:
    """
    Reconcile ML model prediction with aggregated news sentiment.

    Rules:
        Agreement       → + 5 pp  (cap 98)
        Mild Conflict   → − 8 pp
        Strong Conflict → −15 pp + warning string

    Returns a dict with every key always present — never raises.
    """
    sent_label    = agg.get("overall_label",   "Neutral")
    sent_score    = agg.get("weighted_score",  0.0)
    sent_strength = agg.get("signal_strength", "Weak")

    model_dir = (
         1 if "BUY"  in model_signal else
        -1 if "SELL" in model_signal else
         0
    )
    sent_dir = (
         1 if sent_label == "Positive" else
        -1 if sent_label == "Negative" else
         0
    )

    if model_dir == 0 or sent_dir == 0:
        alignment, delta, warning = "Neutral", 0.0, ""
    elif model_dir == sent_dir:
        alignment, delta, warning = "Agreement", +5.0, ""
    else:
        if sent_strength == "Strong":
            alignment = "Strong Conflict"
            delta     = -15.0
            warning   = (
                f"⚠️ Strong Sentiment Conflict: Model signals {model_signal} "
                f"but news is strongly {sent_label.lower()} "
                f"(score: {sent_score:+.3f}). Exercise caution."
            )
        else:
            alignment = "Mild Conflict"
            delta     = -8.0
            warning   = (
                f"⚠️ Mild Sentiment Conflict: Model signals {model_signal} "
                f"while news leans {sent_label.lower()}."
            )

    adjusted_conf = float(np.clip(model_confidence + delta, 0.0, 98.0))

    return {
        "final_signal":    model_signal,
        "adjusted_conf":   round(adjusted_conf,    2),
        "original_conf":   round(model_confidence, 2),
        "delta_conf":      round(delta,            2),
        "alignment":       alignment,
        "warning":         warning,
        "sentiment_label": sent_label,
        "sentiment_score": sent_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CACHED PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)   # refresh every 30 min
def run_sentiment_pipeline() -> dict:
    """
    Execute the full sentiment pipeline end-to-end.
    Pipeline: fetch → clean → score → aggregate.

    CACHE NESTING FIX:
        _load_finbert() (@st.cache_resource) is called HERE — at the top level
        of this @st.cache_data function — before any nested call that might
        try to invoke it. Streamlit allows a cache_resource call from inside
        cache_data as long as it is a direct call (not nested through another
        cache_data). We keep the clf reference and pass it explicitly to
        score_sentiment() and aggregate_sentiment() so they never need to call
        _load_finbert() themselves.

        This pattern ensures:
            1. FinBERT is loaded at most ONCE per session (cache_resource).
            2. The pipeline result is cached for 30 min (cache_data).
            3. No illegal cache nesting occurs.
    """
    # Obtain clf at this call-site — safe to call cache_resource from cache_data
    clf     = _load_finbert()

    raw     = fetch_news()
    cleaned = clean_news(raw)
    scored  = score_sentiment(cleaned, clf=clf)   # clf passed explicitly
    return aggregate_sentiment(scored, clf=clf)    # clf passed explicitly


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT RENDER FUNCTION  (drop-in for dashboard.py)
# ─────────────────────────────────────────────────────────────────────────────

def render_sentiment_section(
    model_signal:     str   = "HOLD",
    model_confidence: float = 50.0,
) -> None:
    """
    Drop-in Streamlit section for the Sentiment Analysis page.

    Handles safely:
        - finbert_score == None      (FinBERT not available)
        - total_articles == 0        (no RSS data)
        - Plotly not installed        (charts section skipped gracefully)

    Usage in dashboard.py:
        from sentiment_pipeline import render_sentiment_section
        render_sentiment_section(live_signal, live_conf)
    """
    try:
        import plotly.graph_objects as go
        _PLOTLY = True
    except ImportError:
        go      = None
        _PLOTLY = False

    with st.spinner("🔄 Fetching live news & scoring sentiment…"):
        agg = run_sentiment_pipeline()

    articles   = agg.get("articles", [])
    no_data    = agg.get("total_articles", 0) == 0
    model_used = agg.get("model_used", "VADER only (FinBERT unavailable)")

    reconciled = adjust_prediction_with_sentiment(model_signal, model_confidence, agg)

    # ── NLP model badge ───────────────────────────────────────────────────────
    badge_color = "#388bfd" if "FinBERT" in model_used else "#d29922"
    st.markdown(
        f'<div style="font-size:0.75rem;color:{badge_color};'
        f'background:{badge_color}18;border:1px solid {badge_color}44;'
        f'border-radius:8px;padding:5px 14px;display:inline-block;margin-bottom:12px;">'
        f'🤖 NLP Model: <b>{model_used}</b></div>',
        unsafe_allow_html=True,
    )

    # ── Summary metrics ───────────────────────────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4)
    oc_map = {"Positive": "#10b981", "Negative": "#ef4444", "Neutral": "#eab308"}
    oc     = oc_map.get(agg.get("overall_label", "Neutral"), "#eab308")

    with mc1:
        st.metric("Articles (24h)", agg.get("total_articles", 0))
    with mc2:
        st.metric(
            "Weighted Sentiment",
            f"{agg.get('weighted_score', 0.0):+.4f}",
            agg.get("overall_label", "Neutral"),
        )
    with mc3:
        st.metric(
            "Distribution",
            f"🟢{agg.get('pct_positive', 0.0):.0f}%  "
            f"🔴{agg.get('pct_negative', 0.0):.0f}%  "
            f"⚪{agg.get('pct_neutral', 100.0):.0f}%",
        )
    with mc4:
        st.metric(
            "Model Alignment",
            reconciled.get("alignment", "Neutral"),
            f"Conf: {reconciled.get('adjusted_conf', 50.0):.1f}% "
            f"({reconciled.get('delta_conf', 0.0):+.0f}pp)",
        )

    if reconciled.get("warning"):
        st.warning(reconciled["warning"])

    # ── Overall sentiment banner ──────────────────────────────────────────────
    strength       = agg.get("signal_strength", "Weak")
    overall_label  = agg.get("overall_label", "Neutral")
    weighted_score = agg.get("weighted_score", 0.0)
    total_articles = agg.get("total_articles", 0)

    strength_icon = {"Strong": "🔥", "Moderate": "📊", "Weak": "🌤️"}.get(strength, "📊")
    st.markdown(f"""
    <div style="background:#111827;border-left:5px solid {oc};
                border-radius:0 14px 14px 0;padding:18px 24px;margin:16px 0;
                display:flex;align-items:center;gap:20px;">
        <div style="font-size:2.4rem;">{strength_icon}</div>
        <div>
            <div style="font-size:0.72rem;color:#64748b;letter-spacing:0.1em;
                        text-transform:uppercase;margin-bottom:4px;">
                Overall Sentiment Signal
            </div>
            <div style="font-size:1.5rem;font-weight:800;color:{oc};">
                {strength} {overall_label}
            </div>
            <div style="font-size:0.8rem;color:#94a3b8;margin-top:4px;">
                Score: {weighted_score:+.4f} &nbsp;·&nbsp;
                {total_articles} articles (24 h) &nbsp;·&nbsp;
                {model_used}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e2d4a;margin:20px 0'>", unsafe_allow_html=True)

    # ── Headlines feed ────────────────────────────────────────────────────────
    st.markdown("#### 📰 Live Financial Headlines")

    if no_data:
        st.info("📭 No recent news found. Check RSS connectivity or install feedparser.")
    else:
        lbl_colors = {"Positive": "#10b981", "Negative": "#ef4444", "Neutral": "#eab308"}

        for art in articles:
            try:
                lbl   = art.get("sentiment_label", "Neutral")
                score = art.get("combined_score",  0.0)
                clr   = lbl_colors.get(lbl, "#eab308")
                fb    = art.get("finbert_score")
                vd    = art.get("vader_score", 0.0)
                dt    = art.get("published_dt")
                ts    = dt.strftime("%d %b %Y, %H:%M UTC") if dt else "Unknown time"
                bar_pct = max(0, min(100, int((score + 1) / 2 * 100)))

                fb_html = (
                    f'<span>FinBERT: <b style="color:#a78bfa">{fb:+.4f}</b></span>'
                    if fb is not None
                    else '<span style="color:#3d444d;font-size:0.68rem;">FinBERT: N/A</span>'
                )

                st.markdown(f"""
                <div style="background:#111827;border-left:4px solid {clr};
                            border-radius:0 12px 12px 0;padding:14px 18px;margin-bottom:10px;">
                    <div style="color:#e2e8f0;font-size:0.9rem;font-weight:600;
                                margin-bottom:6px;">{art.get('title', '')}</div>
                    <div style="color:#64748b;font-size:0.72rem;margin-bottom:8px;">
                        🕐 {ts} &nbsp;·&nbsp; 📰 {art.get('source', 'Unknown')}
                    </div>
                    <div style="height:6px;background:#1e2d4a;border-radius:3px;
                                overflow:hidden;margin-bottom:8px;">
                        <div style="height:100%;width:{bar_pct}%;
                                    background:{clr};border-radius:3px;"></div>
                    </div>
                    <div style="font-size:0.75rem;display:flex;gap:16px;flex-wrap:wrap;">
                        {fb_html}
                        <span>VADER: <b style="color:#38bdf8">{vd:+.4f}</b></span>
                        <span>Combined: <b style="color:#e2e8f0">{score:+.4f}</b></span>
                        <span style="color:{clr};font-weight:700">{lbl}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                continue   # one bad article card must not crash the feed

    st.markdown("<hr style='border-color:#1e2d4a;margin:20px 0'>", unsafe_allow_html=True)

    # ── Charts (Plotly optional) ──────────────────────────────────────────────
    if _PLOTLY and total_articles > 0:
        try:
            col_pie, col_bar = st.columns([1, 1])

            with col_pie:
                st.markdown("#### 🥧 Sentiment Distribution")
                pos_n = round(agg.get("pct_positive", 0.0) / 100 * total_articles)
                neg_n = round(agg.get("pct_negative", 0.0) / 100 * total_articles)
                neu_n = max(total_articles - pos_n - neg_n, 0)
                fig_pie = go.Figure(go.Pie(
                    labels=["Positive", "Negative", "Neutral"],
                    values=[max(pos_n, 0), max(neg_n, 0), neu_n],
                    marker_colors=["#10b981", "#ef4444", "#eab308"],
                    hole=0.55, textfont_size=13,
                ))
                fig_pie.update_layout(
                    paper_bgcolor="#0a0e1a", font_color="#e2e8f0",
                    height=280, margin=dict(l=0, r=0, t=20, b=0),
                    legend=dict(bgcolor="#111827"),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_bar:
                st.markdown("#### 📊 Score Distribution")
                if articles:
                    score_vals   = [a.get("combined_score", 0.0) for a in articles]
                    score_clrs   = [
                        "#10b981" if s >= 0.1 else ("#ef4444" if s <= -0.1 else "#eab308")
                        for s in score_vals
                    ]
                    titles_short = [
                        (a.get("title", "")[:38] + "…") if len(a.get("title", "")) > 40
                        else a.get("title", "")
                        for a in articles
                    ]
                    fig_bar = go.Figure(go.Bar(
                        x=score_vals, y=titles_short, orientation="h",
                        marker_color=score_clrs,
                        text=[f"{s:+.3f}" for s in score_vals],
                        textposition="outside", textfont_color="#e2e8f0",
                    ))
                    fig_bar.add_vline(x=0, line_dash="dash", line_color="#475569", line_width=1)
                    fig_bar.update_layout(
                        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a", font_color="#e2e8f0",
                        height=max(280, len(articles) * 30),
                        margin=dict(l=0, r=60, t=20, b=0), showlegend=False,
                        xaxis=dict(gridcolor="#1e2d4a", range=[-1.1, 1.1]),
                        yaxis=dict(gridcolor="#1e2d4a", automargin=True),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
        except Exception:
            st.info("Charts could not be rendered.")

    st.markdown("<hr style='border-color:#1e2d4a;margin:20px 0'>", unsafe_allow_html=True)

    # ── ML × Sentiment reconciliation panel ──────────────────────────────────
    st.markdown("#### 🤝 ML Prediction × Sentiment Reconciliation")

    alignment    = reconciled.get("alignment", "Neutral")
    adj_conf     = reconciled.get("adjusted_conf", 50.0)
    orig_conf    = reconciled.get("original_conf", 50.0)
    delta_conf   = reconciled.get("delta_conf", 0.0)
    final_signal = reconciled.get("final_signal", "HOLD")
    sent_lbl     = reconciled.get("sentiment_label", "Neutral")
    sent_score   = reconciled.get("sentiment_score", 0.0)

    align_icon   = {
        "Agreement": "✅", "Mild Conflict": "⚠️",
        "Strong Conflict": "🚨", "Neutral": "⚪",
    }.get(alignment, "⚪")
    align_bg     = {
        "Agreement": "#064e3b", "Mild Conflict": "#3b3406",
        "Strong Conflict": "#450a0a", "Neutral": "#1e2d4a",
    }.get(alignment, "#1e2d4a")
    align_border = {
        "Agreement": "#10b981", "Mild Conflict": "#eab308",
        "Strong Conflict": "#ef4444", "Neutral": "#38bdf8",
    }.get(alignment, "#38bdf8")
    dsign = "+" if delta_conf >= 0 else ""

    st.markdown(f"""
    <div style="background:{align_bg};border:2px solid {align_border};
                border-radius:16px;padding:28px;margin-bottom:16px;">
        <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
            <div style="font-size:2.8rem;">{align_icon}</div>
            <div style="flex:1;min-width:200px;">
                <div style="font-size:1.2rem;font-weight:800;
                            color:{align_border};margin-bottom:6px;">{alignment}</div>
                <div style="color:#cbd5e1;font-size:0.88rem;line-height:1.6;">
                    Model: <b style="color:#e2e8f0">{final_signal}</b>
                    &nbsp;|&nbsp;
                    Sentiment: <b style="color:{align_border}">{sent_lbl}</b>
                    (score: {sent_score:+.4f})
                </div>
            </div>
            <div style="text-align:right;min-width:140px;">
                <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;
                            letter-spacing:0.08em;">Adjusted Confidence</div>
                <div style="font-size:2rem;font-weight:900;color:{align_border};">
                    {adj_conf:.1f}%
                </div>
                <div style="font-size:0.78rem;color:#94a3b8;">
                    Original: {orig_conf:.1f}%
                    &nbsp;→&nbsp;
                    <span style="color:{align_border}">{dsign}{delta_conf:.0f}pp</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<div style="color:#475569;font-size:0.72rem;text-align:right;margin-top:8px;">'
        f'🔄 {model_used} · Cached 30 min · '
        f'Last run: {datetime.now().strftime("%H:%M:%S")} · '
        f'Articles from last {HOURS_LOOKBACK} h</div>',
        unsafe_allow_html=True,
    )
