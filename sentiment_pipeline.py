"""
================================================================================
SENTIMENT PIPELINE MODULE — AI Stock Trading Decision Support System
File: sentiment_pipeline.py
================================================================================

ARCHITECTURE:
    fetch_news()               → Raw RSS headlines
    clean_news()               → Deduplicated, time-filtered
    score_sentiment()          → FinBERT (optional) + VADER per headline
    aggregate_sentiment()      → Overall signal + stats
    adjust_prediction_with_sentiment()  → ML output + sentiment reconciliation
    render_sentiment_section() → Drop-in Streamlit UI block

DEPENDENCY STRATEGY:
    vaderSentiment : always used — lightweight, reliable
    transformers   : OPTIONAL — loaded lazily via _safe_import_transformers(),
                     NEVER imported at module level
    torch          : OPTIONAL — pulled in by transformers only if available
    torchvision    : NOT required — text-classification never needs it

    The app NEVER crashes if transformers / torch / torchvision are absent.
    Every import that could fail is wrapped in try/except with a None fallback.
================================================================================
"""

import re
import hashlib
import warnings
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# SAFE LAZY IMPORT — called inside functions, never at module level
# ─────────────────────────────────────────────────────────────────────────────

def _safe_import_transformers():
    """
    Attempt to import transformers.pipeline and torch at call-time.

    By deferring this import we guarantee that missing transformers / torch /
    torchvision packages do NOT raise at module-load time.

    Env vars are set before import so HuggingFace never phones home or
    attempts a network download that could hang on Streamlit Cloud cold starts.

    Returns:
        (hf_pipeline_fn, torch_module)   on success
        (None, None)                     on any ImportError / Exception
    """
    try:
        import os
        os.environ.setdefault("TRANSFORMERS_OFFLINE",      "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM",    "false")  # avoids fork warning

        from transformers import pipeline as hf_pipeline  # noqa: PLC0415
        import torch                                       # noqa: PLC0415
        return hf_pipeline, torch

    except ImportError:
        # transformers or torch simply not installed — expected on Cloud free tier
        return None, None

    except Exception:
        # Catch-all: covers torchvision errors, CUDA init failures, etc.
        return None, None


# ─── VADER — lightweight, attempted once at module load ──────────────────────
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

# Thresholds for sentiment labelling
_POS_THRESH      =  0.05
_NEG_THRESH      = -0.05


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — NEWS FETCH
# ─────────────────────────────────────────────────────────────────────────────

def fetch_news(max_articles: int = MAX_ARTICLES) -> list:
    """
    Fetch raw headlines from RSS feeds.

    Returns list of dicts: { title, published_raw, source, url }
    Returns [] on any failure — never raises.
    """
    if not FEEDPARSER_OK:
        st.warning("⚠️ feedparser not installed — run: pip install feedparser")
        return []

    articles = []
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
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

    return articles


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

    seen    = set()
    cleaned = []
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
        key=lambda a: a["published_dt"] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return pool[:max_articles]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — MODEL LOADERS  (session-cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_finbert():
    """
    Load FinBERT text-classification pipeline (CPU-only).

    Uses _safe_import_transformers() so this function is safe to call even
    when transformers, torch, or torchvision are not installed.

    Returns:
        HuggingFace pipeline object   on success
        None                          on any failure (silent — VADER takes over)
    """
    hf_pipeline, _torch = _safe_import_transformers()
    if hf_pipeline is None:
        return None   # transformers not installed — silent fallback to VADER

    try:
        clf = hf_pipeline(
            "text-classification",
            model=FINBERT_MODEL,
            top_k=None,        # return all 3 class scores per input
            truncation=True,
            max_length=512,
            device=-1,         # CPU only — zero GPU dependency
        )
        return clf
    except Exception:
        return None            # model download failed, OOM, etc.


@st.cache_resource(show_spinner=False)
def _load_vader():
    """
    Load VADER SentimentIntensityAnalyzer.

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
        list[float | None]
            float — valid score in [-1, +1]
            None  — inference failed for this item (treated as "no FinBERT")

    Never raises. Handles every malformed output shape safely.
    """
    if clf is None or not texts:
        return [None] * len(texts)

    try:
        raw_results = clf(texts)
    except Exception:
        return [None] * len(texts)

    # clf(texts) must return a list — one element per input.
    # If it returned anything else (dict, None, etc.) bail out entirely.
    if not isinstance(raw_results, list):
        return [None] * len(texts)

    scores = []
    for result in raw_results:
        try:
            # top_k=None → result is a list of {label, score} dicts for ONE input
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
                    # neutral contributes 0
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
                # Unexpected type — safe neutral fallback, not None, so blend still runs
                scores.append(None)

        except Exception:
            scores.append(None)

    # Safety pad: clf should never return fewer results than inputs, but guard anyway
    while len(scores) < len(texts):
        scores.append(None)

    return scores


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

def score_sentiment(articles: list):
    """
    Score each article dict with VADER (always) and FinBERT (when available).

    Blending:
        FinBERT available  →  combined = 0.70 * finbert + 0.30 * vader
        FinBERT missing    →  combined = vader only

    Adds to each article dict:
        vader_score    : float          always in [-1, +1]
        finbert_score  : float | None   [-1, +1] or None when unavailable
        combined_score : float          always in [-1, +1]
        sentiment_label: str            "Positive" | "Negative" | "Neutral"

    Returns:
        (scored_articles: list, clf: pipeline | None)
        clf is returned so the caller can pass it to aggregate_sentiment()
        without triggering a second _load_finbert() cache lookup.

    Never raises — every failure path produces a neutral (0.0) score.
    """
    vader   = _load_vader()
    finbert = _load_finbert()   # cached — loaded exactly once per session

    if not articles:
        return [], finbert

    titles         = [art["title"] for art in articles]
    finbert_scores = _finbert_batch_score(finbert, titles)

    scored = []
    for art, fb in zip(articles, finbert_scores):

        # ── VADER ─────────────────────────────────────────────────────────────
        vs = _vader_score_single(vader, art["title"])

        # ── Hybrid blend ──────────────────────────────────────────────────────
        if fb is not None:
            combined = float(np.clip(FINBERT_WEIGHT * fb + VADER_WEIGHT * vs, -1.0, 1.0))
        else:
            combined = float(np.clip(vs, -1.0, 1.0))

        # ── Label ─────────────────────────────────────────────────────────────
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

    return scored, finbert


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def _active_model_label(clf) -> str:
    """
    Return a human-readable NLP backend label.

    Accepts the already-loaded clf object so _load_finbert() is NEVER called
    a second time — eliminates the redundant cache-lookup / double-load issue.

    Args:
        clf: the result of _load_finbert() (pipeline object or None)
    """
    if clf is not None:
        return "FinBERT + VADER"
    if VADER_OK:
        return "VADER only (FinBERT unavailable)"
    return "No NLP model loaded"


def aggregate_sentiment(scored_articles: list, clf=None) -> dict:
    """
    Compute overall sentiment from scored articles using recency weighting.

    Args:
        scored_articles: output of score_sentiment()
        clf:             the finbert pipeline object (or None) — passed in so
                         _active_model_label() never calls _load_finbert() again.

    Weights:
        Age ≤ 6 h  → 2.0
        Age ≤ 12 h → 1.5
        Older      → 1.0

    Returns a dict with every key always present — never raises.
    """
    model_used = _active_model_label(clf)

    base = {
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
        return base

    now_utc = datetime.now(timezone.utc)
    scores, weights = [], []
    pos = neg = neu = 0

    for art in scored_articles:
        s = art.get("combined_score", 0.0)
        scores.append(s)

        dt = art.get("published_dt")
        if dt is not None:
            age_h = (now_utc - dt).total_seconds() / 3600
            w = 2.0 if age_h <= 6 else (1.5 if age_h <= 12 else 1.0)
        else:
            w = 1.0
        weights.append(w)

        lbl = art.get("sentiment_label", "Neutral")
        if lbl == "Positive":   pos += 1
        elif lbl == "Negative": neg += 1
        else:                   neu += 1

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
        Agreement       → + 5 pp (cap 98)
        Mild Conflict   → − 8 pp
        Strong Conflict → −15 pp + warning string

    Returns a dict with every key always present — never raises.
    """
    sent_label    = agg.get("overall_label",    "Neutral")
    sent_score    = agg.get("weighted_score",   0.0)
    sent_strength = agg.get("signal_strength",  "Weak")

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

    fetch → clean → score → aggregate

    _load_finbert() is called exactly ONCE inside score_sentiment().
    The resulting clf object is threaded through to aggregate_sentiment()
    so _active_model_label() never triggers a second model load.

    Cached for 30 minutes to avoid hammering RSS endpoints and re-running
    FinBERT inference on every page reload.
    Safe to call even when transformers / torch are not installed.
    """
    raw            = fetch_news()
    cleaned        = clean_news(raw)
    scored, clf    = score_sentiment(cleaned)   # clf loaded exactly once
    return aggregate_sentiment(scored, clf=clf)  # passed in — no reload


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
        - Plotly not installed        (charts section skipped)

    Usage in dashboard.py:
        from sentiment_pipeline import render_sentiment_section
        render_sentiment_section(live_signal, live_conf)
    """
    # ── Plotly is optional ────────────────────────────────────────────────────
    try:
        import plotly.graph_objects as go
        _PLOTLY = True
    except ImportError:
        go      = None
        _PLOTLY = False

    # ── Run pipeline ──────────────────────────────────────────────────────────
    with st.spinner("🔄 Fetching live news & scoring sentiment…"):
        agg = run_sentiment_pipeline()

    articles   = agg.get("articles", [])
    no_data    = agg["total_articles"] == 0
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
    oc     = oc_map.get(agg["overall_label"], "#eab308")

    with mc1:
        st.metric("Articles (24h)", agg["total_articles"])
    with mc2:
        st.metric("Weighted Sentiment", f"{agg['weighted_score']:+.4f}", agg["overall_label"])
    with mc3:
        st.metric(
            "Distribution",
            f"🟢{agg['pct_positive']:.0f}%  🔴{agg['pct_negative']:.0f}%  ⚪{agg['pct_neutral']:.0f}%",
        )
    with mc4:
        align_color_map = {
            "Agreement":       "#10b981",
            "Mild Conflict":   "#eab308",
            "Strong Conflict": "#ef4444",
            "Neutral":         "#64748b",
        }
        st.metric(
            "Model Alignment",
            reconciled["alignment"],
            f"Conf: {reconciled['adjusted_conf']:.1f}% ({reconciled['delta_conf']:+.0f}pp)",
        )

    if reconciled["warning"]:
        st.warning(reconciled["warning"])

    # ── Overall sentiment banner ──────────────────────────────────────────────
    strength_icon = {"Strong": "🔥", "Moderate": "📊", "Weak": "🌤️"}.get(
        agg["signal_strength"], "📊"
    )
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
                {agg['signal_strength']} {agg['overall_label']}
            </div>
            <div style="font-size:0.8rem;color:#94a3b8;margin-top:4px;">
                Score: {agg['weighted_score']:+.4f} &nbsp;·&nbsp;
                {agg['total_articles']} articles (24 h) &nbsp;·&nbsp;
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
            lbl     = art.get("sentiment_label", "Neutral")
            score   = art.get("combined_score",  0.0)
            clr     = lbl_colors.get(lbl, "#eab308")
            fb      = art.get("finbert_score")            # may be None — guard below
            vd      = art.get("vader_score", 0.0)
            dt      = art.get("published_dt")
            ts      = dt.strftime("%d %b %Y, %H:%M UTC") if dt else "Unknown time"
            bar_pct = int((score + 1) / 2 * 100)

            # FinBERT column — show N/A when not computed
            fb_html = (
                f'<span>FinBERT: <b style="color:#a78bfa">{fb:+.4f}</b></span>'
                if fb is not None
                else '<span style="color:#3d444d;font-size:0.68rem;">FinBERT: N/A</span>'
            )

            st.markdown(f"""
            <div style="background:#111827;border-left:4px solid {clr};
                        border-radius:0 12px 12px 0;padding:14px 18px;margin-bottom:10px;">
                <div style="color:#e2e8f0;font-size:0.9rem;font-weight:600;
                            margin-bottom:6px;">{art['title']}</div>
                <div style="color:#64748b;font-size:0.72rem;margin-bottom:8px;">
                    🕐 {ts} &nbsp;·&nbsp; 📰 {art.get('source','Unknown')}
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

    st.markdown("<hr style='border-color:#1e2d4a;margin:20px 0'>", unsafe_allow_html=True)

    # ── Charts (Plotly optional) ──────────────────────────────────────────────
    if _PLOTLY and agg["total_articles"] > 0:
        col_pie, col_bar = st.columns([1, 1])

        with col_pie:
            st.markdown("#### 🥧 Sentiment Distribution")
            pos_n = round(agg["pct_positive"] / 100 * agg["total_articles"])
            neg_n = round(agg["pct_negative"] / 100 * agg["total_articles"])
            neu_n = max(agg["total_articles"] - pos_n - neg_n, 0)
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
                    (a["title"][:38] + "…") if len(a["title"]) > 40 else a["title"]
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

    st.markdown("<hr style='border-color:#1e2d4a;margin:20px 0'>", unsafe_allow_html=True)

    # ── ML × Sentiment reconciliation panel ──────────────────────────────────
    st.markdown("#### 🤝 ML Prediction × Sentiment Reconciliation")

    align_icon   = {"Agreement": "✅", "Mild Conflict": "⚠️",
                    "Strong Conflict": "🚨", "Neutral": "⚪"}.get(reconciled["alignment"], "⚪")
    align_bg     = {"Agreement": "#064e3b", "Mild Conflict": "#3b3406",
                    "Strong Conflict": "#450a0a", "Neutral": "#1e2d4a"}.get(reconciled["alignment"], "#1e2d4a")
    align_border = {"Agreement": "#10b981", "Mild Conflict": "#eab308",
                    "Strong Conflict": "#ef4444", "Neutral": "#38bdf8"}.get(reconciled["alignment"], "#38bdf8")
    dsign        = "+" if reconciled["delta_conf"] >= 0 else ""

    st.markdown(f"""
    <div style="background:{align_bg};border:2px solid {align_border};
                border-radius:16px;padding:28px;margin-bottom:16px;">
        <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
            <div style="font-size:2.8rem;">{align_icon}</div>
            <div style="flex:1;min-width:200px;">
                <div style="font-size:1.2rem;font-weight:800;
                            color:{align_border};margin-bottom:6px;">
                    {reconciled['alignment']}
                </div>
                <div style="color:#cbd5e1;font-size:0.88rem;line-height:1.6;">
                    Model: <b style="color:#e2e8f0">{reconciled['final_signal']}</b>
                    &nbsp;|&nbsp;
                    Sentiment: <b style="color:{align_border}">{reconciled['sentiment_label']}</b>
                    (score: {reconciled['sentiment_score']:+.4f})
                </div>
            </div>
            <div style="text-align:right;min-width:140px;">
                <div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;
                            letter-spacing:0.08em;">Adjusted Confidence</div>
                <div style="font-size:2rem;font-weight:900;color:{align_border};">
                    {reconciled['adjusted_conf']:.1f}%
                </div>
                <div style="font-size:0.78rem;color:#94a3b8;">
                    Original: {reconciled['original_conf']:.1f}%
                    &nbsp;→&nbsp;
                    <span style="color:{align_border}">
                        {dsign}{reconciled['delta_conf']:.0f}pp
                    </span>
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
