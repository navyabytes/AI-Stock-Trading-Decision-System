import streamlit as st
import nltk
from sentiment_pipeline import *

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

import re
import time
import hashlib
import warnings
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")

# ─── Optional heavy imports (graceful fallback) ───────────────────────────────
try:
    import feedparser
    FEEDPARSER_OK = True
except ImportError:
    FEEDPARSER_OK = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_OK = True
except ImportError:
    VADER_OK = False

try:
    from transformers import pipeline as hf_pipeline
    FINBERT_OK = True
except Exception:
    hf_pipeline = None
    FINBERT_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

RSS_FEEDS = [
    "https://news.google.com/rss/search?q=Reliance+Industries&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=RELIANCE+NSE+stock&hl=en-IN&gl=IN&ceid=IN:en",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=RELIANCE.NS&region=US&lang=en-US",
]

FINBERT_MODEL   = "ProsusAI/finbert"
FINBERT_WEIGHT  = 0.70
VADER_WEIGHT    = 0.30
MAX_ARTICLES    = 20
HOURS_LOOKBACK  = 24


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — NEWS PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def fetch_news(max_articles: int = MAX_ARTICLES) -> list[dict]:
    """
    Fetch raw news from Google News RSS.

    Returns list of dicts:
        { title, published_raw, source, url }

    Handles:
        - feedparser not installed  → returns []
        - Network timeout           → returns []
        - Empty feed                → returns []
    """
    if not FEEDPARSER_OK:
        return []

    articles = []
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            if not feed.entries:
                continue
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                if not title:
                    continue
                published = entry.get("published", entry.get("updated", ""))
                source    = entry.get("source", {}).get("title", "Unknown")
                url       = entry.get("link", "")
                articles.append({
                    "title":         title,
                    "published_raw": published,
                    "source":        source,
                    "url":           url,
                })
        except Exception:
            continue  # Don't crash on one bad feed

    return articles[:max_articles * 2]   # Fetch extra before dedup/filter


def _parse_published(raw: str) -> Optional[datetime]:
    """Parse RSS date strings into UTC-aware datetime. Returns None on failure."""
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
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    # Fallback: try email.utils
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _headline_fingerprint(title: str) -> str:
    """Short hash for deduplication — ignores punctuation/case differences."""
    normalized = re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()
    return hashlib.md5(normalized.encode()).hexdigest()


def clean_news(
    raw_articles: list[dict],
    hours_lookback: int = HOURS_LOOKBACK,
    max_articles:   int = MAX_ARTICLES,
) -> list[dict]:
    """
    Deduplicate and time-filter raw news.

    - Removes headlines with identical fingerprints
    - Keeps only articles published within `hours_lookback` hours
    - Attaches parsed `published_dt` (UTC-aware datetime)
    - Returns up to `max_articles` sorted newest-first

    Falls back gracefully:
        - If ALL articles have unparseable dates, keeps newest `max_articles`
          without time filtering (marks published_dt = None).
    """
    if not raw_articles:
        return []

    now_utc = datetime.now(timezone.utc)
    cutoff  = now_utc - timedelta(hours=hours_lookback)

    seen_fingerprints = set()
    cleaned = []

    for art in raw_articles:
        fp = _headline_fingerprint(art["title"])
        if fp in seen_fingerprints:
            continue
        seen_fingerprints.add(fp)

        dt = _parse_published(art["published_raw"])
        art["published_dt"] = dt
        cleaned.append(art)

    # Time filter — only keep recent articles
    time_filtered = [a for a in cleaned if a["published_dt"] is not None
                     and a["published_dt"] >= cutoff]

    # Fallback: if nothing passes time filter, use all (no time gate)
    if not time_filtered and cleaned:
        time_filtered = cleaned   # Show something rather than nothing

    # Sort newest first
    time_filtered.sort(
        key=lambda a: a["published_dt"] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )

    return time_filtered[:max_articles]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — SENTIMENT SCORING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_finbert():
    if not FINBERT_OK or hf_pipeline is None:
        return None
    try:
        clf = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            top_k=None,
            truncation=True
        )
        return clf
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _load_vader():
    """Load VADER once and cache for session lifetime."""
    if not VADER_OK:
        return None
    try:
        return SentimentIntensityAnalyzer()
    except Exception:
        return None


def _finbert_score(clf, title: str) -> float:
    """
    Run FinBERT on a single headline.
    Maps {positive→+1, neutral→0, negative→-1} weighted by probability.
    Returns float in [-1, +1], or 0.0 on failure.
    """
    if clf is None:
        return 0.0 * len(titles)
    try:
        results = clf(title)
        # results is list of list of dicts: [[{label, score}, ...]]
        if results and isinstance(results[0], list):
            results = results[0]
        score_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        total = sum(r["score"] * score_map.get(r["label"].lower(), 0.0)
                    for r in results)
        return float(np.clip(total, -1.0, 1.0))
    except Exception:
        return 0.0


def _vader_score(analyzer, title: str) -> float:
    """
    Run VADER on a single headline.
    Returns compound score in [-1, +1], or 0.0 on failure.
    """
    if analyzer is None:
        return 0.0
    try:
        return float(analyzer.polarity_scores(title)["compound"])
    except Exception:
        return 0.0


def score_sentiment(articles: list[dict]) -> list[dict]:
    """
    Score each article with FinBERT + VADER.

    Adds to each article dict:
        finbert_score  : float [-1, +1]
        vader_score    : float [-1, +1]
        combined_score : float [-1, +1]  (weighted blend)
        sentiment_label: str   (Positive / Negative / Neutral)

    Blending:
        combined = 0.70 * finbert + 0.30 * vader

    If both models unavailable, combined_score = 0.0 (neutral fallback).
    """
    if not articles:
        return []

    clf      = _load_finbert()
    analyzer = _load_vader()

    scored = []
    for art in articles:
        title = art["title"]

        fb   = _finbert_score(clf,      title)
        vd   = _vader_score(analyzer,  title)

        # Weighted blend
        if clf is not None and analyzer is not None:
            combined = FINBERT_WEIGHT * fb + VADER_WEIGHT * vd
        elif clf is not None:
            combined = fb
        elif analyzer is not None:
            combined = vd
        else:
            combined = 0.0

        combined = float(np.clip(combined, -1.0, 1.0))

        # Label thresholds
        if combined >= 0.10:
            label = "Positive"
        elif combined <= -0.10:
            label = "Negative"
        else:
            label = "Neutral"

        scored.append({
            **art,
            "finbert_score":   round(fb,       4),
            "vader_score":     round(vd,       4),
            "combined_score":  round(combined, 4),
            "sentiment_label": label,
        })

    return scored


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_sentiment(scored_articles: list[dict]) -> dict:
    """
    Compute overall sentiment signal from scored articles.

    Weighting:
        Recency-weighted mean — articles from last 6h get weight 2x,
        last 12h get 1.5x, older get 1x.

    Returns:
        {
          total_articles   : int
          avg_score        : float
          weighted_score   : float
          pct_positive     : float
          pct_negative     : float
          pct_neutral      : float
          overall_label    : str   (Positive / Negative / Neutral)
          signal_strength  : str   (Strong / Moderate / Weak)
          articles         : list  (same as input — for convenience)
        }
    """
    if not scored_articles:
        return {
            "total_articles": 0,
            "avg_score":       0.0,
            "weighted_score":  0.0,
            "pct_positive":    0.0,
            "pct_negative":    0.0,
            "pct_neutral":     100.0,
            "overall_label":   "Neutral",
            "signal_strength": "Weak",
            "articles":        [],
        }

    now_utc = datetime.now(timezone.utc)
    scores  = []
    weights = []

    pos = neg = neu = 0

    for art in scored_articles:
        s = art["combined_score"]
        scores.append(s)

        # Recency weight
        dt = art.get("published_dt")
        if dt is not None:
            age_hours = (now_utc - dt).total_seconds() / 3600
            if age_hours <= 6:
                w = 2.0
            elif age_hours <= 12:
                w = 1.5
            else:
                w = 1.0
        else:
            w = 1.0
        weights.append(w)

        lbl = art["sentiment_label"]
        if lbl == "Positive":    pos += 1
        elif lbl == "Negative":  neg += 1
        else:                    neu += 1

    n = len(scores)
    avg_score      = float(np.mean(scores))
    weighted_score = float(np.average(scores, weights=weights))

    pct_pos = round(pos / n * 100, 1)
    pct_neg = round(neg / n * 100, 1)
    pct_neu = round(neu / n * 100, 1)

    # Overall label from weighted score
    if weighted_score >= 0.10:
        overall_label = "Positive"
    elif weighted_score <= -0.10:
        overall_label = "Negative"
    else:
        overall_label = "Neutral"

    # Signal strength
    abs_score = abs(weighted_score)
    if abs_score >= 0.40:
        signal_strength = "Strong"
    elif abs_score >= 0.20:
        signal_strength = "Moderate"
    else:
        signal_strength = "Weak"

    return {
        "total_articles":  n,
        "avg_score":       round(avg_score,      4),
        "weighted_score":  round(weighted_score, 4),
        "pct_positive":    pct_pos,
        "pct_negative":    pct_neg,
        "pct_neutral":     pct_neu,
        "overall_label":   overall_label,
        "signal_strength": signal_strength,
        "articles":        scored_articles,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — PREDICTION ADJUSTMENT
# ─────────────────────────────────────────────────────────────────────────────

def adjust_prediction_with_sentiment(
    model_signal:     str,
    model_confidence: float,
    agg:              dict,
) -> dict:
    """
    Reconcile ML model prediction with aggregated sentiment.

    Rules:
        Agreement   → confidence +5pp (capped at 98)
        Mild conflict   → confidence -8pp
        Strong conflict → confidence -15pp + warning flag

    Returns:
        {
          final_signal      : str    (same as model_signal)
          adjusted_conf     : float
          delta_conf        : float  (change applied)
          alignment         : str    (Agreement / Mild Conflict / Strong Conflict / Neutral)
          warning           : str    (empty string or warning message)
          sentiment_label   : str
          sentiment_score   : float
        }
    """
    sent_label  = agg.get("overall_label",   "Neutral")
    sent_score  = agg.get("weighted_score",  0.0)
    sent_strength = agg.get("signal_strength", "Weak")

    # Directional mapping
    model_dir = (
        1  if "BUY"  in model_signal else
        -1 if "SELL" in model_signal else
        0
    )
    sent_dir = (
        1  if sent_label == "Positive" else
        -1 if sent_label == "Negative" else
        0
    )

    # Determine alignment
    if model_dir == 0 or sent_dir == 0:
        alignment = "Neutral"
        delta     = 0.0
        warning   = ""

    elif model_dir == sent_dir:
        alignment = "Agreement"
        delta     = +5.0
        warning   = ""

    else:
        # Conflict — magnitude depends on sentiment strength
        if sent_strength == "Strong":
            alignment = "Strong Conflict"
            delta     = -15.0
            warning   = (
                f"⚠️ Strong Sentiment Conflict: Model signals {model_signal} "
                f"but sentiment is strongly {sent_label.lower()} "
                f"(score: {sent_score:+.3f}). Exercise caution."
            )
        else:
            alignment = "Mild Conflict"
            delta     = -8.0
            warning   = (
                f"⚠️ Mild Sentiment Conflict: Model signals {model_signal} "
                f"while news sentiment leans {sent_label.lower()}."
            )

    # Safety check
    if model_confidence is None:
        model_confidence = 50.0
    
    if delta is None:
        delta = 0.0
    
    # Ensure numeric safely
    try:
        model_confidence = float(model_confidence)
    except (TypeError, ValueError):
        model_confidence = 50.0
    delta = float(delta)
    
    adjusted_conf = float(np.clip(model_confidence + delta, 0.0, 98.0))

    return {
        "final_signal":   model_signal,
        "adjusted_conf":  round(adjusted_conf, 2),
        "original_conf":  round(model_confidence, 2),
        "delta_conf":     round(delta, 2),
        "alignment":      alignment,
        "warning":        warning,
        "sentiment_label":sent_label,
        "sentiment_score":sent_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CACHED FULL PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)  # Refresh every 30 min
def run_sentiment_pipeline() -> dict:
    """
    Run the full pipeline: fetch → clean → score → aggregate.
    Returns the aggregation dict (which contains .articles list).
    Cached for 30 minutes to avoid hammering RSS + model inference.
    """
    raw      = fetch_news()
    cleaned  = clean_news(raw)
    scored   = score_sentiment(cleaned)
    agg      = aggregate_sentiment(scored)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — STREAMLIT RENDER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def render_sentiment_section(
    model_signal:     str   = "HOLD",
    model_confidence: float = 50.0,
):
    """
    Clean fintech-grade Sentiment Analysis dashboard section.
    Drop-in replacement — same signature, fully refactored UI only.
    """
    import plotly.graph_objects as go
    from datetime import datetime

    # ── Run pipeline ──────────────────────────────────────────────────────────
    with st.spinner("Fetching live news & running sentiment models…"):
        agg = run_sentiment_pipeline()

    articles   = agg.get("articles", [])
    no_data    = agg["total_articles"] == 0
    reconciled = adjust_prediction_with_sentiment(model_signal, model_confidence, agg)

    # ── Design tokens ─────────────────────────────────────────────────────────
    COLORS = {
        "Positive": "#10b981",
        "Negative": "#ef4444",
        "Neutral":  "#f59e0b",
        "bg_card":  "#0f172a",
        "bg_deep":  "#080d18",
        "border":   "#1e293b",
        "text_muted": "#64748b",
        "text_sub":   "#94a3b8",
        "text_main":  "#e2e8f0",
    }
    ALIGN_COLORS = {
        "Agreement":       "#10b981",
        "Mild Conflict":   "#f59e0b",
        "Strong Conflict": "#ef4444",
        "Neutral":         "#64748b",
    }

    sentiment_color  = COLORS.get(agg["overall_label"], COLORS["Neutral"])
    alignment_color  = ALIGN_COLORS.get(reconciled["alignment"], "#64748b")
    strength_pct     = {"Strong": 92, "Moderate": 62, "Weak": 28}.get(agg["signal_strength"], 40)

    # ── Global style injection ────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    .snt-root { font-family: 'Sora', sans-serif; }

    .snt-metric-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 20px 22px;
        height: 100%;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .snt-metric-label {
        font-size: 0.67rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 8px;
    }
    .snt-metric-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #e2e8f0;
        line-height: 1;
        margin-bottom: 4px;
    }
    .snt-metric-sub {
        font-size: 0.75rem;
        color: #64748b;
    }
    .snt-main-card {
        background: linear-gradient(135deg, #0f172a 0%, #0c1628 100%);
        border: 1px solid #1e293b;
        border-radius: 20px;
        padding: 32px 36px;
        margin: 20px 0 8px;
        position: relative;
        overflow: hidden;
    }
    .snt-main-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 4px; height: 100%;
        border-radius: 20px 0 0 20px;
    }
    .snt-headline-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 10px;
        transition: all 0.2s ease;
    }
    .snt-headline-card:hover {
        border-color: #334155;
        transform: translateY(-2px);
        transition: all 0.2s ease;
    }
    .snt-badge {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
    }
    .snt-insight-box {
        background: #0c1628;
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 22px 26px;
        margin: 20px 0;
    }
    .snt-recon-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 24px 28px;
        margin-top: 8px;
    }
    .snt-divider {
        border: none;
        border-top: 1px solid #1e293b;
        margin: 28px 0;
    }
    .snt-section-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 14px;
    }
    </style>
    <div class="snt-root">
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    # BLOCK 1 — KPI ROW
    # ═══════════════════════════════════════════════════════════════════
    k1, k2, k3, k4 = st.columns(4, gap="small")

    with k1:
        st.markdown(f"""
        <div class="snt-metric-card">
            <div class="snt-metric-label">Articles (24h)</div>
            <div class="snt-metric-value">{agg['total_articles']}</div>
            <div class="snt-metric-sub">from live RSS feeds</div>
        </div>""", unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="snt-metric-card">
            <div class="snt-metric-label">Overall Sentiment</div>
            <div class="snt-metric-value" style="color:{sentiment_color}">{agg['overall_label']}</div>
            <div class="snt-metric-sub">{agg['pct_positive']:.0f}% positive · {agg['pct_negative']:.0f}% negative</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        delta_sign = "+" if reconciled["delta_conf"] >= 0 else ""
        st.markdown(f"""
        <div class="snt-metric-card">
            <div class="snt-metric-label">Adjusted Confidence</div>
            <div class="snt-metric-value">{reconciled['adjusted_conf']:.1f}%</div>
            <div class="snt-metric-sub">{delta_sign}{reconciled['delta_conf']:.0f}pp from sentiment</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        strength_colors = {"Strong": "#10b981", "Moderate": "#f59e0b", "Weak": "#64748b"}
        sc = strength_colors.get(agg["signal_strength"], "#64748b")
        st.markdown(f"""
        <div class="snt-metric-card">
            <div class="snt-metric-label">Signal Strength</div>
            <div class="snt-metric-value" style="color:{sc}">{agg['signal_strength']}</div>
            <div class="snt-metric-sub">weighted score: {agg['weighted_score']:+.3f}</div>
        </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    # BLOCK 2 — MAIN SENTIMENT CARD
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    strength_prose = {
        "Strong":   "Strong conviction across recent headlines.",
        "Moderate": "Moderate signal — mixed but directional coverage.",
        "Weak":     "Low conviction — news sentiment is inconclusive.",
    }.get(agg["signal_strength"], "")

    outlook_map = {
        ("Positive", "Strong"):   "Bullish outlook — news momentum supports upside.",
        ("Positive", "Moderate"): "Cautiously bullish — positive lean in recent coverage.",
        ("Positive", "Weak"):     "Slight positive tilt — insufficient news volume to confirm.",
        ("Negative", "Strong"):   "Bearish outlook — consistent negative press.",
        ("Negative", "Moderate"): "Cautiously bearish — negative tone in recent news.",
        ("Negative", "Weak"):     "Slight negative tilt — not enough data to confirm.",
        ("Neutral",  "Strong"):   "Balanced coverage — no clear directional bias.",
        ("Neutral",  "Moderate"): "Mixed sentiment — market awaiting a catalyst.",
        ("Neutral",  "Weak"):     "Quiet news cycle — low informational signal.",
    }
    one_liner = outlook_map.get(
        (agg["overall_label"], agg["signal_strength"]),
        "Sentiment data is being analysed."
    )

    # Confidence ring via SVG
    r = 28
    circ = 2 * 3.14159 * r
    dash_len = (reconciled["adjusted_conf"] / 100) * circ

    st.markdown(f"""
    <div class="snt-main-card" style="border-left: 4px solid {sentiment_color};">
        <div style="display:flex; align-items:center; gap:28px; flex-wrap:wrap;">

            <!-- Confidence ring -->
            <div style="flex-shrink:0; text-align:center;">
                <svg width="80" height="80" viewBox="0 0 80 80">
                    <circle cx="40" cy="40" r="{r}" fill="none" stroke="#1e293b" stroke-width="7"/>
                    <circle cx="40" cy="40" r="{r}" fill="none"
                        stroke="{sentiment_color}" stroke-width="7"
                        stroke-dasharray="{dash_len:.1f} {circ:.1f}"
                        stroke-linecap="round"
                        transform="rotate(-90 40 40)"/>
                    <text x="40" y="44" text-anchor="middle"
                        font-family="Sora,sans-serif" font-size="13"
                        font-weight="800" fill="{sentiment_color}">
                        {reconciled['adjusted_conf']:.0f}%
                    </text>
                </svg>
                <div style="font-size:0.65rem; color:#475569; margin-top:2px; letter-spacing:0.06em;">CONFIDENCE</div>
            </div>

            <!-- Main label -->
            <div style="flex:1; min-width:180px;">
                <div style="font-size:0.68rem; font-weight:600; letter-spacing:0.14em;
                            text-transform:uppercase; color:#475569; margin-bottom:8px;">
                    Market Sentiment
                </div>
                <div style="font-size:2.4rem; font-weight:800; color:{sentiment_color};
                            line-height:1; margin-bottom:10px;">
                    {agg['overall_label']}
                </div>
                <div style="font-size:0.88rem; color:#94a3b8; line-height:1.5;">
                    {one_liner}
                </div>
            </div>

            <!-- Strength pill -->
            <div style="flex-shrink:0; text-align:right; min-width:110px;">
                <div style="font-size:0.68rem; font-weight:600; letter-spacing:0.1em;
                            text-transform:uppercase; color:#475569; margin-bottom:8px;">
                    Signal
                </div>
                <div style="
                    display:inline-block;
                    background: {sentiment_color}18;
                    border: 1.5px solid {sentiment_color}55;
                    color: {sentiment_color};
                    border-radius: 8px;
                    padding: 8px 16px;
                    font-size:1.1rem;
                    font-weight:800;
                    letter-spacing:0.04em;
                ">
                    {agg['signal_strength'].upper()}
                </div>
                <div style="font-size:0.72rem; color:#475569; margin-top:8px;">
                    {agg['total_articles']} articles · 24h window
                </div>
            </div>

        </div>
    </div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    # BLOCK 3 — HEADLINES
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("<hr class='snt-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='snt-section-title'>📰 Live Headlines</div>", unsafe_allow_html=True)

    if no_data:
        st.info("No recent news found. Check RSS feeds or install: `pip install feedparser vaderSentiment transformers torch`")
    else:
        badge_bg = {"Positive": "#10b98122", "Negative": "#ef444422", "Neutral": "#f59e0b22"}
        badge_clr = {"Positive": "#10b981",   "Negative": "#ef4444",   "Neutral": "#f59e0b"}
        bar_bg = {"Positive": "#10b981",       "Negative": "#ef4444",   "Neutral": "#f59e0b"}

        for art in articles:
            lbl   = art["sentiment_label"]
            score = art["combined_score"]
            clr   = badge_clr.get(lbl,  "#f59e0b")
            bbg   = badge_bg.get(lbl,   "#f59e0b22")
            bclr  = bar_bg.get(lbl,     "#f59e0b")
            dt    = art.get("published_dt")
            ts    = dt.strftime("%d %b, %H:%M UTC") if dt else "—"
            src   = art.get("source", "Unknown")
            bar_w = int((score + 1) / 2 * 100)

            st.markdown(f"""
            <div class="snt-headline-card">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px; margin-bottom:10px;">
                    <div style="font-size:0.88rem; font-weight:600; line-height:1.45; flex:1;">
                        <a href="{art.get('url', '#')}" target="_blank"
                           style="color:#e2e8f0; text-decoration:none;">
                            {art['title']}
                        </a>
                    </div>
                    <span class="snt-badge" style="background:{bbg}; color:{clr}; flex-shrink:0;">
                        {lbl}
                    </span>
                </div>
                <div style="display:flex; align-items:center; gap:12px;">
                    <div style="font-size:0.7rem; color:#475569; flex-shrink:0;">
                        {src} · {ts}
                    </div>
                    <div style="flex:1; height:4px; background:#1e293b; border-radius:2px; overflow:hidden;">
                        <div style="width:{bar_w}%; height:100%; background:{bclr}; border-radius:2px;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    # BLOCK 4 — CHARTS (compact)
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("<hr class='snt-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='snt-section-title'>📊 Sentiment Breakdown</div>", unsafe_allow_html=True)

    c_pie, c_bar = st.columns([1, 1.6], gap="medium")

    with c_pie:
        if agg["total_articles"] > 0:
            pos_n = round(agg["pct_positive"] / 100 * agg["total_articles"])
            neg_n = round(agg["pct_negative"] / 100 * agg["total_articles"])
            neu_n = agg["total_articles"] - pos_n - neg_n
            fig_pie = go.Figure(go.Pie(
                labels=["Positive", "Negative", "Neutral"],
                values=[max(pos_n, 0), max(neg_n, 0), max(neu_n, 0)],
                marker_colors=["#10b981", "#ef4444", "#f59e0b"],
                hole=0.65,
                textfont_size=12,
                showlegend=True,
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Sora, sans-serif"),
                height=220,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with c_bar:
        if articles:
            score_vals  = [a["combined_score"] for a in articles]
            score_clrs  = ["#10b981" if s >= 0.1 else ("#ef4444" if s <= -0.1 else "#f59e0b") for s in score_vals]
            titles_short = [(a["title"][:42] + "…") if len(a["title"]) > 44 else a["title"] for a in articles]

            fig_bar = go.Figure(go.Bar(
                x=score_vals,
                y=titles_short,
                orientation="h",
                marker_color=score_clrs,
            ))
            fig_bar.add_vline(x=0, line_dash="dot", line_color="#334155", line_width=1)
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#64748b", family="Sora, sans-serif", size=10),
                height=max(220, len(articles) * 26),
                margin=dict(l=0, r=20, t=10, b=0),
                showlegend=False,
                xaxis=dict(gridcolor="#1e293b", range=[-1.1, 1.1], zeroline=False),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════
    # BLOCK 5 — RECONCILIATION (simplified)
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("<hr class='snt-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='snt-section-title'>🤝 Model vs Sentiment</div>", unsafe_allow_html=True)

    signal_colors = {"BUY": "#10b981", "SELL": "#ef4444", "HOLD": "#f59e0b"}
    sig_clr = next((signal_colors[k] for k in signal_colors if k in model_signal), "#94a3b8")

    align_icon_map = {
        "Agreement":       "✅",
        "Mild Conflict":   "⚠️",
        "Strong Conflict": "🚨",
        "Neutral":         "⚪",
    }
    align_icon = align_icon_map.get(reconciled["alignment"], "⚪")

    r1, r2, r3, r4 = st.columns(4, gap="small")

    with r1:
        st.markdown(f"""
        <div class="snt-metric-card" style="border-color:#1e293b;">
            <div class="snt-metric-label">Model Signal</div>
            <div class="snt-metric-value" style="color:{sig_clr};">{model_signal}</div>
            <div class="snt-metric-sub">ML prediction</div>
        </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="snt-metric-card" style="border-color:#1e293b;">
            <div class="snt-metric-label">News Sentiment</div>
            <div class="snt-metric-value" style="color:{sentiment_color};">{agg['overall_label']}</div>
            <div class="snt-metric-sub">{agg['signal_strength'].lower()} signal</div>
        </div>""", unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
        <div class="snt-metric-card" style="border-color:{alignment_color}40;">
            <div class="snt-metric-label">Alignment</div>
            <div class="snt-metric-value" style="font-size:1.3rem; color:{alignment_color};">
                {align_icon} {reconciled['alignment']}
            </div>
            <div class="snt-metric-sub">model × news</div>
        </div>""", unsafe_allow_html=True)

    with r4:
        delta_sign = "+" if reconciled["delta_conf"] >= 0 else ""
        st.markdown(f"""
        <div class="snt-metric-card" style="border-color:{alignment_color}40;">
            <div class="snt-metric-label">Final Confidence</div>
            <div class="snt-metric-value" style="color:{alignment_color};">{reconciled['adjusted_conf']:.1f}%</div>
            <div class="snt-metric-sub">{delta_sign}{reconciled['delta_conf']:.0f}pp adjustment</div>
        </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════
    # BLOCK 6 — FINAL INSIGHT BOX
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    insight_map = {
        ("Agreement",       "Positive"): ("Signals aligned — bullish conviction",   "Both the ML model and recent news point in the same direction. Positive sentiment reinforces the model's outlook."),
        ("Agreement",       "Negative"): ("Signals aligned — bearish conviction",   "ML model and news sentiment are in agreement on the downside. Proceed with caution."),
        ("Agreement",       "Neutral"):  ("Aligned but indecisive",                 "Both signals are neutral. The market may be waiting for a catalyst before moving."),
        ("Mild Conflict",   "Positive"): ("Minor divergence — news leans positive",  "The model is cautious but news sentiment is positive. Watch for a potential narrative shift."),
        ("Mild Conflict",   "Negative"): ("Minor divergence — news leans negative",  "The model's direction conflicts with negative news sentiment. Monitor closely."),
        ("Strong Conflict", "Positive"): ("Sharp divergence — news strongly positive","The model and news are pulling in opposite directions. High uncertainty — consider waiting for clarity."),
        ("Strong Conflict", "Negative"): ("Sharp divergence — news strongly negative","Strong conflict between model signal and bearish news. Risk is elevated."),
        ("Neutral",         "Positive"): ("No clear conflict — slightly positive",   "News is positive but the model is neutral. Not enough signal to act decisively."),
        ("Neutral",         "Negative"): ("No clear conflict — slightly negative",   "News leans negative but model is neutral. Monitor for deterioration."),
        ("Neutral",         "Neutral"):  ("Low signal environment",                  "Both the model and sentiment are neutral. Await stronger signals before acting."),
    }

    insight_key     = (reconciled["alignment"], agg["overall_label"])
    insight_title, insight_body = insight_map.get(insight_key, (
        "Analysing market conditions",
        f"Sentiment is {agg['overall_label'].lower()} with {agg['signal_strength'].lower()} strength. "
        f"Adjusted model confidence stands at {reconciled['adjusted_conf']:.1f}%."
    ))

    st.markdown(f"""
    <div class="snt-insight-box">
        <div style="display:flex; align-items:flex-start; gap:16px;">
            <div style="font-size:1.6rem; flex-shrink:0; margin-top:2px;">💡</div>
            <div>
                <div style="font-size:0.68rem; font-weight:700; letter-spacing:0.12em;
                            text-transform:uppercase; color:#475569; margin-bottom:6px;">
                    Final Insight
                </div>
                <div style="font-size:1.0rem; font-weight:700; color:#e2e8f0; margin-bottom:6px;">
                    {insight_title}
                </div>
                <div style="font-size:0.85rem; color:#94a3b8; line-height:1.6;">
                    {insight_body}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Footer timestamp ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="color:#334155; font-size:0.68rem; text-align:right; margin-top:16px;
                font-family:'JetBrains Mono', monospace;">
        FinBERT 70% + VADER 30% &nbsp;·&nbsp; Refreshes every 30 min &nbsp;·&nbsp;
        {datetime.now().strftime('%H:%M:%S')}
    </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.title("📊 AI Stock Trading Dashboard")

    render_sentiment_section(
        model_signal="HOLD",
        model_confidence=50.0,
    )

main()
