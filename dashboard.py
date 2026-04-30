import streamlit as st
import nltk

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

st.title("📊 AI Stock Trading Dashboard")

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
    Drop-in replacement for the Sentiment Analysis section in dashboard.py.

    Usage in dashboard.py (inside the `elif nav == "🗞️ Sentiment Analysis":` block):

        from sentiment_pipeline import render_sentiment_section
        render_sentiment_section(live_signal, live_conf)
    """
    import plotly.graph_objects as go

    # ── Run pipeline ─────────────────────────────────────────────────────────
    with st.spinner("🔄 Fetching live news & running NLP models…"):
        agg = run_sentiment_pipeline()

    articles = agg.get("articles", [])
    no_data  = agg["total_articles"] == 0

    # ── Reconcile with ML prediction ─────────────────────────────────────────
    reconciled = adjust_prediction_with_sentiment(model_signal, model_confidence, agg)

    # ── Summary Metrics Row ───────────────────────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4)

    overall_colors = {"Positive": "#10b981", "Negative": "#ef4444", "Neutral": "#eab308"}
    oc = overall_colors.get(agg["overall_label"], "#eab308")

    with mc1:
        st.metric("Articles (24h)", agg["total_articles"])
    with mc2:
        st.metric(
            "Weighted Sentiment",
            f"{agg['weighted_score']:+.4f}",
            agg["overall_label"],
        )
    with mc3:
        st.metric(
            "Distribution",
            f"🟢{agg['pct_positive']:.0f}% 🔴{agg['pct_negative']:.0f}% ⚪{agg['pct_neutral']:.0f}%",
        )
    with mc4:
        alignment_colors = {
            "Agreement":      "#10b981",
            "Mild Conflict":  "#eab308",
            "Strong Conflict":"#ef4444",
            "Neutral":        "#64748b",
        }
        ac = alignment_colors.get(reconciled["alignment"], "#64748b")
        st.metric(
            "Model Alignment",
            reconciled["alignment"],
            f"Conf: {reconciled['adjusted_conf']:.1f}% ({reconciled['delta_conf']:+.0f}pp)",
        )

    # ── Conflict Warning ──────────────────────────────────────────────────────
    if reconciled["warning"]:
        st.warning(reconciled["warning"])

    # ── Overall Sentiment Banner ──────────────────────────────────────────────
    strength_icon = {"Strong": "🔥", "Moderate": "📊", "Weak": "🌤️"}.get(
        agg["signal_strength"], "📊"
    )
    st.markdown(
        f"""
        <div style="
            background: #111827;
            border-left: 5px solid {oc};
            border-radius: 0 14px 14px 0;
            padding: 18px 24px;
            margin: 16px 0;
            display: flex;
            align-items: center;
            gap: 20px;
        ">
            <div style="font-size: 2.4rem;">{strength_icon}</div>
            <div>
                <div style="font-size: 0.72rem; color: #64748b; letter-spacing: 0.1em;
                            text-transform: uppercase; margin-bottom: 4px;">
                    Overall Sentiment Signal
                </div>
                <div style="font-size: 1.5rem; font-weight: 800; color: {oc};">
                    {agg['signal_strength']} {agg['overall_label']}
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">
                    Score: {agg['weighted_score']:+.4f} &nbsp;·&nbsp;
                    Based on {agg['total_articles']} articles in last 24h &nbsp;·&nbsp;
                    FinBERT 70% + VADER 30%
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border-color:#1e2d4a; margin:20px 0'>", unsafe_allow_html=True)

    # ── Headlines Feed ────────────────────────────────────────────────────────
    st.markdown("#### 📰 Live Financial Headlines")

    if no_data:
        st.info("📭 No recent news available. RSS feed returned no results or feedparser is not installed.")
        st.markdown(
            """
            <div class="card" style="border-color:#1e3a5f">
                <div class="card-title">ℹ️ Setup Note</div>
                <div style="color:#94a3b8; font-size:0.85rem; line-height:1.7">
                    Install dependencies: <code>pip install feedparser vaderSentiment transformers torch</code><br>
                    FinBERT model will be downloaded automatically on first run (~500MB).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        lbl_colors = {"Positive": "#10b981", "Negative": "#ef4444", "Neutral": "#eab308"}
        border_colors = {"Positive": "#10b981", "Negative": "#ef4444", "Neutral": "#eab308"}

        for art in articles:
            lbl   = art["sentiment_label"]
            score = art["combined_score"]
            clr   = lbl_colors.get(lbl,    "#eab308")
            bclr  = border_colors.get(lbl, "#1e3a5f")
            fb    = art.get("finbert_score", 0.0)
            vd    = art.get("vader_score",   0.0)

            # Format timestamp
            dt = art.get("published_dt")
            ts = dt.strftime("%d %b %Y, %H:%M UTC") if dt else "Unknown time"

            # Score bar (0–100 mapped from -1..+1)
            bar_pct = int((score + 1) / 2 * 100)
            bar_clr = clr

            st.markdown(
                f"""
                <div style="
                    background: #111827;
                    border-left: 4px solid {bclr};
                    border-radius: 0 12px 12px 0;
                    padding: 14px 18px;
                    margin-bottom: 10px;
                ">
                    <div style="color: #e2e8f0; font-size: 0.9rem; font-weight: 600;
                                margin-bottom: 6px;">{art['title']}</div>
                    <div style="color: #64748b; font-size: 0.72rem; margin-bottom: 8px;">
                        🕐 {ts} &nbsp;·&nbsp; 📰 {art.get('source','Unknown')}
                    </div>
                    <div style="height: 6px; background: #1e2d4a; border-radius: 3px;
                                overflow: hidden; margin-bottom: 8px;">
                        <div style="height: 100%; width: {bar_pct}%;
                                    background: {bar_clr}; border-radius: 3px;"></div>
                    </div>
                    <div style="font-size: 0.75rem; display: flex; gap: 16px; flex-wrap: wrap;">
                        <span>FinBERT: <b style="color:#a78bfa">{fb:+.4f}</b></span>
                        <span>VADER: <b style="color:#38bdf8">{vd:+.4f}</b></span>
                        <span>Combined: <b style="color:#e2e8f0">{score:+.4f}</b></span>
                        <span style="color:{clr}; font-weight:700">{lbl}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<hr style='border-color:#1e2d4a; margin:20px 0'>", unsafe_allow_html=True)

    # ── Sentiment Distribution Chart ──────────────────────────────────────────
    col_pie, col_bar = st.columns([1, 1])

    with col_pie:
        st.markdown("#### 🥧 Sentiment Distribution")
        if agg["total_articles"] > 0:
            pos_n = round(agg["pct_positive"] / 100 * agg["total_articles"])
            neg_n = round(agg["pct_negative"] / 100 * agg["total_articles"])
            neu_n = agg["total_articles"] - pos_n - neg_n
            fig_pie = go.Figure(go.Pie(
                labels=["Positive", "Negative", "Neutral"],
                values=[max(pos_n, 0), max(neg_n, 0), max(neu_n, 0)],
                marker_colors=["#10b981", "#ef4444", "#eab308"],
                hole=0.55,
                textfont_size=13,
            ))
            fig_pie.update_layout(
                paper_bgcolor="#0a0e1a",
                font_color="#e2e8f0",
                height=280,
                margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(bgcolor="#111827"),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No data for chart.")

    with col_bar:
        st.markdown("#### 📊 Score Distribution")
        if articles:
            score_vals = [a["combined_score"] for a in articles]
            score_clrs = [
                "#10b981" if s >= 0.1 else ("#ef4444" if s <= -0.1 else "#eab308")
                for s in score_vals
            ]
            titles_short = [
                (a["title"][:38] + "…") if len(a["title"]) > 40 else a["title"]
                for a in articles
            ]
            fig_bar = go.Figure(go.Bar(
                x=score_vals,
                y=titles_short,
                orientation="h",
                marker_color=score_clrs,
                text=[f"{s:+.3f}" for s in score_vals],
                textposition="outside",
                textfont_color="#e2e8f0",
            ))
            fig_bar.add_vline(x=0, line_dash="dash", line_color="#475569", line_width=1)
            fig_bar.update_layout(
                paper_bgcolor="#0a0e1a",
                plot_bgcolor="#0a0e1a",
                font_color="#e2e8f0",
                height=max(280, len(articles) * 30),
                margin=dict(l=0, r=60, t=20, b=0),
                showlegend=False,
                xaxis=dict(gridcolor="#1e2d4a", range=[-1.1, 1.1]),
                yaxis=dict(gridcolor="#1e2d4a", automargin=True),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No articles scored yet.")

    st.markdown("<hr style='border-color:#1e2d4a; margin:20px 0'>", unsafe_allow_html=True)

    # ── Model × Sentiment Reconciliation Panel ────────────────────────────────
    st.markdown("#### 🤝 ML Prediction × Sentiment Reconciliation")

    align_icon = {
        "Agreement":      "✅",
        "Mild Conflict":  "⚠️",
        "Strong Conflict":"🚨",
        "Neutral":        "⚪",
    }.get(reconciled["alignment"], "⚪")

    align_bg = {
        "Agreement":      "#064e3b",
        "Mild Conflict":  "#3b3406",
        "Strong Conflict":"#450a0a",
        "Neutral":        "#1e2d4a",
    }.get(reconciled["alignment"], "#1e2d4a")

    align_border = {
        "Agreement":      "#10b981",
        "Mild Conflict":  "#eab308",
        "Strong Conflict":"#ef4444",
        "Neutral":        "#38bdf8",
    }.get(reconciled["alignment"], "#38bdf8")

    delta_sign = "+" if reconciled["delta_conf"] >= 0 else ""

    st.markdown(
        f"""
        <div style="
            background: {align_bg};
            border: 2px solid {align_border};
            border-radius: 16px;
            padding: 28px;
            margin-bottom: 16px;
        ">
            <div style="display: flex; align-items: center; gap: 16px; flex-wrap: wrap;">
                <div style="font-size: 2.8rem;">{align_icon}</div>
                <div style="flex: 1; min-width: 200px;">
                    <div style="font-size: 1.2rem; font-weight: 800;
                                color: {align_border}; margin-bottom: 6px;">
                        {reconciled['alignment']}
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6;">
                        Model Signal: <b style="color:#e2e8f0">{reconciled['final_signal']}</b>
                        &nbsp;|&nbsp;
                        Sentiment: <b style="color:{align_border}">{reconciled['sentiment_label']}</b>
                        (score: {reconciled['sentiment_score']:+.4f})
                    </div>
                </div>
                <div style="text-align: right; min-width: 140px;">
                    <div style="font-size: 0.72rem; color: #64748b; text-transform: uppercase;
                                letter-spacing: 0.08em;">Adjusted Confidence</div>
                    <div style="font-size: 2rem; font-weight: 900; color: {align_border};">
                        {reconciled['adjusted_conf']:.1f}%
                    </div>
                    <div style="font-size: 0.78rem; color: #94a3b8;">
                        Original: {reconciled['original_conf']:.1f}%
                        &nbsp;→&nbsp;
                        <span style="color:{align_border}">
                            {delta_sign}{reconciled['delta_conf']:.0f}pp
                        </span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Refresh Info ──────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="color: #475569; font-size: 0.72rem; text-align: right; margin-top: 8px;">
            🔄 Pipeline: FinBERT (70%) + VADER (30%) · Cached 30min ·
            Last run: {datetime.now().strftime('%H:%M:%S')} ·
            Articles from last {HOURS_LOOKBACK}h
        </div>
        """,
        unsafe_allow_html=True,
    )
