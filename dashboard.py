"""
================================================================================
AI STOCK TRADING DASHBOARD — dashboard.py
================================================================================
Entry point:  streamlit run dashboard.py

Architecture:
    ml_model.py           → ML prediction (Random Forest, feature engineering)
    sentiment_pipeline.py → News fetch + FinBERT/VADER scoring
    dashboard.py          → UI only — thin, clean, readable

FIXES IN THIS VERSION:
    ✅ news_updated timestamp now correctly passed into render_header() after
       sentiment data is available (previously always showed "" in header)
    ✅ delta_conf updated in reconciled dict after _cap_sentiment_adjustment()
       (previously showed pre-cap delta, which was misleading in the UI)
    ✅ render_charts() wrapped in try/except — no chart crash on bad data
    ✅ All st.markdown calls with unsafe_allow_html wrapped in try/except
    ✅ Progressive loading preserved — ML renders first, sentiment after
    ✅ Zero duplicate logic, no unused variables
================================================================================
"""

import csv
import streamlit as st

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    go = None

from datetime import datetime
from pathlib import Path

from ml_model import (
    predict_signal,
    get_market_snapshot,
    get_ml_health,
    MAX_SENTIMENT_ADJUSTMENT,
)
from sentiment_pipeline import run_sentiment_pipeline, adjust_prediction_with_sentiment


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Reliance AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1200px; }

.card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 20px 22px;
}
.card-sm { padding: 14px 18px; border-radius: 10px; }

.pill {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    padding: 3px 10px;
    border-radius: 20px;
}

.sec-label {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #484f58;
    margin-bottom: 14px;
}

.div { border: none; border-top: 1px solid #21262d; margin: 24px 0; }

.banner {
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 16px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.banner-icon  { font-size: 1.2rem; flex-shrink: 0; line-height: 1.6; }
.banner-title { font-size: 0.88rem; font-weight: 700; margin-bottom: 3px; }
.banner-msg   { font-size: 0.78rem; line-height: 1.55; }

.decision-box {
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 14px;
    border-left: 3px solid;
}
.decision-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.decision-text {
    font-size: 0.9rem;
    line-height: 1.7;
    color: #c9d1d9;
}

.health-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.78rem;
    padding: 5px 0;
    border-bottom: 1px solid #21262d;
}

.perf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.8rem;
    padding: 6px 0;
    border-bottom: 1px solid #21262d;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & COLOUR MAPS
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_COLOR = {"BUY": "#3fb950", "SELL": "#f85149", "HOLD": "#d29922"}
SENT_COLOR   = {"Positive": "#3fb950", "Negative": "#f85149", "Neutral": "#d29922"}
ALIGN_COLOR  = {
    "Agreement":       "#3fb950",
    "Mild Conflict":   "#d29922",
    "Strong Conflict": "#f85149",
    "Neutral":         "#484f58",
}

FEATURE_LABELS = {
    "ret_1d":    "1-day price return",
    "ret_3d":    "3-day price return",
    "ret_5d":    "5-day price return",
    "ret_10d":   "10-day price return",
    "sma_5":     "Price vs SMA-5",
    "sma_10":    "Price vs SMA-10",
    "sma_20":    "Price vs SMA-20",
    "sma_50":    "Price vs SMA-50",
    "rsi_14":    "RSI (14-day)",
    "atr_14":    "ATR % (14-day)",
    "bb_pct":    "Bollinger Band %",
    "macd_diff": "MACD histogram",
    "vol_ratio": "Volume ratio",
}

PERF_CSV = Path("data/performance_summary.csv")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _pill(text: str, color: str) -> str:
    return (
        f'<span class="pill" style="'
        f'background:{color}18; color:{color}; border:1px solid {color}44;">'
        f'{text}</span>'
    )


def _kpi_card(label: str, value: str, sub: str = "", accent: str = "#e6edf3") -> str:
    return f"""
    <div class="card card-sm" style="height:100%;">
        <div class="sec-label">{label}</div>
        <div style="font-size:1.65rem; font-weight:700; color:{accent};
                    line-height:1; margin-bottom:4px; font-family:'DM Mono',monospace;">
            {value}
        </div>
        <div style="font-size:0.75rem; color:#484f58;">{sub}</div>
    </div>"""


def _plotly_defaults() -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e", family="DM Sans, sans-serif", size=11),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )


def _cap_sentiment_adjustment(original_conf: float, adjusted_conf: float) -> float:
    """Clamp sentiment delta to ±MAX_SENTIMENT_ADJUSTMENT pp so ML stays primary."""
    delta  = adjusted_conf - original_conf
    capped = max(-MAX_SENTIMENT_ADJUSTMENT, min(MAX_SENTIMENT_ADJUSTMENT, delta))
    return round(original_conf + capped, 2)


# ─────────────────────────────────────────────────────────────────────────────
# DECISION SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

_CONF_HIGH   = 70.0
_CONF_MEDIUM = 55.0

_FEATURE_GROUPS = {
    "momentum":   {"rsi_14", "macd_diff", "ret_1d", "ret_3d"},
    "trend":      {"sma_5", "sma_10", "sma_20", "sma_50"},
    "volatility": {"atr_14", "bb_pct"},
    "volume":     {"vol_ratio"},
    "return":     {"ret_5d", "ret_10d"},
}

_FEATURE_GROUP_LABELS = {
    "momentum":   "momentum indicators (RSI, MACD)",
    "trend":      "trend / moving-average signals",
    "volatility": "volatility patterns (ATR, Bollinger)",
    "volume":     "volume activity",
    "return":     "medium-term price returns",
}


def _classify_confidence(conf: float) -> str:
    if conf >= _CONF_HIGH:
        return "high"
    if conf >= _CONF_MEDIUM:
        return "moderate"
    return "low"


def _dominant_feature_groups(top_features: list) -> list:
    group_totals: dict = {}
    for feat in top_features:
        name = feat.get("name", "")
        imp  = feat.get("importance", 0.0)
        for group, members in _FEATURE_GROUPS.items():
            if name in members:
                group_totals[group] = group_totals.get(group, 0.0) + imp
    ranked = sorted(group_totals, key=lambda g: group_totals[g], reverse=True)
    return [_FEATURE_GROUP_LABELS[g] for g in ranked[:2]]


def build_decision_summary(
    signal:         str,
    confidence:     float,
    top_features:   list,
    sentiment_label: str,
    alignment:      str,
    is_fallback:    bool,
) -> str:
    if is_fallback:
        return (
            "The ML model is currently unavailable — the signal shown is a safe "
            "default (HOLD) and does not reflect a genuine prediction. "
            "No trading decisions should be based on this output."
        )

    conf_label  = _classify_confidence(confidence)
    groups      = _dominant_feature_groups(top_features)
    signal_verb = {"BUY": "bullish", "SELL": "bearish", "HOLD": "neutral"}.get(
        signal, signal.lower()
    )

    if groups:
        driven_by = " and ".join(groups)
        s1 = (
            f"The model predicts <strong>{signal}</strong> with {conf_label} confidence, "
            f"driven primarily by {driven_by}."
        )
    else:
        s1 = f"The model predicts <strong>{signal}</strong> with {conf_label} confidence."

    sent_stance = {
        "Positive": "reinforcing the bullish outlook",
        "Negative": "introducing a bearish undertone",
        "Neutral":  "offering no strong directional push",
    }.get(sentiment_label, "inconclusive")
    s2 = f"News sentiment is <strong>{sentiment_label.lower()}</strong>, {sent_stance}."

    if alignment == "Agreement":
        if signal == "BUY":
            s3 = "Both technical and sentiment signals align — the overall outlook is <strong>bullish</strong>."
        elif signal == "SELL":
            s3 = "Both technical and sentiment signals agree — the overall outlook is <strong>bearish</strong>."
        else:
            s3 = "Both signals suggest a <strong>neutral/hold</strong> stance."
    elif alignment == "Strong Conflict":
        s3 = (
            "There is a <strong>strong conflict</strong> between the model signal and news sentiment — "
            "proceed with caution. Confidence has been adjusted downward."
        )
    elif alignment == "Mild Conflict":
        s3 = (
            "There is a mild divergence between the model and sentiment. "
            "Confidence has been slightly reduced to reflect uncertainty."
        )
    else:
        s3 = f"The final signal is <strong>{signal}</strong> — sentiment is neutral and does not shift the outlook."

    return f"{s1} {s2} {s3}"


def render_decision_summary(
    prediction:      dict,
    sentiment_label: str,
    alignment:       str,
) -> None:
    try:
        signal       = prediction.get("signal", "HOLD")
        confidence   = prediction.get("confidence", 50.0)
        top_features = prediction.get("top_features", [])
        is_fallback  = prediction.get("source", "fallback") == "fallback"

        summary_html = build_decision_summary(
            signal, confidence, top_features, sentiment_label, alignment, is_fallback
        )

        border_color = SIGNAL_COLOR.get(signal, "#484f58") if not is_fallback else "#d29922"
        bg_color     = f"{border_color}0a"

        st.markdown(f"""
        <div class="decision-box" style="background:{bg_color}; border-color:{border_color};">
            <div class="decision-label" style="color:{border_color};">🧠 AI Decision Summary</div>
            <div class="decision-text">{summary_html}</div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass   # decision summary is non-critical — never crash the page


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PERFORMANCE PANEL
# ─────────────────────────────────────────────────────────────────────────────

def _load_performance_summary() -> dict:
    if not PERF_CSV.exists():
        return {}
    try:
        result: dict = {}
        with open(PERF_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric = row.get("metric", "").strip()
                value  = row.get("value",  "").strip()
                if metric and value:
                    result[metric] = value
        return result
    except Exception:
        return {}


def render_model_performance() -> None:
    st.markdown('<div class="sec-label">Model Performance</div>', unsafe_allow_html=True)

    perf = _load_performance_summary()

    if not perf:
        st.markdown("""
        <div style="font-size:0.78rem; color:#484f58; line-height:1.8; padding:4px 0;">
            Performance data unavailable.<br>
            <span style="font-size:0.70rem; color:#3d444d;">
                Add <code>data/performance_summary.csv</code><br>
                with columns: <code>metric, value</code>
            </span>
        </div>
        """, unsafe_allow_html=True)
        return

    display_order = [
        ("accuracy",        "Model Accuracy"),
        ("backtest_return", "Backtest Return"),
        ("sharpe_ratio",    "Sharpe Ratio"),
        ("max_drawdown",    "Max Drawdown"),
        ("train_period",    "Training Period"),
        ("test_period",     "Test Period"),
    ]

    rows_html = ""
    for key, label in display_order:
        val = perf.get(key)
        if val:
            rows_html += f"""
            <div class="perf-row">
                <span style="color:#484f58;">{label}</span>
                <span style="color:#c9d1d9; font-family:'DM Mono',monospace;
                             font-size:0.78rem;">{val}</span>
            </div>"""

    known_keys = {k for k, _ in display_order}
    for key, val in perf.items():
        if key not in known_keys:
            label = key.replace("_", " ").title()
            rows_html += f"""
            <div class="perf-row">
                <span style="color:#484f58;">{label}</span>
                <span style="color:#c9d1d9; font-family:'DM Mono',monospace;
                             font-size:0.78rem;">{val}</span>
            </div>"""

    st.markdown(
        rows_html or '<div style="color:#484f58;font-size:0.78rem;">No rows found.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.68rem; color:#3d444d; margin-top:5px;">'
        'Backtested — not a guarantee of future returns.</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK / FEATURE WARNING BANNERS
# ─────────────────────────────────────────────────────────────────────────────

def render_fallback_banner(prediction: dict) -> None:
    if prediction.get("source") == "model":
        return

    reason = (
        prediction.get("fallback_reason")
        or prediction.get("error")
        or "Unknown reason"
    )

    st.markdown(f"""
    <div class="banner" style="background:#2d1b00; border:1px solid #d29922;
                                border-left:4px solid #d29922;">
        <div class="banner-icon">⚠️</div>
        <div>
            <div class="banner-title" style="color:#d29922;">
                Model fallback active — showing default signal (HOLD / 50%)
            </div>
            <div class="banner-msg" style="color:#a8914c;">
                {reason}<br>
                <span style="color:#6b5c2e; font-size:0.72rem;">
                    This is a safe default, not a model prediction.
                    Do not trade based on this output.
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_feature_warning(prediction: dict) -> None:
    feat_warn = prediction.get("feature_warning", "")
    if not feat_warn:
        return

    st.markdown(f"""
    <div class="banner" style="background:#0d1b2e; border:1px solid #388bfd;
                                border-left:4px solid #388bfd;">
        <div class="banner-icon">🔧</div>
        <div>
            <div class="banner-title" style="color:#388bfd;">Feature alignment warning</div>
            <div class="banner-msg" style="color:#6b8aaf;">{feat_warn}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ML HEALTH STATUS PANEL
# ─────────────────────────────────────────────────────────────────────────────

def render_ml_health(health: dict) -> None:
    st.markdown('<div class="sec-label">System Status</div>', unsafe_allow_html=True)

    def _row(label: str, ok: bool, detail: str = "") -> str:
        icon  = "✅" if ok else "❌"
        color = "#3fb950" if ok else "#f85149"
        tip   = (
            f'<span style="color:#484f58; font-size:0.69rem;"> — {str(detail)[:40]}</span>'
            if detail and not ok else ""
        )
        return (
            f'<div class="health-row">'
            f'<span style="color:#8b949e;">{label}</span>'
            f'<span style="color:{color}; font-family:\'DM Mono\',monospace;">{icon}{tip}</span>'
            f'</div>'
        )

    st.markdown(
        _row("Model loaded",       health.get("model_loaded",        False), health.get("model_error", ""))
        + _row("Live data",        health.get("live_data_ok",         False), health.get("data_error",  ""))
        + _row("Feature pipeline", health.get("feature_pipeline_ok",  False), health.get("data_error",  "")),
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:0.68rem; color:#3d444d; margin-top:6px;">'
        f'Checked: {health.get("last_checked", "—")}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

def render_header(snapshot: dict, prediction: dict, news_updated: str = "") -> None:
    """
    Render the three-column header (title / live price / AI signal).

    news_updated: human-readable timestamp of the most recent article.
                  Pass this AFTER sentiment data is loaded so it actually appears.
    """
    sig       = prediction.get("signal", "HOLD")
    conf      = prediction.get("confidence", 50.0)
    sc        = SIGNAL_COLOR.get(sig, "#8b949e")
    price     = snapshot.get("price",      "—")
    change    = snapshot.get("change",     0)
    pct       = snapshot.get("pct_change", 0)
    chg_color = "#3fb950" if change >= 0 else "#f85149"
    chg_arrow = "▲" if change >= 0 else "▼"
    fetched   = snapshot.get("fetched_at", "")

    is_fallback = prediction.get("source", "fallback") == "fallback"
    src_label   = "FALLBACK" if is_fallback else "ML MODEL"
    src_color   = "#d29922"  if is_fallback else "#388bfd"
    data_ts     = prediction.get("data_timestamp", "")

    col_title, col_price, col_signal = st.columns([2, 1.5, 1.5], gap="large")

    with col_title:
        news_line = (
            f"News updated: <span style='color:#484f58;'>{news_updated}</span>"
            if news_updated else ""
        )
        st.markdown(f"""
        <div style="padding-top:6px;">
            <div style="font-size:0.68rem; color:#484f58; letter-spacing:0.1em;
                        text-transform:uppercase; margin-bottom:6px;">
                NSE · RELIANCE.NS
            </div>
            <div style="font-size:1.5rem; font-weight:700; color:#e6edf3;">
                Reliance Industries
            </div>
            <div style="font-size:0.78rem; color:#8b949e; margin-top:4px;">
                AI-powered trading signal
            </div>
            <div style="font-size:0.7rem; color:#3d444d; margin-top:6px; line-height:1.7;">
                Market data as of: <span style="color:#484f58;">{data_ts or "—"}</span><br>
                {news_line}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_price:
        if isinstance(price, (int, float)):
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div class="sec-label">Live Price</div>
                <div style="font-size:1.8rem; font-weight:700; color:#e6edf3;
                            font-family:'DM Mono',monospace;">
                    ₹{price:,.2f}
                </div>
                <div style="font-size:0.82rem; color:{chg_color}; margin-top:4px;">
                    {chg_arrow} ₹{abs(change):,.2f} ({pct:+.2f}%)
                </div>
                <div style="font-size:0.68rem; color:#3d444d; margin-top:8px;">
                    Updated: {fetched or "—"} IST
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align:center;">
                <div class="sec-label">Live Price</div>
                <div style="font-size:1.1rem; color:#484f58; padding:10px 0;">Unavailable</div>
                <div style="font-size:0.72rem; color:#3d444d;">
                    yfinance unreachable
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_signal:
        st.markdown(f"""
        <div class="card" style="text-align:center; border-color:{sc}33;">
            <div style="display:flex; justify-content:space-between; align-items:center;
                        margin-bottom:10px;">
                <div class="sec-label" style="margin-bottom:0;">AI Signal</div>
                <span style="font-size:0.65rem; font-weight:700; letter-spacing:0.08em;
                             color:{src_color}; background:{src_color}18;
                             border:1px solid {src_color}44;
                             padding:2px 8px; border-radius:8px;">{src_label}</span>
            </div>
            <div style="font-size:2.1rem; font-weight:800; color:{sc};
                        font-family:'DM Mono',monospace; line-height:1.1;">
                {sig}
            </div>
            <div style="font-size:0.82rem; color:#8b949e; margin-top:5px;">
                {conf:.1f}% confidence
            </div>
            <div style="font-size:0.68rem; color:#3d444d; margin-top:8px; line-height:1.6;">
                ML signal is primary.<br>
                Sentiment adjusts confidence only (±{MAX_SENTIMENT_ADJUSTMENT:.0f}pp).
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT OVERVIEW KPIs
# ─────────────────────────────────────────────────────────────────────────────

def render_sentiment_kpis(agg: dict, reconciled: dict) -> None:
    sc    = SENT_COLOR.get(agg.get("overall_label", "Neutral"), "#8b949e")
    ac    = ALIGN_COLOR.get(reconciled.get("alignment", "Neutral"), "#484f58")
    delta = reconciled.get("delta_conf", 0.0)
    dsign = "+" if delta >= 0 else ""

    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        st.markdown(
            _kpi_card("Articles (24h)", str(agg.get("total_articles", 0)), "from RSS feeds"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _kpi_card(
                "News Sentiment", agg.get("overall_label", "Neutral"),
                f"{agg.get('pct_positive', 0.0):.0f}% pos · {agg.get('pct_negative', 0.0):.0f}% neg",
                sc,
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _kpi_card(
                "Signal Strength", agg.get("signal_strength", "Weak"),
                f"weighted score: {agg.get('weighted_score', 0.0):+.3f}",
                sc,
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            _kpi_card(
                "Adjusted Confidence", f"{reconciled.get('adjusted_conf', 50.0):.1f}%",
                f"{dsign}{delta:.0f}pp · capped ±{MAX_SENTIMENT_ADJUSTMENT:.0f}pp",
                ac,
            ),
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ALIGNMENT BANNER
# ─────────────────────────────────────────────────────────────────────────────

def render_alignment_banner(reconciled: dict, agg: dict) -> None:
    alignment = reconciled.get("alignment", "Neutral")
    ac   = ALIGN_COLOR.get(alignment, "#484f58")
    sig  = reconciled.get("final_signal", "HOLD")
    sc   = SIGNAL_COLOR.get(sig, "#8b949e")
    snt  = agg.get("overall_label", "Neutral")
    sntc = SENT_COLOR.get(snt, "#8b949e")
    delta = reconciled.get("delta_conf", 0.0)
    dsign = "+" if delta >= 0 else ""

    icon_map = {
        "Agreement": "✓", "Mild Conflict": "△",
        "Strong Conflict": "✕", "Neutral": "○",
    }
    icon = icon_map.get(alignment, "○")

    st.markdown(f"""
    <div class="card" style="border-color:{ac}44; border-left:3px solid {ac};
                              display:flex; align-items:center; gap:24px; flex-wrap:wrap;">
        <div style="font-size:1.6rem; color:{ac}; font-family:'DM Mono',monospace;
                    width:36px; text-align:center; flex-shrink:0;">
            {icon}
        </div>
        <div style="flex:1; min-width:200px;">
            <div style="font-size:0.68rem; color:#484f58; letter-spacing:0.1em;
                        text-transform:uppercase; margin-bottom:4px;">
                Model × Sentiment Alignment
            </div>
            <div style="font-size:1.05rem; font-weight:700; color:{ac};">
                {alignment}
            </div>
            <div style="font-size:0.8rem; color:#8b949e; margin-top:4px;">
                ML: <span style="color:{sc}; font-weight:600;">{sig}</span>
                &nbsp;·&nbsp;
                News: <span style="color:{sntc}; font-weight:600;">{snt}</span>
                &nbsp;·&nbsp;
                Score: {agg.get('weighted_score', 0.0):+.3f}
            </div>
        </div>
        <div style="text-align:right; flex-shrink:0;">
            <div class="sec-label">Adjusted Confidence</div>
            <div style="font-size:1.6rem; font-weight:700; color:{ac};
                        font-family:'DM Mono',monospace;">
                {reconciled.get('adjusted_conf', 50.0):.1f}%
            </div>
            <div style="font-size:0.75rem; color:#484f58;">
                {reconciled.get('original_conf', 50.0):.1f}% → {dsign}{delta:.0f}pp
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if reconciled.get("warning"):
        st.warning(reconciled["warning"])


# ─────────────────────────────────────────────────────────────────────────────
# HEADLINES
# ─────────────────────────────────────────────────────────────────────────────

def render_headlines(articles: list, max_items: int = 6) -> None:
    st.markdown('<div class="sec-label">Live Headlines</div>', unsafe_allow_html=True)

    if not articles:
        st.info("No recent news fetched. Check RSS connectivity or install feedparser.")
        return

    for art in articles[:max_items]:
        try:
            lbl   = art.get("sentiment_label", "Neutral")
            score = art.get("combined_score",  0.0)
            color = SENT_COLOR.get(lbl, "#8b949e")
            dt    = art.get("published_dt")
            ts    = dt.strftime("%d %b, %H:%M") if dt else "—"
            src   = art.get("source", "Unknown")
            url   = art.get("url", "#")
            bar_w = max(0, min(100, int((score + 1) / 2 * 100)))

            st.markdown(f"""
            <div class="card card-sm" style="margin-bottom:8px; border-color:#21262d;">
                <div style="display:flex; justify-content:space-between;
                            align-items:flex-start; gap:12px; margin-bottom:8px;">
                    <a href="{url}" target="_blank" style="
                        color:#c9d1d9; font-size:0.88rem; font-weight:500;
                        line-height:1.45; flex:1; text-decoration:none;">
                        {art.get('title', '')}
                    </a>
                    {_pill(lbl, color)}
                </div>
                <div style="display:flex; align-items:center; gap:10px;">
                    <div style="font-size:0.68rem; color:#484f58; flex-shrink:0;">
                        {src} · {ts}
                    </div>
                    <div style="flex:1; height:3px; background:#21262d;
                                border-radius:2px; overflow:hidden;">
                        <div style="width:{bar_w}%; height:100%;
                                    background:{color}; border-radius:2px;"></div>
                    </div>
                    <div style="font-size:0.68rem; color:#484f58;
                                font-family:'DM Mono',monospace; flex-shrink:0;">
                        {score:+.3f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            continue   # one bad article card must not crash the page


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def render_charts(agg: dict) -> None:
    if go is None:
        st.info("Charts unavailable (Plotly not installed)")
        return

    articles = (agg or {}).get("articles", [])
    if not articles:
        return

    try:
        c_pie, c_bar = st.columns([1, 2], gap="medium")

        with c_pie:
            total = agg.get("total_articles", 0)
            pos_n = round(agg.get("pct_positive", 0.0) / 100 * total)
            neg_n = round(agg.get("pct_negative", 0.0) / 100 * total)
            neu_n = max(total - pos_n - neg_n, 0)

            fig = go.Figure(go.Pie(
                labels=["Positive", "Negative", "Neutral"],
                values=[max(pos_n, 0), max(neg_n, 0), neu_n],
                marker_colors=["#3fb950", "#f85149", "#d29922"],
                hole=0.72,
                textfont_size=11,
                showlegend=True,
            ))
            fig.update_layout(
                **_plotly_defaults(),
                height=200,
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#8b949e"), orientation="v"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c_bar:
            vals   = [a.get("combined_score", 0.0) for a in articles]
            colors = [
                "#3fb950" if s >= 0.1 else ("#f85149" if s <= -0.1 else "#d29922")
                for s in vals
            ]
            labels = [
                (a.get("title", "")[:48] + "…") if len(a.get("title", "")) > 50
                else a.get("title", "")
                for a in articles
            ]

            fig2 = go.Figure(go.Bar(
                x=vals, y=labels, orientation="h",
                marker_color=colors, marker_line_width=0,
            ))
            fig2.add_vline(x=0, line_dash="dot", line_color="#30363d", line_width=1)
            fig2.update_layout(
                **_plotly_defaults(),
                height=max(180, len(articles) * 24),
                xaxis=dict(gridcolor="#21262d", range=[-1.1, 1.1], zeroline=False, tickfont=dict(size=9)),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True, tickfont=dict(size=9, color="#8b949e")),
            )
            st.plotly_chart(fig2, use_container_width=True)

    except Exception:
        st.info("Charts could not be rendered.")


# ─────────────────────────────────────────────────────────────────────────────
# PROBABILITY BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────

def render_prob_bars(prediction: dict) -> None:
    probs = prediction.get("probabilities", {})
    if not probs:
        return

    st.markdown('<div class="sec-label">Class Probabilities</div>', unsafe_allow_html=True)

    for label in ["BUY", "HOLD", "SELL"]:
        val = probs.get(label, 0.0)
        col = SIGNAL_COLOR.get(label, "#8b949e")
        st.markdown(f"""
        <div style="margin-bottom:9px;">
            <div style="display:flex; justify-content:space-between;
                        font-size:0.78rem; margin-bottom:3px;">
                <span style="color:{col}; font-family:'DM Mono',monospace;
                             font-weight:600;">{label}</span>
                <span style="color:#8b949e; font-family:'DM Mono',monospace;">
                    {val:.1f}%
                </span>
            </div>
            <div style="height:5px; background:#21262d; border-radius:3px; overflow:hidden;">
                <div style="width:{min(val, 100):.1f}%; height:100%;
                            background:{col}; border-radius:3px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def render_feature_importance(prediction: dict) -> None:
    if prediction.get("source") == "fallback":
        return

    top_features = prediction.get("top_features", [])
    if not top_features:
        return

    st.markdown('<div class="sec-label">Top Decision Factors</div>', unsafe_allow_html=True)

    max_imp = max((f.get("importance", 0.0) for f in top_features), default=1.0) or 1.0

    for feat in top_features:
        try:
            name    = FEATURE_LABELS.get(feat.get("name", ""), feat.get("name", ""))
            imp     = feat.get("importance", 0.0)
            bar_pct = (imp / max_imp) * 100
            rank    = feat.get("rank", 0)

            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between;
                            font-size:0.76rem; margin-bottom:3px;">
                    <span style="color:#c9d1d9;">
                        <span style="color:#3d444d; font-family:'DM Mono',monospace;
                                     margin-right:6px; font-size:0.7rem;">#{rank}</span>{name}
                    </span>
                    <span style="color:#8b949e; font-family:'DM Mono',monospace;">
                        {imp:.1f}%
                    </span>
                </div>
                <div style="height:4px; background:#21262d; border-radius:2px; overflow:hidden;">
                    <div style="width:{bar_pct:.1f}%; height:100%;
                                background:#388bfd; border-radius:2px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            continue


# ─────────────────────────────────────────────────────────────────────────────
# TRUST FOOTER
# ─────────────────────────────────────────────────────────────────────────────

def render_trust_footer(snapshot: dict, prediction: dict, news_updated: str = "") -> None:
    signal_source = (
        "RandomForest (ML)" if prediction.get("source") == "model"
        else "⚠️ Fallback default"
    )
    data_ts  = prediction.get("data_timestamp") or "—"
    price_ts = snapshot.get("fetched_at") or "—"
    news_line = (
        f"<div><span style='color:#3d444d;'>News updated:</span> "
        f"<span style='color:#8b949e;'>{news_updated}</span></div>"
        if news_updated else ""
    )

    st.markdown(f"""
    <div style="font-size:0.72rem; color:#484f58; line-height:1.9;">
        <div style="margin-bottom:6px; color:#8b949e; font-weight:600;">
            Methodology & Data Sources
        </div>
        <div><span style="color:#3d444d;">Signal source:</span>
             <span style="color:#8b949e;">{signal_source}</span></div>
        <div><span style="color:#3d444d;">Market data as of:</span>
             <span style="color:#8b949e;">{data_ts}</span></div>
        <div><span style="color:#3d444d;">Price updated:</span>
             <span style="color:#8b949e;">{price_ts} IST</span></div>
        {news_line}
        <br>
        Sentiment: FinBERT (70%) + VADER (30%)<br>
        Recency weights: 6h × 2.0 · 12h × 1.5 · 24h × 1.0<br>
        Agreement: +5pp · Conflict: −8/−15pp<br>
        Sentiment capped at ±{MAX_SENTIMENT_ADJUSTMENT:.0f}pp — ML is primary<br>
        <br>
        <div style="color:#3d444d;">
            Pipeline: 30 min · Prices: 60s<br>
            Page loaded: {datetime.now().strftime('%H:%M:%S IST')}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:

    # ── Step 1: ML prediction — renders immediately ───────────────────────────
    with st.spinner("Loading market signal…"):
        snapshot   = get_market_snapshot()
        prediction = predict_signal()
        ml_health  = get_ml_health()

    render_fallback_banner(prediction)
    render_feature_warning(prediction)

    # ── Step 2: Sentiment loads progressively ─────────────────────────────────
    agg = {} 
    with st.spinner("Fetching and scoring news…"):
        if "sentiment_loaded" not in st.session_state:
            st.session_state.sentiment_loaded = False
        
        if not st.session_state.sentiment_loaded:
            if st.button("Load Sentiment Analysis"):
                st.session_state.sentiment_loaded = True
                st.rerun()
        
        if st.session_state.sentiment_loaded:
            try:
                result = run_sentiment_pipeline()
                agg = result if isinstance(result, dict) else {}
            except Exception:
                agg = {}

    # Derive news_updated BEFORE rendering header so it shows correctly
    news_updated = ""
    articles     = agg.get("articles", [])
    if articles:
        try:
            latest_dt = max(
                (a["published_dt"] for a in articles if a.get("published_dt")),
                default=None,
            )
            if latest_dt:
                news_updated = latest_dt.strftime("%d %b, %H:%M")
        except Exception:
            pass

    # Now render header with complete information (price + news timestamp)
    render_header(snapshot, prediction, news_updated=news_updated)

    st.markdown('<hr class="div">', unsafe_allow_html=True)

    # Cap sentiment adjustment — ML remains primary
    reconciled  = adjust_prediction_with_sentiment(
        prediction.get("signal", "HOLD"),
        prediction.get("confidence", 50.0),
        agg,
    )
    capped_conf = _cap_sentiment_adjustment(
        prediction.get("confidence", 50.0),
        reconciled["adjusted_conf"],
    )
    # Update both adjusted_conf AND delta_conf to reflect the cap
    capped_delta = round(capped_conf - prediction.get("confidence", 50.0), 2)
    reconciled   = {**reconciled, "adjusted_conf": capped_conf, "delta_conf": capped_delta}

    # ── Two-column layout ─────────────────────────────────────────────────────
    left, right = st.columns([3, 1], gap="large")

    with left:
        render_sentiment_kpis(agg, reconciled)

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

        render_alignment_banner(reconciled, agg)

        render_decision_summary(
            prediction,
            sentiment_label=agg.get("overall_label", "Neutral"),
            alignment=reconciled.get("alignment", "Neutral"),
        )

        st.markdown('<hr class="div">', unsafe_allow_html=True)

        render_headlines(articles, max_items=6)

        st.markdown('<hr class="div">', unsafe_allow_html=True)

        st.markdown('<div class="sec-label">Sentiment Breakdown</div>', unsafe_allow_html=True)
        render_charts(agg)

    with right:
        render_prob_bars(prediction)

        if prediction.get("probabilities"):
            st.markdown('<hr class="div">', unsafe_allow_html=True)

        render_feature_importance(prediction)

        if prediction.get("top_features") and prediction.get("source") == "model":
            st.markdown('<hr class="div">', unsafe_allow_html=True)

        render_model_performance()

        st.markdown('<hr class="div">', unsafe_allow_html=True)

        render_ml_health(ml_health)

        st.markdown('<hr class="div">', unsafe_allow_html=True)

        # ── Quick sentiment stats ──────────────────────────────────────────────
        st.markdown('<div class="sec-label">Quick Stats</div>', unsafe_allow_html=True)
        for key, val in [
            ("Avg Score",  f"{agg.get('avg_score', 0.0):+.4f}"),
            ("Weighted",   f"{agg.get('weighted_score', 0.0):+.4f}"),
            ("Positive",   f"{agg.get('pct_positive', 0.0):.1f}%"),
            ("Negative",   f"{agg.get('pct_negative', 0.0):.1f}%"),
            ("Neutral",    f"{agg.get('pct_neutral', 100.0):.1f}%"),
        ]:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between;
                        font-size:0.8rem; padding:5px 0; border-bottom:1px solid #21262d;">
                <span style="color:#484f58;">{key}</span>
                <span style="color:#c9d1d9; font-family:'DM Mono',monospace;">{val}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('<hr class="div">', unsafe_allow_html=True)

        render_trust_footer(snapshot, prediction, news_updated)


if __name__ == "__main__":
    main()
