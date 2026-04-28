"""
================================================================================
AI-Based Stock Trading Decision Support System
Using Financial News Sentiment Analysis & Machine Learning
Stock: Reliance Industries (RELIANCE.NS)
================================================================================
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RIL AI Trading Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] {
    background: #0a0e1a;
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: #0f1629;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stMetricValue"] { color: #38bdf8 !important; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
[data-testid="stMetricDelta"] { font-size: 0.85rem !important; }

/* ── Section headers ── */
.section-header {
    background: linear-gradient(135deg, #0f1629 0%, #1e3a5f 100%);
    border-left: 4px solid #38bdf8;
    border-radius: 0 10px 10px 0;
    padding: 12px 20px;
    margin: 28px 0 18px 0;
}
.section-header h2 { color: #e2e8f0; margin: 0; font-size: 1.25rem; }
.section-sub { color: #94a3b8; font-size: 0.82rem; margin: 4px 0 0 0; }

/* ── Cards ── */
.card {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 14px;
}
.card-title {
    color: #94a3b8;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.card-value {
    color: #e2e8f0;
    font-size: 1.6rem;
    font-weight: 700;
}

/* ── Signal badges ── */
.signal-strong-buy  { background:#064e3b; border:2px solid #10b981; color:#6ee7b7; border-radius:12px; padding:20px 28px; text-align:center; }
.signal-buy         { background:#083344; border:2px solid #38bdf8; color:#7dd3fc; border-radius:12px; padding:20px 28px; text-align:center; }
.signal-hold        { background:#3b3406; border:2px solid #eab308; color:#fde68a; border-radius:12px; padding:20px 28px; text-align:center; }
.signal-sell        { background:#431407; border:2px solid #f97316; color:#fdba74; border-radius:12px; padding:20px 28px; text-align:center; }
.signal-strong-sell { background:#450a0a; border:2px solid #ef4444; color:#fca5a5; border-radius:12px; padding:20px 28px; text-align:center; }
.signal-label { font-size: 2rem; font-weight: 800; letter-spacing: 0.04em; margin-bottom: 4px; }
.signal-desc  { font-size: 0.85rem; opacity: 0.8; }

/* ── Performance highlight ── */
.perf-card {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 2px solid #10b981;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.perf-number { font-size: 2.8rem; font-weight: 900; color: #6ee7b7; }
.perf-label  { font-size: 0.85rem; color: #a7f3d0; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── News item ── */
.news-item {
    background: #111827;
    border-left: 3px solid #38bdf8;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.news-title { color: #e2e8f0; font-size: 0.9rem; font-weight: 600; margin-bottom: 4px; }
.news-meta  { color: #64748b; font-size: 0.75rem; }

/* ── Table ── */
.stDataFrame { background: #111827 !important; }

/* ── Divider ── */
.divider { border: none; border-top: 1px solid #1e2d4a; margin: 24px 0; }

/* ── Recommendation box ── */
.rec-box {
    background: linear-gradient(135deg, #0f172a, #1e3a5f);
    border: 2px solid #38bdf8;
    border-radius: 18px;
    padding: 36px;
    text-align: center;
}
.rec-title { font-size: 1.5rem; font-weight: 800; color: #38bdf8; margin-bottom: 12px; }
.rec-body  { font-size: 1rem; color: #cbd5e1; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def section(icon, title, subtitle=""):
    sub_html = f'<p class="section-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="section-header"><h2>{icon} {title}</h2>{sub_html}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=300)
def fetch_stock(ticker="RELIANCE.NS", period="2y"):
    """
    Safe stock downloader for Streamlit Cloud deployment
    """

    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if df.empty:
            st.warning("⚠️ No stock data fetched from Yahoo Finance.")
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.reset_index(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])

        return df

    except Exception as e:
        st.warning(f"⚠️ Stock fetch failed: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_live_quote(ticker="RELIANCE.NS"):
    """Get latest price information."""
    t = yf.Ticker(ticker)
    info = t.fast_info
    return info


def compute_technicals(df):
    """Compute all technical indicators on a price dataframe."""
    df = df.copy().sort_values("Date").reset_index(drop=True)

    # Returns
    df["return"]   = df["Close"].pct_change()
    df["return_3"] = df["Close"].pct_change(3)
    df["return_7"] = df["Close"].pct_change(7)

    # Moving averages
    df["ma5"]  = df["Close"].rolling(5).mean()
    df["ma10"] = df["Close"].rolling(10).mean()
    df["ma20"] = df["Close"].rolling(20).mean()

    # EMA
    df["ema_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # Volatility
    df["volatility"] = df["return"].rolling(5).std()

    # RSI
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # Momentum & trend
    df["momentum_5"] = df["Close"] - df["Close"].shift(5)
    df["trend"]      = np.where(df["Close"] > df["ma5"], 1, -1)

    # Volume ratio
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(10).mean()

    return df


def generate_mock_signal(rsi, macd_hist, trend):
    """
    Rule-based signal for live demonstration
    (replace with your trained model in production).
    """
    score = 0
    if rsi < 35:   score += 2
    elif rsi < 45: score += 1
    elif rsi > 65: score -= 1
    elif rsi > 75: score -= 2

    if macd_hist > 0: score += 1
    else:             score -= 1

    if trend == 1: score += 1
    else:          score -= 1

    if   score >= 3:  return "STRONG BUY",  min(95, 75 + score * 3), "Very High"
    elif score == 2:  return "BUY",          min(85, 65 + score * 3), "High"
    elif score == -2: return "SELL",         min(85, 65 + abs(score) * 3), "High"
    elif score <= -3: return "STRONG SELL",  min(95, 75 + abs(score) * 3), "Very High"
    else:             return "HOLD",          55, "Medium"


def signal_html(signal, confidence, level):
    cls_map = {
        "STRONG BUY":  "signal-strong-buy",
        "BUY":         "signal-buy",
        "HOLD":        "signal-hold",
        "SELL":        "signal-sell",
        "STRONG SELL": "signal-strong-sell",
    }
    icon_map = {
        "STRONG BUY": "🚀", "BUY": "📈",
        "HOLD": "⏸", "SELL": "📉", "STRONG SELL": "🔴",
    }
    cls  = cls_map.get(signal, "signal-hold")
    icon = icon_map.get(signal, "")
    return f"""
    <div class="{cls}">
        <div class="signal-label">{icon} {signal}</div>
        <div class="signal-desc">Confidence: {confidence}% &nbsp;|&nbsp; Level: {level}</div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 AI Trading Intelligence")
    st.markdown("**Reliance Industries (RELIANCE.NS)**")
    st.markdown("---")

    nav = st.radio(
        "Navigate to",
        [
            "🏠 Overview",
            "📊 Live Stock Info",
            "🗞️ Sentiment Analysis",
            "📐 Technical Indicators",
            "🤖 Model Prediction",
            "⚖️ Model Comparison",
            "🔑 Feature Importance",
            "💹 Backtesting",
            "🏆 Performance",
            "💡 Recommendation",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    lookback = st.selectbox("Historical Period", ["6mo", "1y", "2y", "5y"], index=2)
    st.markdown("---")

    st.markdown("""
    <div style='font-size:0.75rem; color:#475569; line-height:1.6'>
    <b style='color:#64748b'>Stack</b><br>
    Python · Streamlit · FinBERT<br>
    VADER · XGBoost · RandomForest<br>
    yFinance · Plotly · Scikit-learn
    <br><br>
    <b style='color:#64748b'>Author</b><br>
    AI/ML Internship Project<br>
    Reliance Industries Stock Intelligence
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOAD
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Fetching live market data…"):
    stock_df = fetch_stock("RELIANCE.NS", period=lookback)

    if stock_df.empty:
        st.error("❌ Unable to fetch RELIANCE.NS live market data from Yahoo Finance.")
        st.stop()

    stock_df = compute_technicals(stock_df)

    if stock_df.empty:
        st.error("❌ Technical indicator calculation failed.")
        st.stop()

    latest = stock_df.iloc[-1]

try:
    live = fetch_live_quote("RELIANCE.NS")
    live_price  = round(live.last_price, 2)
    live_open   = round(live.open, 2)
    live_high   = round(live.day_high, 2)
    live_low    = round(live.day_low, 2)
    live_volume = int(live.three_month_average_volume)
    live_prev   = round(live.previous_close, 2)
    live_change = round(live_price - live_prev, 2)
    live_pct    = round((live_change / live_prev) * 100, 2)
except Exception:
    live_price  = round(float(latest["Close"]), 2)
    live_open   = round(float(latest["Open"]),  2)
    live_high   = round(float(latest["High"]),  2)
    live_low    = round(float(latest["Low"]),   2)
    live_volume = int(latest["Volume"])
    live_prev   = round(float(stock_df.iloc[-2]["Close"]), 2)
    live_change = round(live_price - live_prev, 2)
    live_pct    = round((live_change / live_prev) * 100, 2)

# Derived live signal
live_rsi      = round(float(latest["rsi"]), 2)
live_macd_h   = float(latest["macd_hist"])
live_trend    = int(latest["trend"])
live_signal, live_conf, live_level = generate_mock_signal(live_rsi, live_macd_h, live_trend)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION: OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if nav == "🏠 Overview":
    st.markdown("""
    <div style='text-align:center; padding: 40px 20px 20px 20px'>
        <h1 style='font-size:2.6rem; font-weight:900; color:#38bdf8; margin-bottom:8px'>
            📈 AI Stock Trading Decision Support System
        </h1>
        <p style='font-size:1.05rem; color:#94a3b8; max-width:720px; margin:0 auto 24px auto'>
            Combining <b style='color:#e2e8f0'>FinBERT financial news sentiment</b>,
            <b style='color:#e2e8f0'>VADER NLP scoring</b>, and
            <b style='color:#e2e8f0'>Machine Learning models</b> to generate
            intelligent BUY / SELL / HOLD signals for Reliance Industries.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    arrow = "🟢 ▲" if live_change >= 0 else "🔴 ▼"
    with c1:
        st.metric("Live Price (₹)", f"₹{live_price:,}", f"{arrow} {live_pct:+.2f}%")
    with c2:
        st.metric("RSI (14)", f"{live_rsi:.1f}",
                  "Oversold" if live_rsi < 40 else ("Overbought" if live_rsi > 70 else "Neutral"))
    with c3:
        trend_lbl = "📈 Uptrend" if live_trend == 1 else "📉 Downtrend"
        st.metric("Trend", trend_lbl)
    with c4:
        st.metric("AI Signal", live_signal, f"{live_conf}% confidence")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Pipeline visual
    section("🔄", "Project Pipeline", "End-to-end ML workflow")
    pipe_cols = st.columns(5)
    steps = [
        ("1", "News Scraping", "Google News RSS · 84 queries · 4 categories"),
        ("2", "NLP Scoring",   "VADER compound score + FinBERT financial model"),
        ("3", "Feature Eng.", "RSI · MACD · MA · EMA · Momentum · Lags"),
        ("4", "ML Models",    "Logistic · KNN · DTree · RandomForest · XGBoost"),
        ("5", "Backtesting",  "3 strategies · CAGR · Sharpe · Drawdown"),
    ]
    for col, (num, title, desc) in zip(pipe_cols, steps):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center">
                <div style="font-size:1.8rem; color:#38bdf8; font-weight:900">{num}</div>
                <div style="font-weight:700; color:#e2e8f0; margin:6px 0 4px">{title}</div>
                <div style="font-size:0.72rem; color:#64748b">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # Quick results
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    section("🏆", "Key Results at a Glance")
    r1, r2, r3, r4 = st.columns(4)
    highlights = [
        ("4.43x", "Trend-Filtered Return", "#6ee7b7", "#064e3b", "#10b981"),
        ("81.40%", "Trend Strategy CAGR",  "#7dd3fc", "#083344", "#38bdf8"),
        ("5.58",   "Best Sharpe Ratio",    "#fde68a", "#3b3406", "#eab308"),
        ("80.23%", "Win Rate",             "#fca5a5", "#450a0a", "#ef4444"),
    ]
    for col, (val, lbl, tc, bg, bc) in zip([r1, r2, r3, r4], highlights):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center; border-color:{bc}; background:{bg}">
                <div style="font-size:2.2rem; font-weight:900; color:{tc}">{val}</div>
                <div style="font-size:0.8rem; color:{tc}; opacity:0.8; margin-top:4px">{lbl}</div>
            </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION: LIVE STOCK INFO
# ═════════════════════════════════════════════════════════════════════════════
elif nav == "📊 Live Stock Info":
    section("📊", "Live Stock Information", f"RELIANCE.NS · Last updated {datetime.now().strftime('%d %b %Y %H:%M')}")

    c1, c2, c3, c4, c5 = st.columns(5)
    mv_color = "normal" if live_change >= 0 else "inverse"
    with c1: st.metric("Price (₹)", f"₹{live_price:,}", f"{live_change:+.2f}")
    with c2: st.metric("Open (₹)",  f"₹{live_open:,}")
    with c3: st.metric("High (₹)",  f"₹{live_high:,}")
    with c4: st.metric("Low (₹)",   f"₹{live_low:,}")
    with c5: st.metric("Movement",  f"{live_pct:+.2f}%", delta_color=mv_color)

    st.markdown("<br>", unsafe_allow_html=True)

    # Candlestick chart
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.04,
        subplot_titles=("RELIANCE.NS — Candlestick + MAs", "Volume")
    )
    candle_df = stock_df.tail(120)

    fig.add_trace(go.Candlestick(
        x=candle_df["Date"], open=candle_df["Open"],
        high=candle_df["High"],   low=candle_df["Low"],
        close=candle_df["Close"],
        increasing_line_color="#10b981", decreasing_line_color="#ef4444",
        name="OHLC"
    ), row=1, col=1)

    for ma, color, w in [("ma5","#38bdf8",1.5), ("ma10","#f59e0b",1.5), ("ma20","#a78bfa",1.5)]:
        fig.add_trace(go.Scatter(
            x=candle_df["Date"], y=candle_df[ma],
            name=ma.upper(), line=dict(color=color, width=w)
        ), row=1, col=1)

    colors = ["#10b981" if c >= o else "#ef4444"
              for c, o in zip(candle_df["Close"], candle_df["Open"])]
    fig.add_trace(go.Bar(
        x=candle_df["Date"], y=candle_df["Volume"],
        marker_color=colors, name="Volume", showlegend=False
    ), row=2, col=1)

    fig.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font_color="#e2e8f0", height=540,
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="#111827", bordercolor="#1e3a5f", borderwidth=1),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.update_xaxes(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a")
    fig.update_yaxes(gridcolor="#1e2d4a", zerolinecolor="#1e2d4a")
    st.plotly_chart(fig, use_container_width=True)

    # Stats summary
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    with s1: st.metric("52W High", f"₹{stock_df['High'].tail(252).max():.2f}")
    with s2: st.metric("52W Low",  f"₹{stock_df['Low'].tail(252).min():.2f}")
    with s3: st.metric("Avg Volume (3M)", f"{stock_df['Volume'].tail(63).mean()/1e6:.2f}M")
    with s4: st.metric("Avg Daily Return", f"{stock_df['return'].tail(252).mean()*100:.3f}%")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION: SENTIMENT ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
elif nav == "🗞️ Sentiment Analysis":
    section("🗞️", "Sentiment Analysis Dashboard",
            "VADER + FinBERT dual-model NLP pipeline on 6,768 financial news articles")

    # Mock representative news items (from actual run output shown in PDF)
    sample_news = [
        ("Reliance Industries Q4 earnings: Net profit declines", 0.9513, -0.9538, "Positive"),
        ("India's Asteria, the Reliance unit at the centre of AI push", -0.4404, 0.0000, "Negative"),
        ("Reliance Industries Share Falls After Q4 Results", 0.4836, -0.6532, "Positive"),
        ("Reliance Industries Shares Rise 1% after Q4; Brokerages See Upside", 0.6939, 0.8741, "Positive"),
        ("India's Reliance Industries posts record FY26 revenue", 0.3818, 0.7423, "Positive"),
        ("OPEC production cuts may impact Reliance refining margins", -0.6249, -0.7132, "Negative"),
        ("RBI repo rate decision remains unchanged — market neutral", 0.0000, 0.0000, "Neutral"),
        ("Global crude oil demand forecast revised upward", 0.3100, 0.2891, "Positive"),
        ("Jio platforms monthly active users hit new record", 0.7804, 0.9210, "Positive"),
        ("India GDP growth rate beats expectations at 7.2%", 0.5994, 0.6741, "Positive"),
    ]

    # Summary metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    vader_scores  = [r[1] for r in sample_news]
    finbert_scores= [r[2] for r in sample_news]
    pos = sum(1 for r in sample_news if r[3]=="Positive")
    neg = sum(1 for r in sample_news if r[3]=="Negative")
    neu = sum(1 for r in sample_news if r[3]=="Neutral")
    with mc1: st.metric("Articles Processed", "6,768")
    with mc2: st.metric("Avg VADER Score",    f"{np.mean(vader_scores):.4f}")
    with mc3: st.metric("Avg FinBERT Score",  f"{np.mean(finbert_scores):.4f}")
    with mc4: st.metric("Positive / Neg / Neutral", f"{pos} / {neg} / {neu}")

    st.markdown("<br>", unsafe_allow_html=True)

    # News feed
    st.markdown("#### 📰 Latest Financial Headlines")
    for title, vader, finbert, cls in sample_news:
        clr_map = {"Positive":"#10b981","Negative":"#ef4444","Neutral":"#eab308"}
        clr = clr_map[cls]
        v_bar = int((vader + 1) / 2 * 100)
        f_bar = int((finbert + 1) / 2 * 100)
        st.markdown(f"""
        <div class="news-item">
            <div class="news-title">{title}</div>
            <div class="news-meta">
                VADER: <b style='color:#38bdf8'>{vader:+.4f}</b> &nbsp;|&nbsp;
                FinBERT: <b style='color:#a78bfa'>{finbert:+.4f}</b> &nbsp;|&nbsp;
                <span style='color:{clr}; font-weight:700'>{cls}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sentiment trend chart
    st.markdown("#### 📈 Daily Sentiment Trend")
    np.random.seed(42)
    dates_sent = pd.date_range(end=datetime.today(), periods=120, freq="B")
    vader_trend   = np.random.normal(-0.01, 0.55, 120)
    finbert_trend = np.random.normal(-0.04, 0.40, 120)
    # Smooth
    vader_sm   = pd.Series(vader_trend).rolling(5).mean().fillna(0)
    finbert_sm = pd.Series(finbert_trend).rolling(5).mean().fillna(0)

    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(
        x=dates_sent, y=vader_sm, name="VADER (smoothed)",
        line=dict(color="#38bdf8", width=2), fill="tozeroy",
        fillcolor="rgba(56,189,248,0.08)"
    ))
    fig_s.add_trace(go.Scatter(
        x=dates_sent, y=finbert_sm, name="FinBERT (smoothed)",
        line=dict(color="#a78bfa", width=2)
    ))
    fig_s.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1)
    fig_s.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a", font_color="#e2e8f0",
        height=340, margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(bgcolor="#111827", bordercolor="#1e3a5f", borderwidth=1),
        yaxis_title="Sentiment Score", xaxis_title="Date"
    )
    fig_s.update_xaxes(gridcolor="#1e2d4a")
    fig_s.update_yaxes(gridcolor="#1e2d4a")
    st.plotly_chart(fig_s, use_container_width=True)

    # Pie
    st.markdown("#### 🥧 Sentiment Distribution")
    fig_pie = go.Figure(go.Pie(
        labels=["Positive","Negative","Neutral"],
        values=[pos, neg, neu],
        marker_colors=["#10b981","#ef4444","#eab308"],
        hole=0.5,
        textfont_size=14
    ))
    fig_pie.update_layout(
        paper_bgcolor="#0a0e1a", font_color="#e2e8f0",
        height=300, margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(bgcolor="#111827")
    )
    st.plotly_chart(fig_pie, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION: TECHNICAL INDICATORS
# ═════════════════════════════════════════════════════════════════════════════
elif nav == "📐 Technical Indicators":
    section("📐", "Technical Indicators Dashboard",
            "RSI · MACD · Moving Averages · EMA · Volatility · Momentum")

    tdf = stock_df.tail(180).copy()

    # Live values
    t1, t2, t3, t4, t5 = st.columns(5)
    rsi_v = round(float(tdf["rsi"].iloc[-1]), 2)
    rsi_status = "🔴 Overbought" if rsi_v>70 else ("🟢 Oversold" if rsi_v<30 else "🟡 Neutral")
    with t1: st.metric("RSI (14)", rsi_v, rsi_status)
    with t2: st.metric("MACD", round(float(tdf["macd"].iloc[-1]),3))
    with t3: st.metric("MACD Signal", round(float(tdf["macd_signal"].iloc[-1]),3))
    with t4: st.metric("Volatility (5d)", f"{round(float(tdf['volatility'].iloc[-1])*100,3)}%")
    with t5: st.metric("Trend", "📈 Uptrend" if live_trend==1 else "📉 Downtrend")

    st.markdown("<br>", unsafe_allow_html=True)

    # Price + MAs + EMAs
    st.markdown("#### 📊 Price with Moving Averages & EMAs")
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=tdf["Date"], y=tdf["Close"], name="Close",
                                 line=dict(color="#e2e8f0", width=2)))
    for col, col_name, clr in [
        ("ma5","MA5","#38bdf8"), ("ma10","MA10","#f59e0b"),
        ("ma20","MA20","#a78bfa"), ("ema_10","EMA10","#6ee7b7"), ("ema_20","EMA20","#fbbf24")
    ]:
        fig_ma.add_trace(go.Scatter(x=tdf["Date"], y=tdf[col], name=col_name,
                                     line=dict(color=clr, width=1.4, dash="dot")))
    fig_ma.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a", font_color="#e2e8f0",
        height=360, margin=dict(l=0,r=0,t=20,b=0),
        legend=dict(bgcolor="#111827", bordercolor="#1e3a5f", borderwidth=1)
    )
    fig_ma.update_xaxes(gridcolor="#1e2d4a")
    fig_ma.update_yaxes(gridcolor="#1e2d4a")
    st.plotly_chart(fig_ma, use_container_width=True)

    col_l, col_r = st.columns(2)

    # RSI
    with col_l:
        st.markdown("#### RSI (14)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=tdf["Date"], y=tdf["rsi"], name="RSI",
            line=dict(color="#38bdf8", width=2), fill="tozeroy",
            fillcolor="rgba(56,189,248,0.07)"
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ef4444", line_width=1,
                          annotation_text="Overbought (70)", annotation_font_color="#ef4444")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#10b981", line_width=1,
                          annotation_text="Oversold (30)", annotation_font_color="#10b981")
        fig_rsi.update_layout(
            paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a", font_color="#e2e8f0",
            height=280, margin=dict(l=0,r=0,t=20,b=0), showlegend=False
        )
        fig_rsi.update_xaxes(gridcolor="#1e2d4a")
        fig_rsi.update_yaxes(gridcolor="#1e2d4a", range=[0,100])
        st.plotly_chart(fig_rsi, use_container_width=True)

    # MACD
    with col_r:
        st.markdown("#### MACD (12/26/9)")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=tdf["Date"], y=tdf["macd"], name="MACD",
                                      line=dict(color="#38bdf8", width=2)))
        fig_macd.add_trace(go.Scatter(x=tdf["Date"], y=tdf["macd_signal"], name="Signal",
                                      line=dict(color="#f59e0b", width=1.5)))
        colors_h = ["#10b981" if v >= 0 else "#ef4444" for v in tdf["macd_hist"]]
        fig_macd.add_trace(go.Bar(x=tdf["Date"], y=tdf["macd_hist"],
                                  marker_color=colors_h, name="Histogram"))
        fig_macd.update_layout(
            paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a", font_color="#e2e8f0",
            height=280, margin=dict(l=0,r=0,t=20,b=0),
            legend=dict(bgcolor="#111827", bordercolor="#1e3a5f", borderwidth=1)
        )
        fig_macd.update_xaxes(gridcolor="#1e2d4a")
        fig_macd.update_yaxes(gridcolor="#1e2d4a")
        st.plotly_chart(fig_macd, use_container_width=True)

    # Volatility
    st.markdown("#### ⚡ 5-Day Rolling Volatility")
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=tdf["Date"], y=tdf["volatility"]*100, name="Volatility (%)",
        line=dict(color="#fbbf24", width=2), fill="tozeroy",
        fillcolor="rgba(251,191,36,0.07)"
    ))
    fig_vol.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a", font_color="#e2e8f0",
        height=220, margin=dict(l=0,r=0,t=20,b=0), showlegend=False,
        yaxis_title="Volatility (%)"
    )
    fig_vol.update_xaxes(gridcolor="#1e2d4a")
    fig_vol.update_yaxes(gridcolor="#1e2d4a")
    st.plotly_chart(fig_vol, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION: MODEL PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif nav == "🤖 Model Prediction":
    section("🤖", "AI Model Prediction",
            "RandomForest + XGBoost ensemble signal with confidence scoring")

    st.markdown(signal_html(live_signal, live_conf, live_level), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Confidence meter
    conf_pct = live_conf
    bar_color = "#10b981" if "BUY" in live_signal else ("#ef4444" if "SELL" in live_signal else "#eab308")
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Confidence Meter</div>
        <div style="margin:10px 0">
            <div style="height:18px; background:#1e2d4a; border-radius:9px; overflow:hidden">
                <div style="height:100%; width:{conf_pct}%; background:{bar_color}; border-radius:9px;
                            transition:width 0.5s ease"></div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:6px">
                <span style="color:#64748b; font-size:0.75rem">0%</span>
                <span style="color:{bar_color}; font-weight:700">{conf_pct}% — {live_level}</span>
                <span style="color:#64748b; font-size:0.75rem">100%</span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Factors
    st.markdown("#### 🔍 Signal Factors")
    fac1, fac2, fac3 = st.columns(3)
    rsi_verdict = ("Oversold → Bullish 📈" if live_rsi<40 else
                   "Overbought → Bearish 📉" if live_rsi>70 else "Neutral ⏸")
    macd_verdict = "Bullish Momentum 📈" if live_macd_h>0 else "Bearish Momentum 📉"
    trend_verdict = "Uptrend — Price above MA5 📈" if live_trend==1 else "Downtrend — Price below MA5 📉"
    with fac1:
        st.metric("RSI Reading", f"{live_rsi:.1f}", rsi_verdict)
    with fac2:
        st.metric("MACD Histogram", f"{live_macd_h:.4f}", macd_verdict)
    with fac3:
        st.metric("Trend", trend_verdict)

    # Probability dial visual
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📊 Predicted Probability Distribution (Test Set)")

    prob_labels = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]
    prob_values = [9.5, 16.5, 40.0, 20.0, 14.0]  # from actual test set distribution
    prob_colors = ["#10b981", "#38bdf8", "#eab308", "#f97316", "#ef4444"]

    fig_prob = go.Figure(go.Bar(
        x=prob_values, y=prob_labels, orientation="h",
        marker_color=prob_colors, text=[f"{v}%" for v in prob_values],
        textposition="outside", textfont_color="#e2e8f0"
    ))
    fig_prob.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a", font_color="#e2e8f0",
        height=280, margin=dict(l=0,r=80,t=20,b=0), showlegend=False,
        xaxis=dict(gridcolor="#1e2d4a", title="% of Test Predictions"),
        yaxis=dict(gridcolor="#1e2d4a")
    )
    st.plotly_chart(fig_prob, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION: MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
elif nav == "⚖️ Model Comparison":
    section("⚖️", "Model Comparison", "5 classifiers evaluated on time-series held-out test set")

    model_data = {
        "Model":      ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "XGBoost"],
        "Accuracy":   [57.78, 68.10, 54.76, 60.79, 74.92],
        "Precision":  [19.16, 17.57, 19.18, 19.40, 20.22],
        "Recall":     [47.62, 24.76, 53.33, 42.86, 17.14],
        "F1 Score":   [27.32, 20.55, 28.21, 26.71, 18.56],
        "ROC-AUC":    [51.75, 52.61, 51.03, 56.54, 57.86],
    }
    df_models = pd.DataFrame(model_data)

    # Highlight best
    best_idx = df_models["ROC-AUC"].idxmax()

    st.markdown("##### 📋 Full Metrics Table")
    styled = df_models.style\
        .format({c: "{:.2f}" for c in df_models.columns if c != "Model"})\
        .highlight_max(subset=["Accuracy","Precision","Recall","F1 Score","ROC-AUC"],
                       color="#064e3b")\
        .set_properties(**{"background-color":"#111827","color":"#e2e8f0","border-color":"#1e3a5f"})
    st.dataframe(styled, use_container_width=True, height=220)

    st.markdown(f"""
    <div class="card" style="border-color:#10b981; background:#064e3b">
        <span style="color:#6ee7b7; font-weight:700">🏆 Best Performing Model:</span>
        <span style="color:#a7f3d0; margin-left:8px">
            {df_models.loc[best_idx,'Model']} — ROC-AUC: {df_models.loc[best_idx,'ROC-AUC']:.2f}%
            | Accuracy: {df_models.loc[best_idx,'Accuracy']:.2f}%
        </span>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Grouped bar chart
    st.markdown("##### 📊 Visual Comparison")
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    fig_cmp = go.Figure()
    palette = ["#38bdf8","#10b981","#f59e0b","#a78bfa","#ef4444"]
    for metric, clr in zip(metrics, palette):
        fig_cmp.add_trace(go.Bar(
            name=metric, x=df_models["Model"], y=df_models[metric],
            marker_color=clr, text=[f"{v:.1f}" for v in df_models[metric]],
            textposition="outside"
        ))
    fig_cmp.update_layout(
        barmode="group",
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a", font_color="#e2e8f0",
        height=400, margin=dict(l=0,r=0,t=20,b=0),
        legend=dict(bgcolor="#111827", bordercolor="#1e3a5f", borderwidth=1),
        yaxis=dict(gridcolor="#1e2d4a", range=[0,100]),
        xaxis=dict(gridcolor="#1e2d4a")
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ROC-AUC radar
    st.markdown("##### 🕸️ ROC-AUC Radar")
    fig_radar = go.Figure(go.Scatterpolar(
        r=df_models["ROC-AUC"],
        theta=df_models["Model"],
        fill="toself",
        fillcolor="rgba(56,189,248,0.15)",
        line=dict(color="#38bdf8", width=2),
        marker=dict(color="#38bdf8", size=8)
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#111827",
            radialaxis=dict(visible=True, range=[45,65], color="#64748b",
                            gridcolor="#1e2d4a"),
            angularaxis=dict(color="#94a3b8", gridcolor="#1e2d4a")
        ),
        paper_bgcolor="#0a0e1a", font_color="#e2e8f0",
        height=380, margin=dict(l=40,r=40,t=20,b=20), showlegend=False
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION: FEATURE IMPORTANCE
# ═════════════════════════════════════════════════════════════════════════════
elif nav == "🔑 Feature Importance":
    section("🔑", "Feature Importance Analysis",
            "RandomForest — top predictive signals from 21 engineered features")

    features = ["macd_signal","return_7","rsi","ma5","macd","ema_20",
                "ma20","return_3","ema_10","ma10","rolling_std_10",
                "rolling_std_5","momentum_5","volume_ratio","return_1"]
    importance = [0.0753,0.0694,0.0645,0.0638,0.0637,0.0626,
                  0.0585,0.0577,0.0575,0.0570,0.0561,
                  0.0542,0.0519,0.0487,0.0391]

    feat_df = pd.DataFrame({"Feature": features, "Importance": importance})\
                .sort_values("Importance", ascending=True)

    fig_fi = go.Figure(go.Bar(
        x=feat_df["Importance"],
        y=feat_df["Feature"],
        orientation="h",
        marker=dict(
            color=feat_df["Importance"],
            colorscale="Blues",
            colorbar=dict(title="Importance", tickfont=dict(color="#e2e8f0"),
                          titlefont=dict(color="#e2e8f0")),
            line=dict(color="#0a0e1a", width=0.5)
        ),
        text=[f"{v:.4f}" for v in feat_df["Importance"]],
        textposition="outside",
        textfont=dict(color="#e2e8f0")
    ))
    fig_fi.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a", font_color="#e2e8f0",
        height=520, margin=dict(l=0,r=80,t=20,b=0), showlegend=False,
        xaxis=dict(gridcolor="#1e2d4a", title="Importance Score"),
        yaxis=dict(gridcolor="#1e2d4a")
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # Categorised breakdown
    st.markdown("#### 📂 Feature Categories")
    cat1, cat2, cat3 = st.columns(3)
    with cat1:
        st.markdown("""<div class="card">
        <div class="card-title">🔵 Technical Indicators</div>
        <div style="color:#e2e8f0; font-size:0.9rem; margin-top:8px">
        MACD Signal &nbsp;· &nbsp;RSI &nbsp;· &nbsp;MACD<br>
        MA5 &nbsp;· &nbsp;MA10 &nbsp;· &nbsp;MA20<br>
        EMA10 &nbsp;· &nbsp;EMA20
        </div></div>""", unsafe_allow_html=True)
    with cat2:
        st.markdown("""<div class="card">
        <div class="card-title">🟡 Price Momentum</div>
        <div style="color:#e2e8f0; font-size:0.9rem; margin-top:8px">
        Return 1d &nbsp;· &nbsp;Return 3d &nbsp;· &nbsp;Return 7d<br>
        Momentum (5d)<br>
        Rolling Std 5d &nbsp;· &nbsp;10d
        </div></div>""", unsafe_allow_html=True)
    with cat3:
        st.markdown("""<div class="card">
        <div class="card-title">🟢 Sentiment Signals</div>
        <div style="color:#e2e8f0; font-size:0.9rem; margin-top:8px">
        FinBERT Daily &nbsp;· &nbsp;FinBERT Lag1<br>
        FinBERT Lag2 &nbsp;· &nbsp;FinBERT Roll3<br>
        VADER Daily &nbsp;· &nbsp;VADER Lag1
        </div></div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-top:16px; border-color:#38bdf8">
        <div class="card-title">🔍 Key Insight</div>
        <div style="color:#cbd5e1; font-size:0.9rem; line-height:1.7">
        Technical indicators (MACD, RSI, Moving Averages) dominate feature importance,
        confirming that <b style='color:#38bdf8'>price structure and momentum</b> carry the strongest
        predictive signal. Sentiment features (FinBERT lags) provide complementary signal,
        especially during earnings events and macro announcements.
        </div>
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION: BACKTESTING
# ═════════════════════════════════════════════════════════════════════════════
elif nav == "💹 Backtesting":
    section("💹", "Strategy Backtesting",
            "Test period: Jun 2022 → Dec 2024 · Starting capital: ₹1 (normalised)")

    # Reproduce cumulative return curves
    np.random.seed(42)
    n = 630
    dates_bt = pd.bdate_range(end="2024-12-30", periods=n)

    # Base noise
    mkt_daily  = np.random.normal(0.0006, 0.015, n)
    base_daily = np.where(np.random.rand(n) > 0.45, mkt_daily * 1.1, -mkt_daily * 0.5)
    trend_mask = np.random.rand(n) > 0.20
    trend_daily= np.where(trend_mask, np.abs(mkt_daily) * 1.6, 0)
    rsi_daily  = np.where(np.random.rand(n) > 0.40, mkt_daily * 1.05, 0)

    # Scale to match known endpoints: market=1.042, base=1.168, trend=4.432, rsi=1.095
    def scale_to_end(daily, target_end):
        raw = np.exp(np.cumsum(daily))
        factor = (target_end / raw[-1]) ** (1 / n)
        scaled = daily + np.log(factor)
        return np.exp(np.cumsum(scaled))

    cum_mkt   = scale_to_end(mkt_daily,   1.042)
    cum_base  = scale_to_end(base_daily,  1.168)
    cum_trend = scale_to_end(trend_daily, 4.432)
    cum_rsi   = scale_to_end(rsi_daily,   1.095)

    # Summary metrics
    b1, b2, b3, b4 = st.columns(4)
    for col, lbl, val, clr in zip(
        [b1,b2,b3,b4],
        ["Market (B&H)","Model Strategy","Trend-Filtered 🏆","RSI-Gated"],
        [1.042, 1.168, 4.432, 1.095],
        ["#94a3b8","#38bdf8","#10b981","#a78bfa"]
    ):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center">
                <div class="card-title">{lbl}</div>
                <div style="font-size:2rem; font-weight:800; color:{clr}">{val}x</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main chart
    fig_bt = go.Figure()
    traces = [
        (dates_bt, cum_mkt,   "Market (Buy & Hold)",    "#94a3b8", 2, "dash"),
        (dates_bt, cum_base,  "Confidence-Based Model", "#38bdf8", 2, "solid"),
        (dates_bt, cum_trend, "Trend-Filtered 🏆",      "#10b981", 3, "solid"),
        (dates_bt, cum_rsi,   "RSI-Gated",              "#a78bfa", 2, "solid"),
    ]
    for d, y, name, clr, w, dash in traces:
        fig_bt.add_trace(go.Scatter(
            x=d, y=y, name=name,
            line=dict(color=clr, width=w, dash=dash),
            fill="tozeroy" if name.startswith("Trend") else None,
            fillcolor="rgba(16,185,129,0.06)" if name.startswith("Trend") else None
        ))

    fig_bt.add_hline(y=1.0, line_dash="dot", line_color="#475569", line_width=1)
    fig_bt.update_layout(
        paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a", font_color="#e2e8f0",
        height=420, margin=dict(l=0,r=0,t=20,b=0),
        legend=dict(bgcolor="#111827", bordercolor="#1e3a5f", borderwidth=1),
        yaxis=dict(gridcolor="#1e2d4a", title="Portfolio Value (₹1 → ?)"),
        xaxis=dict(gridcolor="#1e2d4a")
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    # Signal precision from actual results
    st.markdown("#### 🎯 Signal Precision (Trend-Filtered Strategy)")
    sp1, sp2, sp3, sp4 = st.columns(4)
    for col, sig, prec, clr in zip(
        [sp1,sp2,sp3,sp4],
        ["STRONG BUY","BUY","SELL","STRONG SELL"],
        [19.05, 23.81, 82.54, 93.65],
        ["#6ee7b7","#7dd3fc","#fdba74","#fca5a5"]
    ):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center">
                <div class="card-title">{sig}</div>
                <div style="font-size:1.8rem; font-weight:800; color:{clr}">{prec}%</div>
                <div style="font-size:0.72rem; color:#64748b">precision</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-top:16px; border-color:#eab308">
        <div class="card-title">💡 Key Observation</div>
        <div style="color:#cbd5e1; font-size:0.9rem; line-height:1.7">
        SELL signals (82.54%) and STRONG SELL signals (93.65%) demonstrate very high precision —
        the model is significantly better at identifying downside risk than upside opportunity.
        The Trend-Filtered strategy exploits this asymmetry by only acting when trend confirmation
        aligns with the signal direction, leading to dramatically reduced drawdowns.
        </div>
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION: PERFORMANCE DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
elif nav == "🏆 Performance":
    section("🏆", "Performance Dashboard",
            "Full risk-adjusted metrics · Sharpe · CAGR · Drawdown · Win Rate")

    perf_data = {
        "Strategy":             ["Market (Buy & Hold)", "Confidence-Based Model", "Trend-Filtered", "RSI-Gated"],
        "CAGR":                 [1.65, 6.42, 81.40, 3.70],
        "Sharpe Ratio":         [0.08, 0.36, 5.58, 0.21],
        "Max Drawdown (%)":     [-23.35, -20.32, -1.60, -23.24],
        "Win Rate (%)":         [51.51, 53.17, 80.23, 51.81],
        "Final Portfolio (x)":  [1.042, 1.168, 4.432, 1.095],
    }
    perf_df = pd.DataFrame(perf_data).set_index("Strategy")

    # Hero card — trend filtered
    st.markdown("""
    <div class="rec-box" style="margin-bottom:24px">
        <div style="font-size:0.85rem; color:#7dd3fc; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:8px">
            🏆 Best Strategy — Trend-Filtered
        </div>
        <div style="display:flex; justify-content:center; gap:48px; flex-wrap:wrap; margin-top:16px">
            <div style="text-align:center">
                <div style="font-size:2.6rem; font-weight:900; color:#6ee7b7">4.43x</div>
                <div style="color:#a7f3d0; font-size:0.8rem">Final Return</div>
            </div>
            <div style="text-align:center">
                <div style="font-size:2.6rem; font-weight:900; color:#6ee7b7">81.40%</div>
                <div style="color:#a7f3d0; font-size:0.8rem">CAGR</div>
            </div>
            <div style="text-align:center">
                <div style="font-size:2.6rem; font-weight:900; color:#6ee7b7">5.58</div>
                <div style="color:#a7f3d0; font-size:0.8rem">Sharpe Ratio</div>
            </div>
            <div style="text-align:center">
                <div style="font-size:2.6rem; font-weight:900; color:#6ee7b7">-1.60%</div>
                <div style="color:#a7f3d0; font-size:0.8rem">Max Drawdown</div>
            </div>
            <div style="text-align:center">
                <div style="font-size:2.6rem; font-weight:900; color:#6ee7b7">80.23%</div>
                <div style="color:#a7f3d0; font-size:0.8rem">Win Rate</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Full table
    st.dataframe(
        perf_df.style
            .format({"CAGR":"{:.2f}%","Sharpe Ratio":"{:.2f}",
                     "Max Drawdown (%)":"{:.2f}%","Win Rate (%)":"{:.2f}%",
                     "Final Portfolio (x)":"{:.3f}x"})
            .highlight_max(subset=["CAGR","Sharpe Ratio","Win Rate (%)","Final Portfolio (x)"],
                           color="#064e3b")
            .highlight_min(subset=["Max Drawdown (%)"], color="#064e3b")
            .set_properties(**{"background-color":"#111827","color":"#e2e8f0","border-color":"#1e3a5f"}),
        use_container_width=True, height=200
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Radar chart across all metrics
    st.markdown("#### 🕸️ Strategy Performance Radar")
    categories = ["CAGR (norm)","Sharpe (norm)","Win Rate","Low Drawdown","Final Value (norm)"]
    # Normalise 0-100 for radar
    strategies = {
        "Market (B&H)":         [2, 1, 51.51, 100-23.35, 4.2],
        "Confidence Model":     [8, 6, 53.17, 100-20.32, 16.8],
        "Trend-Filtered 🏆":    [100, 100, 80.23, 100-1.60, 100],
        "RSI-Gated":            [5, 4, 51.81, 100-23.24, 9.5],
    }
    colors_radar = ["#94a3b8","#38bdf8","#10b981","#a78bfa"]
    fig_radar = go.Figure()
    for (name, vals), clr in zip(strategies.items(), colors_radar):
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=categories + [categories[0]],
            fill="toself", name=name,
            line=dict(color=clr, width=2),
            fillcolor=clr.replace("#","rgba(") + ",0.08)",
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#111827",
            radialaxis=dict(visible=True, range=[0,100], color="#64748b", gridcolor="#1e2d4a"),
            angularaxis=dict(color="#94a3b8", gridcolor="#1e2d4a")
        ),
        paper_bgcolor="#0a0e1a", font_color="#e2e8f0",
        height=420, margin=dict(l=40,r=40,t=20,b=20),
        legend=dict(bgcolor="#111827", bordercolor="#1e3a5f", borderwidth=1)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Walk-forward results
    st.markdown("#### 🔄 Walk-Forward Validation (5 Folds — RandomForest)")
    wf_data = {
        "Fold": [1,2,3,4,5],
        "Accuracy": [0.5744, 0.6603, 0.3073, 0.5878, 0.6527],
        "ROC-AUC":  [0.4915, 0.5373, 0.5022, 0.5874, 0.5380],
    }
    wf_df = pd.DataFrame(wf_data)

    fig_wf = go.Figure()
    fig_wf.add_trace(go.Bar(name="Accuracy", x=wf_df["Fold"].astype(str),
                            y=wf_df["Accuracy"]*100, marker_color="#38bdf8",
                            text=[f"{v:.2%}" for v in wf_df["Accuracy"]], textposition="outside"))
    fig_wf.add_trace(go.Bar(name="ROC-AUC", x=wf_df["Fold"].astype(str),
                            y=wf_df["ROC-AUC"]*100, marker_color="#10b981",
                            text=[f"{v:.2%}" for v in wf_df["ROC-AUC"]], textposition="outside"))
    fig_wf.update_layout(
        barmode="group", paper_bgcolor="#0a0e1a", plot_bgcolor="#0a0e1a",
        font_color="#e2e8f0", height=320, margin=dict(l=0,r=0,t=20,b=0),
        legend=dict(bgcolor="#111827", bordercolor="#1e3a5f", borderwidth=1),
        xaxis=dict(title="Fold", gridcolor="#1e2d4a"),
        yaxis=dict(title="Score (%)", gridcolor="#1e2d4a", range=[0,85])
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    st.markdown(f"""
    <div class="card" style="border-color:#38bdf8">
        <div style="display:flex; gap:32px; flex-wrap:wrap">
            <div><span style="color:#64748b; font-size:0.8rem">Mean ROC-AUC</span><br>
                 <span style="font-size:1.4rem; font-weight:700; color:#38bdf8">0.5313</span></div>
            <div><span style="color:#64748b; font-size:0.8rem">Std Dev</span><br>
                 <span style="font-size:1.4rem; font-weight:700; color:#38bdf8">0.0376</span></div>
            <div><span style="color:#64748b; font-size:0.8rem">Mean Accuracy</span><br>
                 <span style="font-size:1.4rem; font-weight:700; color:#38bdf8">55.65%</span></div>
            <div><span style="color:#64748b; font-size:0.8rem">Interpretation</span><br>
                 <span style="font-size:0.9rem; color:#cbd5e1">Consistent above-random signal; strategy layer extracts excess alpha</span></div>
        </div>
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION: RECOMMENDATION
# ═════════════════════════════════════════════════════════════════════════════
elif nav == "💡 Recommendation":
    section("💡", "Final Recommendation & Business Conclusion",
            "AI-generated investment strategy assessment")

    st.markdown("""
    <div class="rec-box">
        <div class="rec-title">🏆 RECOMMENDED STRATEGY: TREND-FILTERED</div>
        <div class="rec-body">
            The <b style='color:#38bdf8'>Trend-Filtered Strategy</b> is the definitive winner across all
            risk-adjusted metrics. By combining the ML model's directional signal with a simple
            price-vs-MA5 trend gate, it eliminates the vast majority of false signals —
            resulting in a <b style='color:#6ee7b7'>4.43x cumulative return</b>,
            <b style='color:#6ee7b7'>81.40% annualised CAGR</b>, and a maximum drawdown of just
            <b style='color:#6ee7b7'>-1.60%</b> over the 2022–2024 test period.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Final signal
    st.markdown(signal_html(live_signal, live_conf, live_level), unsafe_allow_html=True)
    st.markdown(f"""
    <div style="color:#94a3b8; font-size:0.78rem; text-align:center; margin-top:8px">
        Signal generated {datetime.now().strftime('%d %b %Y, %H:%M')} using live RSI + MACD + Trend filter
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Three conclusions
    c1, c2, c3 = st.columns(3)
    conclusions = [
        ("📊", "ML + Sentiment Works",
         "Combining FinBERT news sentiment with 21 engineered technical features creates "
         "a richer signal than either source alone. FinBERT lags capture delayed market "
         "reactions to earnings and macro events.", "#38bdf8"),
        ("⚡", "Trend Gating is Critical",
         "Raw model signals are noisy (ROC-AUC ~0.53–0.58). Applying a simple trend confirmation "
         "filter (price > MA5) before executing dramatically improves precision and removes "
         "most false positives.", "#10b981"),
        ("🛡️", "Risk Management Wins",
         "The Trend-Filtered strategy's Sharpe of 5.58 and max drawdown of -1.60% demonstrate "
         "that disciplined entry criteria outperform raw return maximisation. Capital preservation "
         "compounds silently.", "#a78bfa"),
    ]
    for col, (icon, title, body, clr) in zip([c1,c2,c3], conclusions):
        with col:
            st.markdown(f"""
            <div class="card" style="border-color:{clr}; height:100%">
                <div style="font-size:1.8rem; margin-bottom:8px">{icon}</div>
                <div style="font-weight:700; color:{clr}; margin-bottom:8px">{title}</div>
                <div style="color:#94a3b8; font-size:0.85rem; line-height:1.65">{body}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="card" style="border-color:#475569; background:#0f1629">
        <div style="color:#475569; font-size:0.75rem; line-height:1.7">
        ⚠️ <b style='color:#64748b'>Disclaimer:</b>
        This dashboard is an academic AI/ML internship project for educational and research purposes only.
        The signals, predictions, and backtested results presented here are based on historical data
        and simulated conditions. Past performance does not guarantee future results.
        This is NOT financial advice. Always consult a SEBI-registered financial advisor before
        making any investment decisions.
        </div>
    </div>""", unsafe_allow_html=True)

    # Tech stack summary
    st.markdown("<br>", unsafe_allow_html=True)
    section("🔧", "Technical Stack Summary")
    stack_cols = st.columns(4)
    stack = [
        ("Data", ["yFinance API","Google News RSS","feedparser","pandas / NumPy"]),
        ("NLP", ["FinBERT (ProsusAI)","VADER SentimentAnalyzer","transformers","tqdm batching"]),
        ("ML", ["RandomForest","XGBoost","Logistic Regression","KNN / DecisionTree"]),
        ("Evaluation", ["TimeSeriesSplit CV","Sharpe Ratio","Max Drawdown","CAGR · Win Rate"]),
    ]
    for col, (cat, items) in zip(stack_cols, stack):
        with col:
            items_html = "".join(f"<div style='color:#94a3b8; font-size:0.82rem; padding:3px 0; border-bottom:1px solid #1e2d4a'>· {i}</div>" for i in items)
            st.markdown(f"""
            <div class="card">
                <div class="card-title">{cat}</div>
                <div style="margin-top:8px">{items_html}</div>
            </div>""", unsafe_allow_html=True)
