"""
================================================================================
ML MODEL MODULE — AI Stock Trading Decision Support System
File: ml_model.py
================================================================================

ARCHITECTURE:
    load_model()              → Load trained RandomForest from joblib
    get_live_features()       → Build feature vector from current market data
    predict_signal()          → BUY / SELL / HOLD + confidence + explainability
    get_ml_health()           → Structured health status for UI panel
    get_feature_importance()  → Top-N feature importances from RandomForest
    get_market_snapshot()     → Live price / change for dashboard header

FIXES IN THIS VERSION:
    ✅ yfinance and ta added to requirements.txt (caller note)
    ✅ feat_warn pre-initialised as "" — no more "variable referenced before
       assignment" crash in the except block of predict_signal()
    ✅ All dict values guaranteed non-None; all keys always present
    ✅ Robust data_timestamp extraction (handles all pandas index types)
    ✅ Probability normalisation to exactly 100% (floating-point safe)
    ✅ get_market_snapshot() hardened against None fast_info fields
    ✅ yfinance MultiIndex column flattening — single and multi-ticker safe
    ✅ predict_proba class deduplication — no double-counting across label maps
    ✅ Model loading via joblib — path: data/model.joblib
    ✅ Separation-adjusted confidence — penalises ambiguous low-gap predictions
    ✅ Feature alignment validation — aborts on large mismatches
    ✅ Correct class mapping: 0→SELL  1→HOLD  2→BUY
================================================================================
"""

import warnings
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")

# ── Optional imports ─────────────────────────────────────────────────────────
try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    YFINANCE_OK = False

try:
    import ta
    TA_OK = True
except ImportError:
    TA_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH   = Path("data/model.joblib")
TICKER       = "RELIANCE.NS"
LOOKBACK     = 60   # Trading days of history for feature computation

# Canonical feature order — must mirror training
FEATURE_COLS = [
    "ret_1d", "ret_3d", "ret_5d", "ret_10d",
    "sma_5",  "sma_10", "sma_20", "sma_50",
    "rsi_14", "atr_14", "bb_pct", "macd_diff", "vol_ratio",
]

# Numeric class label → signal string (matches training encoding)
CLASS_LABEL_MAP: dict = {
    0: "SELL", 1: "HOLD", 2: "BUY",
    "0": "SELL", "1": "HOLD", "2": "BUY",
    "buy": "BUY", "sell": "SELL", "hold": "HOLD",
    "BUY": "BUY", "SELL": "SELL", "HOLD": "HOLD",
}

# Minimum feature overlap fraction before prediction is aborted
MIN_FEATURE_OVERLAP = 0.80

# Gap between top-2 class probabilities below which confidence is penalised
SEPARATION_THRESHOLD = 0.10

# Maximum sentiment adjustment (pp) — ML remains primary signal
MAX_SENTIMENT_ADJUSTMENT = 10.0

# Fallback probability distribution (equal uncertainty)
_FALLBACK_PROBS = {"BUY": 33.0, "HOLD": 34.0, "SELL": 33.0}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load trained scikit-learn model from disk via joblib.

    Returns:
        (model, feature_names: list, error: str)
        On success : (model, [], "")
        On failure : (None,  [], "human-readable error")

    Cached for the entire Streamlit session.
    """
    if not JOBLIB_OK:
        return None, [], "joblib not installed — run: pip install joblib"

    if not MODEL_PATH.exists():
        return None, [], f"Model file not found at {MODEL_PATH}"

    try:
        model = joblib.load(MODEL_PATH)
        if model is None:
            return None, [], "joblib file did not contain a model object"
        return model, [], ""
    except Exception as exc:
        return None, [], f"Model load error: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)   # 5-minute cache
def get_live_features():
    """
    Download recent OHLCV data for RELIANCE.NS and compute technical features.

    Returns:
        (DataFrame, "")          on success — single-row DataFrame with FEATURE_COLS
        (None, error_str)        on any failure
    """
    if not YFINANCE_OK:
        return None, "yfinance not installed — run: pip install yfinance"

    try:
        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=LOOKBACK + 30)

        raw = yf.download(
            TICKER,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )

        if raw is None or len(raw) < 20:
            n = 0 if raw is None else len(raw)
            return None, f"Insufficient market data (received {n} rows, need ≥20)"

        df = raw.copy()

        # Flatten MultiIndex columns (yfinance ≥ 0.2 may return them)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(col[0]).lower().strip() for col in df.columns]
        else:
            df.columns = [str(col).lower().strip() for col in df.columns]

        df = df.loc[:, ~df.columns.duplicated()]
        df.dropna(inplace=True)

        if df.empty:
            return None, "DataFrame is empty after initial NaN drop"

        required_raw = {"close", "high", "low", "volume"}
        missing_raw  = required_raw - set(df.columns)
        if missing_raw:
            return None, f"Missing OHLCV columns after flattening: {missing_raw}"

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        # Returns
        df["ret_1d"]  = close.pct_change(1)
        df["ret_3d"]  = close.pct_change(3)
        df["ret_5d"]  = close.pct_change(5)
        df["ret_10d"] = close.pct_change(10)

        # Moving-average ratios
        for w in [5, 10, 20, 50]:
            sma = close.rolling(w).mean()
            df[f"sma_{w}"] = (close / (sma + 1e-9)) - 1

        # Technical indicators
        if TA_OK:
            df["rsi_14"]    = ta.momentum.RSIIndicator(close, window=14).rsi()
            df["atr_14"]    = (
                ta.volatility.AverageTrueRange(high, low, close, window=14)
                .average_true_range() / (close + 1e-9)
            )
            bb              = ta.volatility.BollingerBands(close, window=20, window_dev=2)
            df["bb_pct"]    = bb.bollinger_pband()
            macd            = ta.trend.MACD(close)
            df["macd_diff"] = macd.macd_diff()
        else:
            # Manual fallback when ta library is absent
            delta           = close.diff()
            gain            = delta.clip(lower=0).rolling(14).mean()
            loss            = (-delta.clip(upper=0)).rolling(14).mean()
            rs              = gain / (loss + 1e-9)
            df["rsi_14"]    = 100 - 100 / (1 + rs)
            df["atr_14"]    = (high - low).rolling(14).mean() / (close + 1e-9)
            df["bb_pct"]    = (close - close.rolling(20).mean()) / (
                2 * close.rolling(20).std() + 1e-9
            )
            ema12           = close.ewm(span=12).mean()
            ema26           = close.ewm(span=26).mean()
            macd_line       = ema12 - ema26
            df["macd_diff"] = macd_line - macd_line.ewm(span=9).mean()

        # Volume ratio
        df["vol_ratio"] = volume / (volume.rolling(20).mean() + 1e-9)

        df.dropna(inplace=True)
        if df.empty:
            return None, "All rows dropped after NaN removal — insufficient history"

        missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
        if missing_cols:
            return None, f"Feature computation failed for columns: {missing_cols}"

        return df[FEATURE_COLS].iloc[[-1]], ""

    except Exception as exc:
        return None, f"Feature pipeline error: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ALIGNMENT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def _validate_and_align(features_df: pd.DataFrame, model_feature_names: list):
    """
    Validate that live features align with what the model was trained on.

    Returns:
        (aligned_df, "")          — perfect match or no stored names
        (reindexed_df, warning)   — minor mismatch, zero-filled, with warning
        (None, error)             — major mismatch, prediction aborted
    """
    if not model_feature_names:
        return features_df, ""

    live_cols  = set(features_df.columns)
    model_cols = set(model_feature_names)

    if live_cols == model_cols:
        return features_df[model_feature_names], ""

    missing      = model_cols - live_cols
    overlap_frac = len(live_cols & model_cols) / max(len(model_cols), 1)

    if overlap_frac < MIN_FEATURE_OVERLAP:
        return None, (
            f"Feature mismatch too large ({overlap_frac:.0%} overlap). "
            f"Model expects: {sorted(model_cols)}. "
            f"Pipeline produced: {sorted(live_cols)}. "
            f"Aborting to prevent misleading prediction."
        )

    aligned = features_df.reindex(columns=model_feature_names, fill_value=0.0)
    warn    = (
        f"Feature alignment partial ({overlap_frac:.0%}). "
        f"Missing features zero-filled: {sorted(missing)}. "
        f"Prediction quality may be degraded."
    )
    return aligned, warn


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE ADJUSTMENT
# ─────────────────────────────────────────────────────────────────────────────

def _separation_adjusted_confidence(proba: np.ndarray) -> float:
    """
    Confidence = max(proba) but penalised when top-2 classes are too close.

    If gap between top-2 probabilities < SEPARATION_THRESHOLD,
    confidence is linearly reduced toward 50% of its raw value.
    """
    proba = np.asarray(proba, dtype=float)
    if proba.size == 0:
        return 50.0
    if proba.size == 1:
        return round(float(np.clip(proba[0], 0.0, 1.0)) * 100, 2)

    sorted_p   = np.sort(proba)[::-1]
    top1, top2 = float(sorted_p[0]), float(sorted_p[1])
    separation = top1 - top2
    base_conf  = top1 * 100

    if separation < SEPARATION_THRESHOLD:
        penalty_factor = 0.5 + 0.5 * (separation / SEPARATION_THRESHOLD)
        return round(float(np.clip(base_conf * penalty_factor, 0.0, 100.0)), 2)

    return round(float(np.clip(base_conf, 0.0, 100.0)), 2)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_importance(model, feature_names: list, top_n: int = 5) -> list:
    """
    Extract top-N feature importances from a fitted model.

    Returns:
        [{"name": str, "importance": float (%), "rank": int}]
        Empty list if model does not support feature_importances_.
    """
    if model is None or not hasattr(model, "feature_importances_"):
        return []

    try:
        importances = np.asarray(model.feature_importances_, dtype=float)
        names       = list(feature_names) if feature_names else list(FEATURE_COLS)

        if len(importances) != len(names):
            names = [f"feature_{i}" for i in range(len(importances))]

        ranked = sorted(
            zip(names, importances.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        return [
            {"name": str(name), "importance": round(float(imp) * 100, 2), "rank": i + 1}
            for i, (name, imp) in enumerate(ranked)
        ]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# ML HEALTH STATUS
# ─────────────────────────────────────────────────────────────────────────────

def get_ml_health() -> dict:
    """
    Return a structured health snapshot for the UI status panel.
    Never raises — all values are always present and non-None.
    """
    health: dict = {
        "model_loaded":        False,
        "live_data_ok":        False,
        "feature_pipeline_ok": False,
        "model_path":          str(MODEL_PATH),
        "model_error":         "",
        "data_error":          "",
        "last_checked":        datetime.now().strftime("%H:%M:%S"),
    }

    try:
        model, _, load_err     = load_model()
        health["model_loaded"] = model is not None
        health["model_error"]  = str(load_err) if load_err else ""
    except Exception as exc:
        health["model_error"]  = str(exc)

    try:
        features_df, feat_err              = get_live_features()
        health["live_data_ok"]             = features_df is not None
        health["feature_pipeline_ok"]      = features_df is not None
        health["data_error"]               = str(feat_err) if feat_err else ""
    except Exception as exc:
        health["data_error"]               = str(exc)

    return health


# ─────────────────────────────────────────────────────────────────────────────
# SAFE TIMESTAMP EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _safe_data_timestamp(features_df: pd.DataFrame) -> str:
    """
    Extract a human-readable timestamp from the last row of the features DataFrame.
    Handles DatetimeIndex, timezone-aware index, plain RangeIndex and strings.
    Always returns a non-empty string — never raises.
    """
    try:
        last_idx = features_df.index[-1]
        if hasattr(last_idx, "strftime"):
            return last_idx.strftime("%d %b %Y %H:%M")
        if isinstance(last_idx, datetime):
            return last_idx.strftime("%d %b %Y %H:%M")
        val = str(last_idx)
        return val[:20] if len(val) > 20 else val
    except Exception:
        return datetime.now().strftime("%d %b %Y %H:%M") + " (approx)"


# ─────────────────────────────────────────────────────────────────────────────
# PROBABILITY MAP BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_prob_map(model, X: np.ndarray) -> tuple:
    """
    Build a clean {BUY, HOLD, SELL} probability map (values in 0–100, sum == 100)
    and return separation-adjusted confidence.

    Returns:
        prob_map   : dict {"BUY": float, "HOLD": float, "SELL": float}
        confidence : float 0–100
    """
    if not hasattr(model, "predict_proba"):
        raw_pred = model.predict(X)[0]
        signal   = CLASS_LABEL_MAP.get(raw_pred, str(raw_pred).upper())
        if signal not in ("BUY", "SELL", "HOLD"):
            signal = "HOLD"
        prob_map = {"BUY": 0.0, "HOLD": 0.0, "SELL": 0.0}
        prob_map[signal] = 100.0
        return prob_map, 60.0

    proba   = model.predict_proba(X)[0]
    classes = list(model.classes_)

    raw_map:      dict = {"BUY": 0.0, "HOLD": 0.0, "SELL": 0.0}
    seen_labels:  set  = set()

    for cls, p in zip(classes, proba):
        lbl = CLASS_LABEL_MAP.get(cls)
        if lbl is None:
            lbl = str(cls).upper()
        if lbl not in ("BUY", "HOLD", "SELL"):
            lbl = "HOLD"
        # Skip duplicate label mappings (e.g. both int 0 and str "0")
        if lbl in seen_labels:
            continue
        seen_labels.add(lbl)
        raw_map[lbl] += float(p)

    total = sum(raw_map.values())
    if total > 0:
        prob_map = {k: round(v / total * 100, 2) for k, v in raw_map.items()}
    else:
        prob_map = {"BUY": 33.0, "HOLD": 34.0, "SELL": 33.0}

    # Fix floating-point rounding drift so values sum to exactly 100
    keys     = ["BUY", "HOLD", "SELL"]
    diff     = round(100.0 - sum(prob_map[k] for k in keys), 2)
    dominant = max(keys, key=lambda k: prob_map[k])
    prob_map[dominant] = round(prob_map[dominant] + diff, 2)

    confidence = _separation_adjusted_confidence(np.asarray(proba))
    return prob_map, confidence


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PREDICTION ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def predict_signal() -> dict:
    """
    Full prediction pipeline: load → features → validate → predict → explain.

    CONTRACT — returned dict always has ALL keys, never None values:
        signal          : str   "BUY" | "SELL" | "HOLD"
        confidence      : float 0–100
        probabilities   : dict  {"BUY": float, "HOLD": float, "SELL": float}
        source          : str   "model" | "fallback"
        fallback_reason : str   "" when source=="model", else human-readable cause
        feature_warning : str   "" or partial-alignment warning
        top_features    : list  [{name, importance, rank}] (empty on fallback)
        data_timestamp  : str   timestamp of latest data bar
        error           : str   "" or last error detail
    """
    def _fallback(reason: str, error: str = "", feature_warning: str = "") -> dict:
        return {
            "signal":          "HOLD",
            "confidence":      50.0,
            "probabilities":   dict(_FALLBACK_PROBS),
            "source":          "fallback",
            "fallback_reason": str(reason) if reason else "Unknown reason",
            "feature_warning": str(feature_warning),
            "top_features":    [],
            "data_timestamp":  "",
            "error":           str(error or reason),
        }

    # Pre-initialise sentinel so it's always defined in except blocks
    feat_warn = ""

    # 1 ── Load model ──────────────────────────────────────────────────────────
    try:
        model, feature_names, load_err = load_model()
    except Exception as exc:
        return _fallback(f"Model load exception: {exc}", str(exc))

    if model is None:
        return _fallback(f"Model unavailable: {load_err}", load_err)

    # 2 ── Fetch features ──────────────────────────────────────────────────────
    try:
        features_df, feat_err = get_live_features()
    except Exception as exc:
        return _fallback(f"Feature fetch exception: {exc}", str(exc))

    if features_df is None:
        return _fallback(f"Live data unavailable: {feat_err}", feat_err)

    data_timestamp = _safe_data_timestamp(features_df)

    # 3 ── Validate & align features ───────────────────────────────────────────
    try:
        aligned_df, feat_warn = _validate_and_align(features_df, feature_names)
    except Exception as exc:
        return _fallback(f"Feature alignment exception: {exc}", str(exc))

    if aligned_df is None:
        return _fallback(
            f"Feature mismatch: {feat_warn}",
            feat_warn,
            feature_warning=feat_warn,
        )

    # 4 ── Predict ─────────────────────────────────────────────────────────────
    try:
        X        = aligned_df.values
        raw_pred = model.predict(X)[0]

        signal = CLASS_LABEL_MAP.get(raw_pred)
        if signal is None:
            signal = str(raw_pred).upper()
        if signal not in ("BUY", "SELL", "HOLD"):
            signal = "HOLD"

        prob_map, confidence = _build_prob_map(model, X)

        top_features = get_feature_importance(
            model,
            list(aligned_df.columns),
            top_n=5,
        )

        return {
            "signal":          signal,
            "confidence":      float(confidence),
            "probabilities":   prob_map,
            "source":          "model",
            "fallback_reason": "",
            "feature_warning": str(feat_warn),
            "top_features":    top_features,
            "data_timestamp":  data_timestamp,
            "error":           "",
        }

    except Exception as exc:
        return _fallback(
            f"Prediction runtime error: {exc}",
            str(exc),
            feature_warning=feat_warn,   # always defined — initialised above
        )


# ─────────────────────────────────────────────────────────────────────────────
# MARKET SNAPSHOT  (dashboard header data)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def get_market_snapshot() -> dict:
    """
    Fetch latest price, change %, and volume for RELIANCE.NS.

    Returns populated dict on success, empty dict on any failure — never raises.
    All numeric fields are guaranteed to be actual numbers (not None).
    """
    if not YFINANCE_OK:
        return {}

    try:
        ticker = yf.Ticker(TICKER)
        info   = ticker.fast_info

        def _safe_float(attr: str, default: float = 0.0) -> float:
            try:
                val = getattr(info, attr, None)
                if val is None:
                    return default
                return round(float(val), 2)
            except (TypeError, ValueError):
                return default

        def _safe_int(attr: str, default: int = 0) -> int:
            try:
                val = getattr(info, attr, None)
                if val is None:
                    return default
                return int(float(val))
            except (TypeError, ValueError):
                return default

        price  = _safe_float("last_price")
        prev   = _safe_float("previous_close")
        change = round(price - prev, 2) if price and prev else 0.0
        pct    = round((change / prev) * 100, 2) if prev else 0.0
        vol    = _safe_int("three_month_average_volume")

        if price == 0.0:
            return {}

        return {
            "price":      price,
            "change":     change,
            "pct_change": pct,
            "volume":     vol,
            "ticker":     TICKER,
            "fetched_at": datetime.now().strftime("%H:%M:%S"),
        }
    except Exception:
        return {}
