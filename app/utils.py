import pathlib

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from ta import add_all_ta_features

APP_DIR = pathlib.Path(__file__).resolve().parent
NOTEBOOKS_DIR = APP_DIR.parent / "notebooks"

TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

# Original historical data pull range used to build the training CSVs (see 01_data.ipynb).
# Used only as a best-effort way to recover real trading dates for chart x-axes, since the
# saved feature/raw CSVs were written without a Date column.
TRAINING_DATA_START = "2019-01-01"
TRAINING_DATA_END = "2024-01-01"

TA_FEATURE_COLUMNS = [
    "Close",
    "trend_sma_fast", "trend_sma_slow", "trend_ema_fast",
    "trend_macd", "trend_macd_signal", "trend_macd_diff",
    "momentum_rsi",
    "volatility_bbh", "volatility_bbl", "volatility_bbw",
    "volatility_atr",
    "volume_obv",
]


def _ticker_to_fname(ticker: str) -> str:
    return ticker.replace(".", "_")


@st.cache_resource
def load_model():
    model = joblib.load(NOTEBOOKS_DIR / "logistic_regression_model.pkl")
    feature_cols = joblib.load(NOTEBOOKS_DIR / "feature_list.pkl")
    return model, feature_cols


@st.cache_data
def load_features_csv(ticker: str) -> pd.DataFrame:
    fname = _ticker_to_fname(ticker)
    return pd.read_csv(NOTEBOOKS_DIR / f"{fname}_features.csv")


@st.cache_data(ttl=3600)
def fetch_price_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


@st.cache_data(ttl=3600)
def fetch_price_range(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def build_live_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the exact feature set the model was trained on from fresh OHLCV data."""
    data = raw_df.copy()
    data = add_all_ta_features(
        data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
    )
    data = data[TA_FEATURE_COLUMNS]
    data["return_1d"] = data["Close"].pct_change(1)
    data["return_5d"] = data["Close"].pct_change(5)
    data["return_10d"] = data["Close"].pct_change(10)
    data.dropna(inplace=True)
    return data


def get_last_fold_split(X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """Same held-out split used in 04_model_backtest.ipynb: last fold of a 5-way TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    train_idx, test_idx = list(tscv.split(X))[-1]
    return train_idx, test_idx


def run_backtest(df: pd.DataFrame, test_idx, predictions, risk_free: float = 0.06):
    """Reproduces the backtest() logic from 04_model_backtest.ipynb exactly."""
    test = df.iloc[test_idx].copy()
    test["Pred"] = predictions
    test["Return_1d"] = test["Close"].pct_change()
    test["Strategy_Return"] = test["Return_1d"] * test["Pred"].shift(1)
    test["Cum_Strategy"] = (1 + test["Strategy_Return"]).cumprod()
    test["Cum_Market"] = (1 + test["Return_1d"]).cumprod()

    def sharpe(r):
        return np.sqrt(252) * (r.mean() - risk_free / 252) / r.std()

    def max_dd(cum):
        return ((cum - cum.cummax()) / cum.cummax()).min()

    metrics = {
        "Strategy Sharpe": sharpe(test["Strategy_Return"].dropna()),
        "Market Sharpe": sharpe(test["Return_1d"].dropna()),
        "Strategy Drawdown": max_dd(test["Cum_Strategy"]),
        "Market Drawdown": max_dd(test["Cum_Market"]),
        "Strategy Return": test["Cum_Strategy"].iloc[-1] - 1,
        "Market Return": test["Cum_Market"].iloc[-1] - 1,
    }
    return test, metrics


@st.cache_data(ttl=3600)
def align_dates_for_ticker(ticker: str, n_rows: int, offset: int = 10):
    """
    Best-effort recovery of real trading dates for each row of the features CSV.

    The saved CSVs have no Date column, but they were built from a known yfinance pull
    (2019-01-01 to 2024-01-01) with the first `offset` rows dropped by indicator warmup.
    Re-downloading that same range and aligning by position recovers real dates. Returns
    None if the re-download doesn't yield enough rows (e.g. no network), so callers should
    fall back to a positional x-axis.
    """
    try:
        raw = fetch_price_range(ticker, TRAINING_DATA_START, TRAINING_DATA_END)
        if raw is None or raw.empty or len(raw) < offset + n_rows:
            return None
        return raw.index[offset: offset + n_rows]
    except Exception:
        return None


def sidebar_controls():
    st.sidebar.header("Controls")
    ticker = st.sidebar.selectbox("Ticker", TICKERS, key="ticker")
    default_start = (pd.Timestamp.today() - pd.Timedelta(days=180)).date()
    default_end = pd.Timestamp.today().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(default_start, default_end),
        key="date_range",
    )
    st.sidebar.caption("Shared across all pages")
    return ticker, date_range
