import pathlib

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from ta import add_all_ta_features

APP_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"


def _resolve_model_path() -> pathlib.Path:
    """Finds the saved final-model .pkl by pattern instead of a hardcoded filename,
    since the selected model (and its filename, e.g. xgboost_model.pkl vs
    logistic_regression_model.pkl) can change whenever the notebook is rerun."""
    candidates = sorted(MODELS_DIR.glob("*_model.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No *_model.pkl found in {MODELS_DIR}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


MODEL_PATH = _resolve_model_path()
FEATURE_LIST_PATH = MODELS_DIR / "feature_list.pkl"
TEST_PREDICTIONS_PATH = DATA_DIR / "test_predictions.csv"


def get_model_display_name() -> str:
    """Human-readable model name derived from the saved model filename (e.g.
    'logistic_regression_model.pkl' -> 'Logistic Regression'), so the app doesn't need
    a hardcoded name that goes stale when a different model is selected on rerun."""
    stem = MODEL_PATH.stem
    if stem.endswith("_model"):
        stem = stem[: -len("_model")]
    return stem.replace("_", " ").title()

# Fixed calendar train/test split used in notebooks/04_model_backtest.ipynb — a genuine
# out-of-sample holdout rather than a percentage-based slice that shifts on every rerun.
TRAIN_END_DATE = "2024-12-31"
TEST_START_DATE = "2025-01-01"

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NESTLEIND.NS", "TMCV.NS",
]
# TMCV.NS was TATAMOTORS.NS in the training data (2019-2024). Tata Motors demerged in
# 2025: the commercial-vehicles business kept the original "Tata Motors Limited" listing
# under the new symbol TMCV.NS, while passenger vehicles/EV spun off as TMPV.NS. The old
# TATAMOTORS.NS symbol no longer resolves on Yahoo Finance, so live fetches use TMCV.NS
# as the closest continuing entity — historical training/backtest data is unaffected.

# Fixed technical-indicator subset kept from ta.add_all_ta_features output,
# matching notebooks/02_feature_engineering.ipynb::build_features exactly.
TA_COLUMNS_TO_KEEP = [
    "Close",
    "trend_sma_fast", "trend_sma_slow", "trend_ema_fast",
    "trend_macd", "trend_macd_signal", "trend_macd_diff",
    "momentum_rsi",
    "volatility_bbh", "volatility_bbl", "volatility_bbw",
    "volatility_atr",
    "volume_obv",
]

# Reported cross-validation results (see notebooks/04_model_backtest.ipynb, the 5-fold
# TimeSeriesSplit comparison over the full pooled dataset). Hardcoded because recomputing
# this live would mean refitting Dummy/LR/RF/XGBoost across 5 folds on ~37k rows on every
# page load — update these numbers whenever the notebook is rerun. Everything else
# derivable from saved artifacts (classification report, confusion matrix, backtest
# metrics) is computed dynamically below instead of hardcoded.
CV_RESULTS = {
    "Dummy": {"mean": 0.4999, "std": 0.0001},
    "Logistic Regression": {"mean": 0.5059, "std": 0.0094},
    "Random Forest": {"mean": 0.5022, "std": 0.0095},
    "XGBoost": {"mean": 0.5018, "std": 0.0052},
}

# From the probability calibration check in notebooks/04_model_backtest.ipynb — real,
# measured numbers, not estimates. was_recalibrated reflects whether calibration
# actually helped enough to be adopted as the final saved model (see that notebook's
# "Save artifacts" cell for the decision rule and reasoning).
CALIBRATION_INFO = {
    "was_recalibrated": False,
    "cv_accuracy": 0.5059,
    "pre_calibration": {"std": 0.031, "pct_above_70": 0.003, "pct_below_30": 0.001, "accuracy": 0.5119},
    "post_calibration": {"std": 0.044, "pct_above_70": 0.005, "pct_below_30": 0.009, "accuracy": 0.5059},
}

PLOTLY_TEMPLATE = "plotly_white"
COLOR_NAVY = "#2C3E50"
COLOR_SLATE = "#5D7A9C"
COLOR_MUTED_GREEN = "#5B8C6E"
COLOR_MUTED_RED = "#B5545A"
COLOR_GRAY = "#9AA5B1"

# Chart data-color palette — coordinated, distinct, professional (not the flat pale blue
# used before). Bar/line colors only; chart backgrounds stay light per LIGHT_CHART_LAYOUT.
COLOR_ACCENT_GRAY = "#94A3B8"     # de-emphasized baseline (e.g. Dummy classifier)
COLOR_ACCENT_BLUE = "#3B82F6"     # primary model / feature-importance accent
COLOR_ACCENT_AMBER = "#F59E0B"    # secondary model / Market series
COLOR_ACCENT_EMERALD = "#10B981"  # standout model / Strategy series
COLOR_ACCENT_PURPLE = "#8B5CF6"   # tertiary accent, available if needed

# Applied to every Plotly figure's update_layout(**LIGHT_CHART_LAYOUT, ...). Explicit
# colors, not just a light template, because st.plotly_chart's default theme="streamlit"
# re-themes figures on top of whatever template is set — see plotly_chart_kwargs below.
LIGHT_CHART_LAYOUT = dict(
    template=PLOTLY_TEMPLATE,
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(color="#1F2937"),
    xaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB", zerolinecolor="#E5E7EB"),
    yaxis=dict(gridcolor="#E5E7EB", linecolor="#E5E7EB", zerolinecolor="#E5E7EB"),
)

# Pass as **plotly_chart_kwargs to every st.plotly_chart() call. theme=None stops
# Streamlit from re-applying its own (possibly stale/dark) theme on top of the
# figure's own light template and explicit colors set via LIGHT_CHART_LAYOUT.
PLOTLY_CHART_KWARGS = dict(use_container_width=True, theme=None)


def force_light_theme():
    """Streamlit persists the user's last-picked theme (Light/Dark) in the browser's
    localStorage per page path, and canvas/iframe-rendered widgets — st.dataframe's
    data grid, Plotly charts — read that stored theme once at mount to pick their
    colors. A stale "Dark" entry there makes those widgets render dark even though
    CSS forces every plain DOM element light (CSS can't repaint a canvas). This
    corrects any stale entries and reloads once, so those widgets mount light from
    the start. Guarded by a sessionStorage flag so it only runs once per tab and
    won't fight a theme choice the user makes later in the same session."""
    components.html(
        """
        <script>
        (function() {
            try {
                const win = window.parent;
                if (win.sessionStorage.getItem('themeFixAttempted')) return;
                win.sessionStorage.setItem('themeFixAttempted', '1');

                const desired = {
                    name: "Custom Theme",
                    themeInput: {
                        primaryColor: "#2C3E50",
                        backgroundColor: "#FAFAFA",
                        secondaryBackgroundColor: "#F0F2F5",
                        textColor: "#1F2937",
                        font: 0
                    }
                };
                function isDesired(raw) {
                    try {
                        const obj = JSON.parse(raw);
                        return !!(obj && obj.name === "Custom Theme" &&
                            obj.themeInput && obj.themeInput.backgroundColor === "#FAFAFA");
                    } catch (e) { return false; }
                }

                const ls = win.localStorage;
                const keys = [];
                for (let i = 0; i < ls.length; i++) {
                    const k = ls.key(i);
                    if (k && k.startsWith('stActiveTheme-')) keys.push(k);
                }
                if (keys.length === 0) keys.push('stActiveTheme-/-v1');

                let changed = false;
                keys.forEach(function(k) {
                    if (!isDesired(ls.getItem(k))) {
                        ls.setItem(k, JSON.stringify(desired));
                        changed = true;
                    }
                });

                if (changed) {
                    win.location.reload();
                }
            } catch (e) {}
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def inject_base_style():
    st.markdown(
        """
        <style>
        /* Force the light theme regardless of any "Dark" choice the user's browser has
        saved from a previous visit (Streamlit persists that choice in localStorage per
        page path and it takes priority over config.toml on return visits). */
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stHeader"],
        [data-testid="stBottomBlockContainer"],
        .main {
            background-color: #FAFAFA !important;
        }
        [data-testid="stSidebar"] {
            background-color: #F0F2F5 !important;
        }
        [data-testid="stSidebar"] *,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] *,
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] *,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"],
        label, h1, h2, h3, h4, h5, h6, p, li, span {
            color: #1F2937 !important;
        }
        button[kind="primary"], button[kind="primary"] * {
            background-color: #2C3E50 !important;
            color: #FFFFFF !important;
            border-color: #2C3E50 !important;
        }
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            background-color: #FFFFFF !important;
        }
        .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1200px; }
        h1, h2, h3 { font-weight: 600; }
        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            padding: 1rem 1.2rem;
        }
        [data-testid="stMetricLabel"] { color: #6B7280 !important; }
        .info-card {
            background-color: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-left: 3px solid #2C3E50;
            border-radius: 8px;
            padding: 1.1rem 1.3rem;
            height: 100%;
        }
        .info-card h4 { margin: 0 0 0.4rem 0; color: #2C3E50; font-size: 0.95rem; }
        .info-card p { margin: 0; color: #4B5563; font-size: 0.92rem; }
        .callout-note {
            background-color: #F3F4F6;
            border-left: 3px solid #9AA5B1;
            border-radius: 6px;
            padding: 0.85rem 1.1rem;
            font-size: 0.9rem;
            color: #4B5563;
        }
        .disclaimer-box {
            background-color: #FBF3E7;
            border: 1px solid #E8D9BE;
            border-radius: 8px;
            padding: 0.9rem 1.2rem;
            font-size: 0.92rem;
            color: #6B4E1F;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_LIST_PATH)
    return model, feature_cols


def get_feature_importances(model):
    """Extracts feature importances regardless of whether the saved model is a bare
    tree-based classifier, a linear Pipeline, or a CalibratedClassifierCV wrapping
    either — averaging across folds in the calibrated case."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "named_steps"):
        return abs(model.named_steps["clf"].coef_[0])
    if hasattr(model, "calibrated_classifiers_"):
        fold_importances = []
        for calibrated_clf in model.calibrated_classifiers_:
            estimator = calibrated_clf.estimator
            if hasattr(estimator, "feature_importances_"):
                fold_importances.append(estimator.feature_importances_)
            elif hasattr(estimator, "named_steps"):
                fold_importances.append(abs(estimator.named_steps["clf"].coef_[0]))
        if fold_importances:
            return sum(fold_importances) / len(fold_importances)
    return None


@st.cache_data
def load_test_predictions() -> pd.DataFrame:
    if not TEST_PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(TEST_PREDICTIONS_PATH, index_col="Date", parse_dates=True)


def compute_backtest(df: pd.DataFrame, horizon: int = 5, risk_free: float = 0.06):
    """Reconstructs cumulative return curves AND summary metrics directly from the
    saved test-prediction rows, mirroring backtest_cross_sectional() in
    notebooks/04_model_backtest.ipynb exactly (non-overlapping, horizon-day sampling).
    Returns (cum_strategy, cum_market, metrics_dict) — computed fresh from the CSV each
    time rather than hardcoded, so rerunning the notebook flows through automatically."""
    test = df.copy()
    test["Fwd_Return"] = test.groupby("Ticker")["Close"].pct_change(horizon).shift(-horizon)

    daily_strategy = test.groupby(test.index).apply(
        lambda d: d.loc[d["Pred"] == 1, "Fwd_Return"].mean()
    ).dropna()
    daily_market = test.groupby(test.index)["Fwd_Return"].mean().dropna()

    daily_strategy = daily_strategy.iloc[::horizon]
    daily_market = daily_market.iloc[::horizon]

    cum_strategy = (1 + daily_strategy).cumprod()
    cum_market = (1 + daily_market).cumprod()

    def sharpe(r):
        return np.sqrt(252 / horizon) * (r.mean() - risk_free * horizon / 252) / r.std()

    def max_dd(cum):
        return ((cum - cum.cummax()) / cum.cummax()).min()

    metrics = {
        "Strategy Sharpe": sharpe(daily_strategy),
        "Market Sharpe": sharpe(daily_market),
        "Strategy Drawdown": max_dd(cum_strategy),
        "Market Drawdown": max_dd(cum_market),
        "Strategy Return": cum_strategy.iloc[-1] - 1,
        "Market Return": cum_market.iloc[-1] - 1,
    }
    return cum_strategy, cum_market, metrics


def get_backtest_period(df: pd.DataFrame) -> tuple:
    return (df.index.min().date().isoformat(), df.index.max().date().isoformat())


def compute_classification_report(df: pd.DataFrame) -> dict:
    """Precision/recall/f1/support per class + accuracy, computed directly from the
    Target (actual) vs Pred (predicted) columns in the saved test-prediction rows,
    instead of a hardcoded snapshot — so rerunning the notebook updates this
    automatically. Returns sklearn's output_dict format."""
    return sk_classification_report(
        df["Target"], df["Pred"],
        target_names=["Underperform (0)", "Outperform (1)"],
        output_dict=True,
        zero_division=0,
    )


def compute_confusion_matrix(df: pd.DataFrame):
    """Exact confusion matrix (rows=actual, cols=predicted, order=[Underperform,
    Outperform]) computed directly from the saved test-prediction rows."""
    return sk_confusion_matrix(df["Target"], df["Pred"], labels=[0, 1])


def build_live_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Rebuilds the exact feature set the model was trained on from fresh OHLCV data."""
    data = raw_df.copy()
    data = add_all_ta_features(
        data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
    )
    data = data[TA_COLUMNS_TO_KEEP]
    data["return_1d"] = data["Close"].pct_change(1)
    data["return_5d"] = data["Close"].pct_change(5)
    data["return_10d"] = data["Close"].pct_change(10)
    data.dropna(inplace=True)
    return data


def render_ranking_table_html(df: pd.DataFrame) -> str:
    """Renders the live-ranking table as a hand-built HTML string instead of
    st.dataframe(). st.dataframe draws through glide-data-grid (a canvas widget) which
    reads Streamlit's theme once at mount — a stale/dark theme there ignores per-cell
    Styler colors for the header and index columns even after force_light_theme() fixes
    the rest of the app. Passing a pandas Styler to st.dataframe doesn't avoid this
    either: its default HTML template still carries a "@media (prefers-color-scheme:
    dark)" block for header cells that a user's OS-level dark mode can trigger
    independent of Streamlit's own theme. Building plain HTML with explicit inline
    colors sidesteps both failure modes entirely."""
    header_cells = "".join(
        f'<th style="padding:0.6rem 0.9rem; text-align:{align}; color:#1F2937; '
        f'background-color:#F0F2F5; border-bottom:2px solid #E5E7EB; font-weight:600;">{label}</th>'
        for label, align in [
            ("#", "left"),
            ("Ticker", "left"),
            ("Predicted Outperformance Probability", "right"),
            ("Predicted Class", "left"),
        ]
    )

    row_html = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        bg = "#EAF3EC" if row["Predicted Class"] == "Outperform" else "#FBEEEE"
        row_html.append(
            f'<tr style="background-color:{bg};">'
            f'<td style="padding:0.55rem 0.9rem; color:#1F2937; border-bottom:1px solid #E5E7EB;">{i}</td>'
            f'<td style="padding:0.55rem 0.9rem; color:#1F2937; border-bottom:1px solid #E5E7EB;">{row["Ticker"]}</td>'
            f'<td style="padding:0.55rem 0.9rem; color:#1F2937; text-align:right; '
            f'border-bottom:1px solid #E5E7EB;">{row["Predicted Outperformance Probability"]:.1%}</td>'
            f'<td style="padding:0.55rem 0.9rem; color:#1F2937; border-bottom:1px solid #E5E7EB;">{row["Predicted Class"]}</td>'
            f'</tr>'
        )

    return (
        '<div style="overflow-x:auto; border:1px solid #E5E7EB; border-radius:8px;">'
        '<table style="width:100%; border-collapse:collapse; background-color:#FFFFFF; '
        f'font-size:0.92rem;"><thead><tr>{header_cells}</tr></thead>'
        f'<tbody>{"".join(row_html)}</tbody></table></div>'
    )


@st.cache_data(ttl=3600)
def fetch_live_ranking_inputs(tickers: tuple, period: str = "6mo"):
    """Fetches recent OHLCV for each ticker and rebuilds model features.
    Returns (latest_features_by_ticker: dict[str, pd.DataFrame row], errors: dict[str, str])."""
    latest_rows = {}
    errors = {}
    for ticker in tickers:
        try:
            raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                errors[ticker] = "No data returned."
                continue
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
            feat = build_live_features(raw)
            if feat.empty:
                errors[ticker] = "Not enough history to compute indicators."
                continue
            latest_rows[ticker] = feat.iloc[[-1]]
        except Exception as e:
            errors[ticker] = str(e)
    return latest_rows, errors
