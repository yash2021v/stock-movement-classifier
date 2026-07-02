import streamlit as st

from utils import (
    TEST_START_DATE,
    TICKERS,
    TRAIN_END_DATE,
    get_model_display_name,
    inject_base_style,
    load_test_predictions,
)

inject_base_style()

model_name = get_model_display_name()
test_df = load_test_predictions()
data_end = test_df.index.max().date().isoformat() if not test_df.empty else "present"

st.title("Cross-Sectional Alpha Ranking Model")
st.caption(
    "Ranking 20 NSE large-cap stocks against each other to identify relative "
    "outperformers over a 5-trading-day horizon."
)

st.divider()

cards = st.columns(4)
card_content = [
    ("Universe", f"{len(TICKERS)} NSE large-cap stocks"),
    ("Data Period", f"2019 – {data_end} daily OHLCV"),
    ("Target", "Top-half vs. bottom-half relative performer, next 5 trading days"),
    ("Model", f"{model_name} classifier, pooled across all 20 stocks"),
]
for col, (title, body) in zip(cards, card_content):
    col.markdown(
        f"""<div class="info-card"><h4>{title}</h4><p>{body}</p></div>""",
        unsafe_allow_html=True,
    )

st.write("")
st.markdown(
    f"""<div class="callout-note">
    Training data runs through {TRAIN_END_DATE}; the held-out test period covers
    {TEST_START_DATE} through {data_end} — a full calendar year-plus of genuinely
    unseen market conditions, not a percentage-based split of the same window used for
    training. See the Model Performance and Backtest Results pages for what this
    stronger test actually shows.
    </div>""",
    unsafe_allow_html=True,
)

st.write("")
st.write("")

st.subheader("What this model does")
st.markdown(
    """
    Rather than predicting whether a single stock will go up or down in absolute terms,
    this model ranks all 20 stocks in the universe against each other on each trading day
    and predicts which half will outperform the other over the following 5 trading days.
    Framing the problem this way cancels out market-wide moves — a day where every stock
    rises or falls together contributes no information — and isolates the stock-specific
    component of returns, which is a cleaner target for a classifier to learn from.
    """
)

st.subheader("Honest summary of the finding")
st.markdown(
    """
    Cross-validated across five chronological folds on the full 2019–2026 dataset, the
    best-performing model achieves a classification accuracy only marginally above a
    majority-class baseline (roughly a 0.6 percentage point edge), and that edge is
    smaller than it appeared on an earlier, shorter version of this dataset — a sign
    the original edge was partly noise rather than a strong, stable signal. On a fixed,
    genuinely out-of-sample holdout (2025 onward, never seen during training or
    cross-validation), the model's predictions were also noticeably lopsided —
    over-predicting "outperform" relative to the balanced 50/50 base rate — a behavioral
    issue that cross-validation alone did not surface. When this model was used to drive
    a simple long-the-predicted-outperformers portfolio over that same holdout, both the
    strategy and a plain buy-and-hold benchmark lost value, and the strategy did not show
    a clear risk-adjusted advantage over the market. See the Model Performance and
    Backtest Results pages for the full numbers.
    """
)

st.markdown(
    """<div class="callout-note">
    An earlier phase of this project tested absolute (single-stock) direction prediction,
    which found no signal. See the project README for full methodology.
    </div>""",
    unsafe_allow_html=True,
)
