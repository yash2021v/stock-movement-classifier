import plotly.graph_objects as go
import streamlit as st

from utils import (
    COLOR_ACCENT_AMBER,
    COLOR_ACCENT_EMERALD,
    LIGHT_CHART_LAYOUT,
    PLOTLY_CHART_KWARGS,
    compute_backtest,
    get_backtest_period,
    inject_base_style,
    load_test_predictions,
)

inject_base_style()

st.title("Backtest Results")

test_df = load_test_predictions()

if test_df.empty:
    st.warning(
        "No saved backtest time series (data/test_predictions.csv) was found in this "
        "project, so no backtest can be shown."
    )
else:
    period_start, period_end = get_backtest_period(test_df)
    st.caption(
        f"Cross-sectional, equal-weighted portfolio backtest. Test period: "
        f"{period_start} to {period_end} — a fixed calendar holdout the model has "
        f"never seen in any form (see Model Performance page)."
    )

    st.divider()

    st.subheader("Cumulative Return: Strategy vs. Market")

    cum_strategy, cum_market, metrics = compute_backtest(test_df)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cum_strategy.index, y=cum_strategy.values,
            name="Strategy (predicted outperformers)",
            mode="lines", line=dict(color=COLOR_ACCENT_EMERALD, width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cum_market.index, y=cum_market.values,
            name="Market (all 20 stocks, equal-weighted)",
            mode="lines", line=dict(color=COLOR_ACCENT_AMBER, width=2, dash="dash"),
        )
    )
    fig.update_layout(
        **LIGHT_CHART_LAYOUT,
        height=450,
        xaxis_title="Date",
        yaxis_title="Portfolio Value (start = 1)",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, **PLOTLY_CHART_KWARGS)

    st.divider()

    st.subheader("Summary Metrics")

    row1 = st.columns(3)
    row1[0].metric("Strategy Sharpe", f"{metrics['Strategy Sharpe']:.3f}")
    row1[1].metric("Strategy Return", f"{metrics['Strategy Return']:.2%}")
    row1[2].metric("Strategy Drawdown", f"{metrics['Strategy Drawdown']:.2%}")

    row2 = st.columns(3)
    row2[0].metric("Market Sharpe", f"{metrics['Market Sharpe']:.3f}")
    row2[1].metric("Market Return", f"{metrics['Market Return']:.2%}")
    row2[2].metric("Market Drawdown", f"{metrics['Market Drawdown']:.2%}")

    st.write("")
    st.markdown(
        """
**Sharpe Ratio** — a risk-adjusted return measure: return earned per unit of
volatility/risk taken. Higher is generally better; values above ~1 are considered good,
above 2 is unusually strong.

**Cumulative Return** — total percentage growth of ₹1 invested over the test period.

**Maximum Drawdown** — the largest peak-to-trough decline experienced during the test
period, a measure of the worst-case loss an investor would have experienced if invested
at the worst possible time.

**Strategy** vs. **Market** — Strategy is the portfolio of stocks the model predicted as
top-half outperformers; Market is the equal-weighted average of all 20 stocks in the
universe (a buy-and-hold baseline).
"""
    )

    st.divider()

    st.markdown(
        """<div class="callout-note">
        This backtest samples forward returns on a non-overlapping, 5-day basis rather
        than daily overlapping windows — an earlier version of this project used
        overlapping windows, which inflated apparent performance roughly 8-fold. The
        numbers shown here use the corrected, non-overlapping methodology.
        </div>""",
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(
        "This test period runs through a full calendar year-plus (2025 onward) the "
        "model never saw during training or cross-validation — a genuinely "
        "out-of-sample window, not just a different slice of the same multi-year data "
        "used to build the model. Over this period, both the strategy and the "
        "buy-and-hold market benchmark lost value, and the small classification edge "
        "documented on the Model Performance page did not translate into a clear "
        "risk-adjusted advantage: Sharpe ratios and returns remain close between "
        "strategy and market, in a period that was difficult for the underlying market "
        "itself, independent of the model."
    )
