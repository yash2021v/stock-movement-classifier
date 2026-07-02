import plotly.graph_objects as go
import streamlit as st

from utils import (
    align_dates_for_ticker,
    get_last_fold_split,
    load_features_csv,
    load_model,
    run_backtest,
    sidebar_controls,
)

ticker, date_range = sidebar_controls()

st.title("📈 Backtest Results")
st.caption(f"Ticker: {ticker}")

model, feature_cols = load_model()
df = load_features_csv(ticker)
X = df[feature_cols]
y = df["Target"]
train_idx, test_idx = get_last_fold_split(X, y)
X_test = X.iloc[test_idx]
preds = model.predict(X_test)

test_df, metrics = run_backtest(df, test_idx, preds)

st.subheader("Strategy vs. Market")

row1 = st.columns(3)
row1[0].metric("Strategy Sharpe Ratio", f"{metrics['Strategy Sharpe']:.2f}")
row1[1].metric("Strategy Max Drawdown", f"{metrics['Strategy Drawdown']:.2%}")
row1[2].metric("Strategy Total Return", f"{metrics['Strategy Return']:.2%}")

row2 = st.columns(3)
row2[0].metric("Market Sharpe Ratio", f"{metrics['Market Sharpe']:.2f}")
row2[1].metric("Market Max Drawdown", f"{metrics['Market Drawdown']:.2%}")
row2[2].metric("Market Total Return", f"{metrics['Market Return']:.2%}")

st.info(
    "On this test window, the ML strategy underperformed a simple buy-and-hold approach. "
    "This is a common outcome for simple linear models predicting short-horizon (next-day) "
    "price direction — day-to-day returns are dominated by noise, so a small edge in "
    "classification accuracy doesn't reliably translate into better risk-adjusted returns "
    "than just holding the stock."
)

st.divider()

st.subheader("Cumulative Return: Strategy vs. Buy & Hold")

full_dates = align_dates_for_ticker(ticker, len(df))
x_axis = full_dates[test_idx] if full_dates is not None else list(range(len(test_idx)))
x_title = "Date" if full_dates is not None else "Trading Day (test window)"

fig = go.Figure()
fig.add_trace(go.Scatter(x=x_axis, y=test_df["Cum_Strategy"], name="ML Strategy", mode="lines"))
fig.add_trace(go.Scatter(x=x_axis, y=test_df["Cum_Market"], name="Buy & Hold", mode="lines"))
fig.update_layout(
    template="plotly_dark",
    height=450,
    xaxis_title=x_title,
    yaxis_title="Portfolio Value (Start = 1)",
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

if full_dates is None:
    st.caption(
        "Real trading dates could not be recovered for this chart (no network access to "
        "re-fetch history), so the x-axis shows relative trading days within the test window instead."
    )
