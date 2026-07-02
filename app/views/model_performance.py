import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score

from utils import (
    fetch_price_range,
    get_last_fold_split,
    load_features_csv,
    load_model,
    sidebar_controls,
)

ticker, date_range = sidebar_controls()

st.title("📊 Model Performance")
st.caption(f"Ticker: {ticker}")

# --- Candlestick chart ---
st.subheader("Price Chart")

start, end = (date_range if isinstance(date_range, tuple) and len(date_range) == 2 else (None, None))

if start and end:
    with st.spinner("Fetching price data..."):
        try:
            price_df = fetch_price_range(ticker, str(start), str(end))
        except Exception as e:
            price_df = pd.DataFrame()
            st.error(f"Could not fetch price data for {ticker}: {e}")

    if price_df is not None and not price_df.empty:
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=price_df.index,
                    open=price_df["Open"],
                    high=price_df["High"],
                    low=price_df["Low"],
                    close=price_df["Close"],
                    name=ticker,
                )
            ]
        )
        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10),
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No price data available for the selected range.")
else:
    st.info("Pick a start and end date in the sidebar to load the price chart.")

st.divider()

# --- Classification metrics ---
st.subheader("Classification Metrics (held-out test fold)")

model, feature_cols = load_model()
df = load_features_csv(ticker)
X = df[feature_cols]
y = df["Target"]
train_idx, test_idx = get_last_fold_split(X, y)
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, average=None, labels=[0, 1], zero_division=0)
rec = recall_score(y_test, preds, average=None, labels=[0, 1], zero_division=0)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{acc:.1%}")
c2.metric("Precision (Down)", f"{prec[0]:.1%}")
c3.metric("Precision (Up)", f"{prec[1]:.1%}")
c4.metric("Recall (Down)", f"{rec[0]:.1%}")
c5.metric("Recall (Up)", f"{rec[1]:.1%}")

st.caption(
    f"Evaluated on the most recent {len(test_idx)} trading days "
    "(TimeSeriesSplit, last fold) — the same held-out window used in backtesting."
)

st.divider()

# --- Feature importance ---
st.subheader("Feature Importance")
coefs = model.named_steps["clf"].coef_[0]
importance = pd.Series(coefs, index=feature_cols).abs().sort_values()

fig_imp = go.Figure(
    go.Bar(x=importance.values, y=importance.index, orientation="h", marker_color="steelblue")
)
fig_imp.update_layout(
    template="plotly_dark",
    height=500,
    xaxis_title="|Coefficient|",
    margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig_imp, use_container_width=True)
