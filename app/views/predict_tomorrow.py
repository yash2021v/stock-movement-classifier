import streamlit as st

from utils import TICKERS, build_live_features, fetch_price_history, load_model, sidebar_controls

sidebar_ticker, date_range = sidebar_controls()

st.title("🔮 Predict Tomorrow's Move")

ticker = st.selectbox("Select stock ticker", TICKERS, index=TICKERS.index(sidebar_ticker))
predict_clicked = st.button("Predict", type="primary")

st.divider()

if predict_clicked:
    with st.spinner(f"Fetching latest data for {ticker}..."):
        try:
            raw_df = fetch_price_history(ticker, period="6mo")
        except Exception as e:
            raw_df = None
            st.error(f"Failed to fetch data for {ticker}: {e}")

    if raw_df is None or raw_df.empty:
        st.error("No data returned for this ticker. Please try again later.")
    else:
        try:
            model, feature_cols = load_model()
            feat_df = build_live_features(raw_df)
        except Exception as e:
            feat_df = None
            st.error(f"Failed to build features from the fetched data: {e}")

        if feat_df is not None and feat_df.empty:
            st.error("Not enough recent trading history to compute indicators. Try again later.")
        elif feat_df is not None:
            latest = feat_df.iloc[[-1]]
            X_live = latest[feature_cols]
            pred = int(model.predict(X_live)[0])
            proba = model.predict_proba(X_live)[0]
            confidence = proba[pred]

            last_close = latest["Close"].iloc[0]
            rsi = latest["momentum_rsi"].iloc[0]
            macd = latest["trend_macd"].iloc[0]

            col1, col2, col3 = st.columns(3)
            col1.markdown("### 🟢 UP" if pred == 1 else "### 🔴 DOWN")
            col2.metric("Confidence", f"{confidence:.1%}")
            col3.metric("Last Close", f"₹{last_close:,.2f}")

            col4, col5 = st.columns(2)
            col4.metric("RSI (14)", f"{rsi:.1f}")
            col5.metric("MACD", f"{macd:.3f}")
else:
    st.info("Select a ticker and click **Predict** to generate tomorrow's forecast.")

st.caption("For educational purposes only — not financial advice.")
