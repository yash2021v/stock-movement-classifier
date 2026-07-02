import pandas as pd
import streamlit as st

from utils import (
    CALIBRATION_INFO,
    TICKERS,
    fetch_live_ranking_inputs,
    inject_base_style,
    load_model,
    render_ranking_table_html,
)

inject_base_style()

st.title("Live Ranking")
st.caption("Current relative-outperformance ranking across the 20-stock universe.")

st.markdown(
    """<div class="disclaimer-box">
    This is a statistical research tool illustrating model output — not a validated
    trading strategy or financial advice. See the Backtest Results page for performance
    limitations.
    </div>""",
    unsafe_allow_html=True,
)

st.write("")

generate = st.button("Generate Today's Ranking", type="primary")

st.divider()

if generate:
    with st.spinner("Fetching recent data and scoring the universe..."):
        model, feature_cols = load_model()
        latest_rows, errors = fetch_live_ranking_inputs(tuple(TICKERS))

    if not latest_rows:
        st.error("Could not fetch usable data for any ticker. Please try again later.")
    else:
        records = []
        for ticker, row in latest_rows.items():
            X_live = row[feature_cols]
            proba = model.predict_proba(X_live)[0][1]
            pred_class = "Outperform" if proba >= 0.5 else "Underperform"
            records.append(
                {
                    "Ticker": ticker,
                    "Predicted Outperformance Probability": proba,
                    "Predicted Class": pred_class,
                }
            )

        ranking_df = pd.DataFrame(records).sort_values(
            "Predicted Outperformance Probability", ascending=False
        ).reset_index(drop=True)

        st.subheader("Ranking")
        st.markdown(render_ranking_table_html(ranking_df), unsafe_allow_html=True)

        cal = CALIBRATION_INFO
        if cal["was_recalibrated"]:
            st.caption(
                f"Note: this model's cross-validated accuracy is approximately "
                f"{cal['cv_accuracy']:.0%}, only marginally above a random baseline (~50%). "
                "Probabilities shown have been calibrated (isotonic regression) to better "
                "reflect the model's actual accuracy — read them primarily as a relative "
                "ranking signal rather than a precise confidence level."
            )
        else:
            st.caption(
                f"Note: this model's cross-validated accuracy is approximately "
                f"{cal['cv_accuracy']:.0%}, only marginally above a random baseline (~50%). "
                "Probability values shown above may not reflect true confidence levels and "
                "should be read primarily as a relative ranking signal."
            )

        st.write("")
        st.subheader("What does this ranking mean?")
        st.markdown(
            f"""
**What it represents.** Each stock's percentage is the model's estimated probability
that it will be a *relative* outperformer — a top-half performer among these 20 stocks —
over the next 5 trading days. It is **not** a prediction that the stock's price will go
up or down in absolute terms, and it says nothing about the size of any expected move.

**How to read it.** A higher probability means the model leans toward that stock
outperforming its peers over the next 5 trading days; a lower probability means it leans
toward underperforming them. Values near 50% mean the model has little conviction either
way — most of the table will sit close to this line, which is expected given the model's
accuracy (see below).

**Keep it in context.** This reflects a modest, statistically-detected edge (roughly
{cal['cv_accuracy']:.0%} cross-validated accuracy, only marginally above a 50% random
baseline) — not a strong or validated trading signal. Treat it as one input among many,
not a recommendation.
"""
        )

    if errors:
        st.divider()
        st.subheader("Fetch Issues")
        for ticker, msg in errors.items():
            st.warning(f"{ticker}: {msg}")
else:
    st.info("Click **Generate Today's Ranking** to fetch recent data and score all 20 stocks.")
