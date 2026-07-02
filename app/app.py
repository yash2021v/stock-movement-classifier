import streamlit as st

st.set_page_config(page_title="Stock Movement Classifier", layout="wide")

pages = [
    st.Page("views/model_performance.py", title="Model Performance", icon="📊", default=True),
    st.Page("views/backtest_results.py", title="Backtest Results", icon="📈"),
    st.Page("views/predict_tomorrow.py", title="Predict Tomorrow", icon="🔮"),
]

pg = st.navigation(pages)
pg.run()
