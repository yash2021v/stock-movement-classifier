import streamlit as st

from utils import force_light_theme

st.set_page_config(
    page_title="Cross-Sectional Alpha Ranking Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

force_light_theme()

pages = [
    st.Page("views/overview.py", title="Overview", default=True),
    st.Page("views/model_performance.py", title="Model Performance"),
    st.Page("views/backtest_results.py", title="Backtest Results"),
    st.Page("views/live_ranking.py", title="Live Ranking"),
]

st.sidebar.markdown("### Cross-Sectional Alpha Ranking")
st.sidebar.caption("20-stock NSE universe · 5-day horizon")

pg = st.navigation(pages, position="sidebar")
pg.run()
