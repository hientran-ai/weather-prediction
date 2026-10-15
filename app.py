import streamlit as st

st.set_page_config(page_title="WeatherIQ — Cà Mau", page_icon="🌤️",
                   layout="wide", initial_sidebar_state="expanded")

from constants.config import CSS

st.markdown(CSS, unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [("df", None), ("messages", []),
                     ("lgbm_bundle", None), ("ridge_bundle", None),
                     ("baseline_bundle", None), ("feature_cols", []),
                     ("data_fingerprint", None), ("model_fingerprint", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Import components ─────────────────────────────────────────────────────────
from components.sidebar import render_sidebar
from components.tab_analysis import render as render_analysis
from components.tab_overview import render as render_overview
from components.tab_predict import render as render_predict
from components.tab_viz import render as render_viz

# ── Render ────────────────────────────────────────────────────────────────────
render_sidebar()

st.markdown("# 🌤️ WeatherIQ — Dự đoán thời tiết Cà Mau")
st.markdown("*Upload CSV → kiểm tra dữ liệu → train model → dự đoán tempmax ngày mai*")
st.divider()

df   = st.session_state.df
tabs = st.tabs(["📋 Tổng quan", "📊 Biểu đồ", "🔬 Phân tích sâu", "🎯 Dự đoán"])

with tabs[0]: render_overview(df)
with tabs[1]: render_viz(df)
with tabs[2]: render_analysis(df)
with tabs[3]: render_predict(df)
