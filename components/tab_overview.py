import streamlit as st
import pandas as pd
from utils.helpers import auto_insights, models_ready
from constants.config import ACCENT, ACCENT2, ACCENT3


def render(df):
    if df is None:
        st.info("👆 Upload file CSV ở sidebar để bắt đầu.")
        return 
#         st.markdown("""
# **Hướng dẫn:**

# | Bước | Làm gì |
# |------|--------|
# | 1 | Upload CSV lên sidebar |
# | 2 | Bấm **🚀 Train Model** |
# | 3 | Vào tab **🎯 Dự đoán** |
#         """)


    kpi_cols = [c for c in ["tempmax","temp","humidity","windspeed"] if c in df.columns]
    palette  = [ACCENT, ACCENT2, ACCENT3, "#7c3aed"]

    cols = st.columns(len(kpi_cols) + 1)
    with cols[0]:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value' style='color:{ACCENT}'>{len(df):,}</div>
            <div class='metric-label'>Tổng số ngày</div></div>""", unsafe_allow_html=True)
    for i, col in enumerate(kpi_cols):
        with cols[i+1]:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value' style='color:{palette[i]}'>{df[col].mean():.1f}</div>
                <div class='metric-label'>TB {col}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if models_ready():
        st.markdown("<div class='section-header'>🤖 Kết quả Model (Test Set)</div>", unsafe_allow_html=True)
        lm = st.session_state.lgbm_bundle['metrics']
        rm = st.session_state.ridge_bundle['metrics']
        bm = st.session_state.baseline_bundle['metrics'] if st.session_state.baseline_bundle else None

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**📊 Baseline (tempmax hôm nay)**")
            if bm:
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("R²", bm['R2']); mc2.metric("MAE °C", bm['MAE']); mc3.metric("RMSE °C", bm['RMSE'])
        with c2:
            st.markdown("**🔵 Ridge**")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("R²", rm['R2'], delta=f"{round(rm['R2']-(bm['R2'] if bm else 0),4)}")
            mc2.metric("MAE °C", rm['MAE']); mc3.metric("RMSE °C", rm['RMSE'])
        with c3:
            st.markdown("**🟢 LightGBM**")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("R²", lm['R2'], delta=f"{round(lm['R2']-(bm['R2'] if bm else 0),4)}")
            mc2.metric("MAE °C", lm['MAE']); mc3.metric("RMSE °C", lm['RMSE'])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-header'>📋 Xem trước dữ liệu</div>", unsafe_allow_html=True)
        st.dataframe(df.head(10), width='stretch')
    with c2:
        st.markdown("<div class='section-header'>🔍 Chất lượng dữ liệu</div>", unsafe_allow_html=True)
        quality = pd.DataFrame({"Cột": df.columns, "Kiểu": df.dtypes.values,
                                "Thiếu": df.isnull().sum().values,
                                "Thiếu %": (df.isnull().sum().values / len(df) * 100).round(1)})
        st.dataframe(quality, width='stretch', hide_index=True)

    st.markdown("<div class='section-header'>💡 Nhận xét tự động</div>", unsafe_allow_html=True)
    for ins in auto_insights(df):
        st.markdown(f"> {ins}")
