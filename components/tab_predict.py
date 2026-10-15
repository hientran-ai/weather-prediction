import pandas as pd
import streamlit as st

from constants.config import ACCENT2
from utils.helpers import models_ready
from utils.pipeline import build_inference_row
from utils.predict import predict_tomorrow


def render(df):
    st.markdown("<div class='section-header'>🎯 Dự đoán tempmax ngày mai</div>", unsafe_allow_html=True)

    if not models_ready():
        st.warning("⬆️ Upload CSV ở sidebar ")
        return

    lgbm_val = st.session_state.lgbm_bundle['validation_metrics']
    ridge_val = st.session_state.ridge_bundle['validation_metrics']
    if lgbm_val['RMSE'] <= ridge_val['RMSE']:
        model_choice = "LightGBM"
        m_info = st.session_state.lgbm_bundle['metrics']
    else:
        model_choice = "Ridge"
        m_info = st.session_state.ridge_bundle['metrics']
    st.caption(
        f"Tự động chọn **{model_choice}** theo RMSE validation thấp nhất "
        f"(LightGBM: {lgbm_val['RMSE']}°C · Ridge: {ridge_val['RMSE']}°C)."
    )
    st.markdown("Nhập thông số thời tiết **hôm nay** để dự đoán **tempmax ngày mai**:")

    if df is None:
        st.warning("Không còn dữ liệu đầu vào. Vui lòng upload lại CSV.")
        return

    latest = df.sort_values('datetime').iloc[-1]
    latest_date = pd.Timestamp(latest['datetime'])
    st.info(f"📅 Dùng đúng dữ liệu ngày **{latest_date.date()}** làm mặc định")

    raw_inputs = ["tempmax","tempmin","temp","humidity","windspeed",
                  "cloudcover","uvindex","precip","sealevelpressure","dew"]
    raw_inputs = [c for c in raw_inputs if c in df.columns]

    with st.form("prediction_form"):
        input_vals = {}
        cols_ui = st.columns(3)
        for i, feature in enumerate(raw_inputs):
            with cols_ui[i % 3]:
                input_vals[feature] = st.number_input(
                    feature,
                    value=float(latest[feature]),
                    step=0.1,
                    help="Nhập giá trị của ngày hiện tại",
                )
        submitted = st.form_submit_button(
            "⚡ Dự đoán tempmax ngày mai", type="primary", width='stretch'
        )

    if submitted:
        try:
            feature_row = build_inference_row(
                df, input_vals, st.session_state.feature_cols
            )
            pred = predict_tomorrow(feature_row, model_choice)
            forecast_date = latest_date + pd.Timedelta(days=1)
            st.markdown(f"""
            <div class='predict-box'>
                <div style='font-size:0.85rem;color:#64748b;text-transform:uppercase;letter-spacing:2px;'>
                    Tempmax {forecast_date:%d/%m/%Y} — {model_choice}
                </div>
                <div style='font-size:4rem;font-weight:700;color:{ACCENT2};
                            font-family:JetBrains Mono,monospace;margin:12px 0;'>
                    {pred:.1f}°C
                </div>
                <div style='font-size:0.9rem;color:#64748b;'>
                    MAE trên tập test tham khảo: {m_info['MAE']}°C
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:  # noqa: BLE001 - surface inference errors in the UI
            st.error(f"Lỗi: {e}")
