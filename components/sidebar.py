import hashlib
import io

import pandas as pd
import streamlit as st

from utils.helpers import models_ready
from utils.pipeline import validate_weather_data
from utils.train import train_on_web


@st.cache_data(show_spinner=False)
def _read_csv(content: bytes):
    return validate_weather_data(pd.read_csv(io.BytesIO(content)))


def _clear_models():
    st.session_state.lgbm_bundle = None
    st.session_state.ridge_bundle = None
    st.session_state.baseline_bundle = None
    st.session_state.feature_cols = []
    st.session_state.model_fingerprint = None


def render_sidebar():
    with st.sidebar:
        st.markdown("## 🌤️ WeatherIQ")
        st.markdown("<span class='tag'>Cà Mau</span> <span class='tag'>Dự đoán ngày mai</span>",
                    unsafe_allow_html=True)
        st.divider()

        uploaded_file = st.file_uploader("📂 Upload CSV thời tiết", type=["csv"])
        if uploaded_file:
            try:
                content = uploaded_file.getvalue()
                fingerprint = hashlib.sha256(content).hexdigest()
                df_raw = _read_csv(content)
                if fingerprint != st.session_state.data_fingerprint:
                    st.session_state.df = df_raw
                    st.session_state.data_fingerprint = fingerprint
                    _clear_models()
                    st.session_state.messages = []
                st.success(f"✅ {len(df_raw):,} dòng hợp lệ")
            except Exception as exc:  # noqa: BLE001 - UI boundary must show parse errors
                st.session_state.df = None
                st.session_state.data_fingerprint = None
                _clear_models()
                st.error(f"CSV không hợp lệ: {exc}")

        if st.session_state.df is not None:
            model_is_current = (
                models_ready()
                and st.session_state.model_fingerprint == st.session_state.data_fingerprint
            )
            if model_is_current:
                candidates = {
                    "LightGBM": st.session_state.lgbm_bundle['validation_metrics'],
                    "Ridge": st.session_state.ridge_bundle['validation_metrics'],
                }
                best_name = min(candidates, key=lambda name: candidates[name]['RMSE'])
                metrics = candidates[best_name]
                st.success(f"✅ {best_name} sẵn sàng — val MAE: {metrics['MAE']}°C")
            if st.button(
                "🔄 Train lại Model" if model_is_current else "🚀 Train Model",
                type="primary",
                width='stretch',
            ):
                try:
                    with st.spinner("⚙️ Đang train LightGBM + Ridge..."):
                        lb, rb, bb, fc = train_on_web(st.session_state.df)
                    st.session_state.lgbm_bundle     = lb
                    st.session_state.ridge_bundle    = rb
                    st.session_state.baseline_bundle = bb
                    st.session_state.feature_cols    = fc
                    st.session_state.model_fingerprint = st.session_state.data_fingerprint
                    st.rerun()
                except Exception as e:  # noqa: BLE001 - UI boundary must keep app alive
                    _clear_models()
                    st.error(f"Không thể train model: {e}")

        st.divider()
        _render_chatbot()


def _render_chatbot():
    from utils.helpers import auto_insights, models_ready

    st.markdown("### 🤖 Chatbot")
    chat_container = st.container(height=300)
    with chat_container:
        for msg in st.session_state.messages[-20:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Hỏi về dữ liệu..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        df_ctx = st.session_state.df
        p = prompt.lower()

        if df_ctx is None and not any(w in p for w in ["model","train","dự đoán","hello","hi","chào"]):
            reply = "⚠️ Upload CSV để tôi phân tích dữ liệu nhé!"
        elif any(w in p for w in ["hello","hi","chào","xin chào","help","giúp"]):
            reply = ("👋 Xin chào! Tôi là **WeatherIQ Chatbot**.\n\n"
                     "Tôi có thể:\n"
                     "- 🌡️ Phân tích nhiệt độ, độ ẩm, gió\n"
                     "- 🤖 Thông tin về model đã train\n"
                     "- 🎯 Hướng dẫn dự đoán ngày mai\n"
                     "- 💡 Tóm tắt insights từ dữ liệu")
        elif any(w in p for w in ["model","r2","mae","rmse","độ chính xác","train","lightgbm","ridge"]):
            if models_ready():
                lm = st.session_state.lgbm_bundle['metrics']
                rm = st.session_state.ridge_bundle['metrics']
                bm = st.session_state.baseline_bundle['metrics'] if st.session_state.baseline_bundle else None
                baseline_line = f"**📊 Baseline:**\n- R²: **{bm['R2']}** | MAE: **{bm['MAE']}°**\n\n" if bm else ""
                reply = (f"🤖 **Kết quả Model (Test Set)**\n\n"
                         f"{baseline_line}"
                         f"**🔵 Ridge:**\n- R²: **{rm['R2']}** | MAE: **{rm['MAE']}°** | RMSE: **{rm['RMSE']}°**\n\n"
                         f"**🟢 LightGBM:**\n- R²: **{lm['R2']}** | MAE: **{lm['MAE']}°** | RMSE: **{lm['RMSE']}°**")
            else:
                reply = "Chưa có model. Upload CSV rồi bấm **🚀 Train Model** nhé!"
        elif any(w in p for w in ["dự đoán","ngày mai","predict"]):
            reply = "🎯 Vào tab **Dự đoán** → chọn model → nhập thông số hôm nay → bấm Dự đoán!"
        elif df_ctx is not None and any(w in p for w in ["temp","nhiệt","nóng","lạnh"]):
            if "temp" in df_ctx.columns:
                t = df_ctx["temp"]
                reply = (f"🌡️ **Nhiệt độ**\n\n"
                         f"- TB: **{t.mean():.1f}°** | Max: **{t.max():.1f}°** | Min: **{t.min():.1f}°**\n"
                         + (f"- Tempmax TB: **{df_ctx['tempmax'].mean():.1f}°**" if "tempmax" in df_ctx.columns else ""))
            else:
                reply = "Không tìm thấy cột nhiệt độ."
        elif df_ctx is not None and any(w in p for w in ["insight","nhận xét","tóm tắt","phân tích"]):
            reply = "🔍 **Nhận xét tự động**\n\n" + "\n\n".join(auto_insights(df_ctx))
        elif df_ctx is not None and any(w in p for w in ["dữ liệu","data","bao nhiêu","cột","dòng"]):
            reply = (f"📊 **Tổng quan**\n\n"
                     f"- Số dòng: **{len(df_ctx):,}** | Số cột: **{len(df_ctx.columns)}**\n"
                     + (f"- Thời gian: {df_ctx['datetime'].min().date()} → {df_ctx['datetime'].max().date()}"
                        if "datetime" in df_ctx.columns else ""))
        else:
            reply = "Hãy hỏi tôi về nhiệt độ, độ ẩm, model, hoặc dự đoán nhé! 😊"

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()
