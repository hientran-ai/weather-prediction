import streamlit as st
from utils.train import train_on_web
from utils.helpers import models_ready


def render_sidebar():
    with st.sidebar:
        st.markdown("## 🌤️ WeatherIQ")
        st.markdown("<span class='tag'>Cà Mau</span> <span class='tag'>Dự đoán ngày mai</span>",
                    unsafe_allow_html=True)
        st.divider()

        uploaded_file = st.file_uploader("📂 Upload CSV thời tiết", type=["csv"])
        if uploaded_file:
            import pandas as pd
            df_raw = pd.read_csv(uploaded_file)
            df_raw['datetime'] = pd.to_datetime(df_raw['datetime'], errors='coerce')
            df_raw = df_raw.sort_values('datetime').reset_index(drop=True)
            st.session_state.df = df_raw.copy()
            st.success(f"✅ {len(df_raw):,} dòng")

        # if st.session_state.df is not None:
        #     if st.button("🚀 Train Model", type="primary", width='stretch'):
        #         with st.spinner("⚙️ Đang train LightGBM + Ridge..."):
            if not models_ready() and st.session_state.df is not None:
                try:
                    lb, rb, bb, fc = train_on_web(st.session_state.df)
                    st.session_state.lgbm_bundle     = lb
                    st.session_state.ridge_bundle    = rb
                    st.session_state.baseline_bundle = bb
                    st.session_state.feature_cols    = fc
                    # st.success("✅ Train xong!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi: {e}")

        # if models_ready():
        #     lgbm_m = st.session_state.lgbm_bundle['metrics']
        #     st.success(f"✅ Model sẵn sàng — MAE: {lgbm_m['MAE']}° | R²: {lgbm_m['R2']}")
        # else:
        #     st.warning("⬆️ Upload CSV rồi bấm **🚀 Train Model**")

        st.divider()
        _render_chatbot()


def _render_chatbot():
    import numpy as np
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
