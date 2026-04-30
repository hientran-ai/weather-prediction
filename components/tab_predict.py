import streamlit as st
from utils.helpers import preprocess, feature_engineering, models_ready
from utils.predict import predict_tomorrow
from constants.config import ACCENT2


def render(df):
    st.markdown("<div class='section-header'>🎯 Dự đoán tempmax ngày mai</div>", unsafe_allow_html=True)

    if not models_ready():
        st.warning("⬆️ Upload CSV ở sidebar ")
        return

    # model_choice = st.radio("Chọn model", ["LightGBM", "Ridge"], horizontal=True,
    #                          help="LightGBM thường chính xác hơn Ridge")
    # m_info = (st.session_state.lgbm_bundle['metrics'] if model_choice == "LightGBM"
    #           else st.session_state.ridge_bundle['metrics'])
    # st.caption(f"R²: {m_info['R2']} | MAE: {m_info['MAE']}° | RMSE: {m_info['RMSE']}°")
    # st.divider()
    if st.session_state.lgbm_bundle['metrics']['R2'] >= st.session_state.ridge_bundle['metrics']['R2']:
        model_choice = "LightGBM"
        m_info = st.session_state.lgbm_bundle['metrics']
    else:
        model_choice = "Ridge"
        m_info = st.session_state.ridge_bundle['metrics']
    st.markdown("Nhập thông số thời tiết **hôm nay** để dự đoán **tempmax ngày mai**:")

    # Dùng dòng cuối CSV làm giá trị mặc định
    if df is not None:
        df_feat  = feature_engineering(preprocess(df.copy()))
        last_row = df_feat.iloc[-1][st.session_state.feature_cols].to_dict()
        st.info(f"📅 Dùng dữ liệu ngày **{df['datetime'].iloc[-1].date()}** làm mặc định")
    else:
        last_row = {f: 0.0 for f in st.session_state.feature_cols}

    raw_inputs = ["tempmax","tempmin","temp","humidity","windspeed",
                  "cloudcover","uvindex","precip","sealevelpressure","dew"]
    raw_inputs = [c for c in raw_inputs if c in st.session_state.feature_cols]

    input_vals = dict(last_row)
    cols_ui = st.columns(len(raw_inputs))
    for i, f in enumerate(raw_inputs):
        with cols_ui[i]:
            input_vals[f] = st.number_input(f, value=float(last_row.get(f, 0.0)),
                                             step=0.1, help="Nhập giá trị hôm nay")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚡ Dự đoán tempmax ngày mai", type="primary", width='stretch'):
        try:
            pred = predict_tomorrow(input_vals, model_choice)
            st.markdown(f"""
            <div class='predict-box'>
                <div style='font-size:0.85rem;color:#64748b;text-transform:uppercase;letter-spacing:2px;'>
                    Dự đoán Tempmax Ngày Mai — {model_choice}
                </div>
                <div style='font-size:4rem;font-weight:700;color:{ACCENT2};
                            font-family:JetBrains Mono,monospace;margin:12px 0;'>
                    {pred:.1f}°C
                </div>
                <div style='font-size:0.9rem;color:#64748b;'>
                    Sai số dự kiến: ± {m_info['MAE']}°C
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Lỗi: {e}")
