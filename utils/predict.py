import pandas as pd
import streamlit as st


def predict_tomorrow(row_dict, model_choice="LightGBM"):
    """Dự đoán tempmax ngày mai từ 1 dòng input."""
    fc = st.session_state.feature_cols
    X  = pd.DataFrame([row_dict])[fc]
    if model_choice == "LightGBM":
        pred = st.session_state.lgbm_bundle['model'].predict(X)[0]
    else:
        scaler = st.session_state.ridge_bundle['scaler']
        pred   = st.session_state.ridge_bundle['model'].predict(scaler.transform(X))[0]
    return round(float(pred), 2)
