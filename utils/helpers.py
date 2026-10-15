import matplotlib.pyplot as plt
import streamlit as st

from constants.config import CARD_BG, TEXT


def models_ready():
    return (
        st.session_state.get("lgbm_bundle") is not None
        and st.session_state.get("ridge_bundle") is not None
        and bool(st.session_state.get("feature_cols"))
    )


def styled_fig():
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values(): spine.set_edgecolor("#c8d0f0")
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    return fig, ax


def auto_insights(df):
    insights = []
    if "temp" in df.columns:
        avg_t, max_t, min_t = df["temp"].mean(), df["temp"].max(), df["temp"].min()
        insights.append(f"🌡️ Nhiệt độ dao động **{min_t:.1f}° – {max_t:.1f}°**, trung bình **{avg_t:.1f}°**.")
        hot = (df["temp"] > avg_t + df["temp"].std()).sum()
        insights.append(f"☀️ **{hot}** ngày có nhiệt độ bất thường cao.")
    if "humidity" in df.columns:
        avg_h = df["humidity"].mean()
        insights.append(f"💧 Độ ẩm trung bình **{avg_h:.1f}%** — {'ẩm cao' if avg_h > 70 else 'vừa phải'}.")
    if "tempmax" in df.columns:
        insights.append(f"🔆 Tempmax trung bình: **{df['tempmax'].mean():.1f}°**, cao nhất: **{df['tempmax'].max():.1f}°**.")
    if "precip" in df.columns:
        rainy = (df["precip"] > 0).sum()
        insights.append(f"🌧️ **{rainy}** ngày có mưa ({rainy/len(df)*100:.0f}% tổng số ngày).")
    return insights
