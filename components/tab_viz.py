import matplotlib.pyplot as plt
import streamlit as st

from constants.config import ACCENT, ACCENT2, ACCENT3, CARD_BG, TEXT
from utils.helpers import styled_fig


def render(df):
    if df is None:
        st.info("Upload CSV để xem biểu đồ.")
        return

    num_cols = df.select_dtypes(include='number').columns.tolist()

    # Nhiệt độ theo thời gian
    if "tempmax" in df.columns and "datetime" in df.columns:
        st.markdown("<div class='section-header'>🔆 Nhiệt độ theo thời gian</div>", unsafe_allow_html=True)
        fig, ax = styled_fig()
        ax.plot(df["datetime"], df["tempmax"], color=ACCENT2, linewidth=1.2, label="tempmax")
        if "temp" in df.columns:
            ax.plot(df["datetime"], df["temp"], color=ACCENT, linewidth=1, alpha=0.7, label="temp TB")
        if "tempmin" in df.columns:
            ax.fill_between(df["datetime"], df["tempmin"], df["tempmax"], alpha=0.1, color=ACCENT2)
        ax.set_xlabel("Ngày"); ax.set_ylabel("Nhiệt độ (°C)"); ax.legend()
        plt.xticks(rotation=30); st.pyplot(fig); plt.close()

    # Các chỉ số khác
    plot_cols = [c for c in ["humidity","windspeed","cloudcover","uvindex","precip","sealevelpressure"] if c in df.columns]
    if plot_cols and "datetime" in df.columns:
        st.markdown("<div class='section-header'>📈 Các chỉ số khác</div>", unsafe_allow_html=True)
        sel = st.multiselect("Chọn chỉ số", plot_cols, default=plot_cols[:2])
        if sel:
            fig, ax = styled_fig()
            colors = [ACCENT, ACCENT2, ACCENT3, "#7c3aed", "#f59e0b", "#06b6d4"]
            for i, c in enumerate(sel):
                ax.plot(df["datetime"], df[c], label=c, color=colors[i % len(colors)], linewidth=1.2)
            ax.legend(); plt.xticks(rotation=30); st.pyplot(fig); plt.close()

    # Phân phối & box plot
    if num_cols:
        st.markdown("<div class='section-header'>📊 Phân phối</div>", unsafe_allow_html=True)
        default_col = next((c for c in ["tempmax","temp","humidity"] if c in num_cols), num_cols[0])
        sel_col = st.selectbox("Chọn cột", num_cols, index=num_cols.index(default_col))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor("#ffffff")
        for ax in axes:
            ax.set_facecolor(CARD_BG)
            for spine in ax.spines.values(): spine.set_edgecolor("#c8d0f0")
        axes[0].hist(df[sel_col].dropna(), bins=40, color=ACCENT, edgecolor="white", alpha=0.85)
        axes[0].set_title(f"Phân phối {sel_col}", color=TEXT)
        axes[1].boxplot(df[sel_col].dropna(), patch_artist=True,
                        boxprops={"facecolor": ACCENT, "color": ACCENT},
                        whiskerprops={"color": TEXT}, capprops={"color": TEXT},
                        medianprops={"color": ACCENT2, "linewidth": 2})
        axes[1].set_title(f"Box plot {sel_col}", color=TEXT)
        st.pyplot(fig); plt.close()
