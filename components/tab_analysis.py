import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.helpers import styled_fig, models_ready
from constants.config import ACCENT, ACCENT2, ACCENT3, CARD_BG, TEXT


def render(df):
    if df is None:
        st.info("Upload CSV để phân tích.")
        return

    num_cols = df.select_dtypes(include='number').columns.tolist()

    # Ma trận tương quan
    base_cols = [c for c in ["tempmax","tempmin","temp","humidity","windspeed",
                              "cloudcover","uvindex","precip","sealevelpressure","dew"] if c in num_cols]
    if len(base_cols) >= 3:
        st.markdown("<div class='section-header'>🔥 Ma trận tương quan</div>", unsafe_allow_html=True)
        corr = df[base_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#ffffff")
        sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)),
                    annot=True, fmt=".2f", cmap="RdYlBu_r", ax=ax,
                    linewidths=0.5, linecolor="white", annot_kws={"size": 9})
        ax.set_title("Tương quan giữa các biến", pad=15, fontsize=13)
        plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
        st.pyplot(fig); plt.close()

    # Tempmax theo tháng
    if "datetime" in df.columns and "tempmax" in df.columns:
        df = df.copy()
        df["month"] = df["datetime"].dt.month
        st.markdown("<div class='section-header'>📅 Tempmax theo tháng</div>", unsafe_allow_html=True)
        monthly = df.groupby("month")["tempmax"].agg(["mean","min","max"]).reset_index()
        month_names = ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12"]
        monthly["tháng"] = monthly["month"].apply(lambda x: month_names[x-1])
        fig, ax = styled_fig()
        ax.fill_between(monthly["month"], monthly["min"], monthly["max"], alpha=0.15, color=ACCENT2, label="Min–Max")
        ax.plot(monthly["month"], monthly["mean"], marker="o", color=ACCENT2, linewidth=2, markersize=6, label="TB tháng")
        ax.set_xticks(monthly["month"]); ax.set_xticklabels(monthly["tháng"])
        ax.set_xlabel("Tháng"); ax.set_ylabel("Tempmax (°C)"); ax.legend()
        st.pyplot(fig); plt.close()

    # So sánh models + Feature Importance
    if models_ready():
        st.markdown("<div class='section-header'>📊 So sánh Models</div>", unsafe_allow_html=True)
        lm = st.session_state.lgbm_bundle['metrics']
        rm = st.session_state.ridge_bundle['metrics']
        bm = st.session_state.baseline_bundle['metrics'] if st.session_state.baseline_bundle else None

        rows = []
        if bm: rows.append({"Model": "📊 Baseline", "R²": bm['R2'], "MAE (°C)": bm['MAE'], "RMSE (°C)": bm['RMSE']})
        rows.append({"Model": "🔵 Ridge",    "R²": rm['R2'], "MAE (°C)": rm['MAE'], "RMSE (°C)": rm['RMSE']})
        rows.append({"Model": "🟢 LightGBM", "R²": lm['R2'], "MAE (°C)": lm['MAE'], "RMSE (°C)": lm['RMSE']})
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        fig, ax = styled_fig()
        colors = ["#94a3b8", ACCENT, ACCENT3][:len(rows)]
        bars = ax.bar([r["Model"] for r in rows], [r["MAE (°C)"] for r in rows],
                      color=colors, alpha=0.85, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}°", ha='center', va='bottom', fontsize=9)
        ax.set_ylabel("MAE (°C)"); ax.set_title("So sánh MAE — thấp hơn = tốt hơn")
        st.pyplot(fig); plt.close()

        st.markdown("<div class='section-header'>🏆 Feature Importance (LightGBM)</div>", unsafe_allow_html=True)
        fi = pd.Series(st.session_state.lgbm_bundle['model'].feature_importances_,
                       index=st.session_state.lgbm_bundle['feature_cols']).nlargest(15).sort_values()
        fig, ax = styled_fig()
        bars = ax.barh(fi.index, fi.values, color=ACCENT, alpha=0.8)
        bars[-1].set_color(ACCENT2)
        ax.set_xlabel("Importance"); ax.set_title("Top 15 Features quan trọng nhất")
        st.pyplot(fig); plt.close()

    # Thống kê & Outliers
    base_num = [c for c in ["tempmax","tempmin","temp","humidity","windspeed","precip",
                             "cloudcover","uvindex","sealevelpressure"] if c in num_cols]
    st.markdown("<div class='section-header'>📐 Thống kê mô tả</div>", unsafe_allow_html=True)
    st.dataframe(df[base_num].describe().round(3).T, width='stretch')

    st.markdown("<div class='section-header'>🚨 Outlier Detection (IQR)</div>", unsafe_allow_html=True)
    outlier_data = []
    for col in base_num:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        outlier_data.append({"Cột": col, "Q1": round(Q1,2), "Q3": round(Q3,2),
                             "IQR": round(IQR,2), "Outliers": n_out,
                             "Outlier %": round(n_out/len(df)*100, 1)})
    st.dataframe(pd.DataFrame(outlier_data), width='stretch', hide_index=True)
