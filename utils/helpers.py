import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from constants.config import ACCENT, ACCENT2, CARD_BG, TEXT


def models_ready():
    return st.session_state.lgbm_bundle is not None


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


def preprocess(df):
    keep_cols = ['datetime','tempmax','tempmin','temp','feelslikemax','feelslikemin',
                 'feelslike','dew','humidity','precip','precipprob','windspeed','windgust',
                 'sealevelpressure','cloudcover','solarradiation','solarenergy','uvindex',
                 'sunrise','sunset']
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
    for col in ['snow','snowdepth','severerisk']:
        if col in df.columns: df.drop(columns=[col], inplace=True)
    temp_cols = [c for c in ['tempmax','tempmin','temp','dew','humidity','sealevelpressure',
                              'feelslikemax','feelslikemin','feelslike'] if c in df.columns]
    df[temp_cols] = df[temp_cols].interpolate(method='linear')
    fill_zero = [c for c in ['precip','precipprob','windgust','cloudcover'] if c in df.columns]
    df[fill_zero] = df[fill_zero].fillna(0)
    return df


def feature_engineering(df):
    df['month']     = df['datetime'].dt.month
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['sin_day']   = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['cos_day']   = np.cos(2 * np.pi * df['dayofyear'] / 365)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    df['is_rainy_season'] = df['month'].isin([5,6,7,8,9,10]).astype(int)
    df['is_peak_rain']    = df['month'].isin([7,8,9]).astype(int)
    df['is_dry_peak']     = df['month'].isin([2,3,4]).astype(int)
    df['temp_range']      = df['tempmax'] - df['tempmin']
    df['pressure_change'] = df['sealevelpressure'] - df['sealevelpressure'].shift(1)
    df['tempmax_diff1']   = df['tempmax'] - df['tempmax'].shift(1)
    if 'sunrise' in df.columns and 'sunset' in df.columns:
        df['day_length'] = (pd.to_datetime(df['sunset']) - pd.to_datetime(df['sunrise'])).dt.seconds / 3600
        df.drop(columns=['sunrise','sunset'], inplace=True)
    if 'solarradiation' in df.columns and 'cloudcover' in df.columns:
        df['solar_cloud'] = df['solarradiation'] * (1 - df['cloudcover'] / 100)
    if 'temp' in df.columns and 'humidity' in df.columns:
        df['heat_index'] = df['temp'] * df['humidity'] / 100
    lag_cols = {'tempmax':[1,2,3],'tempmin':[1],'temp':[1,2],'humidity':[1,2],
                'sealevelpressure':[1,2],'precip':[1],'cloudcover':[1],'dew':[1]}
    for col, lags in lag_cols.items():
        if col in df.columns:
            for lag in lags: df[f'{col}_lag{lag}'] = df[col].shift(lag)
    df['tempmax_roll3'] = df['tempmax'].shift(1).rolling(3).mean()
    df['tempmax_roll7'] = df['tempmax'].shift(1).rolling(7).mean()
    df['temp_std3']     = df['temp'].shift(1).rolling(3).std()
    df['precip_roll7']  = df['precip'].shift(1).rolling(7).sum()
    # Target: tempmax ngày MAI — tạo TRƯỚC dropna
    df['target'] = df['tempmax'].shift(-1)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
