"""Data validation and feature engineering for the weather models.

This module deliberately has no Streamlit dependency so that the ML pipeline can
be tested and reused from a batch job or an API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "datetime",
    "tempmax",
    "tempmin",
    "temp",
    "precip",
    "sealevelpressure",
}

WEATHER_COLUMNS = [
    "datetime",
    "tempmax",
    "tempmin",
    "temp",
    "feelslikemax",
    "feelslikemin",
    "feelslike",
    "dew",
    "humidity",
    "precip",
    "precipprob",
    "windspeed",
    "windgust",
    "sealevelpressure",
    "cloudcover",
    "solarradiation",
    "solarenergy",
    "uvindex",
    "sunrise",
    "sunset",
]

NON_NUMERIC_COLUMNS = {"datetime", "sunrise", "sunset"}


def validate_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate uploaded data and return a sorted, normalized copy."""
    if df.empty:
        raise ValueError("CSV không có dòng dữ liệu nào.")

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"CSV thiếu các cột bắt buộc: {', '.join(missing)}")

    result = df.copy()
    result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce")
    invalid_dates = int(result["datetime"].isna().sum())
    if invalid_dates:
        raise ValueError(f"Cột datetime có {invalid_dates} giá trị không hợp lệ.")

    if result["datetime"].duplicated().any():
        duplicates = int(result["datetime"].duplicated().sum())
        raise ValueError(f"Cột datetime có {duplicates} ngày bị trùng.")

    ordered_dates = result["datetime"].sort_values()
    gaps = ordered_dates.diff().dropna()
    irregular_gaps = int((gaps != pd.Timedelta(days=1)).sum())
    if irregular_gaps:
        raise ValueError(
            f"Chuỗi thời gian không liên tục theo ngày ({irregular_gaps} khoảng bị thiếu)."
        )

    for column in set(WEATHER_COLUMNS).intersection(result.columns) - NON_NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    empty_required = [
        column
        for column in REQUIRED_COLUMNS - {"datetime"}
        if result[column].isna().all()
    ]
    if empty_required:
        raise ValueError(
            "Các cột bắt buộc không có dữ liệu số: " + ", ".join(sorted(empty_required))
        )

    return result.sort_values("datetime").reset_index(drop=True)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Select model columns and fill gaps using past information only."""
    result = validate_weather_data(df)
    keep_cols = [column for column in WEATHER_COLUMNS if column in result.columns]
    result = result[keep_cols].copy()

    numeric_cols = [column for column in keep_cols if column not in NON_NUMERIC_COLUMNS]
    # Forward-fill is causal: unlike linear interpolation it never reads a future day.
    result[numeric_cols] = result[numeric_cols].ffill()
    empty_optional = [
        column
        for column in result.columns
        if column not in REQUIRED_COLUMNS and result[column].isna().all()
    ]
    result.drop(columns=empty_optional, inplace=True)
    return result


def feature_engineering(df: pd.DataFrame, *, include_target: bool = True) -> pd.DataFrame:
    """Create features for day t; optionally attach tempmax(t+1) as target."""
    result = df.copy()
    result["month"] = result["datetime"].dt.month
    result["dayofyear"] = result["datetime"].dt.dayofyear
    result["sin_day"] = np.sin(2 * np.pi * result["dayofyear"] / 365.25)
    result["cos_day"] = np.cos(2 * np.pi * result["dayofyear"] / 365.25)
    result["sin_month"] = np.sin(2 * np.pi * result["month"] / 12)
    result["cos_month"] = np.cos(2 * np.pi * result["month"] / 12)
    result["is_rainy_season"] = result["month"].isin([5, 6, 7, 8, 9, 10]).astype(int)
    result["is_peak_rain"] = result["month"].isin([7, 8, 9]).astype(int)
    result["is_dry_peak"] = result["month"].isin([2, 3, 4]).astype(int)
    result["temp_range"] = result["tempmax"] - result["tempmin"]
    result["pressure_change"] = result["sealevelpressure"].diff()
    result["tempmax_diff1"] = result["tempmax"].diff()

    if {"sunrise", "sunset"}.issubset(result.columns):
        sunrise = pd.to_datetime(result["sunrise"], errors="coerce")
        sunset = pd.to_datetime(result["sunset"], errors="coerce")
        result["day_length"] = (sunset - sunrise).dt.total_seconds() / 3600
        result.drop(columns=["sunrise", "sunset"], inplace=True)

    if {"solarradiation", "cloudcover"}.issubset(result.columns):
        result["solar_cloud"] = result["solarradiation"] * (
            1 - result["cloudcover"] / 100
        )
    if {"temp", "humidity"}.issubset(result.columns):
        result["heat_index"] = result["temp"] * result["humidity"] / 100

    lag_cols = {
        "tempmax": [1, 2, 3],
        "tempmin": [1],
        "temp": [1, 2],
        "humidity": [1, 2],
        "sealevelpressure": [1, 2],
        "precip": [1],
        "cloudcover": [1],
        "dew": [1],
    }
    for column, lags in lag_cols.items():
        if column in result.columns:
            for lag in lags:
                result[f"{column}_lag{lag}"] = result[column].shift(lag)

    result["tempmax_roll3"] = result["tempmax"].shift(1).rolling(3).mean()
    result["tempmax_roll7"] = result["tempmax"].shift(1).rolling(7).mean()
    result["temp_std3"] = result["temp"].shift(1).rolling(3).std()
    result["precip_roll7"] = result["precip"].shift(1).rolling(7).sum()

    if include_target:
        result["target"] = result["tempmax"].shift(-1)

    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    required = [column for column in result.columns if column != "datetime"]
    result.dropna(subset=required, inplace=True)
    return result.reset_index(drop=True)


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build a supervised frame where each row predicts the following day."""
    return feature_engineering(preprocess(df), include_target=True)


def build_inference_row(
    df: pd.DataFrame,
    overrides: Mapping[str, float],
    feature_cols: Sequence[str],
) -> dict[str, float]:
    """Build the latest feature row after applying edits to the latest raw day."""
    prepared = preprocess(df)
    latest_index = prepared.index[-1]
    for column, value in overrides.items():
        if column in prepared.columns and column not in NON_NUMERIC_COLUMNS:
            prepared.loc[latest_index, column] = float(value)

    featured = feature_engineering(prepared, include_target=False)
    if featured.empty:
        raise ValueError("Cần ít nhất 8 ngày dữ liệu hợp lệ để tạo đặc trưng dự báo.")

    missing = sorted(set(feature_cols) - set(featured.columns))
    if missing:
        raise ValueError("Không thể tạo các feature: " + ", ".join(missing))

    latest = featured.iloc[-1]
    return {column: float(latest[column]) for column in feature_cols}


def chronological_split(
    df: pd.DataFrame,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    min_partition_size: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split an ordered time series without shuffling or fixed calendar dates."""
    if not 0 < validation_fraction < 0.5 or not 0 < test_fraction < 0.5:
        raise ValueError("Tỷ lệ validation/test phải nằm trong khoảng (0, 0.5).")

    ordered = df.sort_values("datetime").reset_index(drop=True)
    minimum_total = min_partition_size * 3
    if len(ordered) < minimum_total:
        raise ValueError(
            f"Cần ít nhất {minimum_total} dòng sau feature engineering; "
            f"hiện chỉ có {len(ordered)}."
        )

    test_size = max(min_partition_size, round(len(ordered) * test_fraction))
    validation_size = max(min_partition_size, round(len(ordered) * validation_fraction))
    train_end = len(ordered) - validation_size - test_size
    if train_end < min_partition_size:
        raise ValueError("Tập train quá nhỏ sau khi chia dữ liệu theo thời gian.")

    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end : train_end + validation_size].copy()
    test = ordered.iloc[train_end + validation_size :].copy()
    return train, validation, test
