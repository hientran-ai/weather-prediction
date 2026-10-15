import numpy as np
import pandas as pd
import pytest

from utils.pipeline import (
    build_inference_row,
    build_training_frame,
    chronological_split,
    feature_engineering,
    preprocess,
    validate_weather_data,
)


def make_weather_data(rows=120):
    index = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=rows, freq="D"),
            "tempmax": 30 + index / 100,
            "tempmin": 22 + index / 200,
            "temp": 26 + index / 150,
            "humidity": 75 - index / 500,
            "precip": index % 5,
            "windspeed": 10 + index / 100,
            "sealevelpressure": 1010 + index / 1000,
            "cloudcover": 50 + index / 100,
            "dew": 23 + index / 300,
        }
    )


def test_validation_sorts_dates_and_rejects_duplicates():
    data = make_weather_data(10).sort_values("datetime", ascending=False)
    validated = validate_weather_data(data)
    assert validated["datetime"].is_monotonic_increasing

    duplicated = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="trùng"):
        validate_weather_data(duplicated)


def test_validation_reports_missing_required_columns():
    with pytest.raises(ValueError, match="tempmax"):
        validate_weather_data(make_weather_data(10).drop(columns="tempmax"))


def test_validation_rejects_missing_calendar_days():
    data = make_weather_data(10).drop(index=4)
    with pytest.raises(ValueError, match="không liên tục"):
        validate_weather_data(data)


def test_training_target_is_the_following_day():
    raw = make_weather_data()
    trained = build_training_frame(raw)
    last = trained.iloc[-1]

    assert last["datetime"] == raw.iloc[-2]["datetime"]
    assert last["target"] == pytest.approx(raw.iloc[-1]["tempmax"])


def test_inference_keeps_latest_day_and_recomputes_derived_features():
    raw = make_weather_data()
    prepared = preprocess(raw)
    featured = feature_engineering(prepared, include_target=False)
    assert featured.iloc[-1]["datetime"] == raw.iloc[-1]["datetime"]

    feature_cols = [column for column in featured.columns if column != "datetime"]
    row = build_inference_row(raw, {"tempmax": 40, "tempmin": 20}, feature_cols)

    assert row["temp_range"] == pytest.approx(20)
    assert row["tempmax_diff1"] == pytest.approx(40 - raw.iloc[-2]["tempmax"])
    assert row["tempmax_lag1"] == pytest.approx(raw.iloc[-2]["tempmax"])


def test_chronological_split_preserves_order_and_keeps_all_rows():
    frame = build_training_frame(make_weather_data(240))
    train, validation, test = chronological_split(frame)

    assert len(train) + len(validation) + len(test) == len(frame)
    assert train["datetime"].max() < validation["datetime"].min()
    assert validation["datetime"].max() < test["datetime"].min()


def test_chronological_split_rejects_small_dataset():
    frame = build_training_frame(make_weather_data(50))
    with pytest.raises(ValueError, match="ít nhất"):
        chronological_split(frame)
