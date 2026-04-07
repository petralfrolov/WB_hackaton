import pandas as pd
import numpy as np
import joblib
import pytest

from ml.prediction import (
    build_feature_cols,
    encode_id_categoricals,
    load_models,
    make_features,
    predict_for_route_timestamp,
)


class DummyModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, df):
        return np.array([self.value])


def test_make_features_adds_expected_columns():
    ts = pd.date_range("2025-05-01", periods=3, freq="30min")
    df = pd.DataFrame({
        "route_id": ["1", "1", "1"],
        "timestamp": ts,
        "target_2h": [1.0, 2.0, 3.0],
        "office_from_id": ["10", "10", "10"],
    })

    out = make_features(df, extended=True)

    # Time & lag features
    assert {"hour", "day_of_week", "halfhour_slot", "target_lag_1"}.issubset(out.columns)
    # Rolling/statistical features
    assert "target_roll_mean_4" in out.columns
    # Deconvolution and route encodings are created even on small data
    assert "deconv_s_t0" in out.columns
    assert "route_target_mean" in out.columns
    # Row count is preserved
    assert len(out) == len(df)


def test_build_feature_cols_excludes_targets_and_timestamp():
    df = pd.DataFrame({
        "target_2h": [0.5],
        "timestamp": pd.to_datetime(["2025-05-01 00:00:00"]),
        "target_step_1": [0.1],
        "f1": [1],
    })
    cols = build_feature_cols(df)
    assert "target_2h" not in cols
    assert "timestamp" not in cols
    assert "target_step_1" not in cols
    assert "f1" in cols


def test_predict_for_route_timestamp_averages_models():
    ts = pd.Timestamp("2025-05-01 12:00:00")
    X_all = pd.DataFrame({
        "route_id": ["42"],
        "timestamp": [ts],
        "feature": [10],
    })
    feature_cols = ["route_id", "feature"]

    models = [
        {
            "target_step_4": DummyModel(1.0),
            "target_step_8": DummyModel(2.0),
            "target_step_12": DummyModel(3.0),
        },
        {
            "target_step_4": DummyModel(3.0),
            "target_step_8": DummyModel(4.0),
            "target_step_12": DummyModel(5.0),
        },
    ]

    result = predict_for_route_timestamp(
        X_all=X_all,
        feature_cols=feature_cols,
        models=models,
        route_id="42",
        timestamp=str(ts),
    )

    assert result["pred_0_2h"] == pytest.approx(2.0)
    assert result["pred_2_4h"] == pytest.approx(3.0)
    assert result["pred_4_6h"] == pytest.approx(4.0)
    # Std dev of [1,3] is 1.0 (population std)
    assert result["pred_0_2h_std"] == pytest.approx(1.0)
    assert result["n_models"] == 2


def test_load_models_requires_all_keys(tmp_path):
    bad = tmp_path / "bad.pkl"
    joblib.dump({"target_step_1": 1}, bad)

    with pytest.raises(KeyError):
        load_models(tmp_path)
