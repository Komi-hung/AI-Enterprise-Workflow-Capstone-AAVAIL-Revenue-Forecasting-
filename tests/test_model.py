"""
tests/test_model.py
===================
Unit tests for the AAVAIL model training & inference pipeline.

Test coverage:
  - Data loading from processed CSV.
  - compute_metrics correctness.
  - Candidate pipeline construction.
  - _LastValueBaseline fit/predict.
  - cross_validate_models returns correct structure.
  - select_best_model picks minimum RMSE.
  - train_final_model fits and returns predictions.
  - save_model / load_model round-trip.
  - predict() helper — happy path and missing feature error.
  - Full run_training_pipeline integration test.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))
sys.path.insert(0, str(_PROJ_ROOT / "src"))

from model import (
    FEATURE_COLS,
    TARGET_COL,
    _LastValueBaseline,
    _build_candidate_pipelines,
    compute_metrics,
    cross_validate_models,
    load_model,
    predict,
    run_training_pipeline,
    save_model,
    select_best_model,
    train_final_model,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def synthetic_agg_df() -> pd.DataFrame:
    """
    Build a small synthetic monthly aggregated DataFrame (36 rows)
    matching the schema expected by the model pipeline.
    """
    rng = np.random.default_rng(42)
    n   = 36
    dates = pd.date_range("2022-01-01", periods=n, freq="MS")
    revenue = 500_000 + rng.normal(0, 30_000, n).cumsum()

    df = pd.DataFrame({"date": dates, "total_revenue": revenue})
    df["month"]           = df["date"].dt.month
    df["quarter"]         = df["date"].dt.quarter
    df["month_sin"]       = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]       = np.cos(2 * np.pi * df["month"] / 12)
    df["lag_1"]           = df["total_revenue"].shift(1)
    df["lag_3"]           = df["total_revenue"].shift(3)
    df["rolling_3"]       = df["total_revenue"].shift(1).rolling(3).mean()
    df["rolling_std_3"]   = df["total_revenue"].shift(1).rolling(3).std()
    df.dropna(inplace=True)
    return df.reset_index(drop=True)


@pytest.fixture(scope="module")
def X_y(synthetic_agg_df):
    """Extract feature matrix and target vector."""
    X = synthetic_agg_df[FEATURE_COLS].values
    y = synthetic_agg_df[TARGET_COL].values
    return X, y


@pytest.fixture(scope="module")
def tmp_model_dir(synthetic_agg_df):
    """
    Train a model, save it to a temp directory, and yield the dir path.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "models"
        data_dir  = Path(tmpdir) / "data" / "processed"
        data_dir.mkdir(parents=True)

        # Write the synthetic CSV
        synthetic_agg_df.to_csv(data_dir / "aavail_monthly.csv", index=False)

        run_training_pipeline(data_dir=data_dir, model_dir=model_dir)
        yield model_dir


# ===========================================================================
# Tests — compute_metrics
# ===========================================================================

class TestComputeMetrics:

    def test_perfect_prediction_r2_is_one(self):
        y = np.array([100.0, 200.0, 300.0])
        metrics = compute_metrics(y, y.copy())
        assert metrics["MAE"]  == pytest.approx(0.0, abs=1e-9)
        assert metrics["RMSE"] == pytest.approx(0.0, abs=1e-9)
        assert metrics["R2"]   == pytest.approx(1.0, abs=1e-9)

    def test_constant_prediction_mae(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([200.0, 200.0, 200.0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["MAE"] == pytest.approx(200 / 3, rel=1e-6)

    def test_metrics_keys_present(self):
        y = np.ones(10) * 500_000
        metrics = compute_metrics(y, y)
        for key in ("MAE", "RMSE", "MAPE", "R2"):
            assert key in metrics

    def test_mape_correct(self):
        y_true = np.array([100.0, 200.0, 400.0])
        y_pred = np.array([110.0, 210.0, 420.0])
        metrics = compute_metrics(y_true, y_pred)
        # Expected MAPE = mean([10/100, 10/200, 20/400]) * 100 = mean([0.1, 0.05, 0.05]) * 100 = 6.67%
        assert metrics["MAPE"] == pytest.approx(100 / 15, rel=1e-4)

    def test_zero_std_y_pred_r2_not_nan(self):
        """Constant predictions should not produce NaN R²."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        metrics = compute_metrics(y_true, y_pred)
        assert not np.isnan(metrics["R2"])


# ===========================================================================
# Tests — _LastValueBaseline
# ===========================================================================

class TestLastValueBaseline:

    def test_fit_returns_self(self):
        baseline = _LastValueBaseline()
        X = np.ones((5, 8))
        y = np.arange(5, dtype=float)
        result = baseline.fit(X, y)
        assert result is baseline

    def test_predict_uses_lag1_column(self):
        """Baseline should return the lag_1 column (index 0 of features)."""
        baseline = _LastValueBaseline()
        X = np.zeros((5, 8))
        y = np.ones(5) * 100
        baseline.fit(X, y)

        X_test = np.zeros((3, 8))
        X_test[:, 0] = [10.0, 20.0, 30.0]   # lag_1 values
        preds = baseline.predict(X_test)
        np.testing.assert_array_almost_equal(preds, [10.0, 20.0, 30.0])

    def test_predict_fallback_on_nan(self):
        """NaN lag_1 should fall back to the last training value."""
        baseline = _LastValueBaseline()
        X_train = np.zeros((5, 8))
        y_train = np.array([100, 200, 300, 400, 500], dtype=float)
        baseline.fit(X_train, y_train)

        X_test      = np.zeros((2, 8))
        X_test[0, 0] = np.nan
        X_test[1, 0] = 250.0
        preds = baseline.predict(X_test)
        assert preds[0] == pytest.approx(500.0)
        assert preds[1] == pytest.approx(250.0)


# ===========================================================================
# Tests — Candidate Pipelines
# ===========================================================================

class TestBuildCandidatePipelines:

    def test_all_expected_models_present(self):
        pipelines = _build_candidate_pipelines()
        for name in ("Ridge", "RandomForest", "GradientBoosting", "Baseline_LastValue"):
            assert name in pipelines, f"Missing pipeline: {name}"

    def test_pipelines_fit_and_predict(self, X_y):
        X, y = X_y
        for name, pipeline in _build_candidate_pipelines().items():
            pipeline.fit(X, y)
            preds = pipeline.predict(X)
            assert preds.shape == (len(y),), f"{name}: unexpected prediction shape"
            assert np.all(np.isfinite(preds)), f"{name}: non-finite predictions"


# ===========================================================================
# Tests — Cross-Validation
# ===========================================================================

class TestCrossValidateModels:

    def test_returns_all_model_names(self, X_y):
        X, y = X_y
        results = cross_validate_models(X, y, n_splits=3)
        for name in ("Ridge", "RandomForest", "GradientBoosting", "Baseline_LastValue"):
            assert name in results

    def test_metrics_are_finite(self, X_y):
        X, y = X_y
        results = cross_validate_models(X, y, n_splits=3)
        for name, metrics in results.items():
            for key, val in metrics.items():
                assert np.isfinite(val) or np.isnan(val), f"{name}.{key} = {val}"

    def test_rmse_is_non_negative(self, X_y):
        X, y = X_y
        results = cross_validate_models(X, y, n_splits=3)
        for name, metrics in results.items():
            assert metrics["RMSE"] >= 0, f"{name} has negative RMSE"


# ===========================================================================
# Tests — Model Selection
# ===========================================================================

class TestSelectBestModel:

    def test_selects_minimum_rmse(self):
        cv_results = {
            "ModelA": {"MAE": 100, "RMSE": 500, "MAPE": 5.0, "R2": 0.8},
            "ModelB": {"MAE": 200, "RMSE": 300, "MAPE": 3.0, "R2": 0.9},
            "ModelC": {"MAE": 150, "RMSE": 700, "MAPE": 7.0, "R2": 0.7},
        }
        best = select_best_model(cv_results)
        assert best == "ModelB"

    def test_single_model_returns_that_model(self):
        cv_results = {
            "OnlyModel": {"MAE": 50, "RMSE": 80, "MAPE": 2.0, "R2": 0.95},
        }
        assert select_best_model(cv_results) == "OnlyModel"


# ===========================================================================
# Tests — Train Final Model
# ===========================================================================

class TestTrainFinalModel:

    def test_ridge_trains_without_error(self, X_y):
        X, y = X_y
        model = train_final_model(X, y, "Ridge")
        preds = model.predict(X)
        assert preds.shape == (len(y),)

    def test_unknown_model_name_raises(self, X_y):
        X, y = X_y
        with pytest.raises(ValueError, match="Unknown model name"):
            train_final_model(X, y, "NonExistentModel")

    def test_gradient_boosting_trains(self, X_y):
        X, y = X_y
        model = train_final_model(X, y, "GradientBoosting")
        assert model is not None

    def test_random_forest_trains(self, X_y):
        X, y = X_y
        model = train_final_model(X, y, "RandomForest")
        assert model is not None


# ===========================================================================
# Tests — Save / Load Round-trip
# ===========================================================================

class TestSaveLoadModel:

    def test_save_creates_files(self, X_y):
        X, y = X_y
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "models"
            model  = train_final_model(X, y, "Ridge")
            cv_res = {"Ridge": {"MAE": 1, "RMSE": 2, "MAPE": 3, "R2": 0.9}}
            save_model(model, "Ridge", cv_res, FEATURE_COLS, model_dir)

            assert (model_dir / "aavail_best_model.joblib").exists()
            assert (model_dir / "model_metadata.json").exists()

    def test_load_returns_correct_model_name(self, X_y):
        X, y = X_y
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "models"
            model  = train_final_model(X, y, "Ridge")
            cv_res = {"Ridge": {"MAE": 1, "RMSE": 2, "MAPE": 3, "R2": 0.9}}
            save_model(model, "Ridge", cv_res, FEATURE_COLS, model_dir)

            loaded_model, metadata = load_model(model_dir)
            assert metadata["model_name"] == "Ridge"
            preds = loaded_model.predict(X)
            assert preds.shape == (len(y),)

    def test_load_missing_model_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_model(Path(tmpdir) / "empty_dir")


# ===========================================================================
# Tests — predict() Helper
# ===========================================================================

class TestPredictHelper:

    @pytest.fixture()
    def fitted_ridge(self, X_y):
        X, y = X_y
        return train_final_model(X, y, "Ridge")

    def test_predict_returns_float(self, fitted_ridge):
        fv = {col: 0.5 for col in FEATURE_COLS}
        result = predict(fitted_ridge, fv)
        assert isinstance(result, float)

    def test_predict_missing_feature_raises_value_error(self, fitted_ridge):
        fv = {"month": 6}   # many features missing
        with pytest.raises(ValueError, match="Missing features"):
            predict(fitted_ridge, fv)

    def test_predict_is_deterministic(self, fitted_ridge):
        fv = {col: float(i) for i, col in enumerate(FEATURE_COLS)}
        p1 = predict(fitted_ridge, fv)
        p2 = predict(fitted_ridge, fv)
        assert p1 == pytest.approx(p2)


# ===========================================================================
# Tests — Full Training Pipeline Integration
# ===========================================================================

class TestRunTrainingPipeline:

    def test_pipeline_creates_model_file(self, tmp_model_dir):
        assert (tmp_model_dir / "aavail_best_model.joblib").exists()

    def test_pipeline_creates_metadata_file(self, tmp_model_dir):
        meta_path = tmp_model_dir / "model_metadata.json"
        assert meta_path.exists()
        with open(meta_path) as fh:
            meta = json.load(fh)
        assert "model_name" in meta
        assert "feature_cols" in meta

    def test_pipeline_model_is_loadable(self, tmp_model_dir):
        model, metadata = load_model(tmp_model_dir)
        assert model is not None
        assert metadata["model_name"] in (
            "Ridge", "RandomForest", "GradientBoosting", "Baseline_LastValue"
        )

    def test_pipeline_best_model_can_predict(self, tmp_model_dir, X_y):
        X, _ = X_y
        model, _ = load_model(tmp_model_dir)
        preds = model.predict(X)
        assert preds.shape[0] == X.shape[0]
        assert np.all(np.isfinite(preds))
