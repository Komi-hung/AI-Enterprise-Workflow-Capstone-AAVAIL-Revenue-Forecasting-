"""
tests/test_api.py
=================
Unit tests for the AAVAIL Flask inference API endpoints.

Test coverage:
  - POST /predict         (happy path, missing fields, bad JSON)
  - POST /predict/batch   (happy path, empty list, missing key)
  - GET  /health
  - GET  /metrics
  - GET  /drift
  - GET  /history
  - GET  /model-info

All tests use a pre-trained mock model so they are fully self-contained
and do NOT require the full data pipeline to have been run.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup — allow imports from project root
# ---------------------------------------------------------------------------
_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))
sys.path.insert(0, str(_PROJ_ROOT / "src"))

# We patch the model load before importing app to avoid FS side effects
# in environments without a pre-trained model.
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def tmp_dirs():
    """
    Create temporary model and log directories, train a tiny Ridge model,
    and return the paths dict.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "models"
        log_dir   = Path(tmpdir) / "logs"
        model_dir.mkdir()
        log_dir.mkdir()

        # Train a minimal model
        X = np.random.default_rng(0).random((20, 8))
        y = np.random.default_rng(0).random(20) * 500_000

        model = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
        model.fit(X, y)
        joblib.dump(model, model_dir / "aavail_best_model.joblib")

        metadata = {
            "model_name":  "Ridge",
            "trained_at":  "2024-01-01T00:00:00Z",
            "feature_cols": [
                "month", "quarter", "month_sin", "month_cos",
                "lag_1", "lag_3", "rolling_3", "rolling_std_3",
            ],
            "cv_results":        {},
            "best_cv_metrics":   {"MAE": 1000, "RMSE": 1200, "MAPE": 5.0, "R2": 0.85},
        }
        (model_dir / "model_metadata.json").write_text(json.dumps(metadata))

        yield {"model_dir": model_dir, "log_dir": log_dir, "tmp": Path(tmpdir)}


@pytest.fixture()
def client(tmp_dirs):
    """
    Provide a Flask test client with model/log paths patched to tmp directories.
    """
    import src.app as app_module
    import src.logger as logger_module

    # Reset globals
    app_module._model    = None
    app_module._metadata = None
    app_module._pred_logger = None
    logger_module.reset_logger()

    # Patch directory constants inside the app module
    with (
        patch.object(app_module, "_MODEL_DIR", tmp_dirs["model_dir"]),
        patch.object(app_module, "_LOG_DIR",   tmp_dirs["log_dir"]),
    ):
        app_module.app.config["TESTING"] = True
        with app_module.app.test_client() as c:
            yield c

        # Clean up globals after each test
        app_module._model    = None
        app_module._metadata = None
        app_module._pred_logger = None
        logger_module.reset_logger()


@pytest.fixture()
def valid_payload():
    """A complete, valid prediction payload."""
    return {
        "month":          6,
        "quarter":        2,
        "month_sin":      0.866,
        "month_cos":     -0.5,
        "lag_1":          580_000.0,
        "lag_3":          550_000.0,
        "rolling_3":      560_000.0,
        "rolling_std_3":  15_000.0,
    }


# ===========================================================================
# Tests — POST /predict
# ===========================================================================

class TestPredictEndpoint:

    def test_predict_happy_path(self, client, valid_payload):
        """Valid request returns 200 with required response keys."""
        resp = client.post(
            "/predict",
            data=json.dumps(valid_payload),
            content_type="application/json",
        )
        assert resp.status_code == 200, resp.get_data(as_text=True)
        body = resp.get_json()
        assert "prediction"     in body
        assert "request_id"     in body
        assert "model_name"     in body
        assert "latency_ms"     in body
        assert "drift_detected" in body
        assert isinstance(body["prediction"], float)
        assert body["prediction"] > 0

    def test_predict_missing_features_defaults_to_zero(self, client):
        """Partial payload should be accepted; missing features default to 0."""
        resp = client.post(
            "/predict",
            data=json.dumps({"month": 3, "lag_1": 400_000}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert "prediction" in body

    def test_predict_empty_body_returns_400(self, client):
        """Empty JSON body should return 400."""
        resp = client.post(
            "/predict",
            data="",
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_predict_response_has_valid_latency(self, client, valid_payload):
        """Latency should be a non-negative float."""
        resp = client.post(
            "/predict",
            data=json.dumps(valid_payload),
            content_type="application/json",
        )
        body = resp.get_json()
        assert body["latency_ms"] >= 0

    def test_predict_request_id_is_unique(self, client, valid_payload):
        """Each request should produce a unique request_id."""
        ids = set()
        for _ in range(5):
            resp = client.post(
                "/predict",
                data=json.dumps(valid_payload),
                content_type="application/json",
            )
            ids.add(resp.get_json()["request_id"])
        assert len(ids) == 5


# ===========================================================================
# Tests — POST /predict/batch
# ===========================================================================

class TestPredictBatchEndpoint:

    def test_batch_happy_path(self, client, valid_payload):
        """Batch request with two observations returns two results."""
        payload = {"observations": [valid_payload, valid_payload]}
        resp = client.post(
            "/predict/batch",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["total"] == 2
        assert len(body["results"]) == 2

    def test_batch_result_has_required_keys(self, client, valid_payload):
        """Each batch result should contain index, prediction, request_id."""
        payload = {"observations": [valid_payload]}
        resp = client.post(
            "/predict/batch",
            data=json.dumps(payload),
            content_type="application/json",
        )
        result = resp.get_json()["results"][0]
        assert "index"      in result
        assert "prediction" in result
        assert "request_id" in result

    def test_batch_missing_observations_key_returns_400(self, client):
        """Payload without 'observations' key should return 400."""
        resp = client.post(
            "/predict/batch",
            data=json.dumps({"data": []}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_batch_empty_observations_returns_400(self, client):
        """Empty observations list should return 400."""
        resp = client.post(
            "/predict/batch",
            data=json.dumps({"observations": []}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_batch_indices_are_sequential(self, client, valid_payload):
        """Batch result indices should be 0-based sequential integers."""
        n = 4
        payload = {"observations": [valid_payload] * n}
        resp = client.post(
            "/predict/batch",
            data=json.dumps(payload),
            content_type="application/json",
        )
        indices = [r["index"] for r in resp.get_json()["results"]]
        assert indices == list(range(n))


# ===========================================================================
# Tests — GET /health
# ===========================================================================

class TestHealthEndpoint:

    def test_health_returns_ok(self, client):
        """Health endpoint should return 200 and status OK."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "OK"
        assert body["model_loaded"] is True

    def test_health_contains_model_name(self, client):
        """Health response should include model_name."""
        resp = client.get("/health")
        body = resp.get_json()
        assert "model_name" in body
        assert body["model_name"] == "Ridge"


# ===========================================================================
# Tests — GET /metrics
# ===========================================================================

class TestMetricsEndpoint:

    def test_metrics_returns_200(self, client, valid_payload):
        """After at least one prediction, metrics endpoint should return 200."""
        # Make a prediction first
        client.post("/predict", data=json.dumps(valid_payload), content_type="application/json")
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_has_prediction_count(self, client, valid_payload):
        """Metrics response should include total_predictions."""
        client.post("/predict", data=json.dumps(valid_payload), content_type="application/json")
        body = client.get("/metrics").get_json()
        assert "total_predictions" in body
        assert body["total_predictions"] >= 1


# ===========================================================================
# Tests — GET /drift
# ===========================================================================

class TestDriftEndpoint:

    def test_drift_returns_200(self, client):
        """Drift endpoint should always return 200."""
        resp = client.get("/drift")
        assert resp.status_code == 200

    def test_drift_response_has_list(self, client):
        """Drift response should contain drift_alerts list."""
        body = client.get("/drift").get_json()
        assert "drift_alerts" in body
        assert isinstance(body["drift_alerts"], list)


# ===========================================================================
# Tests — GET /history
# ===========================================================================

class TestHistoryEndpoint:

    def test_history_returns_200(self, client):
        resp = client.get("/history")
        assert resp.status_code == 200

    def test_history_default_limit(self, client, valid_payload):
        """Make several predictions and check they appear in history."""
        for _ in range(3):
            client.post("/predict", data=json.dumps(valid_payload), content_type="application/json")
        body = client.get("/history").get_json()
        assert body["total"] >= 3

    def test_history_n_param(self, client, valid_payload):
        """n=1 should return at most 1 record."""
        client.post("/predict", data=json.dumps(valid_payload), content_type="application/json")
        body = client.get("/history?n=1").get_json()
        assert len(body["history"]) <= 1


# ===========================================================================
# Tests — GET /model-info
# ===========================================================================

class TestModelInfoEndpoint:

    def test_model_info_returns_200(self, client):
        resp = client.get("/model-info")
        assert resp.status_code == 200

    def test_model_info_has_feature_cols(self, client):
        body = client.get("/model-info").get_json()
        assert "feature_cols" in body
        assert isinstance(body["feature_cols"], list)
        assert len(body["feature_cols"]) == 8
