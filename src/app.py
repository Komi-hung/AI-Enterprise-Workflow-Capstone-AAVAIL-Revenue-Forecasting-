"""
AAVAIL Revenue Forecasting — Flask Inference API
=================================================
Endpoints:
  POST /predict          → Single-observation revenue prediction.
  POST /predict/batch    → Batch prediction for multiple observations.
  GET  /health           → Liveness / readiness probe.
  GET  /metrics          → Prediction monitoring summary.
  GET  /drift            → Drift alert log.
  GET  /history          → Last N prediction records.

Usage:
    python src/app.py                          # development
    gunicorn -w 4 -b 0.0.0.0:5000 src.app:app # production

Author : AAVAIL MLOps Team
"""

from __future__ import annotations

import os
import time
import uuid
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Path resolution — works whether run from project root or src/
# ---------------------------------------------------------------------------
_BASE_DIR   = Path(__file__).resolve().parent.parent
_MODEL_DIR  = _BASE_DIR / "models"
_DATA_DIR   = _BASE_DIR / "data" / "processed"
_LOG_DIR    = _BASE_DIR / "logs"

import sys
sys.path.insert(0, str(_BASE_DIR / "src"))

from model  import load_model, predict, FEATURE_COLS
from logger import get_logger, reset_logger

# ---------------------------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_app_log = logging.getLogger("aavail.api")

# ---------------------------------------------------------------------------
# Lazy model / logger loading (avoids import-time side effects in tests)
# ---------------------------------------------------------------------------
_model    = None
_metadata = None
_pred_logger = None


def _load_assets() -> None:
    """Load model and initialise the prediction logger (idempotent)."""
    global _model, _metadata, _pred_logger

    if _model is None:
        _model, _metadata = load_model(_MODEL_DIR)
        model_name = _metadata.get("model_name", "unknown")
        reset_logger()   # ensure a fresh singleton
        _pred_logger = get_logger(log_dir=_LOG_DIR, model_name=model_name)
        _app_log.info("Model '%s' loaded and ready.", model_name)


# ===========================================================================
# Helper — Build feature vector from JSON payload
# ===========================================================================

def _extract_features(payload: dict) -> Dict[str, Any]:
    """
    Extract and coerce features from a raw request payload.

    Expected payload keys (all numeric):
        month, quarter, month_sin, month_cos,
        lag_1, lag_3, rolling_3, rolling_std_3

    Missing keys default to 0.0 (with a warning logged).
    """
    features: Dict[str, Any] = {}
    missing = []
    for col in FEATURE_COLS:
        if col in payload:
            features[col] = float(payload[col])
        else:
            features[col] = 0.0
            missing.append(col)

    if missing:
        _app_log.warning("Missing features defaulted to 0: %s", missing)

    return features


# ===========================================================================
# Endpoint — POST /predict
# ===========================================================================

@app.route("/predict", methods=["POST"])
def predict_endpoint() -> Response:
    """
    Predict total revenue for a single observation.

    Request body (JSON):
    --------------------
    {
      "month":          6,
      "quarter":        2,
      "month_sin":      0.866,
      "month_cos":     -0.5,
      "lag_1":          580000.0,
      "lag_3":          550000.0,
      "rolling_3":      560000.0,
      "rolling_std_3":  15000.0
    }

    Response (JSON):
    ----------------
    {
      "request_id":  "uuid",
      "prediction":  612345.67,
      "model_name":  "GradientBoosting",
      "latency_ms":  3.21,
      "drift_detected": false,
      "drift_flags": []
    }
    """
    try:
        _load_assets()
        payload = request.get_json(force=True, silent=True)

        if payload is None:
            return jsonify({"error": "Empty or invalid JSON body."}), 400

        request_id = str(uuid.uuid4())
        features   = _extract_features(payload)

        t0         = time.perf_counter()
        prediction = predict(_model, features, FEATURE_COLS)
        latency_ms = (time.perf_counter() - t0) * 1000

        log_record = _pred_logger.log_prediction(
            request_id=request_id,
            feature_vector=features,
            prediction=prediction,
            latency_ms=latency_ms,
            ground_truth=payload.get("ground_truth"),
            metadata={"source_ip": request.remote_addr},
        )

        return jsonify({
            "request_id":     request_id,
            "prediction":     round(prediction, 2),
            "model_name":     _metadata.get("model_name", "unknown"),
            "latency_ms":     round(latency_ms, 3),
            "drift_detected": log_record["drift_detected"],
            "drift_flags":    log_record["drift_flags"],
        }), 200

    except Exception as exc:  # noqa: BLE001
        _app_log.exception("Prediction failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ===========================================================================
# Endpoint — POST /predict/batch
# ===========================================================================

@app.route("/predict/batch", methods=["POST"])
def predict_batch_endpoint() -> Response:
    """
    Predict revenue for multiple observations in one request.

    Request body (JSON):
    --------------------
    {
      "observations": [
        { "month": 6, "lag_1": 580000, ... },
        { "month": 7, "lag_1": 600000, ... }
      ]
    }

    Response (JSON):
    ----------------
    {
      "results": [
        { "index": 0, "prediction": 612345.67, "request_id": "...", ... },
        ...
      ],
      "total": 2
    }
    """
    try:
        _load_assets()
        payload = request.get_json(force=True, silent=True)

        if not payload or "observations" not in payload:
            return jsonify({"error": "Payload must contain 'observations' list."}), 400

        observations = payload["observations"]
        if not isinstance(observations, list) or len(observations) == 0:
            return jsonify({"error": "'observations' must be a non-empty list."}), 400

        results = []
        for idx, obs in enumerate(observations):
            request_id = str(uuid.uuid4())
            features   = _extract_features(obs)

            t0         = time.perf_counter()
            prediction = predict(_model, features, FEATURE_COLS)
            latency_ms = (time.perf_counter() - t0) * 1000

            log_record = _pred_logger.log_prediction(
                request_id=request_id,
                feature_vector=features,
                prediction=prediction,
                latency_ms=latency_ms,
                ground_truth=obs.get("ground_truth"),
                metadata={"batch_index": idx},
            )

            results.append({
                "index":          idx,
                "request_id":     request_id,
                "prediction":     round(prediction, 2),
                "latency_ms":     round(latency_ms, 3),
                "drift_detected": log_record["drift_detected"],
            })

        return jsonify({"results": results, "total": len(results)}), 200

    except Exception as exc:  # noqa: BLE001
        _app_log.exception("Batch prediction failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ===========================================================================
# Endpoint — GET /health
# ===========================================================================

@app.route("/health", methods=["GET"])
def health_endpoint() -> Response:
    """
    Liveness & readiness probe.

    Returns 200 if the model is loaded and the API is ready.
    Returns 503 if the model has not been loaded yet.
    """
    try:
        _load_assets()
        model_name = _metadata.get("model_name", "unknown")

        details = {
            "model_loaded": True,
            "model_name":   model_name,
            "trained_at":   _metadata.get("trained_at", "N/A"),
        }
        _pred_logger.log_health_check(status="OK", details=details)

        return jsonify({"status": "OK", **details}), 200

    except FileNotFoundError as exc:
        return jsonify({"status": "UNHEALTHY", "error": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        _app_log.exception("Health check failed: %s", exc)
        return jsonify({"status": "UNHEALTHY", "error": str(exc)}), 503


# ===========================================================================
# Endpoint — GET /metrics
# ===========================================================================

@app.route("/metrics", methods=["GET"])
def metrics_endpoint() -> Response:
    """
    Return aggregated monitoring statistics for all logged predictions.

    Response includes: total_predictions, avg_prediction, avg_latency_ms,
    drift_rate, drift_count.
    """
    try:
        _load_assets()
        summary = _pred_logger.compute_summary_statistics()
        return jsonify(summary), 200
    except Exception as exc:  # noqa: BLE001
        _app_log.exception("Metrics retrieval failed: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ===========================================================================
# Endpoint — GET /drift
# ===========================================================================

@app.route("/drift", methods=["GET"])
def drift_endpoint() -> Response:
    """
    Return all drift alert records.
    """
    try:
        _load_assets()
        alerts = _pred_logger.get_drift_alerts()
        return jsonify({"drift_alerts": alerts, "total": len(alerts)}), 200
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


# ===========================================================================
# Endpoint — GET /history
# ===========================================================================

@app.route("/history", methods=["GET"])
def history_endpoint() -> Response:
    """
    Return recent prediction log records.

    Query params:
        n  (int, default=50)  — Number of most-recent records to return.
    """
    try:
        _load_assets()
        n = int(request.args.get("n", 50))
        n = max(1, min(n, 1000))   # clamp to [1, 1000]
        records = _pred_logger.get_prediction_history(n_last=n)
        return jsonify({"history": records, "total": len(records)}), 200
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


# ===========================================================================
# Endpoint — GET /model-info
# ===========================================================================

@app.route("/model-info", methods=["GET"])
def model_info_endpoint() -> Response:
    """
    Return model metadata (training time, feature columns, CV metrics).
    """
    try:
        _load_assets()
        return jsonify(_metadata), 200
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


# ===========================================================================
# Application Entry Point
# ===========================================================================

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    _app_log.info("Starting AAVAIL Inference API on port %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
