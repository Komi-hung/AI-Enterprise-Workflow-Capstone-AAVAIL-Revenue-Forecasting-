"""
AAVAIL Revenue Forecasting — Prediction & Monitoring Logger
============================================================
Responsibilities:
  1. Log every prediction request with its inputs and output.
  2. Persist structured log entries as JSONL to logs/predictions.jsonl.
  3. Detect data-drift by comparing incoming feature distributions
     against a reference baseline stored in models/.
  4. Expose a health-check logging function for API readiness probes.

Author : AAVAIL MLOps Team
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Module-level stdlib logger (for operational messages about the logger itself)
# ---------------------------------------------------------------------------
_internal_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_LOG_DIR = Path("logs")
PREDICTION_LOG  = "predictions.jsonl"
HEALTH_LOG      = "health_checks.jsonl"
DRIFT_LOG       = "drift_alerts.jsonl"

# Drift thresholds
Z_SCORE_THRESHOLD   = 3.0    # flag if |z| > 3
MISSING_RATE_LIMIT  = 0.10   # flag if > 10 % of feature values are NaN


# ===========================================================================
# Section 1 — JSONL Writer (thread-safe)
# ===========================================================================

class _JsonlWriter:
    """
    Append-only JSONL log writer with a threading lock for concurrency safety.

    Parameters
    ----------
    log_path : Full path (including filename) to the .jsonl file.
    """

    def __init__(self, log_path: Path) -> None:
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        """Append *record* as one JSON line to the log file."""
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")

    def read_all(self) -> List[dict]:
        """Read and return all records from the log file."""
        if not self._path.exists():
            return []
        records = []
        with open(self._path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        _internal_log.warning("Skipping malformed log line: %s", line)
        return records


# ===========================================================================
# Section 2 — Prediction Logger
# ===========================================================================

class PredictionLogger:
    """
    Structured prediction logger for the AAVAIL inference API.

    Each call to :meth:`log_prediction` writes one JSONL record containing:
      - timestamp (ISO-8601 UTC)
      - request_id (unique per call)
      - feature_vector (input)
      - prediction (model output)
      - model_name
      - latency_ms (end-to-end inference latency)
      - optional drift_flag

    Parameters
    ----------
    log_dir    : Directory where log files are written.
    model_name : Human-readable model identifier (embedded in every record).
    """

    def __init__(
        self,
        log_dir: Path | str = DEFAULT_LOG_DIR,
        model_name: str = "unknown",
    ) -> None:
        log_dir = Path(log_dir)
        self._prediction_writer = _JsonlWriter(log_dir / PREDICTION_LOG)
        self._drift_writer      = _JsonlWriter(log_dir / DRIFT_LOG)
        self._health_writer     = _JsonlWriter(log_dir / HEALTH_LOG)
        self._model_name        = model_name
        self._reference_stats: Optional[Dict[str, Dict[str, float]]] = None
        _internal_log.info(
            "PredictionLogger initialised. Log directory: %s", log_dir
        )

    # -----------------------------------------------------------------------
    # 2.1  Reference baseline (for drift detection)
    # -----------------------------------------------------------------------

    def set_reference_stats(self, stats: Dict[str, Dict[str, float]]) -> None:
        """
        Store a reference feature distribution for drift detection.

        Parameters
        ----------
        stats : dict mapping feature_name → {"mean": float, "std": float}.
        """
        self._reference_stats = stats
        _internal_log.info(
            "Reference statistics set for %d features.", len(stats)
        )

    def compute_reference_stats(
        self,
        X_train: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute and store mean/std from training data.

        Parameters
        ----------
        X_train       : 2-D numpy array of training features.
        feature_names : Ordered list of feature column names.

        Returns
        -------
        Stats dict (also stored internally).
        """
        stats: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(feature_names):
            col = X_train[:, i]
            stats[name] = {
                "mean": float(np.nanmean(col)),
                "std":  float(np.nanstd(col)),
            }
        self.set_reference_stats(stats)
        return stats

    # -----------------------------------------------------------------------
    # 2.2  Drift detection
    # -----------------------------------------------------------------------

    def _detect_drift(
        self,
        feature_vector: Dict[str, Any],
        request_id: str,
    ) -> List[str]:
        """
        Compare incoming features against reference statistics using z-score.

        Returns a list of flagged feature names (empty if no drift).
        """
        if self._reference_stats is None:
            return []

        flagged: List[str] = []
        for feat, ref in self._reference_stats.items():
            value = feature_vector.get(feat)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                flagged.append(feat)
                continue
            if ref["std"] > 0:
                z = abs((float(value) - ref["mean"]) / ref["std"])
                if z > Z_SCORE_THRESHOLD:
                    flagged.append(feat)

        if flagged:
            drift_record = {
                "timestamp":  datetime.utcnow().isoformat() + "Z",
                "request_id": request_id,
                "flagged_features": flagged,
                "z_threshold": Z_SCORE_THRESHOLD,
                "severity": "HIGH" if len(flagged) > 2 else "MEDIUM",
            }
            self._drift_writer.write(drift_record)
            _internal_log.warning(
                "Drift detected [%s] — flagged: %s", request_id, flagged
            )

        return flagged

    # -----------------------------------------------------------------------
    # 2.3  Core logging method
    # -----------------------------------------------------------------------

    def log_prediction(
        self,
        request_id: str,
        feature_vector: Dict[str, Any],
        prediction: float,
        latency_ms: float,
        ground_truth: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        Write one prediction record to the JSONL log.

        Parameters
        ----------
        request_id     : Unique identifier for the request (UUID or similar).
        feature_vector : Dict of input features used for inference.
        prediction     : Model output (predicted revenue).
        latency_ms     : Inference latency in milliseconds.
        ground_truth   : Actual revenue, if known at log time (may be None).
        metadata       : Arbitrary extra metadata to embed.

        Returns
        -------
        The dict that was written to the log.
        """
        drift_flags = self._detect_drift(feature_vector, request_id)

        record = {
            "timestamp":     datetime.utcnow().isoformat() + "Z",
            "request_id":    request_id,
            "model_name":    self._model_name,
            "feature_vector": feature_vector,
            "prediction":    round(prediction, 4),
            "ground_truth":  ground_truth,
            "latency_ms":    round(latency_ms, 3),
            "drift_flags":   drift_flags,
            "drift_detected": len(drift_flags) > 0,
            "metadata":      metadata or {},
        }
        self._prediction_writer.write(record)
        _internal_log.debug(
            "Logged prediction [%s] → %.2f  (drift=%s)",
            request_id, prediction, len(drift_flags) > 0
        )
        return record

    # -----------------------------------------------------------------------
    # 2.4  Health-check logging
    # -----------------------------------------------------------------------

    def log_health_check(
        self,
        status: str = "OK",
        details: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        Write a health-check event (called by /health endpoint).

        Parameters
        ----------
        status  : "OK" | "DEGRADED" | "UNHEALTHY"
        details : Extra context dict (model version, uptime, etc.)

        Returns
        -------
        The dict written to the log.
        """
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status":    status,
            "details":   details or {},
        }
        self._health_writer.write(record)
        return record

    # -----------------------------------------------------------------------
    # 2.5  Query / analytics helpers
    # -----------------------------------------------------------------------

    def get_prediction_history(self, n_last: int = 100) -> List[dict]:
        """Return the last *n_last* prediction log records."""
        all_records = self._prediction_writer.read_all()
        return all_records[-n_last:]

    def compute_summary_statistics(self) -> Dict[str, Any]:
        """
        Compute aggregate summary of all logged predictions.

        Returns
        -------
        Dict with keys: total_predictions, avg_prediction, std_prediction,
        avg_latency_ms, drift_rate, model_name.
        """
        records = self._prediction_writer.read_all()
        if not records:
            return {"total_predictions": 0, "model_name": self._model_name}

        predictions = [r["prediction"] for r in records if r.get("prediction") is not None]
        latencies   = [r["latency_ms"]  for r in records if r.get("latency_ms")  is not None]
        drift_count = sum(1 for r in records if r.get("drift_detected"))

        return {
            "model_name":       self._model_name,
            "total_predictions": len(records),
            "avg_prediction":   round(float(np.mean(predictions)),  2) if predictions else None,
            "std_prediction":   round(float(np.std(predictions)),   2) if predictions else None,
            "min_prediction":   round(float(np.min(predictions)),   2) if predictions else None,
            "max_prediction":   round(float(np.max(predictions)),   2) if predictions else None,
            "avg_latency_ms":   round(float(np.mean(latencies)),    3) if latencies   else None,
            "drift_rate":       round(drift_count / len(records),   4),
            "drift_count":      drift_count,
        }

    def get_drift_alerts(self) -> List[dict]:
        """Return all drift alert records."""
        return self._drift_writer.read_all()


# ===========================================================================
# Section 3 — Module-level Singleton Factory
# ===========================================================================

_logger_instance: Optional[PredictionLogger] = None
_logger_lock = threading.Lock()


def get_logger(
    log_dir: Path | str = DEFAULT_LOG_DIR,
    model_name: str = "unknown",
) -> PredictionLogger:
    """
    Return (or create) the module-level singleton PredictionLogger.

    Calling this function multiple times returns the same instance, which
    avoids opening duplicate file handles.

    Parameters
    ----------
    log_dir    : Log directory (used only on first call).
    model_name : Model identifier (used only on first call).
    """
    global _logger_instance
    if _logger_instance is None:
        with _logger_lock:
            if _logger_instance is None:
                _logger_instance = PredictionLogger(log_dir, model_name)
    return _logger_instance


def reset_logger() -> None:
    """Reset the singleton (used in unit tests to get a fresh instance)."""
    global _logger_instance
    with _logger_lock:
        _logger_instance = None
