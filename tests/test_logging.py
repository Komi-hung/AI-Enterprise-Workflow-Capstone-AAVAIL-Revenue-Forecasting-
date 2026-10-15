"""
tests/test_logging.py
=====================
Unit tests for the AAVAIL prediction & monitoring logger.

Test coverage:
  - _JsonlWriter: write, read_all, thread safety.
  - PredictionLogger: log_prediction record structure.
  - PredictionLogger: reference stats & drift detection.
  - PredictionLogger: log_health_check.
  - PredictionLogger: compute_summary_statistics.
  - PredictionLogger: get_prediction_history (n_last).
  - PredictionLogger: get_drift_alerts.
  - Singleton get_logger / reset_logger.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest

_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))
sys.path.insert(0, str(_PROJ_ROOT / "src"))

from logger import (
    PredictionLogger,
    _JsonlWriter,
    get_logger,
    reset_logger,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def tmp_log_dir(tmp_path):
    """Provide a fresh temporary log directory for each test."""
    return tmp_path / "logs"


@pytest.fixture()
def logger_instance(tmp_log_dir) -> PredictionLogger:
    """Return a PredictionLogger writing to a fresh temp directory."""
    return PredictionLogger(log_dir=tmp_log_dir, model_name="TestModel")


@pytest.fixture()
def sample_features():
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
# Tests — _JsonlWriter
# ===========================================================================

class TestJsonlWriter:

    def test_write_creates_file(self, tmp_path):
        writer = _JsonlWriter(tmp_path / "test.jsonl")
        writer.write({"key": "value"})
        assert (tmp_path / "test.jsonl").exists()

    def test_write_and_read_all_roundtrip(self, tmp_path):
        writer  = _JsonlWriter(tmp_path / "test.jsonl")
        records = [{"id": i, "value": i * 100} for i in range(5)]
        for r in records:
            writer.write(r)
        read_back = writer.read_all()
        assert len(read_back) == 5
        for i, r in enumerate(read_back):
            assert r["id"]    == i
            assert r["value"] == i * 100

    def test_empty_file_returns_empty_list(self, tmp_path):
        writer = _JsonlWriter(tmp_path / "empty.jsonl")
        assert writer.read_all() == []

    def test_read_nonexistent_file_returns_empty_list(self, tmp_path):
        writer = _JsonlWriter(tmp_path / "does_not_exist.jsonl")
        assert writer.read_all() == []

    def test_thread_safety(self, tmp_path):
        """50 concurrent writes should produce exactly 50 records."""
        writer  = _JsonlWriter(tmp_path / "concurrent.jsonl")
        n       = 50
        barrier = threading.Barrier(n)

        def write_record(idx: int) -> None:
            barrier.wait()          # all threads start simultaneously
            writer.write({"thread": idx})

        threads = [threading.Thread(target=write_record, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        records = writer.read_all()
        assert len(records) == n

    def test_write_serialises_datetime(self, tmp_path):
        """Writer should handle datetime-like values via json default=str."""
        from datetime import datetime
        writer = _JsonlWriter(tmp_path / "dt.jsonl")
        writer.write({"ts": datetime(2024, 1, 1, 12, 0, 0)})
        records = writer.read_all()
        assert records[0]["ts"] == "2024-01-01 12:00:00"


# ===========================================================================
# Tests — PredictionLogger.log_prediction
# ===========================================================================

class TestLogPrediction:

    def test_returns_dict_with_required_keys(self, logger_instance, sample_features):
        record = logger_instance.log_prediction(
            request_id="req-001",
            feature_vector=sample_features,
            prediction=612_345.67,
            latency_ms=3.21,
        )
        for key in ("timestamp", "request_id", "model_name", "feature_vector",
                    "prediction", "latency_ms", "drift_flags", "drift_detected"):
            assert key in record, f"Missing key: {key}"

    def test_prediction_value_is_rounded(self, logger_instance, sample_features):
        record = logger_instance.log_prediction(
            request_id="req-002",
            feature_vector=sample_features,
            prediction=999_999.9999,
            latency_ms=1.0,
        )
        # prediction should be rounded to 4 decimal places
        assert round(record["prediction"], 4) == record["prediction"]

    def test_model_name_in_record(self, logger_instance, sample_features):
        record = logger_instance.log_prediction(
            request_id="req-003",
            feature_vector=sample_features,
            prediction=500_000.0,
            latency_ms=2.0,
        )
        assert record["model_name"] == "TestModel"

    def test_ground_truth_stored_when_provided(self, logger_instance, sample_features):
        record = logger_instance.log_prediction(
            request_id="req-004",
            feature_vector=sample_features,
            prediction=500_000.0,
            latency_ms=2.0,
            ground_truth=510_000.0,
        )
        assert record["ground_truth"] == 510_000.0

    def test_ground_truth_none_when_not_provided(self, logger_instance, sample_features):
        record = logger_instance.log_prediction(
            request_id="req-005",
            feature_vector=sample_features,
            prediction=500_000.0,
            latency_ms=2.0,
        )
        assert record["ground_truth"] is None

    def test_records_persisted_to_disk(self, logger_instance, sample_features, tmp_log_dir):
        logger_instance.log_prediction(
            request_id="req-persist",
            feature_vector=sample_features,
            prediction=600_000.0,
            latency_ms=4.0,
        )
        pred_file = tmp_log_dir / "predictions.jsonl"
        assert pred_file.exists()
        lines = pred_file.read_text().strip().split("\n")
        assert any("req-persist" in line for line in lines)


# ===========================================================================
# Tests — Drift Detection
# ===========================================================================

class TestDriftDetection:

    def test_no_reference_stats_no_drift(self, logger_instance, sample_features):
        record = logger_instance.log_prediction(
            request_id="drift-none",
            feature_vector=sample_features,
            prediction=500_000.0,
            latency_ms=1.0,
        )
        assert record["drift_detected"] is False
        assert record["drift_flags"]   == []

    def test_drift_detected_when_far_from_reference(self, tmp_log_dir, sample_features):
        logger = PredictionLogger(log_dir=tmp_log_dir, model_name="DriftTest")
        # Set reference stats such that lag_1 is "normal" at mean=500, std=10
        logger.set_reference_stats({
            "lag_1": {"mean": 500.0, "std": 10.0},
        })
        # Send lag_1 = 580_000 → z ≈ 57_950 → far above threshold
        record = logger.log_prediction(
            request_id="drift-yes",
            feature_vector={"lag_1": 580_000.0},
            prediction=600_000.0,
            latency_ms=1.0,
        )
        assert record["drift_detected"] is True
        assert "lag_1" in record["drift_flags"]

    def test_no_drift_when_within_threshold(self, tmp_log_dir):
        logger = PredictionLogger(log_dir=tmp_log_dir, model_name="NoDrift")
        logger.set_reference_stats({
            "lag_1": {"mean": 580_000.0, "std": 50_000.0},
        })
        # z = (580_000 - 580_000) / 50_000 = 0 → no drift
        record = logger.log_prediction(
            request_id="no-drift",
            feature_vector={"lag_1": 580_000.0},
            prediction=600_000.0,
            latency_ms=1.0,
        )
        assert record["drift_detected"] is False

    def test_compute_reference_stats_from_training_data(self, tmp_log_dir):
        logger       = PredictionLogger(log_dir=tmp_log_dir, model_name="StatsTest")
        feature_names = ["f1", "f2"]
        X_train       = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        stats         = logger.compute_reference_stats(X_train, feature_names)
        assert stats["f1"]["mean"] == pytest.approx(3.0)
        assert stats["f2"]["mean"] == pytest.approx(4.0)
        assert stats["f1"]["std"]  == pytest.approx(np.std([1, 3, 5]))

    def test_drift_alerts_written_to_disk(self, tmp_log_dir):
        logger = PredictionLogger(log_dir=tmp_log_dir, model_name="AlertTest")
        logger.set_reference_stats({
            "lag_1": {"mean": 100.0, "std": 1.0},
        })
        logger.log_prediction(
            request_id="alert-001",
            feature_vector={"lag_1": 999_999.0},  # massive z-score
            prediction=100.0,
            latency_ms=0.5,
        )
        drift_file = tmp_log_dir / "drift_alerts.jsonl"
        assert drift_file.exists()
        alerts = json.loads(drift_file.read_text().strip().split("\n")[0])
        assert "flagged_features" in alerts


# ===========================================================================
# Tests — Health Check Logging
# ===========================================================================

class TestHealthCheckLogging:

    def test_health_check_returns_dict(self, logger_instance):
        record = logger_instance.log_health_check(status="OK", details={"uptime_s": 100})
        assert record["status"]  == "OK"
        assert "timestamp"       in record

    def test_health_check_written_to_disk(self, logger_instance, tmp_log_dir):
        logger_instance.log_health_check(status="DEGRADED")
        health_file = tmp_log_dir / "health_checks.jsonl"
        assert health_file.exists()

    def test_health_check_custom_status(self, logger_instance):
        record = logger_instance.log_health_check(status="UNHEALTHY")
        assert record["status"] == "UNHEALTHY"


# ===========================================================================
# Tests — Summary Statistics
# ===========================================================================

class TestSummaryStatistics:

    def test_empty_logger_returns_zero_predictions(self, logger_instance):
        summary = logger_instance.compute_summary_statistics()
        assert summary["total_predictions"] == 0

    def test_summary_after_predictions(self, logger_instance, sample_features):
        for i in range(5):
            logger_instance.log_prediction(
                request_id=f"sum-{i}",
                feature_vector=sample_features,
                prediction=500_000.0 + i * 1_000,
                latency_ms=float(i),
            )
        summary = logger_instance.compute_summary_statistics()
        assert summary["total_predictions"] == 5
        assert "avg_prediction" in summary
        assert "avg_latency_ms" in summary
        assert "drift_rate"     in summary

    def test_avg_prediction_correct(self, tmp_log_dir, sample_features):
        logger = PredictionLogger(log_dir=tmp_log_dir, model_name="AvgTest")
        for val in [100.0, 200.0, 300.0]:
            logger.log_prediction(
                request_id=f"avg-{val}",
                feature_vector=sample_features,
                prediction=val,
                latency_ms=1.0,
            )
        summary = logger.compute_summary_statistics()
        assert summary["avg_prediction"] == pytest.approx(200.0)


# ===========================================================================
# Tests — get_prediction_history
# ===========================================================================

class TestGetPredictionHistory:

    def test_returns_last_n_records(self, logger_instance, sample_features):
        for i in range(10):
            logger_instance.log_prediction(
                request_id=f"hist-{i}",
                feature_vector=sample_features,
                prediction=float(i * 10_000),
                latency_ms=1.0,
            )
        history = logger_instance.get_prediction_history(n_last=5)
        assert len(history) == 5

    def test_returns_most_recent_records(self, logger_instance, sample_features):
        for i in range(10):
            logger_instance.log_prediction(
                request_id=f"order-{i}",
                feature_vector=sample_features,
                prediction=float(i),
                latency_ms=1.0,
            )
        history = logger_instance.get_prediction_history(n_last=3)
        # Last 3 predictions should have predictions 7, 8, 9
        preds = [r["prediction"] for r in history]
        assert 9.0 in preds

    def test_n_larger_than_total_returns_all(self, logger_instance, sample_features):
        for i in range(3):
            logger_instance.log_prediction(
                request_id=f"small-{i}",
                feature_vector=sample_features,
                prediction=float(i),
                latency_ms=1.0,
            )
        history = logger_instance.get_prediction_history(n_last=1000)
        assert len(history) >= 3


# ===========================================================================
# Tests — Singleton Factory
# ===========================================================================

class TestSingletonFactory:

    def test_reset_then_get_returns_new_instance(self, tmp_log_dir):
        reset_logger()
        a = get_logger(log_dir=tmp_log_dir, model_name="A")
        b = get_logger(log_dir=tmp_log_dir, model_name="B")   # same instance
        assert a is b
        reset_logger()
        c = get_logger(log_dir=tmp_log_dir, model_name="C")
        assert a is not c

    def test_singleton_is_thread_safe(self, tmp_log_dir):
        """Multiple threads calling get_logger should all receive the same object."""
        reset_logger()
        instances = []
        lock      = threading.Lock()

        def fetch() -> None:
            inst = get_logger(log_dir=tmp_log_dir, model_name="ThreadTest")
            with lock:
                instances.append(inst)

        threads = [threading.Thread(target=fetch) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(inst is instances[0] for inst in instances)
        reset_logger()
