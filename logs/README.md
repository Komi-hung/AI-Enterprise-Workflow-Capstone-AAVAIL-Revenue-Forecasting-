This directory stores structured JSONL log files produced by the inference API.

Files are created automatically when the Flask API processes requests:
  - predictions.jsonl  : One record per inference call (request, features,
                         prediction, latency, drift status).
  - health_checks.jsonl: One record per /health endpoint call.
  - drift_alerts.jsonl : Drift detection events (written when z-score > 3.0).

Log record schema (predictions.jsonl):
{
  "timestamp":       "2024-06-01T12:00:00.000000Z",
  "request_id":      "uuid-v4",
  "model_name":      "GradientBoosting",
  "feature_vector":  { "month": 6, "lag_1": 580000.0, ... },
  "prediction":      612345.6789,
  "ground_truth":    null,
  "latency_ms":      3.21,
  "drift_flags":     [],
  "drift_detected":  false,
  "metadata":        {}
}
