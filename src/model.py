"""
AAVAIL Revenue Forecasting — Model Training, Evaluation & Serving Pipeline
===========================================================================
Responsibilities:
  1. Load processed data from data/processed/.
  2. Train multiple candidate models (Baseline, Ridge, RandomForest, GradientBoosting).
  3. Evaluate models with time-series cross-validation.
  4. Select & persist the best model to models/.
  5. Expose load / predict helpers used by the Flask API.

Usage (CLI):
    python src/model.py --data-dir data/processed --model-dir models

Author : AAVAIL MLOps Team
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_FILENAME = "aavail_best_model.joblib"
METADATA_FILENAME = "model_metadata.json"
FEATURE_COLS: List[str] = [
    "month", "quarter", "month_sin", "month_cos",
    "lag_1", "lag_3", "rolling_3", "rolling_std_3",
]
TARGET_COL = "total_revenue"
N_SPLITS = 5          # TimeSeriesSplit folds
RANDOM_STATE = 42


# ===========================================================================
# Section 1 — Data Loading
# ===========================================================================

def load_processed_data(data_dir: Path) -> pd.DataFrame:
    """
    Load the aggregated monthly CSV and drop rows with NaN features.

    Parameters
    ----------
    data_dir : Path to processed data directory (must contain aavail_monthly.csv).

    Returns
    -------
    DataFrame ready for modelling (date, features, target).

    Raises
    ------
    FileNotFoundError  if the CSV is missing.
    """
    csv_path = data_dir / "aavail_monthly.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {csv_path}. "
            "Run data_ingestion.py first."
        )

    df = pd.read_csv(csv_path, parse_dates=["date"])
    before = len(df)
    df.dropna(subset=FEATURE_COLS + [TARGET_COL], inplace=True)
    after = len(df)
    if before != after:
        log.warning("Dropped %d rows with NaN features/target.", before - after)

    log.info("Loaded %d rows from %s", after, csv_path)
    return df.sort_values("date").reset_index(drop=True)


# ===========================================================================
# Section 2 — Metrics Helper
# ===========================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute MAE, RMSE, MAPE and R² between *y_true* and *y_pred*.

    Returns a dict of metric_name → float value.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    # MAPE — guard against zero targets
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


# ===========================================================================
# Section 3 — Model Definitions
# ===========================================================================

def _build_candidate_pipelines() -> Dict[str, Pipeline]:
    """
    Return a dict of name → sklearn Pipeline objects covering four candidate
    models, each prefixed with a StandardScaler for numerical stability.
    """
    candidates: Dict[str, Pipeline] = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Baseline_LastValue": _LastValueBaseline(),
    }
    return candidates


class _LastValueBaseline:
    """
    Naïve persistence baseline: predict y[t] = y[t-1] (i.e. lag_1 feature).

    This is a sklearn-compatible estimator that simply reads the "lag_1"
    feature column, so it can participate in the same evaluation loop.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LastValueBaseline":
        # Store the last seen value for inference when lag_1 is unavailable.
        self._last_value = float(y[-1]) if len(y) else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # lag_1 is the first column (see FEATURE_COLS order)
        lag1 = X[:, 0]
        fallback = np.where(np.isnan(lag1), self._last_value, lag1)
        return fallback

    # sklearn-style get/set params
    def get_params(self, deep: bool = True) -> dict:
        return {}

    def set_params(self, **params: Any) -> "_LastValueBaseline":
        return self


# ===========================================================================
# Section 4 — Cross-Validated Training & Evaluation
# ===========================================================================

def cross_validate_models(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_SPLITS,
) -> Dict[str, Dict[str, float]]:
    """
    Run TimeSeriesSplit cross-validation for all candidate models.

    Parameters
    ----------
    X        : Feature matrix (numpy array).
    y        : Target vector.
    n_splits : Number of CV folds.

    Returns
    -------
    Dict mapping model_name → averaged metric dict.
    """
    candidates = _build_candidate_pipelines()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results: Dict[str, Dict[str, List[float]]] = {
        name: {"MAE": [], "RMSE": [], "MAPE": [], "R2": []}
        for name in candidates
    }

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        for name, model in candidates.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            m = compute_metrics(y_val, preds)
            for k, v in m.items():
                cv_results[name][k].append(v)

        log.debug("Fold %d complete.", fold_idx + 1)

    # Average across folds
    averaged: Dict[str, Dict[str, float]] = {}
    for name, metrics in cv_results.items():
        averaged[name] = {k: float(np.nanmean(v)) for k, v in metrics.items()}
        log.info(
            "%-20s | MAE=%.0f | RMSE=%.0f | MAPE=%.2f%% | R²=%.4f",
            name,
            averaged[name]["MAE"],
            averaged[name]["RMSE"],
            averaged[name]["MAPE"],
            averaged[name]["R2"],
        )

    return averaged


# ===========================================================================
# Section 5 — Model Selection & Final Training
# ===========================================================================

def select_best_model(cv_results: Dict[str, Dict[str, float]]) -> str:
    """
    Return the name of the model with the lowest average RMSE.

    Parameters
    ----------
    cv_results : Output of cross_validate_models().

    Returns
    -------
    Name string of the winning model.
    """
    best_name = min(cv_results, key=lambda n: cv_results[n]["RMSE"])
    log.info("Best model selected: %s  (RMSE=%.0f)", best_name, cv_results[best_name]["RMSE"])
    return best_name


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
) -> Any:
    """
    Retrain the chosen model on the full dataset and return the fitted object.

    Parameters
    ----------
    X          : Full feature matrix.
    y          : Full target vector.
    model_name : Key from _build_candidate_pipelines().

    Returns
    -------
    Fitted sklearn pipeline (or _LastValueBaseline).
    """
    candidates = _build_candidate_pipelines()
    if model_name not in candidates:
        raise ValueError(f"Unknown model name: {model_name}")
    model = candidates[model_name]
    model.fit(X, y)
    log.info("Final model '%s' trained on %d samples.", model_name, len(y))
    return model


# ===========================================================================
# Section 6 — Persistence
# ===========================================================================

def save_model(
    model: Any,
    model_name: str,
    cv_results: Dict[str, Dict[str, float]],
    feature_cols: List[str],
    model_dir: Path,
) -> Path:
    """
    Persist the fitted model with joblib and write a JSON metadata sidecar.

    Parameters
    ----------
    model       : Fitted sklearn-compatible model.
    model_name  : Human-readable model name.
    cv_results  : Full CV result dict (all candidates).
    feature_cols: Ordered list of feature column names.
    model_dir   : Directory to write artefacts.

    Returns
    -------
    Path to the saved .joblib file.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / MODEL_FILENAME
    joblib.dump(model, model_path)
    log.info("Model saved → %s", model_path)

    # Metadata
    metadata = {
        "model_name": model_name,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "feature_cols": feature_cols,
        "cv_results": cv_results,
        "best_cv_metrics": cv_results[model_name],
    }
    meta_path = model_dir / METADATA_FILENAME
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    log.info("Metadata saved → %s", meta_path)

    return model_path


def load_model(model_dir: Path) -> Tuple[Any, dict]:
    """
    Load the persisted model and its metadata from *model_dir*.

    Returns
    -------
    (fitted_model, metadata_dict)

    Raises
    ------
    FileNotFoundError  if the model file is absent.
    """
    model_path = model_dir / MODEL_FILENAME
    meta_path  = model_dir / METADATA_FILENAME

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            "Run model.py first to train and save a model."
        )

    model = joblib.load(model_path)
    metadata: dict = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

    log.info("Model loaded from %s", model_path)
    return model, metadata


# ===========================================================================
# Section 7 — Prediction Helper
# ===========================================================================

def predict(
    model: Any,
    feature_vector: Dict[str, Any],
    feature_cols: List[str] | None = None,
) -> float:
    """
    Run inference for a single observation.

    Parameters
    ----------
    model          : Fitted sklearn-compatible model / pipeline.
    feature_vector : Dict mapping feature name → value.
    feature_cols   : Ordered feature column names.  Defaults to FEATURE_COLS.

    Returns
    -------
    Predicted total_revenue (float).

    Raises
    ------
    ValueError  if any required feature is missing.
    """
    cols = feature_cols or FEATURE_COLS
    missing = set(cols) - set(feature_vector.keys())
    if missing:
        raise ValueError(f"Missing features for prediction: {missing}")

    X_row = np.array([[feature_vector[c] for c in cols]])
    prediction = float(model.predict(X_row)[0])
    return prediction


# ===========================================================================
# Section 8 — Full Training Pipeline
# ===========================================================================

def run_training_pipeline(
    data_dir: Path,
    model_dir: Path,
) -> Tuple[Any, dict, Dict[str, Dict[str, float]]]:
    """
    Orchestrate the end-to-end training workflow:
      load → split features/target → CV evaluation → select best →
      retrain on full set → save.

    Returns
    -------
    (fitted_model, metadata, cv_results)
    """
    log.info("=== Loading processed data ===")
    df = load_processed_data(data_dir)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    log.info("=== Cross-validating all candidate models (%d folds) ===", N_SPLITS)
    cv_results = cross_validate_models(X, y)

    log.info("=== Selecting best model ===")
    best_name = select_best_model(cv_results)

    log.info("=== Training final model on full dataset ===")
    final_model = train_final_model(X, y, best_name)

    log.info("=== Evaluating final model on full dataset ===")
    preds = final_model.predict(X)
    full_metrics = compute_metrics(y, preds)
    log.info(
        "Full-dataset metrics | MAE=%.0f | RMSE=%.0f | MAPE=%.2f%% | R²=%.4f",
        full_metrics["MAE"], full_metrics["RMSE"],
        full_metrics["MAPE"], full_metrics["R2"],
    )

    log.info("=== Saving model & metadata ===")
    save_model(final_model, best_name, cv_results, FEATURE_COLS, model_dir)

    _, metadata = load_model(model_dir)   # re-load to verify
    return final_model, metadata, cv_results


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AAVAIL Model Training Pipeline"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed"),
        help="Directory containing processed CSV files.",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("models"),
        help="Directory to save trained model artefacts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_training_pipeline(data_dir=args.data_dir, model_dir=args.model_dir)
