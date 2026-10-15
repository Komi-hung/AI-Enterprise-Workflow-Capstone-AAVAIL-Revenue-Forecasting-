"""
AAVAIL Revenue Forecasting — Data Ingestion & Preprocessing Pipeline
=====================================================================
Responsibilities:
  1. Simulate / load raw JSON transaction files per country/channel.
  2. Merge all JSON fragments into a single master DataFrame.
  3. Clean, validate and feature-engineer the dataset.
  4. Persist the processed CSV to data/processed/.

Usage (CLI):
    python src/data_ingestion.py --raw-dir data/raw --out-dir data/processed

Author : AAVAIL MLOps Team
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level logger  (file-level, NOT the prediction logger)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COUNTRIES: List[str] = ["US", "UK", "DE", "FR", "SG", "AU"]
CHANNELS: List[str] = ["direct", "partner", "online", "enterprise"]
REVENUE_BASE: dict = {
    "US": 120_000,
    "UK": 80_000,
    "DE": 70_000,
    "FR": 65_000,
    "SG": 50_000,
    "AU": 45_000,
}
RANDOM_SEED = 42
N_MONTHS = 36          # 3 years of monthly data


# ===========================================================================
# Section 1 — Raw Data Simulation (replaces real data feeds in demo)
# ===========================================================================

def generate_monthly_revenue(
    country: str,
    channel: str,
    base: float,
    n_months: int = N_MONTHS,
    seed: int = RANDOM_SEED,
) -> List[dict]:
    """
    Simulate n_months of monthly revenue records for one country-channel pair.

    Parameters
    ----------
    country  : ISO-2 country code.
    channel  : Subscription / sales channel name.
    base     : Baseline monthly revenue in USD.
    n_months : Number of monthly observations to generate.
    seed     : Random seed for reproducibility.

    Returns
    -------
    List of dicts, each representing one monthly observation.
    """
    rng = np.random.default_rng(seed + hash(country + channel) % 10_000)
    start_date = datetime(2022, 1, 1)
    records = []

    trend_slope = rng.uniform(0.005, 0.015)    # 0.5%–1.5% monthly growth
    seasonality_amp = rng.uniform(0.05, 0.20)  # ±5%–20% seasonal swing

    for i in range(n_months):
        current_date = start_date + timedelta(days=30 * i)
        month = current_date.month

        # Additive components
        trend = base * (1 + trend_slope) ** i
        seasonal = trend * seasonality_amp * np.sin(2 * np.pi * (month - 1) / 12)
        noise = rng.normal(0, base * 0.03)

        revenue = max(0.0, trend + seasonal + noise)

        # Introduce occasional churn dip
        if rng.random() < 0.05:
            revenue *= rng.uniform(0.75, 0.90)

        records.append(
            {
                "date": current_date.strftime("%Y-%m-%d"),
                "country": country,
                "channel": channel,
                "revenue_usd": round(revenue, 2),
                "subscribers": int(revenue // rng.uniform(80, 120)),
                "churn_rate": round(rng.uniform(0.01, 0.08), 4),
            }
        )
    return records


def simulate_raw_json_files(raw_dir: Path, seed: int = RANDOM_SEED) -> List[Path]:
    """
    Write one JSON file per (country, channel) to *raw_dir*.

    Returns the list of paths written.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    for country in COUNTRIES:
        for channel in CHANNELS:
            records = generate_monthly_revenue(
                country=country,
                channel=channel,
                base=REVENUE_BASE[country] * random.uniform(0.8, 1.2),
                n_months=N_MONTHS,
                seed=seed,
            )
            file_path = raw_dir / f"{country}_{channel}.json"
            with open(file_path, "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2)
            paths.append(file_path)
            log.info("Wrote %d records → %s", len(records), file_path)

    log.info("Simulation complete. %d JSON files written to %s", len(paths), raw_dir)
    return paths


# ===========================================================================
# Section 2 — Data Loading & Merging
# ===========================================================================

def load_json_files(raw_dir: Path) -> pd.DataFrame:
    """
    Read all *.json files in *raw_dir* and concatenate into one DataFrame.

    Raises
    ------
    FileNotFoundError  if raw_dir does not exist.
    ValueError         if no JSON files are found.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    json_files = sorted(raw_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {raw_dir}")

    frames: List[pd.DataFrame] = []
    for fp in json_files:
        with open(fp, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        frames.append(pd.DataFrame(data))
        log.debug("Loaded %s  (%d rows)", fp.name, len(data))

    merged = pd.concat(frames, ignore_index=True)
    log.info(
        "Merged %d files → DataFrame shape: %s", len(json_files), merged.shape
    )
    return merged


# ===========================================================================
# Section 3 — Cleaning & Validation
# ===========================================================================

def validate_schema(df: pd.DataFrame) -> None:
    """Assert required columns exist and dtypes are broadly correct."""
    required = {"date", "country", "channel", "revenue_usd", "subscribers", "churn_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in raw data: {missing}")
    log.info("Schema validation passed.")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning rules:
      - Parse dates.
      - Remove duplicates.
      - Clip negative revenues to 0.
      - Drop rows with null revenue.
      - Clip churn_rate to [0, 1].
    """
    df = df.copy()

    # Parse date column
    df["date"] = pd.to_datetime(df["date"])

    # Remove exact duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    if before != after:
        log.warning("Dropped %d duplicate rows.", before - after)

    # Clip / drop invalid revenue values
    null_revenue = df["revenue_usd"].isna().sum()
    if null_revenue:
        log.warning("Dropping %d rows with null revenue_usd.", null_revenue)
    df.dropna(subset=["revenue_usd"], inplace=True)
    df["revenue_usd"] = df["revenue_usd"].clip(lower=0.0)

    # Clip churn rate
    df["churn_rate"] = df["churn_rate"].clip(0.0, 1.0)

    # Clip subscriber count
    df["subscribers"] = df["subscribers"].clip(lower=0)

    log.info("Cleaning complete. Final shape: %s", df.shape)
    return df


# ===========================================================================
# Section 4 — Feature Engineering
# ===========================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based and lag features for modelling.

    New columns
    -----------
    year, month, quarter
        Calendar decomposition.
    revenue_lag_1, revenue_lag_3
        1- and 3-month lagged revenues per (country, channel).
    revenue_rolling_3
        3-month rolling mean.
    revenue_rolling_std_3
        3-month rolling std dev.
    month_sin, month_cos
        Cyclic encoding of month-of-year.
    """
    df = df.sort_values(["country", "channel", "date"]).reset_index(drop=True)

    # Calendar features
    df["year"]    = df["date"].dt.year
    df["month"]   = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    # Cyclic month encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Lag & rolling per group
    grp = df.groupby(["country", "channel"])["revenue_usd"]

    df["revenue_lag_1"]        = grp.shift(1)
    df["revenue_lag_3"]        = grp.shift(3)
    df["revenue_rolling_3"]    = grp.transform(lambda s: s.shift(1).rolling(3).mean())
    df["revenue_rolling_std_3"]= grp.transform(lambda s: s.shift(1).rolling(3).std())

    log.info("Feature engineering complete. Shape: %s", df.shape)
    return df


# ===========================================================================
# Section 5 — Aggregate Monthly Total (used by model pipeline)
# ===========================================================================

def build_aggregated_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse all countries/channels into a single monthly total revenue Series.
    Returns a DataFrame with columns [date, total_revenue, month, quarter,
    month_sin, month_cos, lag_1, lag_3, rolling_3].
    """
    agg = (
        df.groupby("date")["revenue_usd"]
        .sum()
        .reset_index()
        .rename(columns={"revenue_usd": "total_revenue"})
        .sort_values("date")
        .reset_index(drop=True)
    )

    agg["month"]             = agg["date"].dt.month
    agg["quarter"]           = agg["date"].dt.quarter
    agg["month_sin"]         = np.sin(2 * np.pi * agg["month"] / 12)
    agg["month_cos"]         = np.cos(2 * np.pi * agg["month"] / 12)
    agg["lag_1"]             = agg["total_revenue"].shift(1)
    agg["lag_3"]             = agg["total_revenue"].shift(3)
    agg["rolling_3"]         = agg["total_revenue"].shift(1).rolling(3).mean()
    agg["rolling_std_3"]     = agg["total_revenue"].shift(1).rolling(3).std()

    log.info("Aggregated series shape: %s", agg.shape)
    return agg


# ===========================================================================
# Section 6 — Persistence
# ===========================================================================

def save_processed(df: pd.DataFrame, out_dir: Path, filename: str) -> Path:
    """Write DataFrame to CSV in *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.to_csv(out_path, index=False)
    log.info("Saved processed data → %s  (%d rows)", out_path, len(df))
    return out_path


# ===========================================================================
# Section 7 — High-level Pipeline Entry Point
# ===========================================================================

def run_ingestion_pipeline(
    raw_dir: Path,
    out_dir: Path,
    simulate: bool = True,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end ingestion pipeline.

    Parameters
    ----------
    raw_dir  : Directory with raw JSON files (or where they will be simulated).
    out_dir  : Directory to write processed CSVs.
    simulate : If True, generate synthetic raw data before loading.
    seed     : Random seed for simulation.

    Returns
    -------
    detail_df   : Per (country, channel, date) engineered DataFrame.
    agg_df      : Monthly total revenue time-series DataFrame.
    """
    if simulate:
        log.info("=== Simulating raw JSON data ===")
        simulate_raw_json_files(raw_dir, seed=seed)

    log.info("=== Loading & merging JSON files ===")
    raw_df = load_json_files(raw_dir)

    log.info("=== Validating schema ===")
    validate_schema(raw_df)

    log.info("=== Cleaning data ===")
    clean_df = clean_dataframe(raw_df)

    log.info("=== Engineering features ===")
    detail_df = engineer_features(clean_df)

    log.info("=== Building aggregated monthly series ===")
    agg_df = build_aggregated_series(detail_df)

    log.info("=== Saving to disk ===")
    save_processed(detail_df, out_dir, "aavail_detail.csv")
    save_processed(agg_df,    out_dir, "aavail_monthly.csv")

    return detail_df, agg_df


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AAVAIL Data Ingestion & Preprocessing Pipeline"
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw"),
        help="Directory containing (or to receive) raw JSON files.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/processed"),
        help="Output directory for processed CSV files.",
    )
    parser.add_argument(
        "--no-simulate", action="store_true",
        help="Skip data simulation (assume JSON files already present).",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed for simulation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ingestion_pipeline(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        simulate=not args.no_simulate,
        seed=args.seed,
    )
