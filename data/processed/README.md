This directory stores processed CSV files output by the ingestion pipeline.

Files are generated automatically by running:
    python src/data_ingestion.py

Output files:
  - aavail_detail.csv   : Per (country, channel, date) engineered features.
  - aavail_monthly.csv  : Aggregated monthly total revenue time-series
                          with lag, rolling and cyclic features.

aavail_monthly.csv columns:
  date, total_revenue, month, quarter, month_sin, month_cos,
  lag_1, lag_3, rolling_3, rolling_std_3
