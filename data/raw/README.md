This directory stores raw JSON transaction files, one per (country, channel) pair.

Files are generated automatically by running:
    python src/data_ingestion.py

Expected filename pattern: {COUNTRY}_{CHANNEL}.json
Example files: US_direct.json, UK_partner.json, DE_online.json, ...

Each JSON file contains an array of monthly revenue records:
[
  {
    "date":         "2022-01-01",
    "country":      "US",
    "channel":      "direct",
    "revenue_usd":  125432.50,
    "subscribers":  1243,
    "churn_rate":   0.034
  },
  ...
]
