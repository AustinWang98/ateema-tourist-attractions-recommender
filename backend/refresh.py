"""CLI: refresh local CSV files from BigQuery.

Usage:
    python -m backend.refresh                       # uses .env values
    python -m backend.refresh --project X --dataset Y
    python -m backend.refresh --inspect             # only list tables, no fetch

Prereqs:
    1. `pip install -r requirements.txt`
    2. `gcloud auth application-default login`
    3. BQ_PROJECT and BQ_DATASET set in .env (or via flags)

This writes three CSVs into data/:
    data/user_location_features.csv
    data/location_dim.csv
    data/events.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("refresh")


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Refresh local data from BigQuery.")
    ap.add_argument("--project", default=os.getenv("BQ_PROJECT"), help="GCP project id")
    ap.add_argument("--dataset", default=os.getenv("BQ_DATASET"), help="BQ dataset name")
    ap.add_argument("--out", default="data", help="Output directory for CSVs")
    ap.add_argument("--inspect", action="store_true",
                    help="List tables in the dataset and exit (no fetch).")
    args = ap.parse_args()

    if not args.project or not args.dataset:
        ap.error("BQ_PROJECT and BQ_DATASET must be set (env or --flag).")

    # Import lazily so the module loads even if google-cloud-bigquery is absent
    # (e.g. on someone's clean clone before they install the new deps).
    try:
        from backend.sources.bq_source import BQConfig, export_to_csv, list_dataset_tables
    except ImportError as exc:
        print(f"ERROR: {exc}\nRun: pip install -r requirements.txt", file=sys.stderr)
        return 1

    cfg = BQConfig(
        project=args.project,
        dataset=args.dataset,
        table_features=os.getenv("BQ_TABLE_FEATURES", "user_location_full_features"),
        table_location_dim=os.getenv("BQ_TABLE_LOCATION_DIM", "location_dim"),
        table_events=os.getenv("BQ_TABLE_EVENTS", "user_location_category_events"),
    )

    if args.inspect:
        tables = list_dataset_tables(cfg)
        print(f"Tables in {cfg.project}.{cfg.dataset} ({len(tables)}):")
        for t in tables:
            print(f"  - {t}")
        return 0

    print(f"Pulling from {cfg.project}.{cfg.dataset} into {args.out}/ ...")
    try:
        summary = export_to_csv(cfg, out_dir=args.out)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Refresh complete:")
    for logical, info in summary.items():
        print(f"  {logical:<14s} {info['rows']:>6d} rows -> {info['csv']}")
    print("\nIf the server is running, hit POST /api/refresh to reload it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
