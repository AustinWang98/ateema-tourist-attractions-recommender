"""Load warehouse seeds and rebuild the BigQuery transformation pipeline.

This utility never stores credentials. It uses the identity already configured
for the Google Cloud ``bq`` CLI.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = ROOT / "warehouse"
SQL_PROJECT_PLACEHOLDER = "__PROJECT_ID__"
SQL_DATASET_PLACEHOLDER = "__DATASET_ID__"
DEFAULT_DATASET = "analytics_demo"

PIPELINE = (
    "01_location_dim.sql",
    "02_location_category_bridge_final.sql",
    "03_location_category_dim.sql",
    "04_user_location_category_events.sql",
    "05_user_location_full_features.sql",
)

SEEDS = (
    (
        "category_mapping",
        WAREHOUSE / "seeds" / "category_mapping.csv",
        WAREHOUSE / "schemas" / "category_mapping.json",
    ),
    (
        "location_category_bridge",
        WAREHOUSE / "seeds" / "location_category_bridge.csv",
        WAREHOUSE / "schemas" / "location_category_bridge.json",
    ),
    (
        "location_dim",
        ROOT / "data" / "location_dim.csv",
        WAREHOUSE / "schemas" / "location_dim.json",
    ),
)


def _run(command: list[str], *, label: str, input_text: str | None = None) -> None:
    print(f"==> {label}")
    subprocess.run(command, check=True, input=input_text, text=True)


def _target_sql(path: Path, project: str, dataset: str) -> str:
    sql = path.read_text(encoding="utf-8")
    source = f"`{SQL_PROJECT_PLACEHOLDER}.{SQL_DATASET_PLACEHOLDER}."
    target = f"`{project}.{dataset}."
    return sql.replace(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the ChicagoDoes BigQuery warehouse from versioned SQL."
    )
    parser.add_argument("--project", required=True, help="Destination Google Cloud project ID")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Destination dataset ID")
    parser.add_argument("--location", default="US", help="BigQuery dataset/job location")
    parser.add_argument(
        "--create-dataset",
        action="store_true",
        help="Create the destination dataset before loading data",
    )
    parser.add_argument(
        "--load-seeds",
        action="store_true",
        help="Replace the three non-personal dimension tables from repository CSVs",
    )
    parser.add_argument(
        "--include-candidates",
        action="store_true",
        help="Also build the optional full user-by-location candidate table",
    )
    parser.add_argument(
        "--refresh-only",
        action="store_true",
        help="Run only the event and feature steps after new GA4 exports arrive",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate SQL without creating tables; dataset/seed writes are skipped",
    )
    args = parser.parse_args()

    if args.refresh_only and args.load_seeds:
        parser.error("--refresh-only and --load-seeds cannot be used together")

    if shutil.which("bq") is None:
        parser.error("Google Cloud's `bq` command is required and was not found on PATH")

    if args.create_dataset and not args.dry_run:
        _run(
            [
                "bq",
                "--quiet",
                f"--project_id={args.project}",
                f"--location={args.location}",
                "mk",
                "--dataset",
                f"{args.project}:{args.dataset}",
            ],
            label=f"Create dataset {args.project}:{args.dataset}",
        )

    if args.load_seeds and not args.dry_run:
        for table, csv_path, schema_path in SEEDS:
            _run(
                [
                    "bq",
                    "--quiet",
                    f"--project_id={args.project}",
                    f"--location={args.location}",
                    "load",
                    "--replace",
                    "--source_format=CSV",
                    "--skip_leading_rows=1",
                    f"{args.project}:{args.dataset}.{table}",
                    str(csv_path),
                    str(schema_path),
                ],
                label=f"Load seed table {table}",
            )

    if args.refresh_only:
        steps = [
            "04_user_location_category_events.sql",
            "05_user_location_full_features.sql",
        ]
    else:
        steps = list(PIPELINE)
    if args.load_seeds:
        # location_dim was loaded directly from data/location_dim.csv.
        steps.remove("01_location_dim.sql")
    if args.include_candidates:
        steps.append("06_candidate_user_location_table.sql")

    for filename in steps:
        sql_path = WAREHOUSE / "sql" / filename
        command = [
            "bq",
            "query",
            "--quiet",
            f"--project_id={args.project}",
            f"--location={args.location}",
            "--use_legacy_sql=false",
        ]
        if args.dry_run:
            command.append("--dry_run=true")
        sql = _target_sql(sql_path, args.project, args.dataset)
        _run(command, label=f"Run {filename}", input_text=sql)

    print("Warehouse pipeline complete.")


if __name__ == "__main__":
    main()
