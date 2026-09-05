# BigQuery Warehouse Template

This directory preserves the complete warehouse structure used by the
recommender while removing production resource identifiers and table contents.
The six SQL stages, eight schemas, non-personal dimension seeds, and executable
pipeline runner remain available for technical review and reproducibility.

## Contents

```text
warehouse/
├── INVENTORY.md
├── README.md
├── run_pipeline.py
├── seeds/
│   ├── category_mapping.csv
│   └── location_category_bridge.csv
├── schemas/
│   └── *.json
└── sql/
    ├── 01_location_dim.sql
    ├── 02_location_category_bridge_final.sql
    ├── 03_location_category_dim.sql
    ├── 04_user_location_category_events.sql
    ├── 05_user_location_full_features.sql
    └── 06_candidate_user_location_table.sql
```

The SQL uses `__PROJECT_ID__.__DATASET_ID__` placeholders.
`run_pipeline.py` substitutes the project and dataset supplied on the command
line before submitting a query.

## Warehouse layers

| Order | Table | Purpose | Live input |
| --- | --- | --- | --- |
| Seed | `location_dim` | Canonical place identifiers and names | Yes |
| Seed | `category_mapping` | Canonical category mapping | Indirectly |
| Seed | `location_category_bridge` | Place-to-category relationships | Indirectly |
| 2 | `location_category_bridge_final` | Cleaned relationship bridge | Indirectly |
| 3 | `location_category_dim` | Enriched category row per place | Indirectly |
| 4 | `user_location_category_events` | Parsed and weighted analytics events | Yes |
| 5 | `user_location_full_features` | User-place features and global priors | Yes |
| 6 | `candidate_user_location_table` | Optional user-by-place candidate grid | No |

Stages 4 through 6 operate on pseudonymous behavioral data. The public
repository includes schemas and transformations, not production rows.

## Rebuild

Install the Google Cloud CLI, authenticate with an approved identity, and run:

```bash
python warehouse/run_pipeline.py \
  --project your-project-id \
  --dataset analytics_demo \
  --create-dataset \
  --load-seeds \
  --include-candidates
```

Omit `--include-candidates` for the smaller production path. To validate the SQL
without creating tables, add `--dry-run`.

After a new analytics export arrives, rebuild the event and feature layers:

```bash
python warehouse/run_pipeline.py \
  --project your-project-id \
  --dataset analytics_demo \
  --refresh-only
```

Use a dedicated least-privilege service account for deployments. Do not commit
service-account files, credentials, query results, or visitor-level exports.
