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

The third non-personal seed, the 350-place catalog, is
[`data/location_dim.csv`](../data/location_dim.csv). Raw GA4 exports,
pseudonymous-user tables, query results, credentials, and private aggregates
are not published.

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
| 6 | `candidate_user_location_table` | Optional user-by-place candidate grid | No; offline analysis only |

Stages 4 through 6 operate on pseudonymous behavioral data. The public
repository includes schemas and transformations, not production rows.

## Prerequisites

The warehouse expects a billing-enabled BigQuery environment containing
compatible GA4 daily tables named `events_YYYYMMDD`. The local
`data/demo_events.csv` file is not a BigQuery GA4 export and cannot replace
those source tables.

Install the Google Cloud CLI, authenticate the `bq` command with an approved
identity, and use a dedicated development dataset first. The runner executes
`CREATE OR REPLACE TABLE` statements.

For a new environment:

1. Create or select an organization-owned project and enable BigQuery.
2. Link an authorized GA4 property so daily `events_*` exports reach the target
   dataset, or provision compatible authorized source tables.
3. Load the public dimension seeds and run the derived-table pipeline after
   the event source exists.

## Build or validate the tables

For an existing dataset with its `events_*` source already present:

```bash
python warehouse/run_pipeline.py \
  --project your-project-id \
  --dataset your-dataset-id \
  --load-seeds \
  --include-candidates
```

Omit `--include-candidates` for the smaller live-application build. The
application does not read the candidate grid.

`--create-dataset` is available only when the named dataset does not yet exist;
it is not an idempotent flag. Creating an empty dataset does not provide the
required GA4 source tables.

To validate query parsing and referenced schemas without writing destination
tables, add `--dry-run`. The referenced source tables must still exist.

After a new analytics export arrives, rebuild the event and feature layers:

```bash
python warehouse/run_pipeline.py \
  --project your-project-id \
  --dataset your-dataset-id \
  --refresh-only
```

`--refresh-only` rebuilds the event and user-location feature layers. Rebuild
the optional candidate table separately when an offline workflow needs it.

## Connect the application

Configure the private runtime environment after the tables are ready:

```text
BQ_PROJECT=your-project-id
BQ_DATASET=your-dataset-id
BQ_TABLE_FEATURES=user_location_full_features
BQ_TABLE_LOCATION_DIM=location_dim
BQ_TABLE_EVENTS=user_location_category_events
```

The Python backend uses Application Default Credentials, which are separate
from the credentials used by the `bq` CLI. Attach a dedicated, least-privilege
service account in hosted environments. It needs permission to run BigQuery
jobs and read the three application input tables.

Restart the application and inspect `GET /api/health`. `load_mode` should report
`bigquery` and the location, user, and event counts should be plausible for the
selected environment.

For an authorized local cache after a warehouse refresh:

```bash
python -m backend.refresh --out data/private
```

The resulting pseudonymous aggregates remain restricted. The public
`.gitignore`, `.gcloudignore`, and `.dockerignore` exclude the private data
directory and common export paths, but operators must still inspect every
deployment context and staged change.

## Operating boundary

No scheduled-query configuration is included. A deployment owner must decide
how and when to orchestrate refreshes, monitor data quality, validate location
matching, and cut over traffic. Exact production resource names, identities,
billing/ownership details, and operational history remain in the private
handoff documentation.
