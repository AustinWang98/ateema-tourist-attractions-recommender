# Technical guide

[← Project overview](../README.md)

This guide covers the runnable public demo and the paths into the full
implementation. Synthetic inputs demonstrate the engineering workflow; they
do not reproduce the original capstone metrics.

## Local configuration

Start with the [README setup steps](../README.md#run-locally) and
[`.env.example`](../.env.example). Keep `.env` local and leave `BQ_PROJECT`,
`BQ_DATASET`, and `OPENAI_API_KEY` empty for the credential-free demo.

The application selects data sources in this order:

1. BigQuery when a project and dataset are configured.
2. A compatible local feature cache at `DATA_CSV_PATH`.
3. The public fallback built from `data/demo_events.csv`,
   `data/location_dim.csv`, and `data/locations_geo.csv`.

Existing shell environment variables can affect configuration. On a clean demo
setup, [`/api/health`](http://localhost:8000/api/health) should report
`load_mode: public_demo_fallback`, with 80 users, 350 locations, and 1,920 events.

The original product screenshot is illustrative. The repository preserves
media metadata but does not bundle the original photography and video files.
The optional AI itinerary path needs a separately configured provider; the
application includes a deterministic fallback.

## Verify the public fixture and application

From the repository root with the virtual environment activated:

```bash
python -m pip install pytest
python -m pytest
python -m py_compile backend/main.py backend/data_loader.py backend/recommender.py
```

If Node.js is installed, check the frontend syntax as CI does:

```bash
node --check frontend/app.js
```

To regenerate the deterministic fixture intentionally:

```bash
python scripts/generate_demo_events.py
python -m pytest tests/test_demo_fixture.py
```

The generator rewrites `data/demo_events.csv`. The fixture uses
`synthetic-user-` identities and reserved `.invalid` domains; it contains no
original visitor records.

## Navigate the implementation

| Layer | Main entry points |
| --- | --- |
| Data-source selection and normalization | [Application lifecycle](../backend/main.py) · [Data loader](../backend/data_loader.py) |
| Score blending, cold start, and MMR | [Recommender](../backend/recommender.py) |
| Item co-visitation and user neighbors | [Collaborative models](../backend/collab.py) |
| Session and transition models | [Behavioral signals](../backend/behavior.py) |
| Request/response contracts | [API schemas](../backend/schemas.py) |
| Browser experience | [HTML](../frontend/index.html) · [JavaScript](../frontend/app.js) |
| Temporal holdouts and baseline evaluation | [Evaluation utilities](../backend/evaluation.py) · [Runner](../backend/evaluation_runner.py) |
| Weight experiments | [Randomized search](../backend/weight_search.py) · [Cross-validation](../backend/weight_search_cv.py) |
| Warehouse implementation | [SQL stages](../warehouse/sql/) · [Schemas](../warehouse/schemas/) · [Runner](../warehouse/run_pipeline.py) |

## Run the evaluation workflow

**Use a separate disposable clone for experiments.** The current scripts write
results into tracked paths under `data/`; running them in your primary checkout
will replace some preserved capstone artifacts. They do not have a general
output-directory option.

Create a separate checkout (choose another directory name if this exists):

```bash
git clone https://github.com/AustinWang98/ateema-tourist-attractions-recommender.git chicagodoes-eval
cd chicagodoes-eval
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-ml.txt
```

Run the workflow explicitly against synthetic events:

```bash
python -m backend.evaluation_runner --events data/demo_events.csv
python -m backend.weight_search --events data/demo_events.csv
python -m backend.weight_search_cv --events data/demo_events.csv --k 20 --folds 5
```

Weight search can take substantially longer than the basic demo. For a quick
pipeline check, add `--n-samples 100` to either weight-search command; this is
not equivalent to the full search.

| Command | Files it writes or replaces |
| --- | --- |
| `backend.evaluation_runner` | `data/offline_eval_results.csv` |
| `backend.weight_search` | `data/weight_search_results.csv`, `data/best_weights.json` |
| `backend.weight_search_cv` | `data/weight_cv_results.csv` |

Keep newly generated results labeled as **synthetic-demo experiments**.
Different inputs, cutoffs, and search settings produce different metrics.

### Read the saved results correctly

- [`robust_weights.json`](../data/robust_weights.json) records the original
  NDCG@20 validation summary: 120 returning users, +3.2% relative gain, and
  3/5 folds won.
- [`best_weights.json`](../data/best_weights.json) records a separate NDCG@10
  tuning result and the relevance–diversity comparison with MMR.
- [`weight_cv_results.csv`](../data/weight_cv_results.csv) is a separate saved
  cross-validation run, not the fold-level evidence for the NDCG@20 summary.
- The application uses `WEIGHTS_NEW` and `WEIGHTS_RETURNING` in
  [`recommender.py`](../backend/recommender.py). Saved tuning JSON files are
  not automatically applied at startup.

The demo lets readers inspect and rerun the methods. Reproducing the historical
capstone numbers requires the original authorized dataset and matching
experiment settings, not only the synthetic fixture.

## Connect an approved private BigQuery deployment

The public repository includes the transformations and schemas, not the
original analytics export or a provisioned cloud environment.

Before running the warehouse pipeline, you need:

- An approved Google Cloud project, a billing-enabled BigQuery environment,
  and an identity with the required dataset and query permissions.
- The Google Cloud CLI and appropriate authentication.
- Compatible GA4 `events_*` source tables in the configured dataset, with
  event fields and parameters expected by
  [the event transformation](../warehouse/sql/04_user_location_category_events.sql).

**The local demo CSV is not a GA4 BigQuery export.** Creating an empty dataset
and loading the public dimension seeds alone does not supply the analytics
events needed by the pipeline.

For authorized local development, authenticate the `bq` CLI used by the
warehouse runner, and separately configure Application Default Credentials
for the Python backend. These are distinct authentication contexts.
In hosted environments, prefer an organization-owned service account rather
than a downloaded key. Never commit credentials or private query results.

With source tables already available, adapt the placeholders below:

```bash
python warehouse/run_pipeline.py \
  --project your-project-id \
  --dataset your-dataset-id \
  --load-seeds \
  --include-candidates
```

Omit `--create-dataset` when the dataset already exists, as in this example.
The runner creates or replaces warehouse tables; target a dedicated development
dataset first. Review the [stage-by-stage runbook](../warehouse/README.md) and
[warehouse inventory](../warehouse/INVENTORY.md) before running it.

Then set the matching resources in your private `.env` or deployment secrets:

```text
BQ_PROJECT=your-project-id
BQ_DATASET=your-dataset-id
BQ_TABLE_FEATURES=user_location_full_features
BQ_TABLE_LOCATION_DIM=location_dim
BQ_TABLE_EVENTS=user_location_category_events
```

Restart the application and check `/api/health` to confirm the actual data source.

## API surface

With the server running, open
[FastAPI's interactive documentation](http://localhost:8000/docs) for the
complete schemas and request examples.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Service and data-source status |
| `GET` | `/api/categories` | Available recommendation categories |
| `GET` | `/api/trending?limit=N` | Time-aware trending places |
| `POST` | `/api/recommend` | Ranked and explained recommendations |
| `POST` | `/api/itinerary` | AI-assisted or deterministic itinerary |
| `POST` | `/api/location/info` | Place details and media |
| `POST` | `/api/explain` | Recommendation explanation |
| `POST` | `/api/refine` | Natural-language itinerary refinement |
| `POST` | `/api/outbound/click` | Separate recommendation click log |
| `POST` | `/api/refresh` | Reload configured data sources |

Some endpoints use configured services or write local operational logs.
This table describes the application surface, not an authorization policy for
a production deployment.

## Private handoff

Full operational handoff documents, original datasets, and internal deployment
details are available to authorized stakeholders through the private team
repository. Contact [Yiou (Austin) Wang](mailto:yiouwang@uchicago.edu) to request
access. Do not include credentials or visitor-level data in public issues.
