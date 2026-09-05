# Ateema Tourist Attractions Recommender

A full-stack Chicago attractions recommender with a FastAPI backend and a
browser-based frontend. Visitors can browse popular places immediately or share
their interests to receive personalized recommendations and an itinerary.

## Public data and privacy

Due to data privacy requirements, the public version of this project uses
deterministic synthetic/demo data rather than the original production dataset
used during the capstone. The replacement follows the same schema and exercises
the same application workflows; the system architecture, BigQuery SQL and
schemas, recommendation algorithms, and evaluation tooling remain intact.

The public repository contains no production credentials, visitor identifiers,
session histories, private analytics exports, or internal operating records.
The final academic paper and presentation are published unchanged, preserving
their original technical narrative, authorship, and reported results.

## Internal handoff access

This repository is intended for public technical review, reproducibility, and
academic reference. Authorized project stakeholders who need the complete
operational handoff documentation, original or private data exports, deployment
details, private environment configuration, or internal repository history can
request access to the private internal repository by contacting Austin Wang at
[yiouwang@uchicago.edu](mailto:yiouwang@uchicago.edu). Please do not post
credentials or visitor-level data in public issues.

## Features

- Hybrid ranking with content similarity, popularity, collaborative signals,
  session behavior, trending, and diversity re-ranking
- Guided cold-start preferences for visitors without prior history
- Media-rich recommendation cards and Chicago map coordinates
- Optional OpenAI-compatible itinerary generation with a deterministic fallback
- Synthetic demo data that can be regenerated and audited
- Six-stage BigQuery transformation pipeline with schemas and public seeds
- Leakage-safe offline evaluation, weight search, cross-validation, and MLflow
- Final paper PDF and the team's editable Word manuscript
- Complete final PowerPoint deck, generation source, and visual assets
- Reproducible paper source, figures, and algorithm diagrams

## Repository layout

```text
backend/                    FastAPI application and recommendation engine
frontend/                   Browser interface and static assets
data/
  demo_events.csv           Deterministic synthetic interactions
  location_dim.csv          Public place catalog
  locations_geo.csv         Place coordinates
  location_cards.json       Public card metadata
  *_results.csv             Complete aggregate evaluation and tuning results
warehouse/
  sql/                      Six BigQuery transformation stages
  schemas/                  Eight table schemas
  seeds/                    Non-personal dimension seeds
  run_pipeline.py           Parameterized BigQuery rebuild utility
docs/
  paper/                    Final PDF, Word manuscript, LaTeX, and figures
  presentation/             Final PowerPoint, builder, and assets
scripts/
  generate_demo_events.py   Rebuilds the synthetic fixture
  build_location_cards.py   Rebuilds card metadata
  enrich_geocode_firecrawl.py
  evaluate_engagement_impact.py
tests/                      Unit and privacy checks
```

## Recommendation architecture

The ranker blends TF-IDF content similarity, global popularity and engagement,
item co-visitation, returning-user neighbors, session co-visitation, transition
probabilities, and event-time trending. Maximal Marginal Relevance then balances
relevance against category diversity. Cold-start visitors receive a profile
constructed from selected interests and optional free text. Recommendation
traffic is tagged and separately logged so it can be excluded or down-weighted
during later model refreshes.

The complete implementation remains under `backend/`. The public/private split
changes data sources and identifiers, not the ranking architecture.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn backend.main:app --reload --port 8000
```

Open <http://localhost:8000>.

The default configuration loads `data/demo_events.csv`. Every identity begins
with `synthetic-user-`, every session derives from a synthetic identity, and
every demo URL uses the reserved `.invalid` domain.

Regenerate and verify the fixture:

```bash
python scripts/generate_demo_events.py
python -m pytest tests/test_demo_fixture.py
```

## Configuration

Copy `.env.example` to `.env`. The app works without external credentials.

For optional LLM itineraries, set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and
`OPENAI_MODEL`. Keep real values in the deployment platform's secret manager,
never in source control.

For a private BigQuery deployment, set:

```text
BQ_PROJECT=your-project-id
BQ_DATASET=your-dataset-id
BQ_TABLE_FEATURES=user_location_full_features
BQ_TABLE_LOCATION_DIM=location_dim
BQ_TABLE_EVENTS=user_location_category_events
```

The browser never queries BigQuery directly. The server reads configured tables
at startup and builds its in-memory ranking indexes.

The full parameterized warehouse is documented in
[`warehouse/README.md`](warehouse/README.md). Its SQL retains user/session field
logic and table relationships, while project and dataset identifiers are generic
placeholders. No production rows are included.

## Paper and presentation

The complete capstone materials are available directly in the repository:

- [`docs/paper/ChicagoDoes_Capstone_Paper.pdf`](docs/paper/ChicagoDoes_Capstone_Paper.pdf)
- [`docs/paper/ChicagoDoes_Capstone_Paper.docx`](docs/paper/ChicagoDoes_Capstone_Paper.docx)
- [`docs/presentation/ChicagoDoes_Final_Presentation.pptx`](docs/presentation/ChicagoDoes_Final_Presentation.pptx)

Their source material is included beside them: the full LaTeX manuscript,
paper figures and diagram generator, presentation builder, formula images,
branding assets, and screenshots. These academic deliverables are preserved
unchanged from the team edition.

## Evaluation tooling

The repository retains the temporal holdout evaluator, randomized weight search,
five-fold validation, engagement-policy comparison, and optional MLflow tracking.
The checked-in historical tuning artifacts are aggregate and non-identifying,
including the complete randomized-search results, cross-validation summary,
selected weights, and offline evaluation table. Use an approved private export
under `data/private/` to reproduce the historical measurements, or use the
synthetic fixture to exercise the pipeline without credentials.

```bash
python -m backend.evaluation_runner --events data/demo_events.csv
python -m backend.weight_search --events data/demo_events.csv
python -m backend.weight_search_cv --events data/demo_events.csv
python scripts/evaluate_engagement_impact.py
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Service and data-source status |
| `GET` | `/api/categories` | Available categories |
| `GET` | `/api/trending?limit=N` | Trending places |
| `POST` | `/api/recommend` | Ranked recommendations |
| `POST` | `/api/itinerary` | AI or deterministic itinerary |
| `POST` | `/api/location/info` | Place details |
| `POST` | `/api/explain` | Recommendation explanation |
| `POST` | `/api/refine` | Natural-language itinerary update |
| `POST` | `/api/outbound/click` | Separate recommendation click log |
| `POST` | `/api/refresh` | Reload configured data sources |

## Privacy and security

The checked-in fixture is synthetic and intentionally omits production visitor
fields such as GA4 pseudonymous IDs, account IDs, device fingerprints, location
history, and raw referrers. Private exports belong under `data/private/`, which
is ignored by Git. BigQuery schemas may name these fields because the warehouse
must define them, but the repository contains no production values.

See [PRIVACY.md](PRIVACY.md) for the public-data boundary and
[SECURITY.md](SECURITY.md) for vulnerability reporting and secret handling.

## Development checks

```bash
python -m py_compile backend/main.py backend/data_loader.py backend/recommender.py
node --check frontend/app.js
python -m pytest
```

Optional evaluation and MLflow tooling is installed separately:

```bash
pip install -r requirements-ml.txt
```

## Deployment

The included `Dockerfile`, `Procfile`, and `render.yaml` support common Python
hosting platforms. Deploy the synthetic public edition first, then connect a
private data source only in an access-controlled environment.

## License

No open-source license is currently provided. Copyright remains with the
project owners and contributors.
