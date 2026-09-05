# Ateema Tourist Attractions Recommender

A full-stack Chicago place recommender with a FastAPI backend and a browser
frontend. Visitors can ask for immediate popular picks or provide preferences
for a more personalized list and itinerary.

This public edition runs entirely from a deterministic synthetic event fixture.
It contains no production credentials, visitor identifiers, session histories,
private analytics exports, cloud resource names, or internal operating records.

## Features

- Hybrid ranking with content similarity, popularity, collaborative signals,
  session behavior, trending, and diversity re-ranking
- Guided cold-start preferences for visitors without prior history
- Media-rich recommendation cards and Chicago map coordinates
- Optional OpenAI-compatible itinerary generation with a deterministic fallback
- Synthetic demo data that can be regenerated and audited
- BigQuery support through environment variables for private deployments

## Repository layout

```text
backend/                    FastAPI application and recommendation engine
frontend/                   Browser interface and static assets
data/
  demo_events.csv           Deterministic synthetic interactions
  location_dim.csv          Public place catalog
  locations_geo.csv         Place coordinates
  location_cards.json       Public card metadata
scripts/
  generate_demo_events.py   Rebuilds the synthetic fixture
  build_location_cards.py   Rebuilds card metadata
tests/                      Unit and privacy checks
```

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
is ignored by Git.

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
