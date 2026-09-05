# Ateema Tourist Attractions Recommender

**From behavioral signals to a personalized day in Chicago.**

An end-to-end recommendation system built for Ateema's ChicagoDoes catalog. It
turns GA4-style interactions and curated place data into ranked, diverse
recommendations, then delivers them through a FastAPI backend, a browser-based
product experience, and an optional AI itinerary layer.

[![CI](https://github.com/AustinWang98/ateema-tourist-attractions-recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/AustinWang98/ateema-tourist-attractions-recommender/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![BigQuery](https://img.shields.io/badge/Google_BigQuery-Warehouse-4285F4?logo=googlebigquery&logoColor=white)
![Public data](https://img.shields.io/badge/Public_Data-Synthetic-2E7D32)

[Read the final paper](docs/paper/ChicagoDoes_Capstone_Paper.pdf) ·
[Open the presentation](docs/presentation/ChicagoDoes_Final_Presentation.pptx) ·
[Explore the BigQuery warehouse](warehouse/README.md) ·
[Run it locally](#run-locally)

![ChicagoDoes Recommender landing page with personalization controls and a 350-place catalog](docs/readme/product-overview.jpg)

*Capstone product interface. The public repository runs the same product and
ranking pipeline with deterministic synthetic/demo data.*

> **Public release note:** Due to data privacy requirements, the public version
> of this project uses synthetic/demo data rather than the original production
> dataset used during the capstone. The substitution changes the data source,
> not the system's complexity: the BigQuery SQL and schemas, ranking algorithms,
> evaluation tooling, API, frontend, paper, and presentation remain available.

## Why this project stands out

- **End-to-end scope:** data modeling, feature engineering, recommendation
  logic, API design, frontend delivery, evaluation, and deployment live in one
  reproducible repository.
- **Real recommender-system constraints:** the design addresses sparse implicit
  feedback, anonymous and cold-start visitors, short sessions, popularity bias,
  temporal behavior, and repetitive top-K results.
- **Measured product tradeoffs:** relevance is evaluated alongside coverage and
  diversity instead of optimizing a single offline score.
- **Responsible public release:** the technical implementation stays complete
  while credentials and visitor-level production records remain private.

## Project at a glance

| Area | What is included |
| --- | --- |
| Product | Guided preference flow, immediate popular picks, recommendation explanations, rich place cards, and day itineraries |
| Catalog | 350 Chicago places with canonical IDs, categories, coordinates, media metadata, and map links |
| Data engineering | Six-stage BigQuery transformation pipeline, eight JSON schemas, public dimension seeds, and a parameterized runner |
| Ranking | Six-signal hybrid score, separate new/returning-user weights, cold-start profiles, and MMR diversity re-ranking |
| Evaluation | Temporal holdout, baseline comparison, randomized weight search, five-fold validation, diversity analysis, and optional MLflow tracking |
| Engineering | Python 3.11, FastAPI, pandas, scikit-learn, vanilla JavaScript/CSS, Docker, Render configuration, and GitHub Actions |
| Public fixture | 1,920 deterministic events across 80 synthetic users, exercising the same 350-place application surface |

## System architecture

The system carries information from analytics events to warehouse features,
ranking signals, diversified results, and finally the user-facing application.
The browser never queries BigQuery directly.

![Architecture from GA4-style events through BigQuery, hybrid ranking, FastAPI, and the frontend](docs/paper/figures/architecture.png)

The end-to-end flow is:

1. Parse and qualify GA4-style interaction events.
2. Build user-location features and catalog dimensions in BigQuery.
3. Generate content, popularity, collaborative, session, transition, and
   trending signals.
4. Blend signals with weights that change for new and returning visitors.
5. Apply Maximal Marginal Relevance (MMR) to balance relevance and variety.
6. Return explainable recommendations and an optional itinerary through the
   FastAPI application.

## How the recommender works

| Signal | Role in the final ranking |
| --- | --- |
| Content similarity | TF-IDF cosine similarity between a visitor profile and location-category tokens |
| Popularity | Leakage-safe distinct-user and engagement priors |
| Item-item collaborative filtering | Sparse co-visitation similarity between places |
| User-user neighbors | Category-profile similarity for returning visitors |
| Session and transition behavior | Same-session co-visitation plus observed previous-to-next movement |
| Trending | Recent engagement rate compared with an earlier time window |
| MMR re-ranking | Diversifies the final list so one category does not dominate |

Cold-start visitors are not forced into a popularity-only experience. Their
selected interests, traveler type, vibe, and optional free text create a TF-IDF
pseudo-profile and a behavioral archetype that can seed collaborative scoring.

The optional LLM is intentionally a presentation layer: deterministic ranking
selects the places first, and the model turns that approved pool into readable
itinerary text. If no API key is configured, the application returns a
deterministic itinerary instead.

## Evaluation highlights

The values below are preserved aggregate results from the original capstone
evaluation. The synthetic public fixture makes the workflow runnable, but it is
not presented as production evidence.

| Experiment | Recorded result |
| --- | --- |
| Randomized weight search | 4,004 evaluated configurations retained for audit and comparison |
| Five-fold robust validation | Held-out NDCG@20 increased from 0.2479 to 0.2557 (+3.2%); the tuned configuration won 3 of 5 folds |
| MMR diversity study | Intra-list diversity increased from 0.2769 to 0.6038, with the relevance tradeoff explicitly recorded (NDCG@10: 0.2559 to 0.2402) |

Explore the evidence in
[`weight_search_results.csv`](data/weight_search_results.csv),
[`weight_cv_results.csv`](data/weight_cv_results.csv),
[`best_weights.json`](data/best_weights.json), and
[`robust_weights.json`](data/robust_weights.json).

## Engineering decisions worth exploring

- **Feedback-loop guard:** recommender-origin clicks are logged separately and
  can be excluded or down-weighted during future model refreshes.
- **Leakage-aware features:** global priors are separated from per-user history,
  and evaluation uses time-aware holdouts.
- **Authoritative catalog:** all 350 official places remain eligible, including
  locations with no observed clicks.
- **Graceful degradation:** the product runs without BigQuery credentials or an
  LLM key by falling back to checked-in demo data and deterministic planning.
- **Explainable delivery:** the API exposes the evidence behind ranked cards
  instead of returning opaque scores alone.

## Repository tour

| Path | Start here for |
| --- | --- |
| [`backend/recommender.py`](backend/recommender.py) | Hybrid score construction, user regimes, cold start, and MMR |
| [`backend/collab.py`](backend/collab.py) | Item co-visitation and user-neighbor models |
| [`backend/behavior.py`](backend/behavior.py) | Session co-visitation and transition signals |
| [`backend/main.py`](backend/main.py) | FastAPI lifecycle, data-source selection, and product endpoints |
| [`warehouse/sql/`](warehouse/sql/) | Complete six-stage BigQuery transformation logic |
| [`warehouse/schemas/`](warehouse/schemas/) | Eight deployable table schemas |
| [`frontend/`](frontend/) | Responsive browser experience and recommendation UI |
| [`backend/evaluation_runner.py`](backend/evaluation_runner.py) | Temporal holdout and baseline evaluation |
| [`backend/weight_search_cv.py`](backend/weight_search_cv.py) | Five-fold robust weight validation |
| [`docs/`](docs/) | Final paper, presentation, editable sources, figures, and diagrams |

## Run locally

The public edition works without external credentials:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn backend.main:app --reload --port 8000
```

Open <http://localhost:8000>. The default configuration loads
`data/demo_events.csv`; every identity begins with `synthetic-user-`, and every
demo URL uses the reserved `.invalid` domain.

Regenerate and verify the public fixture:

```bash
python scripts/generate_demo_events.py
python -m pytest tests/test_demo_fixture.py
```

## Connect a private BigQuery deployment

Copy `.env.example` to `.env` and configure approved project resources:

```text
BQ_PROJECT=your-project-id
BQ_DATASET=your-dataset-id
BQ_TABLE_FEATURES=user_location_full_features
BQ_TABLE_LOCATION_DIM=location_dim
BQ_TABLE_EVENTS=user_location_category_events
```

The parameterized warehouse runner can rebuild the public template without
hard-coded infrastructure identifiers:

```bash
python warehouse/run_pipeline.py \
  --project your-project-id \
  --dataset analytics_demo \
  --create-dataset \
  --load-seeds \
  --include-candidates
```

See [`warehouse/README.md`](warehouse/README.md) for the full stage-by-stage
runbook.

## Reproduce the evaluation workflow

```bash
pip install -r requirements-ml.txt
python -m backend.evaluation_runner --events data/demo_events.csv
python -m backend.weight_search --events data/demo_events.csv
python -m backend.weight_search_cv --events data/demo_events.csv
python scripts/evaluate_engagement_impact.py
```

## API surface

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

## Paper and presentation

The complete academic deliverables are published unchanged from the team
edition:

- [Final capstone paper — PDF](docs/paper/ChicagoDoes_Capstone_Paper.pdf)
- [Editable paper manuscript — DOCX](docs/paper/ChicagoDoes_Capstone_Paper.docx)
- [Final presentation — PPTX](docs/presentation/ChicagoDoes_Final_Presentation.pptx)
- [LaTeX source and research figures](docs/paper/)
- [Presentation builder and visual assets](docs/presentation/)

## Team and context

This project was completed as a Spring 2026 University of Chicago MS in Applied
Data Science capstone for Ateema / ChicagoDoes.

**Capstone team:** Yiou Wang · RJ Xia · Kennedy Damtse

## Public data and privacy

The public repository contains no production credentials, GA4 pseudonymous
identifiers, account IDs, device fingerprints, visitor location histories, raw
referrers, session histories, or private analytics exports. Production resource
identifiers in operational code and SQL are replaced with parameters.

The final paper and presentation remain unchanged as the historical academic
record, including their authorship, reported aggregate results, and technical
narrative. See [PRIVACY.md](PRIVACY.md) for the exact public-data boundary and
[SECURITY.md](SECURITY.md) for secret handling and vulnerability reporting.

## Internal handoff access

Authorized project stakeholders who need complete operational handoff
documentation, original or private data exports, deployment details, private
environment configuration, or internal repository history can request access
to the private internal repository by contacting Austin Wang at
[yiouwang@uchicago.edu](mailto:yiouwang@uchicago.edu). Please do not post
credentials or visitor-level data in public issues.

## License

No open-source license is currently provided. Copyright remains with the
project owners and contributors.
