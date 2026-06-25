# Ateema Tourist Attractions Recommender

ChicagoDoes recommendation website for ranking visitor attractions, restaurants,
bars, shops, hotels, parks, museums, and other Chicago places from Ateema /
ChicagoDoes behavioral data.

The product has two entry paths:

- **Personalize my picks**: a guided profile wizard asks for traveler type,
  vibe, interests, avoid categories, trip length, number of places, and optional
  notes.
- **Show top places now**: skips setup and returns the strongest baseline
  ChicagoDoes picks immediately.

The redesigned UI uses a dark, media-forward product shell, location cards with
verified media, a clickable Trending now module, optional AI day plans, and links
back to corresponding ChicagoDoes location pages.

## Project Layout

```text
.
├── backend/
│   ├── main.py                  # FastAPI app, startup loading, API routes
│   ├── data_loader.py           # BigQuery/local/public-demo frame builders
│   ├── recommender.py           # hybrid ranker, MMR, feedback guard
│   ├── engagement.py            # qualified engagement filtering
│   ├── behavior.py              # session co-visitation + transition graph
│   ├── collab.py                # item and user collaborative signals
│   ├── trends.py                # recent-vs-early trending score
│   ├── itinerary_llm.py         # AI and deterministic itinerary assembly
│   ├── llm_service.py           # OpenAI-compatible LLM client + fallbacks
│   ├── location_enrich.py       # card enrichment helpers
│   └── sources/bq_source.py     # BigQuery loader/exporter
├── frontend/
│   ├── index.html
│   ├── app.js
│   ├── styles.css
│   ├── ateema-logo.png
│   └── ateema-tab-logo.png
├── data/
│   ├── location_dim.csv         # official ChicagoDoes location universe
│   ├── events.csv               # event-level public/demo signals
│   ├── locations_geo.csv        # map coordinates
│   └── location_cards.json      # prebuilt card metadata
├── scripts/build_location_cards.py
├── tests/
├── render.yaml                  # free Render web-service blueprint
└── requirements.txt
```

Generated media and crawler caches are intentionally not committed:
`data/location_images/`, `data/location_videos/`, `data/firecrawl_cache/`,
`data/enrich_cache.sqlite`, and outbound click logs are local artifacts.

## Data Modes

Startup tries data sources in this order:

1. **Production BigQuery** when `BQ_PROJECT` and `BQ_DATASET` are set.
2. **Full local cache** when `data/user_location_features.csv` exists.
3. **Public demo fallback** from checked-in `location_dim.csv`, `events.csv`,
   and `locations_geo.csv`.

The public fallback makes free hosting possible without private credentials. It
keeps the official 350-location catalog, categories and popularity derived from
events, trending, session co-visitation, and map coordinates. Production
personalization is stronger because it uses the full user-location aggregate
table from BigQuery.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn backend.main:app --reload --port 8000
```

Open <http://localhost:8000>.

For production data, set:

```bash
BQ_PROJECT=your-gcp-project
BQ_DATASET=analytics_459092297
BQ_TABLE_FEATURES=user_location_full_features
BQ_TABLE_LOCATION_DIM=location_dim
BQ_TABLE_EVENTS=user_location_category_events
```

Then authenticate locally:

```bash
gcloud auth application-default login
```

## Recommendation Algorithm

The ranker is a hybrid model:

- **Content similarity**: TF-IDF over location category tokens.
- **Popularity/engagement priors**: leakage-safe `*_all_users` fields.
- **Item co-visitation**: locations that are engaged by the same users.
- **User-neighbor collaborative filtering**: returning visitors only.
- **Session behavior**: session co-visitation plus next-location transitions.
- **Trending**: recent engagement compared with earlier engagement.
- **MMR diversity**: re-ranks the candidate pool to avoid repetitive cards.
- **Duplicate suppression**: canonical place names prevent the same place from
  appearing multiple times in Top picks even if the source data has duplicate
  IDs or variant names.

Top-picks mode intentionally does not infer interests from vibe/traveler type
unless the visitor selects interests or enters free text. That keeps the
baseline list honest.

## Feedback Loop Guard

Outbound clicks from this recommendation website can create a feedback loop:
if they are written back into GA4/BigQuery as ordinary ChicagoDoes engagement,
recommended places may look artificially more popular, then get recommended
even more often in future model updates.

The code prevents that in three ways:

1. **Separate logging**: `/api/outbound/click` records recommender-origin clicks
   to `data/outbound_clicks.jsonl` instead of treating them as positive
   engagement labels.
2. **Attribution/filtering**: ChicagoDoes outbound links are tagged from the
   frontend so GA4/BigQuery training pipelines can exclude or down-weight
   recommender-origin sessions.
3. **Runtime penalty**: `ContentRecommender._build_feedback_penalty()` reads
   recent outbound clicks and applies a small normalized penalty:

   ```text
   final_score =
     hybrid_relevance_score
     - FEEDBACK_GUARD_STRENGTH * normalized_recent_recommender_clicks
   ```

   Defaults:

   ```bash
   FEEDBACK_GUARD_ENABLED=true
   FEEDBACK_GUARD_WINDOW_DAYS=30
   FEEDBACK_GUARD_STRENGTH=0.08
   RECOMMENDER_OUTBOUND_CLICKS_PATH=data/outbound_clicks.jsonl
   ```

In production model refreshes, recommender-origin outbound traffic should remain
an attribution/control signal, not a direct popularity label.

## Media Rules

For locations in the ChicagoDoes dataset, cards try to show verified media from
the prebuilt card store and local media folders. Official and ChicagoDoes media
should be preferred over generic web images. If heavy media folders are omitted
from deployment, the app still runs and uses available metadata/fallback media.

AI itinerary stops may include outside places when useful. Outside-of-database
stops are allowed to appear without media; the media guarantee only applies to
ChicagoDoes catalog locations.

## API Summary

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Load mode, data counts, LLM status, feedback guard |
| `GET` | `/api/categories` | Known categories |
| `GET` | `/api/trending?limit=N` | Trending locations |
| `POST` | `/api/recommend` | Top picks, scores, evidence, archetype |
| `POST` | `/api/itinerary` | AI or deterministic day plan |
| `POST` | `/api/location/info` | Enriched place modal |
| `POST` | `/api/explain` | Personalized “why this?” |
| `POST` | `/api/refine` | Natural-language itinerary tweak |
| `POST` | `/api/outbound/click` | Separate recommender click logging |
| `POST` | `/api/refresh` | Reload warehouse frames |

## LLM Configuration

The site works without an LLM. If no provider is reachable, itinerary and copy
fall back to deterministic templates.

OpenAI:

```bash
OPENAI_BASE_URL=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

Local Ollama:

```bash
ollama serve
ollama pull llama3.1:8b

OPENAI_BASE_URL=ollama
OPENAI_API_KEY=ollama
OPENAI_MODEL=llama3.1:8b
```

## Free Deployment

This repo includes `render.yaml` for Render’s free web service tier.

1. Push this repository to GitHub.
2. In Render, choose **New → Blueprint**.
3. Connect `AustinWang98/ateema-tourist-attractions-recommender`.
4. Render reads `render.yaml` and starts:

   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port $PORT
   ```

5. Optional environment variables:
   - `OPENAI_API_KEY` for hosted AI itinerary quality.
   - `BQ_PROJECT`, `BQ_DATASET`, and Google credentials for production data.

Without BigQuery credentials, the deployed app uses the public demo fallback and
still serves the redesigned website.

Render free instances may sleep after inactivity. The first request after sleep
can be slow.

## Development Checks

```bash
python -m py_compile backend/main.py backend/data_loader.py backend/recommender.py
node --check frontend/app.js
pytest
```

