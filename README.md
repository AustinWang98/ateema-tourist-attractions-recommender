# ChicagoDoes Content Recommender

A content-based recommendation website built on the Ateema / ChicagoDoes
capstone data warehouse (see `../SKILL.md` for the upstream SQL pipeline and
table definitions).

Visitors set preferences on the left (or simulate a returning `user_key`).
The app returns:

1. **Top-K location recommendations** ranked by a hybrid content + collaboration
   model with warehouse-backed evidence on every card.
2. **An optional AI itinerary** that schedules only from those Top picks
   (data first, LLM second).
3. **LLM concierge copy** for “Why this?” and “More info” on each place, with
   deterministic fallbacks when the LLM is off.

The recommender is **leakage-aware**: it does not use a user’s own interaction
counts to score whether they would click a place. Ranking blends content
similarity, global `*_all_users` priors, item/user collaborative signals,
session co-visitation, and event-time trending.

---

## 1. Project layout

```text
chicagodoes-recsys/
├── data/
│   ├── *.csv                    # offline cache (refresh via backend.refresh)
│   ├── locations_geo.csv        # geocoded lat/lon for maps
│   └── llm_cache.sqlite         # optional LLM response cache
├── backend/
│   ├── data_loader.py           # warehouse frames + engagement policy
│   ├── recommender.py           # hybrid ranker + MMR diversity
│   ├── collab.py                # item + user collaborative signals
│   ├── behavior.py              # session co-visitation + transitions
│   ├── trends.py                # event-time trending
│   ├── engagement.py            # qualified-event filtering
│   ├── segments.py              # user archetypes
│   ├── itinerary_llm.py         # assemble AI / fallback day plans
│   ├── geo.py                   # coords, route ordering, legs
│   ├── geocode.py               # Nominatim backfill for locations_geo.csv
│   ├── llm_service.py           # Ollama / OpenAI-compatible LLM + fallbacks
│   ├── refresh.py               # pull BQ → local CSV cache
│   ├── sources/bq_source.py     # BigQuery loader
│   ├── schemas.py
│   └── main.py                  # FastAPI app
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── requirements.txt
├── .env.example
└── README.md
```

---

## 2. Data source

**Primary:** BigQuery on startup and `POST /api/refresh`.

Set in `.env`:

```bash
BQ_PROJECT=your-gcp-project
BQ_DATASET=analytics_459092297
BQ_TABLE_FEATURES=user_location_full_features
BQ_TABLE_LOCATION_DIM=location_dim
BQ_TABLE_EVENTS=user_location_category_events
```

Authenticate once:

```bash
gcloud auth application-default login
```

**Fallback:** if BQ is unreachable, the app loads CSV snapshots under `data/`.
Refresh them manually:

```bash
python -m backend.refresh
```

**Maps:** coordinates come from `data/locations_geo.csv`. To backfill missing
locations:

```bash
python -m backend.geocode
```

---

## 3. Install and run

### 3.1 Python app

```bash
cd chicagodoes-recsys
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit BQ_* and LLM settings if needed

./.venv/bin/uvicorn backend.main:app --reload --port 8000
```

Open <http://localhost:8000>.

### 3.2 LLM (default: local Ollama, free)

The app uses the **OpenAI Python SDK** against any compatible base URL.
Defaults in `.env.example` target **Ollama**:

```bash
# Terminal 1 — keep running
ollama serve

# once
ollama pull llama3.1:8b

# Terminal 2
./.venv/bin/uvicorn backend.main:app --reload --port 8000
```

`.env`:

```bash
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3.1:8b
LLM_CACHE_PATH=data/llm_cache.sqlite
LLM_CACHE_ENABLED=1
```

**Switch to OpenAI (paid):**

```bash
OPENAI_BASE_URL=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
```

**Other hosts (e.g. Groq):** set `OPENAI_BASE_URL` to their OpenAI-compatible
endpoint and the model name they document.

If the LLM is unreachable, recommendations still work; itinerary falls back to an
automatic schedule from Top picks; descriptions and “Why this?” use templates.

---

## 4. Website flow

1. Choose **traveler type**, **vibe**, **interests**, **avoid**, **trip length**.
   Optional: simulate a returning visitor (`user_key`), free-text notes, and
   **Plan my days with AI**.
2. **Get recommendations** — paginated Top picks (about 3×2 cards per page,
   arrow navigation), score breakdown, evidence, similar users / archetype when
   available.
3. **Build AI itinerary** (if enabled) — only uses `location_id`s from the
   current Top picks pool. Day maps show numbered pins (no misleading straight
   route lines).
4. Per card: **Why this?** and **More info** (concierge blurb, highlights, tips,
   optional links).
5. **Tweak your plan** — natural-language refinement re-runs recommend +
   itinerary.

---

## 5. How ranking works

### 5.1 Location representation

- **Content vector:** TF-IDF over category tokens (`HOT SPOTS` kept as one token).
- **Global priors:** popularity and engagement from `*_all_users` columns
  (leakage-safe for new users).

### 5.2 User profile

- **Returning user:** weighted mean of interacted location vectors.
- **New user:** pseudo-profile from form (interests, vibe, traveler type,
  optional `free_text` keyword hints).
- **Blend** when both exist: `0.6 · behavioural + 0.4 · form` (see
  `recommender.py`).

### 5.3 Score blend (simplified)

New visitors emphasize content + session signals; returning users add
user–user collaborative similarity. Terms include:

- content similarity (cosine),
- popularity / engagement norms,
- item co-visitation,
- user-neighbor collaborative score,
- trending (event-time),
- session co-visitation + transition graph.

Results are filtered by interests/avoid lists, then **MMR** diversifies the
Top-K list.

### 5.4 Evidence

Each card exposes warehouse stats (`n_users_engaged`, sessions, trending flag,
etc.) so copy is grounded in real map activity, not hallucinated counts.

---

## 6. Itinerary

**Product rule:** the AI may only schedule stops from the recommendation pool
(`itinerary_pool_ids` from the frontend’s Top picks).

1. User checks **Plan my days with AI** and runs **Get recommendations**, then
   **Build AI itinerary**.
2. `POST /api/itinerary` re-ranks if needed, builds a candidate list, and calls
   `LLMService.generate_itinerary()` (compact JSON prompt, retry, lenient parse).
3. If the local model returns invalid JSON, a **deterministic schedule** spreads
   Top picks across days so the UI still shows a plan.
4. `itinerary_llm.assemble_itinerary_plan()` attaches coordinates, sorts slots
   (breakfast → … → drinks), and computes route legs.

There is no fixed clock-time or `daily_hours` budgeting in the UI.

---

## 7. API summary

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET | `/api/health` | Liveness, warehouse stats, LLM flag |
| POST | `/api/refresh` | Reload from BigQuery |
| GET | `/api/categories` | All categories |
| GET | `/api/trending?limit=N` | Trending locations |
| GET | `/api/users?limit=N` | Sample `user_key` values (demo) |
| POST | `/api/recommend` | Top-K cards + archetype + similar users |
| GET | `/api/similar_users` | Similar visitors for a `user_key` |
| POST | `/api/itinerary` | AI (+ fallback) multi-day plan |
| POST | `/api/location/info` | Concierge description for one place |
| POST | `/api/explain` | Personalised “why this?” for one place |
| POST | `/api/refine` | NL tweak → delta → re-rank + itinerary |
| POST | `/api/parse_intent` | Free text → `RecommendRequest` (API only; not used by UI) |

### `RecommendRequest` (main fields)

```jsonc
{
  "user_key": null,
  "interests": ["Attractions", "Parks"],
  "traveler_type": "family",
  "vibe": "outdoorsy",
  "avoid_categories": ["Bars"],
  "trip_days": 2,
  "free_text": "we love sculptures and deep-dish",
  "top_k": 40,
  "use_ai_itinerary": true,
  "itinerary_pool_ids": ["uuid-1", "uuid-2"]
}
```

`itinerary_pool_ids` should be the `location_id` list from the latest
recommend response when building an AI itinerary from Top picks.

---

## 8. LLM features

All LLM calls share one cache (`data/llm_cache.sqlite`) and template fallbacks.

| Feature | Endpoint / UI | Notes |
| ------- | ------------- | ----- |
| **Itinerary layout** | `POST /api/itinerary` | JSON day plan from Top picks only; fallback scheduler if parse fails |
| **Why this?** | `POST /api/explain` · card button | Uses interests + evidence from the card |
| **More info** | `POST /api/location/info` · modal | Description, highlights, tips, Maps link |
| **Refine plan** | `POST /api/refine` · itinerary panel | NL delta → merge → re-run |
| **Intent parse** | `POST /api/parse_intent` | Still available for scripts; removed from homepage UI |

Delete `data/llm_cache.sqlite` after changing models or prompts. Set
`LLM_CACHE_ENABLED=0` while iterating.

---

## 9. Troubleshooting

| Symptom | What to check |
| ------- | ------------- |
| `LLM off` in health | `ollama serve` running; `ollama list` shows your model |
| `Connection error` in logs | Ollama not running in a **second** terminal while uvicorn runs |
| Itinerary empty / generic notice | Model returned bad JSON — retry; clear LLM cache; fallback schedule should still appear |
| Slow first itinerary | Local `llama3.1:8b` load can take 30–90s on first call |
| BQ errors on startup | ADC login, `BQ_PROJECT` / `BQ_DATASET`, or use cached CSV via `refresh` |

---

## 10. Roadmap ideas

- Offline Precision@K / Recall@K on held-out interactions.
- Stronger itinerary validation / repair loop for small local models.
- Deploy with a hosted OpenAI-compatible API for faster JSON itinerary generation.
