"""FastAPI entry point for the ChicagoDoes recommender website.

Endpoints
---------
GET  /api/health          - liveness probe
GET  /api/categories      - all categories known to the recommender
GET  /api/users           - sample of known user_keys (for the demo selector)
POST /api/recommend       - top-K recommendations + inferred interests
POST /api/itinerary       - optional LLM day plan (when use_ai_itinerary=true)
POST /api/location/info   - LLM-enriched description for one location

The frontend is served from the `frontend/` directory mounted at `/`.
"""
from __future__ import annotations

import logging
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus
from urllib.parse import urlparse

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .data_loader import WarehouseFrames, load_public_demo_warehouse, load_warehouse
from .itinerary_llm import (
    PLAN_AI,
    assemble_itinerary_plan,
    build_geo_index,
    build_location_indexes,
    count_scheduled_stops,
    deterministic_itinerary_payload,
    filter_to_recommendation_pool,
    recommendations_for_prompt,
    _norm_name,
)
from .llm_service import LLMService, _fallback_blurb
from .location_enrich import LocationEnricher
from .recommender import ContentRecommender
from .schemas import (
    Archetype,
    ExplainRequest,
    ExplainResponse,
    IntentParseRequest,
    IntentParseResponse,
    ItineraryDay,
    ItineraryLeg,
    ItineraryResponse,
    ItineraryStop,
    LocationCard,
    LocationEvidence,
    LocationCardRequest,
    LocationCardResponse,
    CardMediaItem,
    LocationInfoRequest,
    LocationInfoResponse,
    OutboundClickRequest,
    RecommendRequest,
    RecommendResponse,
    RefineRequest,
    RefineResponse,
    SimilarUser,
)

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("chicagodoes")


# --------------------------------------------------------------------------- #
# App + globals
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV_PATH = os.getenv("DATA_CSV_PATH", str(PROJECT_ROOT / "data" / "user_location_features.csv"))
LOCATION_DIM_PATH = os.getenv("LOCATION_DIM_PATH", str(PROJECT_ROOT / "data" / "location_dim.csv"))
EVENTS_PATH = os.getenv("EVENTS_PATH", str(PROJECT_ROOT / "data" / "events.csv"))
GEO_PATH = os.getenv("GEO_PATH", str(PROJECT_ROOT / "data" / "locations_geo.csv"))
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Prebuilt card enrichment (photo + one-line blurb + official site), generated
# offline by `scripts/build_location_cards.py`. Served to users with NO OpenAI
# or external calls — only the AI itinerary hits the LLM at request time.
CARD_STORE_PATH = os.getenv("CARD_STORE_PATH", str(PROJECT_ROOT / "data" / "location_cards.json"))
CARD_IMAGES_DIR = os.getenv("CARD_IMAGES_DIR", str(PROJECT_ROOT / "data" / "location_images"))
CARD_VIDEOS_DIR = os.getenv("CARD_VIDEOS_DIR", str(PROJECT_ROOT / "data" / "location_videos"))
OUTBOUND_CLICKS_PATH = os.getenv(
    "RECOMMENDER_OUTBOUND_CLICKS_PATH",
    str(PROJECT_ROOT / "data" / "outbound_clicks.jsonl"),
)
os.environ.setdefault("RECOMMENDER_OUTBOUND_CLICKS_PATH", OUTBOUND_CLICKS_PATH)

BQ_PROJECT = os.getenv("BQ_PROJECT", "").strip()
BQ_DATASET = os.getenv("BQ_DATASET", "").strip()
BQ_TABLE_FEATURES = os.getenv("BQ_TABLE_FEATURES", "user_location_full_features")
BQ_TABLE_LOCATION_DIM = os.getenv("BQ_TABLE_LOCATION_DIM", "location_dim")
BQ_TABLE_EVENTS = os.getenv("BQ_TABLE_EVENTS", "user_location_category_events")

# Where last-known-good warehouse rows live on disk, so the demo still
# boots if BigQuery is briefly unavailable.
CACHE_DIR = PROJECT_ROOT / "data"


# Track the last load mode for /api/health.
_LAST_LOAD_MODE: dict = {"mode": "unknown", "warning": None}


def _build_bq_config():
    """Return BQConfig when BQ_PROJECT / BQ_DATASET are set."""
    if not (BQ_PROJECT and BQ_DATASET):
        return None
    from .sources.bq_source import BQConfig
    return BQConfig(
        project=BQ_PROJECT,
        dataset=BQ_DATASET,
        table_features=BQ_TABLE_FEATURES,
        table_location_dim=BQ_TABLE_LOCATION_DIM,
        table_events=BQ_TABLE_EVENTS,
    )


def _load_frames() -> WarehouseFrames:
    """Load warehouse frames from BigQuery, local cache, or public demo data.

    Strategy:
      1. Try BigQuery when BQ_PROJECT and BQ_DATASET are configured.
      2. Fall back to full local cache from `python -m backend.refresh`.
      3. Fall back to public CSVs (`location_dim`, `events`, `geo`) so free
         hosts can boot without private credentials.

    To keep the offline cache fresh, run:
        python -m backend.refresh
    """
    cfg = _build_bq_config()
    bq_error: Optional[Exception] = None
    if cfg is not None:
        try:
            logger.info(
                "Loading warehouse from BigQuery (project=%s dataset=%s)",
                cfg.project, cfg.dataset,
            )
            frames = load_warehouse(
                source="bq", bq_config=cfg,
                geo_path=GEO_PATH if Path(GEO_PATH).exists() else None,
            )
            _LAST_LOAD_MODE.update({"mode": "bigquery", "warning": None})
            return frames
        except Exception as exc:  # noqa: BLE001
            bq_error = exc
            logger.warning("BigQuery load failed: %s", exc)

    if _local_cache_present():
        warning = None
        if bq_error:
            warning = f"BigQuery unavailable: {bq_error}"
        elif cfg is None:
            warning = "BigQuery env vars not set; using local cache."
        frames = load_warehouse(
            csv_path=DATA_CSV_PATH,
            location_dim_path=LOCATION_DIM_PATH,
            events_path=EVENTS_PATH,
            geo_path=GEO_PATH if Path(GEO_PATH).exists() else None,
        )
        _LAST_LOAD_MODE.update({
            "mode": "local_cache_fallback",
            "warning": warning,
        })
        return frames

    if Path(LOCATION_DIM_PATH).exists():
        warning = "BigQuery env vars not set; using public demo CSVs."
        if bq_error:
            warning = f"BigQuery unavailable: {bq_error}; using public demo CSVs."
        frames = load_public_demo_warehouse(
            location_dim_path=LOCATION_DIM_PATH,
            events_path=EVENTS_PATH,
            geo_path=GEO_PATH if Path(GEO_PATH).exists() else None,
        )
        _LAST_LOAD_MODE.update({
            "mode": "public_demo_fallback",
            "warning": warning,
        })
        return frames

    if bq_error:
        raise bq_error
    raise RuntimeError(
        "No data source available. Configure BQ_PROJECT/BQ_DATASET or provide "
        "data/location_dim.csv."
    )


def _local_cache_present() -> bool:
    return Path(DATA_CSV_PATH).exists() and Path(LOCATION_DIM_PATH).exists()


def _rebuild_state(frames: WarehouseFrames) -> None:
    """Rebuild every model that depends on warehouse frames."""
    from .engagement import apply_engagement_policy

    apply_engagement_policy(frames)
    state.frames = frames
    state.recommender = ContentRecommender(frames)
    state.geo = build_geo_index(frames)

app = FastAPI(
    title="ChicagoDoes Content Recommender",
    description="Content-based recommendations + itinerary for the ChicagoDoes map.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AppState:
    frames: Optional[WarehouseFrames] = None
    recommender: Optional[ContentRecommender] = None
    geo: Optional[object] = None
    llm: Optional[LLMService] = None
    enricher: Optional[LocationEnricher] = None
    card_store: dict = {}


state = AppState()


def _load_card_store() -> dict:
    """Load the prebuilt per-location card enrichment from disk (if present)."""
    path = Path(CARD_STORE_PATH)
    if not path.exists():
        logger.warning(
            "Card store %s not found — cards will use offline fallbacks. "
            "Build it with: python -m scripts.build_location_cards", path,
        )
        return {}
    try:
        import json as _json
        store = _json.loads(path.read_text(encoding="utf-8"))
        logger.info("Loaded %d prebuilt location cards from %s", len(store), path)
        return store if isinstance(store, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read card store %s: %s", path, exc)
        return {}


@app.on_event("startup")
def _startup() -> None:
    state.llm = LLMService()
    state.enricher = LocationEnricher(
        cache_path=os.getenv("ENRICH_CACHE_PATH", "data/enrich_cache.sqlite"),
    )
    state.card_store = _load_card_store()
    _rebuild_state(_load_frames())
    logger.info("Recommender ready. LLM enabled=%s, load_mode=%s",
                state.llm.enabled, _LAST_LOAD_MODE["mode"])


# --------------------------------------------------------------------------- #
# API endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict:
    f = state.frames
    rec = state.recommender
    return {
        "ok": True,
        "load_mode":            _LAST_LOAD_MODE["mode"],   # "bigquery" or "local_cache_fallback"
        "load_warning":         _LAST_LOAD_MODE["warning"],
        "bq_project":           BQ_PROJECT or None,
        "bq_dataset":           BQ_DATASET or None,
        "n_users":              0 if f is None else len(f.users),
        "n_locations":          0 if f is None else len(f.locations),
        "n_observed_locations": 0 if f is None else int(f.locations.get("observed", pd.Series(dtype=bool)).sum()),
        "n_events":             0 if f is None or f.events is None else len(f.events),
        "llm_enabled":          bool(state.llm and state.llm.enabled),
        "trending_locations":   0 if rec is None else int((rec._trending > 0).sum()),
        "session_pairs":        0 if rec is None else int(rec.session_coviz.jaccard.nnz),
        "transitions":          0 if rec is None else int(rec.transitions.transitions.nnz),
        "feedback_guard":       (rec.feedback_guard if rec else {}),
        "data_quality":         (rec.data_quality if rec else {}),
        "engagement":           (f.engagement_report if f else {}),
    }


@app.post("/api/outbound/click")
def outbound_click(req: OutboundClickRequest) -> dict:
    """Record recommender-origin outbound clicks separately from engagement.

    These clicks are attribution/control data, not positive training labels.
    The recommender reads this file on rebuild to correct for self-generated
    popularity, and the UTM params let GA4/BigQuery filter the same traffic.
    """
    payload = req.model_dump()
    event = {
        "event_type": "outbound_click",
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "location_id": str(payload.get("location_id") or "").strip() or None,
        "location_name": str(payload.get("location_name") or "").strip() or None,
        "surface": str(payload.get("surface") or "unknown").strip() or "unknown",
        "rank": payload.get("rank"),
        "link_type": str(payload.get("link_type") or "").strip() or None,
        "href": str(payload.get("href") or "").strip()[:1000] or None,
    }
    try:
        path = Path(OUTBOUND_CLICKS_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Failed to record outbound click: %s", exc)
        raise HTTPException(500, "Could not record outbound click.")
    return {"ok": True}


@app.post("/api/refresh")
def refresh_data() -> dict:
    """Pull the latest rows from BigQuery and rebuild every model.

    Falls back to the local cache (data/*.csv) if BQ is briefly down,
    same policy as the startup loader.
    """
    try:
        frames = _load_frames()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Refresh failed: {exc}")
    _rebuild_state(frames)
    return {
        "ok": True,
        "load_mode":    _LAST_LOAD_MODE["mode"],
        "load_warning": _LAST_LOAD_MODE["warning"],
        "n_users":      len(frames.users),
        "n_locations":  len(frames.locations),
        "n_observed":   int(frames.locations.get("observed", pd.Series(dtype=bool)).sum()),
        "n_events":     0 if frames.events is None else len(frames.events),
    }


@app.get("/api/categories")
def categories() -> dict:
    if state.frames is None:
        raise HTTPException(503, "Warehouse not loaded yet.")
    return {"categories": state.frames.all_categories}


@app.get("/api/trending")
def trending(limit: int = 6) -> dict:
    """Top-N currently-trending locations from event-time signal.

    Used by the welcome card to show "what's hot right now" so the
    landing page doesn't look empty before the user submits anything.
    """
    if state.recommender is None:
        raise HTTPException(503, "Recommender not ready.")
    rec = state.recommender
    locs = state.frames.locations  # type: ignore[union-attr]
    scores = rec._trending
    order = scores.argsort()[::-1]
    out = []
    for pos in order[: max(limit * 4, 12)]:
        if scores[pos] <= 0:
            break
        row = locs.iloc[int(pos)]
        out.append({
            "location_id":      str(row["location_id"]),
            "location_name":    str(row.get("location_name") or ""),
            "primary_category": str(row.get("primary_category") or ""),
            "trending_score":   round(float(scores[pos]), 4),
        })
        if len(out) >= limit:
            break
    return {"locations": out}


@app.get("/api/users")
def sample_users(limit: int = 50) -> dict:
    """Return known users with human-readable labels for the demo selector.

    Each entry includes:
        user_key          - the raw id (UUID)
        label             - e.g. "Foodie Explorer · 6 visits · Bars, Restaurants"
        archetype         - cluster name from UserSegmenter, e.g. "Foodie Explorer"
        n_interactions    - distinct locations they touched
        top_categories    - up to 3 categories they engaged with most
    """
    if state.frames is None or state.recommender is None:
        raise HTTPException(503, "Warehouse not loaded yet.")

    users = state.frames.users.copy()
    if "total_user_interaction_score" in users.columns:
        users = users.sort_values("total_user_interaction_score", ascending=False)

    enriched = getattr(state.frames, "interactions_enriched", None)
    interactions = enriched if enriched is not None else state.frames.interactions
    segmenter = state.recommender.segmenter

    items: list[dict] = []
    for _, row in users.head(limit).iterrows():
        uid = str(row["user_key"])
        sub = interactions[interactions["user_key"].astype(str) == uid]
        if "is_qualified" in sub.columns:
            sub = sub[sub["is_qualified"]]
        cat_counts: dict[str, int] = {}
        for cats in sub.get("categories", pd.Series(dtype=object)):
            for c in (cats if isinstance(cats, list) else []):
                cat_counts[c] = cat_counts.get(c, 0) + 1
        top_cats = [c for c, _ in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:3]]
        n_inter = int(len(sub))
        try:
            arche = segmenter.assign_returning(uid)
            arche_name = arche.archetype if arche else "Explorer"
        except Exception:
            arche_name = "Explorer"
        cats_txt = ", ".join(top_cats) if top_cats else "no clicks yet"
        label = f"{arche_name} · {n_inter} visit{'s' if n_inter != 1 else ''} · {cats_txt}"
        items.append({
            "user_key": uid,
            "label": label,
            "archetype": arche_name,
            "n_interactions": n_inter,
            "top_categories": top_cats,
        })
    return {"users": items}


SIMILAR_USERS_POOL_MULT = 6        # fetch this × k candidates first
SIMILAR_USERS_MAX_PER_ARCHETYPE = 2


def _find_similar_users(
    is_returning: bool,
    user_key: Optional[str],
    inferred_interests: List[str],
    k: int = 5,
) -> List[SimilarUser]:
    """Return up to `k` real users behaviourally similar to this visitor,
    diversified across behavioural archetypes.

    Why diversify? Our cluster sizes are very unbalanced — "Trend Hunter"
    alone owns ~49% of users, so a vanilla cosine kNN almost always
    returns 5 Trend Hunters in a row, which makes the "5 closest real
    visitors" panel look like a bug. We instead pull a larger pool
    (k * SIMILAR_USERS_POOL_MULT) and greedily fill the slate with at
    most SIMILAR_USERS_MAX_PER_ARCHETYPE per archetype, picking the
    highest-similarity candidate within each archetype first.

    - Returning user: cosine similarity over the user-category matrix.
    - New visitor: project inferred interests into the category space
      and find the nearest real users.
    """
    if state.recommender is None or state.frames is None:
        return []
    nbrs_module = state.recommender.neighbours
    segmenter = state.recommender.segmenter

    pool_size = max(k * SIMILAR_USERS_POOL_MULT, k + 5)
    if is_returning and user_key:
        raw = nbrs_module.neighbours_for(user_key, k=pool_size)
    else:
        raw = nbrs_module.neighbours_from_categories(inferred_interests, k=pool_size)
    if not raw:
        return []

    enriched = getattr(state.frames, "interactions_enriched", None)
    interactions = enriched if enriched is not None else state.frames.interactions

    # Build full candidate list with archetype + label.
    candidates: List[SimilarUser] = []
    for uid, sim in raw:
        sub = interactions[interactions["user_key"].astype(str) == uid]
        if "is_qualified" in sub.columns:
            sub = sub[sub["is_qualified"]]
        cat_counts: dict[str, int] = {}
        for cats in sub.get("categories", pd.Series(dtype=object)):
            for c in (cats if isinstance(cats, list) else []):
                cat_counts[c] = cat_counts.get(c, 0) + 1
        top_cats = [c for c, _ in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:3]]
        n_inter = int(len(sub))
        try:
            arche = segmenter.assign_returning(uid)
            arche_name = arche.archetype if arche else "Explorer"
        except Exception:
            arche_name = "Explorer"
        cats_txt = ", ".join(top_cats) if top_cats else "no clicks yet"
        label = f"{arche_name} · {n_inter} visit{'s' if n_inter != 1 else ''} · {cats_txt}"
        candidates.append(SimilarUser(
            user_key=uid,
            label=label,
            archetype=arche_name,
            similarity=round(float(sim), 4),
            n_interactions=n_inter,
            top_categories=top_cats,
        ))

    # De-dupe visually-identical candidates (same archetype + same n_visits
    # + same top-3 cats). They come from different user_keys but look
    # interchangeable in the chip UI, which feels like a bug to users.
    seen_sig: set[tuple] = set()
    deduped: List[SimilarUser] = []
    for c in candidates:
        sig = (c.archetype, c.n_interactions, tuple(c.top_categories))
        if sig in seen_sig:
            continue
        seen_sig.add(sig)
        deduped.append(c)
    candidates = deduped

    # Diversity pass — round-robin across archetypes (best-of each first),
    # then keep going up to MAX_PER_ARCHETYPE per archetype.
    by_arche: dict[str, List[SimilarUser]] = {}
    for c in candidates:
        by_arche.setdefault(c.archetype, []).append(c)
    # Order archetypes by their top candidate's similarity (best first).
    arche_order = sorted(by_arche.keys(),
                         key=lambda a: by_arche[a][0].similarity,
                         reverse=True)

    out: List[SimilarUser] = []
    arche_count: dict[str, int] = {}
    # Round 1 — one per distinct archetype.
    for arche in arche_order:
        if len(out) >= k:
            break
        out.append(by_arche[arche][0])
        arche_count[arche] = 1
    # Round 2 — fill remaining slots up to MAX_PER_ARCHETYPE per arche,
    # preserving original similarity order.
    if len(out) < k:
        picked_keys = {u.user_key for u in out}
        for cand in candidates:
            if len(out) >= k:
                break
            if cand.user_key in picked_keys:
                continue
            if arche_count.get(cand.archetype, 0) >= SIMILAR_USERS_MAX_PER_ARCHETYPE:
                continue
            out.append(cand)
            arche_count[cand.archetype] = arche_count.get(cand.archetype, 0) + 1
            picked_keys.add(cand.user_key)
    return out


@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    if state.recommender is None:
        raise HTTPException(503, "Recommender not ready.")

    results, inferred, is_returning, archetype = state.recommender.recommend(
        user_key=req.user_key,
        interests=req.interests,
        traveler_type=req.traveler_type,
        vibe=req.vibe,
        avoid_categories=req.avoid_categories,
        free_text=req.free_text,
        top_k=req.top_k,
    )
    cards: List[LocationCard] = [
        LocationCard(
            **{**r, "evidence": LocationEvidence(**(r.get("evidence") or {}))}
        )
        for r in results
    ]
    arch_model = (
        Archetype(
            archetype=archetype.archetype,
            cluster_id=archetype.cluster_id,
            confidence=archetype.confidence,
            cluster_size=archetype.cluster_size,
            top_categories=archetype.top_categories,
        )
        if archetype
        else None
    )
    similar = _find_similar_users(is_returning, req.user_key, inferred, k=5)
    return RecommendResponse(
        user_key=req.user_key,
        is_returning_user=is_returning,
        inferred_interests=inferred,
        archetype=arch_model,
        similar_users=similar,
        recommendations=cards,
    )


@app.get("/api/similar_users")
def similar_users(user_key: Optional[str] = None, interests: Optional[str] = None,
                  k: int = 5) -> dict:
    """Find real users whose behaviour resembles either an existing user or
    a comma-separated list of interest category names.

    Useful for the demo to ad-hoc inspect "who would the system match me to?".
    """
    if state.recommender is None:
        raise HTTPException(503, "Recommender not ready.")
    if user_key:
        sims = _find_similar_users(True, user_key, [], k=k)
    elif interests:
        cats = [c.strip() for c in interests.split(",") if c.strip()]
        sims = _find_similar_users(False, None, cats, k=k)
    else:
        raise HTTPException(400, "Provide either user_key or interests.")
    return {"similar_users": [s.model_dump() for s in sims]}


def _pool_notice(pool_size: int, pool_stops: int, ai_stops: int, used_client_pool: bool) -> str:
    src = "your **Top picks**" if used_client_pool else "your recommendations"
    msg = (
        f"This plan is built around {src} ({pool_size} data-ranked places): "
        f"**{pool_stops}** of them are scheduled into your days."
    )
    if ai_stops:
        msg += f" The AI also added **{ai_stops}** outside Chicago stop{'s' if ai_stops != 1 else ''} to round out the route."
    return msg


def _itinerary_response_from_plan(
    req: RecommendRequest,
    plan: dict,
    *,
    narrative_source: str,
    notice: Optional[str] = None,
    feasible: bool = True,
    skip_reason: Optional[str] = None,
    recommendation_pool_size: int = 0,
    stops_from_pool: int = 0,
    used_client_pool: bool = False,
    warehouse_by_id: Optional[Dict[str, Dict]] = None,
    warehouse_by_name: Optional[Dict[str, str]] = None,
) -> ItineraryResponse:
    warehouse_by_id = warehouse_by_id or {}
    warehouse_by_name = warehouse_by_name or {}
    card_store = state.card_store or {}
    official_by_name = _official_site_by_name(card_store)
    days_payload = plan.get("days") or []
    days = [
        ItineraryDay(
            day_number=d["day_number"],
            theme=d["theme"],
            stops=[
                ItineraryStop(
                    **_enrich_itinerary_stop_link(
                        s, warehouse_by_id, warehouse_by_name, official_by_name,
                    )
                )
                for s in d["stops"]
            ],
            legs=[ItineraryLeg(**leg) for leg in d.get("legs", [])],
            narrative=d.get("narrative"),
            n_stops=d.get("n_stops", len(d["stops"])),
        )
        for d in days_payload
    ]
    pool_n = recommendation_pool_size or pool_size_from_plan(plan)
    stops_n = stops_from_pool or count_scheduled_stops(days_payload)
    ai_n = int(plan.get("ai_added_stops") or 0)
    pool_notice = (
        _pool_notice(pool_n, stops_n, ai_n, used_client_pool) if feasible and pool_n else None
    )
    merged_notice = _join_itinerary_notices(notice, pool_notice)

    return ItineraryResponse(
        user_key=req.user_key,
        trip_days=len(days) if days else 0,
        days=days,
        summary=plan.get("summary") or "",
        notice=merged_notice,
        plan_mode=plan.get("plan_mode", PLAN_AI),
        feasible=feasible,
        skip_reason=skip_reason,
        narrative_source=narrative_source,
        recommendation_pool_size=pool_n,
        stops_from_pool=stops_n,
    )


def pool_size_from_plan(plan: dict) -> int:
    return int(plan.get("recommendation_pool_size") or 0)


def _join_itinerary_notices(*parts: Optional[str]) -> Optional[str]:
    bits = [p.strip() for p in parts if p and str(p).strip()]
    return " ".join(bits) if bits else None


def _disabled_itinerary_response(req: RecommendRequest) -> ItineraryResponse:
    return ItineraryResponse(
        user_key=req.user_key,
        trip_days=0,
        days=[],
        summary="AI day plan is off. Turn on **Plan my days with AI** to generate a route.",
        notice=None,
        plan_mode=PLAN_AI,
        feasible=False,
        skip_reason="ai_itinerary_disabled",
        narrative_source="fallback",
    )


@app.post("/api/itinerary", response_model=ItineraryResponse)
def itinerary(req: RecommendRequest) -> ItineraryResponse:
    if state.recommender is None or state.llm is None:
        raise HTTPException(503, "Service not ready.")

    if not req.use_ai_itinerary:
        return _disabled_itinerary_response(req)

    used_client_pool = bool(req.itinerary_pool_ids)
    if used_client_pool:
        top_k = max(len(req.itinerary_pool_ids), req.top_k, 15)
    else:
        top_k = max(req.top_k, req.trip_days * 10, 36)

    results, inferred, _, _ = state.recommender.recommend(
        user_key=req.user_key,
        interests=req.interests,
        traveler_type=req.traveler_type,
        vibe=req.vibe,
        avoid_categories=req.avoid_categories,
        free_text=req.free_text,
        top_k=top_k,
    )

    pool_results = filter_to_recommendation_pool(results, req.itinerary_pool_ids)
    if used_client_pool and not pool_results:
        return ItineraryResponse(
            user_key=req.user_key,
            trip_days=0,
            days=[],
            summary="Could not match your Top picks to schedule.",
            notice=(
                "Run **Get recommendations** first, then **Build my AI day plan** "
                "so we only arrange places you already saw."
            ),
            plan_mode=PLAN_AI,
            feasible=False,
            skip_reason="pool_mismatch",
            narrative_source="fallback",
        )

    scheduling_pool = pool_results if used_client_pool else results
    pool_size = len(scheduling_pool)

    candidates = recommendations_for_prompt(scheduling_pool, geo=state.geo)  # entire Top-picks pool
    rec_by_id = {
        str(r["location_id"]): r
        for r in scheduling_pool
        if r.get("location_id")
    }

    llm_out = state.llm.generate_itinerary(
        candidates=candidates,
        trip_days=req.trip_days,
        interests=req.interests,
        avoid_categories=req.avoid_categories,
        inferred_interests=inferred,
        traveler_type=req.traveler_type,
        vibe=req.vibe,
        free_text=req.free_text,
        planner_note=req.planner_note,
    )
    source = str(llm_out.get("source") or "fallback")
    llm_notice = llm_out.get("notice")

    if source != "llm" or not llm_out.get("days"):
        llm_out = deterministic_itinerary_payload(
            candidates,
            req.trip_days,
            list(dict.fromkeys([*req.interests, *inferred])),
        )
        source = str(llm_out.get("source") or "fallback_schedule")
        llm_notice = llm_notice or (
            "AI formatting failed — showing an automatic day-by-day layout from your Top picks."
        )

    plan = assemble_itinerary_plan(
        llm_out,
        rec_by_id,
        state.geo,
        req.trip_days,
        *build_location_indexes(
            state.recommender.frames.locations if state.recommender else None
        ),
    )
    if not plan.get("feasible"):
        return ItineraryResponse(
            user_key=req.user_key,
            trip_days=0,
            days=[],
            summary=plan.get("summary") or "AI returned no valid stops.",
            notice="The model picked locations we could not match — try again.",
            plan_mode=PLAN_AI,
            feasible=False,
            skip_reason=plan.get("skip_reason") or "llm_empty_plan",
            narrative_source=source,
        )

    plan["recommendation_pool_size"] = pool_size
    warehouse_by_id, warehouse_by_name = build_location_indexes(
        state.recommender.frames.locations if state.recommender else None
    )
    return _itinerary_response_from_plan(
        req,
        plan,
        notice=llm_notice,
        narrative_source=source,
        recommendation_pool_size=pool_size,
        stops_from_pool=int(plan.get("stops_from_pool") or 0),
        used_client_pool=used_client_pool,
        warehouse_by_id=warehouse_by_id,
        warehouse_by_name=warehouse_by_name,
    )


def _warehouse_context(row: pd.Series) -> Dict[str, Any]:
    """Real engagement stats from the warehouse — grounds the concierge LLM."""
    eng_ms = float(row.get("avg_location_engagement_all_users_msec_capped") or 0)
    return {
        "distinct_users": int(row.get("distinct_users_interacted_location") or 0),
        "total_interactions": int(row.get("total_location_interactions_all_users") or 0),
        "total_sessions": int(row.get("distinct_sessions_interacted_location") or 0),
        "is_hot_spot": bool(int(row.get("is_hot_spot_location") or 0)),
        "avg_engagement_sec": round(eng_ms / 1000.0, 1) if eng_ms > 0 else 0.0,
    }


def _maps_search_url(location_name: str) -> str:
    q = quote_plus(f"{location_name}, Chicago, IL")
    return f"https://www.google.com/maps/search/?api=1&query={q}"


# Image hosts we are willing to proxy. Strict allowlist prevents SSRF — the
# proxy must never fetch arbitrary user-supplied URLs.
_IMG_PROXY_HOSTS = frozenset({
    "upload.wikimedia.org",
    "api.openverse.org",
})
_IMG_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    # Wikimedia 400s requests without these; real browsers always send them.
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _proxy_image_url(upstream: Optional[str]) -> Optional[str]:
    """Rewrite a same-origin proxy URL so images load reliably in the browser."""
    if not upstream:
        return None
    return "/api/img?u=" + quote_plus(upstream)


@app.get("/api/img")
def image_proxy(u: str = Query(..., description="Upstream image URL (allowlisted hosts only).")):
    """Stream an allowlisted remote image same-origin.

    Avoids client-side hotlink quirks (Wikimedia header rules), mixed-content,
    and referrer issues by fetching server-side, where it works reliably.
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    host = (urllib.parse.urlparse(u).hostname or "").lower()
    if host not in _IMG_PROXY_HOSTS:
        raise HTTPException(400, "Image host not allowed.")
    try:
        req = urllib.request.Request(u, headers=_IMG_FETCH_HEADERS)
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            data = resp.read(6_000_000)  # cap ~6 MB
            ctype = resp.headers.get("Content-Type", "image/jpeg")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.info("image_proxy failed for %s: %s", u[:80], exc)
        raise HTTPException(502, "Could not fetch image.")
    if not ctype.startswith("image/"):
        raise HTTPException(415, "Upstream is not an image.")
    return Response(
        content=data,
        media_type=ctype,
        headers={"Cache-Control": "public, max-age=604800"},  # 7 days
    )


CHICAGODOES_HOME = "https://www.chicagodoes.com/"
# ChicagoDoes runs on mapme; each venue deep-links by its location_id (a UUID
# that matches our warehouse location_id exactly).
MAPME_LOCATION_BASE = "https://viewer.mapme.com/chicagodoesinteractivevideomaps/location/"


def _looks_like_uuid(value: str) -> bool:
    parts = value.split("-")
    return len(value) == 36 and len(parts) == 5 and all(
        c in "0123456789abcdefABCDEF" for c in value.replace("-", "")
    )


def _chicagodoes_url(location_id: str) -> Optional[str]:
    """Deep link straight to this venue on the ChicagoDoes interactive map."""
    lid = str(location_id or "").strip()
    return MAPME_LOCATION_BASE + lid if _looks_like_uuid(lid) else None


def _media_item_url(item: dict) -> Optional[str]:
    """Resolve a stored media item to a browser-loadable URL (local files only for images)."""
    if not item:
        return None
    f = item.get("file")
    if f:
        if item.get("type") == "video":
            if not (Path(CARD_VIDEOS_DIR) / str(f)).exists():
                url = item.get("url")
                if url:
                    host = (urlparse(str(url)).hostname or "").lower()
                    if "youtube" in host or host == "youtu.be" or host == "media.mapme.com":
                        return str(url)
                return None
            return f"/card-videos/{f}"
        if not (Path(CARD_IMAGES_DIR) / str(f)).exists():
            return None
        return f"/cards/{f}"
    if item.get("type") == "video":
        url = item.get("url")
        if url:
            host = (urlparse(str(url)).hostname or "").lower()
            if "youtube" in host or host == "youtu.be" or host == "media.mapme.com":
                return str(url)
    return None


def _media_item_key(media_type: str, url: str) -> str:
    clean = str(url or "").strip()
    parsed = urlparse(clean)
    if parsed.scheme and parsed.netloc:
        clean = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path}"
    return f"{str(media_type or '').lower()}::{clean.lower()}"


def _append_unique_media(
    media_items: List[CardMediaItem],
    seen: set[str],
    *,
    media_type: str,
    url: str,
    source: Optional[str] = None,
    attribution: Optional[str] = None,
) -> None:
    key = _media_item_key(media_type, url)
    if key in seen:
        return
    seen.add(key)
    media_items.append(CardMediaItem(
        type=media_type,
        url=url,
        source=source,
        attribution=attribution,
    ))


def _card_media_from_store(rec: Optional[dict]) -> Dict[str, Any]:
    """Resolve local or proxied image/video URLs from a prebuilt card record."""
    empty = {
        "image_url": None,
        "image_attribution": None,
        "video_url": None,
        "video_attribution": None,
        "media_source": None,
        "media_items": [],
    }
    if not rec:
        return empty

    media_items: List[CardMediaItem] = []
    seen_media: set[str] = set()
    for raw in rec.get("media_items") or []:
        url = _media_item_url(raw)
        if not url:
            continue
        _append_unique_media(
            media_items,
            seen_media,
            media_type=str(raw.get("type") or "image"),
            url=url,
            source=raw.get("source"),
            attribution=raw.get("attribution"),
        )

    if not media_items:
        img_file = rec.get("image_file")
        vid_file = rec.get("video_file")
        attr = rec.get("image_attribution")
        if vid_file:
            _append_unique_media(
                media_items,
                seen_media,
                media_type="video",
                url=f"/card-videos/{vid_file}",
                source=rec.get("video_source"),
                attribution=attr,
            )
        elif rec.get("video_url"):
            _append_unique_media(
                media_items,
                seen_media,
                media_type="video",
                url=str(rec["video_url"]),
                source=rec.get("video_source"),
                attribution=attr,
            )
        if img_file:
            _append_unique_media(
                media_items,
                seen_media,
                media_type="image",
                url=f"/cards/{img_file}",
                source=rec.get("image_source"),
                attribution=attr,
            )

    first_image = next((m for m in media_items if m.type == "image"), None)
    first_video = next((m for m in media_items if m.type == "video"), None)
    attr = rec.get("image_attribution")
    if first_video and first_video.attribution:
        attr = first_video.attribution
    elif first_image and first_image.attribution:
        attr = first_image.attribution
    return {
        "image_url": first_image.url if first_image else None,
        "image_attribution": attr,
        "video_url": first_video.url if first_video else None,
        "video_attribution": attr if first_video else None,
        "media_source": rec.get("media_source") or rec.get("image_source"),
        "media_items": media_items,
    }


def _card_media_for_location(location_id: str) -> Dict[str, Any]:
    rec = (state.card_store or {}).get(str(location_id))
    return _card_media_from_store(rec)


def _card_link(location_id: str, location_name: str) -> tuple:
    """The card title/photo always links to the venue's ChicagoDoes page."""
    cd = _chicagodoes_url(location_id)
    if cd:
        return cd, "chicagodoes"
    return CHICAGODOES_HOME, "chicagodoes"


def _valid_http_url(url: Any) -> Optional[str]:
    s = str(url or "").strip()
    if s.lower().startswith(("http://", "https://")):
        return s
    return None


def _warehouse_location_id(
    location_id: str,
    location_name: str,
    warehouse_by_id: Dict[str, Dict],
    warehouse_by_name: Dict[str, str],
) -> Optional[str]:
    """Return a warehouse id when this stop is in the ChicagoDoes location universe."""
    lid = str(location_id or "").strip()
    if lid and lid in warehouse_by_id:
        return lid
    nk = _norm_name(location_name)
    if nk and nk in warehouse_by_name:
        return warehouse_by_name[nk]
    return None


def _official_site_by_name(card_store: Dict[str, dict]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for card in card_store.values():
        site = _valid_http_url(card.get("official_site"))
        if not site:
            continue
        nk = _norm_name(card.get("location_name"))
        if nk and nk not in out:
            out[nk] = site
    return out


def _card_store_id_by_name(card_store: Dict[str, dict]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for lid, card in card_store.items():
        nk = _norm_name(card.get("location_name"))
        if nk and nk not in out:
            out[nk] = str(lid)
    return out


def _itinerary_stop_link(
    stop: dict,
    warehouse_by_id: Dict[str, Dict],
    warehouse_by_name: Dict[str, str],
    official_by_name: Dict[str, str],
) -> tuple[Optional[str], Optional[str]]:
    """Choose a link for an itinerary stop.

    Top-picks stops always deep-link to ChicagoDoes. AI-added stops only link
    when we can verify the place is in our warehouse (the ChicagoDoes universe);
    otherwise we fall back to a prebuilt official site, or no link.
    """
    source = str(stop.get("source") or "recommended")
    name = str(stop.get("location_name") or "")
    lid = str(stop.get("location_id") or "").strip()

    if source != "ai":
        wh_id = _warehouse_location_id(lid, name, warehouse_by_id, warehouse_by_name)
        if wh_id:
            return _chicagodoes_url(wh_id), "chicagodoes"
        cd = _chicagodoes_url(lid)
        return cd or CHICAGODOES_HOME, "chicagodoes"

    wh_id = _warehouse_location_id(lid, name, warehouse_by_id, warehouse_by_name)
    if wh_id:
        return _chicagodoes_url(wh_id), "chicagodoes"

    official = official_by_name.get(_norm_name(name))
    if official:
        return official, "official"
    return None, None


def _enrich_itinerary_stop_link(
    stop: dict,
    warehouse_by_id: Dict[str, Dict],
    warehouse_by_name: Dict[str, str],
    official_by_name: Dict[str, str],
) -> dict:
    link_url, link_type = _itinerary_stop_link(
        stop, warehouse_by_id, warehouse_by_name, official_by_name,
    )
    out = {**stop, "link_url": link_url, "link_type": link_type}
    card_store = state.card_store or {}
    card_by_name = _card_store_id_by_name(card_store)
    lid = _warehouse_location_id(
        str(stop.get("location_id") or "").strip(),
        str(stop.get("location_name") or ""),
        warehouse_by_id,
        warehouse_by_name,
    )
    if not lid:
        lid = card_by_name.get(_norm_name(stop.get("location_name")))
    if lid:
        media = _card_media_for_location(lid)
        if media.get("image_url"):
            out["image_url"] = media["image_url"]
        if media.get("video_url"):
            out["video_url"] = media["video_url"]
        if media.get("media_items"):
            out["media_items"] = [m.model_dump() for m in media["media_items"]]
        if not out.get("location_id"):
            out["location_id"] = lid
    return out


@app.post("/api/location/card", response_model=LocationCardResponse)
def location_card(req: LocationCardRequest) -> LocationCardResponse:
    """Per-card photo + one-line specialty blurb + best link.

    Served from the LOCAL prebuilt store (data/location_cards.json + images)
    with NO OpenAI or external calls — that work is done once, offline, by
    `scripts/build_location_cards.py`. (Only the AI itinerary calls the LLM at
    request time.)
    """
    if state.recommender is None:
        raise HTTPException(503, "Service not ready.")

    row = state.recommender.location_lookup(req.location_id)
    if row is None:
        raise HTTPException(404, f"Unknown location_id: {req.location_id}")

    name = str(row["location_name"])
    primary = row.get("primary_category")
    cats = list(row.get("categories") or [])

    rec = (state.card_store or {}).get(req.location_id)
    media_items: List[CardMediaItem] = []
    if rec is not None:
        media = _card_media_from_store(rec)
        image_url = media["image_url"]
        image_attr = media["image_attribution"]
        video_url = media["video_url"]
        video_attr = media["video_attribution"]
        media_source = media["media_source"]
        media_items = media["media_items"]
        blurb = rec.get("blurb") or _fallback_blurb(name, primary, cats)
        blurb_source = rec.get("blurb_source", "fallback")
    else:
        image_url = None
        image_attr = None
        video_url = None
        video_attr = None
        media_source = None
        blurb = _fallback_blurb(name, primary, cats)
        blurb_source = "fallback"

    link_url, link_type = _card_link(req.location_id, name)
    return LocationCardResponse(
        location_id=req.location_id,
        location_name=name,
        image_url=image_url,
        image_attribution=image_attr,
        video_url=video_url,
        video_attribution=video_attr,
        media_items=media_items,
        blurb=blurb,
        blurb_source=blurb_source,
        link_url=link_url,
        link_type=link_type,
        media_source=media_source,
    )


@app.post("/api/location/info", response_model=LocationInfoResponse)
def location_info(req: LocationInfoRequest) -> LocationInfoResponse:
    if state.recommender is None or state.llm is None:
        raise HTTPException(503, "Service not ready.")

    row = state.recommender.location_lookup(req.location_id)
    if row is None:
        raise HTTPException(404, f"Unknown location_id: {req.location_id}")

    loc_name = str(row["location_name"])
    info = state.llm.describe_location(
        location_name=loc_name,
        primary_category=row.get("primary_category"),
        categories=list(row.get("categories") or []),
        style=req.style or "friendly",
        warehouse_context=_warehouse_context(row),
    )
    return LocationInfoResponse(
        location_id=req.location_id,
        location_name=loc_name,
        primary_category=row.get("primary_category"),
        description=info["description"],
        highlights=info.get("highlights") or [],
        tips=info.get("tips") or [],
        neighborhood=info.get("neighborhood"),
        best_for=info.get("best_for"),
        website_url=info.get("website_url"),
        maps_search_url=_maps_search_url(loc_name),
        source=info["source"],
    )


# --------------------------------------------------------------------------- #
# LLM-powered endpoints (Phase A)
# --------------------------------------------------------------------------- #
@app.post("/api/parse_intent", response_model=IntentParseResponse)
def parse_intent(req: IntentParseRequest) -> IntentParseResponse:
    """A1 - natural-language intent parsing.

    Returns a structured RecommendRequest-shaped payload that the
    frontend can either auto-submit or let the user tweak first.
    """
    if state.llm is None or state.frames is None:
        raise HTTPException(503, "LLM service not ready.")
    parsed = state.llm.parse_intent(
        free_text=req.free_text,
        valid_categories=state.frames.all_categories,
    )
    return IntentParseResponse(**parsed)


@app.post("/api/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest) -> ExplainResponse:
    """A3 - personalised 'why is this for me?' answer."""
    if state.recommender is None or state.llm is None:
        raise HTTPException(503, "Service not ready.")
    row = state.recommender.location_lookup(req.location_id)
    if row is None:
        raise HTTPException(404, f"Unknown location_id: {req.location_id}")
    rationale = state.llm.rationale_for(
        location_name=str(row["location_name"]),
        location_categories=list(row.get("categories") or []),
        user_interests=req.interests,
        inferred_interests=req.inferred_interests,
        vibe=req.vibe,
        traveler_type=req.traveler_type,
        rank=req.rank,
        system_reason=req.system_reason,
        evidence_summary=req.evidence_summary,
        final_score=req.final_score,
        is_trending=req.is_trending,
        is_hot_spot=req.is_hot_spot,
        similarity_score=req.similarity_score,
        item_collab_score=req.item_collab_score,
        trending_score=req.trending_score,
    )
    return ExplainResponse(
        location_id=req.location_id,
        location_name=str(row["location_name"]),
        rationale=rationale,
        source="llm" if state.llm.enabled else "fallback",
    )


@app.post("/api/refine", response_model=RefineResponse)
def refine(req: RefineRequest) -> RefineResponse:
    """A4 - apply a natural-language tweak and re-rank in one call.

    The LLM returns a *delta* (add_interests, remove_interests, set_vibe,
    ...). We merge it into the previous request, then re-run
    /recommend and /itinerary so the response is fully self-contained.
    The recommender stays the source of truth; the LLM only translates
    English into structured preference changes.
    """
    if state.recommender is None or state.llm is None or state.frames is None:
        raise HTTPException(503, "Service not ready.")

    delta = state.llm.refine_request(
        instruction=req.instruction,
        previous_request=req.previous_request.model_dump(),
        valid_categories=state.frames.all_categories,
    )
    new_req = _merge_delta(req.previous_request, delta)

    rec_resp = recommend(new_req)
    if new_req.use_ai_itinerary:
        pool_ids = [c.location_id for c in rec_resp.recommendations if c.location_id]
        itin_req = new_req.model_copy(
            update={
                "itinerary_pool_ids": pool_ids,
                "use_ai_itinerary": True,
                "planner_note": req.instruction,
            },
        )
        itin_resp = itinerary(itin_req)
    else:
        itin_resp = _disabled_itinerary_response(new_req)
    return RefineResponse(
        delta=delta,
        new_request=new_req,
        recommendations=rec_resp,
        itinerary=itin_resp,
        source=delta.get("source", "fallback"),
    )


def _merge_delta(prev: RecommendRequest, delta: dict) -> RecommendRequest:
    """Apply a refinement delta to a RecommendRequest, immutably."""
    interests = list(prev.interests)
    for c in delta.get("add_interests", []):
        if c not in interests:
            interests.append(c)
    for c in delta.get("remove_interests", []):
        if c in interests:
            interests.remove(c)

    avoid = list(prev.avoid_categories)
    for c in delta.get("add_avoid", []):
        if c not in avoid:
            avoid.append(c)
    for c in delta.get("remove_avoid", []):
        if c in avoid:
            avoid.remove(c)

    return RecommendRequest(
        user_key=prev.user_key,
        interests=interests,
        avoid_categories=avoid,
        traveler_type=delta.get("set_traveler") or prev.traveler_type,
        vibe=delta.get("set_vibe") or prev.vibe,
        trip_days=delta.get("set_trip_days") or prev.trip_days,
        free_text=prev.free_text,
        top_k=prev.top_k,
        use_ai_itinerary=prev.use_ai_itinerary,
    )


# --------------------------------------------------------------------------- #
# Static frontend + locally-stored card photos
# --------------------------------------------------------------------------- #
Path(CARD_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
Path(CARD_VIDEOS_DIR).mkdir(parents=True, exist_ok=True)
app.mount("/cards", StaticFiles(directory=CARD_IMAGES_DIR), name="cards")
app.mount("/card-videos", StaticFiles(directory=CARD_VIDEOS_DIR), name="card-videos")

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(FRONTEND_DIR / "index.html"))
