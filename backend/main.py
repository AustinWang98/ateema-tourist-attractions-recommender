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
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .data_loader import WarehouseFrames, load_warehouse
from .itinerary_llm import (
    PLAN_AI,
    assemble_itinerary_plan,
    build_geo_index,
    count_scheduled_stops,
    deterministic_itinerary_payload,
    filter_to_recommendation_pool,
    recommendations_for_prompt,
)
from .llm_service import LLMService
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
    LocationInfoRequest,
    LocationInfoResponse,
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
    """Return BQConfig (BQ_PROJECT / BQ_DATASET must be set)."""
    if not (BQ_PROJECT and BQ_DATASET):
        raise RuntimeError(
            "BigQuery is the only supported data source. Set BQ_PROJECT and "
            "BQ_DATASET in .env (and run `gcloud auth application-default "
            "login` once)."
        )
    from .sources.bq_source import BQConfig
    return BQConfig(
        project=BQ_PROJECT,
        dataset=BQ_DATASET,
        table_features=BQ_TABLE_FEATURES,
        table_location_dim=BQ_TABLE_LOCATION_DIM,
        table_events=BQ_TABLE_EVENTS,
    )


def _load_frames() -> WarehouseFrames:
    """Load warehouse frames from BigQuery, with a local-cache fallback.

    Strategy:
      1. Try BigQuery (using Application Default Credentials).
      2. If BigQuery fails (no creds, network down, ...) AND a local
         cache from a previous `python -m backend.refresh` exists,
         fall back to that and surface a warning via /api/health.
      3. If neither works, propagate the original error.

    To keep the offline cache fresh, run:
        python -m backend.refresh
    """
    cfg = _build_bq_config()
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
        if _local_cache_present():
            logger.warning(
                "BigQuery load failed (%s). Falling back to local cache "
                "under %s. Run `python -m backend.refresh` after fixing "
                "BQ access to refresh the offline cache.", exc, CACHE_DIR,
            )
            frames = load_warehouse(
                csv_path=DATA_CSV_PATH,
                location_dim_path=LOCATION_DIM_PATH,
                events_path=EVENTS_PATH,
                geo_path=GEO_PATH if Path(GEO_PATH).exists() else None,
            )
            _LAST_LOAD_MODE.update({
                "mode": "local_cache_fallback",
                "warning": f"BigQuery unavailable: {exc}",
            })
            return frames
        raise


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


state = AppState()


@app.on_event("startup")
def _startup() -> None:
    state.llm = LLMService()
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
        "data_quality":         (rec.data_quality if rec else {}),
        "engagement":           (f.engagement_report if f else {}),
    }


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


def _pool_notice(pool_size: int, stops: int, used_client_pool: bool) -> str:
    src = "your **Top picks**" if used_client_pool else "this request's recommendations"
    return (
        f"Every scheduled stop comes from {src} ({pool_size} data-ranked places) — "
        f"AI only arranges **{stops}** of them into days; it does not add new venues."
    )


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
) -> ItineraryResponse:
    days_payload = plan.get("days") or []
    days = [
        ItineraryDay(
            day_number=d["day_number"],
            theme=d["theme"],
            stops=[ItineraryStop(**s) for s in d["stops"]],
            legs=[ItineraryLeg(**leg) for leg in d.get("legs", [])],
            narrative=d.get("narrative"),
            n_stops=d.get("n_stops", len(d["stops"])),
        )
        for d in days_payload
    ]
    pool_n = recommendation_pool_size or pool_size_from_plan(plan)
    stops_n = stops_from_pool or count_scheduled_stops(days_payload)
    pool_notice = _pool_notice(pool_n, stops_n, used_client_pool) if feasible and pool_n else None
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
        summary="AI itinerary is off. Turn on **Plan my days with AI** to generate a schedule.",
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
                "Run **Get recommendations** first, then **Build AI itinerary** "
                "so we only arrange places you already saw."
            ),
            plan_mode=PLAN_AI,
            feasible=False,
            skip_reason="pool_mismatch",
            narrative_source="fallback",
        )

    scheduling_pool = pool_results if used_client_pool else results
    pool_size = len(scheduling_pool)

    candidates = recommendations_for_prompt(scheduling_pool)  # entire Top-picks pool
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
        llm_out, rec_by_id, state.geo, req.trip_days,
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
    return _itinerary_response_from_plan(
        req,
        plan,
        notice=llm_notice,
        narrative_source=source,
        recommendation_pool_size=pool_size,
        stops_from_pool=int(plan.get("stops_from_pool") or 0),
        used_client_pool=used_client_pool,
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
            update={"itinerary_pool_ids": pool_ids, "use_ai_itinerary": True},
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
# Static frontend
# --------------------------------------------------------------------------- #
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(FRONTEND_DIR / "index.html"))
