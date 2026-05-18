"""Assemble LLM-generated itineraries for the API (coords, route order, legs)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

from .geo import GeoIndex, order_stops_by_route, route_legs

PLAN_AI = "ai_generated"

SLOT_SORT_KEY: Dict[str, int] = {
    "breakfast": 0,
    "morning": 1,
    "lunch": 2,
    "afternoon": 3,
    "dinner": 4,
    "evening": 5,
    "drinks": 6,
}

_EXCLUDE_CATS = frozenset({"hotels", "neighborhood organizations", "favorites"})


def build_geo_index(frames) -> GeoIndex:
    if frames is not None and frames.locations is not None:
        return GeoIndex.from_locations_frame(frames.locations)
    return GeoIndex({})


def filter_to_recommendation_pool(
    recommendations: Sequence[Dict],
    pool_ids: Sequence[str],
) -> List[Dict]:
    """Keep only locations the user already saw in Top picks (preserve rank order)."""
    if not pool_ids:
        return list(recommendations)
    by_id = {str(r.get("location_id", "")): r for r in recommendations if r.get("location_id")}
    ordered: List[Dict] = []
    for lid in pool_ids:
        key = str(lid).strip()
        if key and key in by_id:
            ordered.append(by_id[key])
    return ordered


def recommendations_for_prompt(
    recommendations: Sequence[Dict],
    limit: Optional[int] = None,
) -> List[Dict]:
    """Send the full ranked Top-picks pool to the LLM (optional cap)."""
    pool = list(recommendations)
    if limit is not None:
        pool = pool[:limit]
    out: List[Dict] = []
    for rank, rec in enumerate(pool, start=1):
        lid = str(rec.get("location_id", ""))
        if not lid:
            continue
        cats = rec.get("categories") or []
        primary = rec.get("primary_category") or (cats[0] if cats else "")
        pl = str(primary).strip().lower()
        if pl in _EXCLUDE_CATS:
            continue
        reason = str(rec.get("reason") or "").strip()
        if len(reason) > 120:
            reason = reason[:117] + "..."
        out.append({
            "rank": rank,
            "location_id": lid,
            "location_name": str(rec.get("location_name", "")),
            "primary_category": primary,
            "categories": list(cats)[:4],
            "score": round(float(rec.get("final_score") or 0.0), 3),
            "why_recommended": reason,
        })
    return out


def count_scheduled_stops(days: List[Dict]) -> int:
    return sum(len(d.get("stops") or []) for d in days)


_SLOT_CYCLE = ("breakfast", "morning", "lunch", "afternoon", "dinner", "drinks")
_SLOT_LABELS = {
    "breakfast": "Breakfast",
    "morning": "Morning",
    "lunch": "Lunch",
    "afternoon": "Afternoon",
    "dinner": "Dinner",
    "drinks": "Drinks",
}


def deterministic_itinerary_payload(
    candidates: Sequence[Dict],
    trip_days: int,
    interests: Sequence[str],
) -> Dict[str, Any]:
    """Spread Top picks across days when the LLM cannot return valid JSON."""
    days_n = max(1, min(int(trip_days), 7))
    pool = [c for c in candidates if c.get("location_id")]
    if not pool:
        return {"summary": "", "days": [], "source": "fallback"}

    per_day = max(4, min(6, (len(pool) + days_n - 1) // days_n))
    interest_txt = ", ".join(list(interests)[:3]) if interests else "Chicago highlights"
    days_out: List[Dict] = []
    idx = 0

    for day_num in range(1, days_n + 1):
        stops: List[Dict] = []
        for slot in _SLOT_CYCLE:
            if len(stops) >= per_day or idx >= len(pool):
                break
            c = pool[idx]
            idx += 1
            stops.append({
                "location_id": str(c["location_id"]),
                "slot": slot,
                "slot_label": _SLOT_LABELS[slot],
                "note": str(c.get("why_recommended") or c.get("location_name", ""))[:80],
            })
        if stops:
            days_out.append({
                "day_number": day_num,
                "theme": f"Day {day_num}",
                "narrative": f"Top picks for {interest_txt}.",
                "stops": stops,
            })
        if idx >= len(pool):
            break

    summary = (
        f"A {len(days_out)}-day schedule from your ranked recommendations "
        f"({interest_txt}) — auto-arranged while the AI formatter recovers."
    )
    return {"summary": summary, "days": days_out, "source": "fallback_schedule"}


def assemble_itinerary_plan(
    llm_payload: Dict[str, Any],
    rec_by_id: Dict[str, Dict],
    geo: GeoIndex,
    trip_days: int,
) -> Dict[str, Any]:
    """Validate LLM JSON, attach coordinates, sort stops, compute legs."""
    days_in = llm_payload.get("days") or []
    days_out: List[Dict] = []
    used_global: Set[str] = set()

    for raw_day in days_in:
        if len(days_out) >= trip_days:
            break
        day_num = int(raw_day.get("day_number") or len(days_out) + 1)
        stops_raw = raw_day.get("stops") or []
        stop_dicts: List[Dict] = []
        seen_day: Set[str] = set()

        for s in stops_raw:
            lid = str(s.get("location_id", ""))
            if not lid or lid in seen_day or lid in used_global:
                continue
            rec = rec_by_id.get(lid)
            if rec is None:
                continue
            primary = rec.get("primary_category") or s.get("primary_category")
            if str(primary or "").strip().lower() in _EXCLUDE_CATS:
                continue
            slot = str(s.get("slot") or "afternoon").strip().lower()
            slot_label = str(s.get("slot_label") or slot.title()).strip()
            coord = geo.get(lid)
            stop_dicts.append({
                "slot": slot,
                "slot_label": slot_label,
                "location_id": lid,
                "location_name": str(rec.get("location_name", s.get("location_name", ""))),
                "primary_category": primary,
                "note": str(s.get("note") or rec.get("reason") or "").strip(),
                "lat": coord.lat if coord else None,
                "lon": coord.lon if coord else None,
                "geo_source": coord.source if coord else None,
            })
            seen_day.add(lid)
            used_global.add(lid)

        if not stop_dicts:
            continue

        stop_dicts.sort(key=lambda x: SLOT_SORT_KEY.get(x["slot"], 99))
        visit_idxs = [
            i for i, st in enumerate(stop_dicts)
            if st["slot"] in ("morning", "afternoon")
        ]
        if len(visit_idxs) >= 2:
            visit_stops = [stop_dicts[i] for i in visit_idxs]
            ordered = order_stops_by_route(visit_stops, geo)
            for j, idx in enumerate(visit_idxs):
                stop_dicts[idx] = ordered[j]

        legs = route_legs(stop_dicts, geo)
        days_out.append({
            "day_number": day_num,
            "theme": str(raw_day.get("theme") or f"Day {day_num}"),
            "stops": stop_dicts,
            "legs": legs,
            "narrative": str(raw_day.get("narrative") or "").strip() or None,
            "n_stops": len(stop_dicts),
        })

    summary = str(llm_payload.get("summary") or "").strip()
    if not summary and days_out:
        summary = f"A {len(days_out)}-day Chicago plan from your top recommendations."

    return {
        "days": days_out,
        "summary": summary or "AI itinerary",
        "plan_mode": PLAN_AI,
        "feasible": len(days_out) > 0,
        "skip_reason": None if days_out else "llm_empty_plan",
        "stops_from_pool": count_scheduled_stops(days_out),
    }
