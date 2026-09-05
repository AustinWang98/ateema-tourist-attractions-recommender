"""Assemble LLM-generated itineraries for the API (coords, route order, legs)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set

from .geo import Coord, GeoIndex, haversine_km

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

# --------------------------------------------------------------------------- #
# Slot / meal logic
# --------------------------------------------------------------------------- #
# Meal slots that must be filled by an eating/drinking venue.
MEAL_SLOTS = frozenset({"breakfast", "lunch", "dinner"})
FOOD_CATS = frozenset({"restaurants"})
DRINK_CATS = frozenset({"bars"})

# Which time slots each category is a sensible fit for. Used both to guide the
# LLM (in the prompt) and to sanity-correct its output afterwards. Keys are
# lower-cased category names.
SLOTS_BY_CATEGORY: Dict[str, List[str]] = {
    "restaurants":                ["breakfast", "lunch", "dinner"],
    "bars":                       ["drinks", "evening"],
    "museums":                    ["morning", "afternoon"],
    "parks":                      ["morning", "afternoon"],
    "attractions":                ["morning", "afternoon"],
    "shops":                      ["morning", "afternoon"],
    "murals":                     ["morning", "afternoon"],
    "movie & tv locations":       ["morning", "afternoon"],
    "big bus tours stops":        ["morning", "afternoon"],
    "sports venues":              ["afternoon", "evening"],
    "theaters and music venues":  ["evening", "drinks"],
    "hot spots":                  ["morning", "afternoon", "evening"],
}
_DEFAULT_SLOTS = ["morning", "afternoon"]


def suitable_slots(categories: Sequence[str]) -> List[str]:
    """Time slots a place can reasonably occupy, given its categories."""
    out: List[str] = []
    for c in categories or []:
        for slot in SLOTS_BY_CATEGORY.get(str(c).strip().lower(), []):
            if slot not in out:
                out.append(slot)
    return out or list(_DEFAULT_SLOTS)


def _is_food(categories: Sequence[str]) -> bool:
    return any(str(c).strip().lower() in FOOD_CATS for c in (categories or []))


def _is_drink(categories: Sequence[str]) -> bool:
    return any(str(c).strip().lower() in DRINK_CATS for c in (categories or []))


def assign_area_groups(
    candidates: Sequence[Dict],
    geo: GeoIndex,
    radius_km: float = 2.0,
) -> Dict[str, str]:
    """Greedy geographic clustering so the LLM can keep each day compact.

    Walks candidates in rank order; each location joins the first existing
    cluster whose centre is within ``radius_km``, otherwise it seeds a new
    cluster. Returns ``{location_id: "Area N"}``. Locations without coords get
    ``"Area ?"`` so the model knows their proximity is unknown.
    """
    centres: List[Coord] = []
    labels: Dict[str, str] = {}
    for c in candidates:
        lid = str(c.get("location_id", ""))
        if not lid:
            continue
        coord = geo.get(lid)
        if coord is None:
            labels[lid] = "Area ?"
            continue
        assigned: Optional[int] = None
        for i, ctr in enumerate(centres):
            if haversine_km(coord.lat, coord.lon, ctr.lat, ctr.lon) <= radius_km:
                assigned = i
                break
        if assigned is None:
            centres.append(coord)
            assigned = len(centres) - 1
        labels[lid] = f"Area {assigned + 1}"
    return labels


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
    seen_names: Set[str] = set()
    for lid in pool_ids:
        key = str(lid).strip()
        if key and key in by_id:
            rec = by_id[key]
            name_key = _norm_name(rec.get("location_name"))
            if name_key and name_key in seen_names:
                continue
            ordered.append(rec)
            if name_key:
                seen_names.add(name_key)
    return ordered


def recommendations_for_prompt(
    recommendations: Sequence[Dict],
    limit: Optional[int] = None,
    geo: Optional[GeoIndex] = None,
) -> List[Dict]:
    """Send the ranked Top-picks pool to the LLM as enriched candidates.

    Each candidate carries the fields the planner needs to reason well:
    its categories, the time `slots` it suits (so the model never schedules
    a bar for breakfast), and a geographic `area` label (so it can keep each
    day's stops close together for a convenient route).
    """
    pool = list(recommendations)
    if limit is not None:
        pool = pool[:limit]

    # Pre-compute geographic clusters across the whole pool (rank order).
    area_by_id = assign_area_groups(pool, geo) if geo is not None else {}

    out: List[Dict] = []
    seen_names: Set[str] = set()
    for rank, rec in enumerate(pool, start=1):
        lid = str(rec.get("location_id", ""))
        if not lid:
            continue
        name_key = _norm_name(rec.get("location_name"))
        if name_key and name_key in seen_names:
            continue
        cats = rec.get("categories") or []
        primary = rec.get("primary_category") or (cats[0] if cats else "")
        pl = str(primary).strip().lower()
        if pl in _EXCLUDE_CATS:
            continue
        reason = str(rec.get("reason") or "").strip()
        if len(reason) > 120:
            reason = reason[:117] + "..."
        coord = geo.get(lid) if geo is not None else None
        out.append({
            "rank": rank,
            "location_id": lid,
            "location_name": str(rec.get("location_name", "")),
            "primary_category": primary,
            "categories": list(cats)[:4],
            "slots": suitable_slots([primary, *cats]),
            "area": area_by_id.get(lid, "Area ?"),
            "lat": round(coord.lat, 4) if coord else None,
            "lon": round(coord.lon, 4) if coord else None,
            "score": round(float(rec.get("final_score") or 0.0), 3),
            "why_recommended": reason,
        })
        if name_key:
            seen_names.add(name_key)
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
    pool: List[Dict] = []
    seen_names: Set[str] = set()
    for c in candidates:
        if not c.get("location_id"):
            continue
        name_key = _norm_name(c.get("location_name"))
        if name_key and name_key in seen_names:
            continue
        pool.append(c)
        if name_key:
            seen_names.add(name_key)
    if not pool:
        return {"summary": "", "days": [], "source": "fallback"}

    per_day = max(MIN_STOPS_PER_DAY, min(MAX_STOPS_PER_DAY, (len(pool) + days_n - 1) // days_n))
    interest_txt = ", ".join(list(interests)[:3]) if interests else "Chicago highlights"
    days_out: List[Dict] = []
    idx = 0

    for day_num in range(1, days_n + 1):
        stops: List[Dict] = []
        used_slots: Set[str] = set()
        for _ in range(len(pool)):
            if len(stops) >= per_day or idx >= len(pool):
                break
            c = pool[idx]
            idx += 1
            cats = [c.get("primary_category"), *(c.get("categories") or [])]
            options = [s for s in suitable_slots(cats) if s not in used_slots]
            slot = options[0] if options else next(
                (s for s in _SLOT_CYCLE if s not in used_slots), "afternoon"
            )
            used_slots.add(slot)
            stops.append({
                "location_id": str(c["location_id"]),
                "slot": slot,
                "slot_label": _SLOT_LABELS.get(slot, slot.title()),
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


MIN_STOPS_PER_DAY = 3
MAX_STOPS_PER_DAY = 5
# The planner may add a small number of real Chicago places outside the
# ChicagoDoes catalog. Those stops can render without media; catalog stops still
# get verified card media attached by the API response.
MAX_AI_STOPS_PER_DAY = 2

# Rough Chicago bounding box — used to sanity-check LLM-provided coordinates for
# places that are not in our warehouse (so a bad number can't break the map).
_CHI_LAT = (41.5, 42.2)
_CHI_LON = (-88.1, -87.3)


def _valid_chicago_coord(lat: Any, lon: Any) -> Optional[tuple]:
    try:
        latf, lonf = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if _CHI_LAT[0] <= latf <= _CHI_LAT[1] and _CHI_LON[0] <= lonf <= _CHI_LON[1]:
        return (latf, lonf)
    return None


def _norm_name(name: str) -> str:
    text = str(name or "").lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _clean_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def build_location_indexes(
    locations_df,
) -> tuple[Dict[str, Dict], Dict[str, str]]:
    """Map warehouse location_id and normalized name -> recommendation-shaped dict."""
    by_id: Dict[str, Dict] = {}
    by_name: Dict[str, str] = {}
    if locations_df is None or getattr(locations_df, "empty", True):
        return by_id, by_name

    for _, row in locations_df.iterrows():
        lid = str(row.get("location_id") or "").strip()
        if not lid:
            continue
        name = str(row.get("location_name") or "")
        cats = row.get("categories")
        if hasattr(cats, "tolist"):
            cats = cats.tolist()
        elif not cats:
            cats = []
        by_id[lid] = {
            "location_id": lid,
            "location_name": name,
            "primary_category": _clean_optional_text(row.get("primary_category")),
            "categories": [
                c for c in (_clean_optional_text(c) for c in list(cats)) if c
            ],
            "reason": "",
        }
        nk = _norm_name(name)
        if nk and nk not in by_name:
            by_name[nk] = lid
    return by_id, by_name


def _resolve_location_record(
    raw: Dict,
    rec_by_id: Dict[str, Dict],
    warehouse_by_id: Dict[str, Dict],
    warehouse_by_name: Dict[str, str],
) -> tuple[Optional[Dict], str]:
    """Match an LLM stop to a warehouse row by id or name.

    The LLM often returns only a name (or an id that is not in the Top-picks
    pool). Without this, those stops keep ``location_id=""`` and the UI cannot
    deep-link to the venue's ChicagoDoes page.
    """
    lid = str(raw.get("location_id") or "").strip()
    name = str(raw.get("location_name") or raw.get("name") or "").strip()

    if lid and lid in rec_by_id:
        return rec_by_id[lid], lid
    if lid and lid in warehouse_by_id:
        return warehouse_by_id[lid], lid

    nk = _norm_name(name)
    if nk and nk in warehouse_by_name:
        rid = warehouse_by_name[nk]
        rec = rec_by_id.get(rid) or warehouse_by_id.get(rid)
        if rec:
            return rec, rid

    if nk:
        for rec in rec_by_id.values():
            if _norm_name(rec.get("location_name")) == nk:
                return rec, str(rec["location_id"])
        for rec in warehouse_by_id.values():
            if _norm_name(rec.get("location_name")) == nk:
                return rec, str(rec["location_id"])

    return None, ""


def _coord_of(stop: Dict) -> Optional[tuple]:
    lat, lon = stop.get("lat"), stop.get("lon")
    if lat is None or lon is None:
        return None
    return (float(lat), float(lon))


def _order_visits_by_coords(stops: List[Dict]) -> List[Dict]:
    """Greedy nearest-neighbour ordering using each stop's attached lat/lon.

    Works for both warehouse picks and LLM-added places (coords already on the
    stop dict), so the route stays sensible regardless of where a stop came from.
    """
    remaining = list(stops)
    with_coord = [s for s in remaining if _coord_of(s)]
    if len(with_coord) < 2:
        return stops
    ordered: List[Dict] = [with_coord.pop(0)]
    while with_coord:
        last = _coord_of(ordered[-1])
        nxt = min(with_coord, key=lambda s: haversine_km(last[0], last[1], *_coord_of(s)))
        with_coord.remove(nxt)
        ordered.append(nxt)
    # Re-insert any coordless stops at the end, preserving original order.
    ordered.extend([s for s in remaining if not _coord_of(s)])
    return ordered


def _legs_from_coords(stops: List[Dict]) -> List[Dict]:
    """Leg-by-leg travel info computed from coords attached to each stop."""
    from .geo import choose_mode, travel_minutes  # local import avoids cycle churn
    legs: List[Dict] = []
    for i in range(len(stops) - 1):
        a, b = stops[i], stops[i + 1]
        ca, cb = _coord_of(a), _coord_of(b)
        if ca is None or cb is None or a.get("geo_source") == "fallback" or b.get("geo_source") == "fallback":
            legs.append({
                "from_id": a.get("location_id", ""),
                "to_id": b.get("location_id", ""),
                "from_name": a.get("location_name", ""),
                "to_name": b.get("location_name", ""),
                "km": None, "mode": "unknown", "minutes": None,
            })
            continue
        km = haversine_km(ca[0], ca[1], cb[0], cb[1])
        mode = choose_mode(km)
        legs.append({
            "from_id": a.get("location_id", ""),
            "to_id": b.get("location_id", ""),
            "from_name": a.get("location_name", ""),
            "to_name": b.get("location_name", ""),
            "km": round(km, 2), "mode": mode, "minutes": travel_minutes(km, mode),
        })
    return legs


def _correct_slot(slot: str, categories: Sequence[str]) -> str:
    """Keep the LLM's slot when it makes sense; fix obvious mistakes.

    Rules: a meal slot (breakfast/lunch/dinner) must be a place you can eat at;
    bars belong in drinks/evening, never at breakfast/lunch. Non-food daytime
    places are pushed to morning/afternoon.
    """
    s = (slot or "afternoon").strip().lower()
    food, drink = _is_food(categories), _is_drink(categories)
    if s in MEAL_SLOTS and not food:
        if drink:
            return "drinks"
        return "morning" if s == "breakfast" else "afternoon"
    if s == "drinks" and not (drink or food):
        return "evening"
    if s not in SLOT_SORT_KEY:
        return "afternoon"
    return s


def _slot_label(slot: str) -> str:
    return _SLOT_LABELS.get(slot, slot.title())


def _make_stop(lid: str, slot: str, rec: Dict, geo: GeoIndex, note: str = "") -> Dict:
    coord = geo.get(lid)
    categories = [
        c for c in (_clean_optional_text(c) for c in (rec.get("categories") or [])) if c
    ]
    primary = _clean_optional_text(rec.get("primary_category")) or (
        categories[0] if categories else None
    )
    return {
        "slot": slot,
        "slot_label": _slot_label(slot),
        "location_id": lid,
        "location_name": str(rec.get("location_name", "")),
        "primary_category": primary,
        "note": str(note or rec.get("reason") or "").strip(),
        "lat": coord.lat if coord else None,
        "lon": coord.lon if coord else None,
        "geo_source": coord.source if coord else None,
        "source": "recommended",
    }


def _make_ai_stop(raw: Dict, slot: str) -> Optional[Dict]:
    """Build a stop for a real Chicago place the LLM added (not in our pool).

    Requires a name; coordinates are used only if they look like valid Chicago
    points, otherwise the stop still shows but without a map pin.
    """
    name = str(raw.get("location_name") or raw.get("name") or "").strip()
    if not name:
        return None
    primary = str(raw.get("primary_category") or raw.get("category") or "").strip() or None
    if str(primary or "").strip().lower() in _EXCLUDE_CATS:
        return None
    coord = _valid_chicago_coord(raw.get("lat"), raw.get("lon"))
    return {
        "slot": slot,
        "slot_label": _slot_label(slot),
        "location_id": "",
        "location_name": name,
        "primary_category": primary,
        "note": str(raw.get("note") or "").strip(),
        "lat": coord[0] if coord else None,
        "lon": coord[1] if coord else None,
        "geo_source": "ai" if coord else None,
        "source": "ai",
    }


# Distance thresholds (km) that keep a single day geographically coherent.
NEARBY_KM = 4.0     # backfill prefers picks this close to the day's core
OUTLIER_KM = 7.0    # a stop farther than this from the day's core is a zig-zag


def _medoid_coord(stops: List[Dict]) -> Optional[tuple]:
    """The 'most central' stop coordinate (minimizes total distance to others).

    Using a medoid instead of a mean makes the day's core robust to a single
    far-flung outlier, which is exactly the case we want to detect.
    """
    coords = [c for c in (_coord_of(s) for s in stops) if c]
    if not coords:
        return None
    if len(coords) == 1:
        return coords[0]
    best, best_sum = None, None
    for ci in coords:
        tot = sum(haversine_km(ci[0], ci[1], cj[0], cj[1]) for cj in coords)
        if best_sum is None or tot < best_sum:
            best, best_sum = ci, tot
    return best


def _enforce_day_compactness(stop_dicts: List[Dict]) -> List[Dict]:
    """Drop stops that sit far outside the day's geographic core.

    Prevents an itinerary that hops north -> far-south -> north in one day.
    Only fires when there is a clear core (>=2 stops clustered together), so a
    legitimately spread-out day is left alone; backfill later refills near the
    core to keep a reasonable number of stops.
    """
    coord_stops = [s for s in stop_dicts if _coord_of(s)]
    if len(coord_stops) < 3:
        return stop_dicts
    medoid = _medoid_coord(stop_dicts)
    if medoid is None:
        return stop_dicts
    core = [
        s for s in coord_stops
        if haversine_km(medoid[0], medoid[1], *_coord_of(s)) <= OUTLIER_KM
    ]
    if len(core) < 2:
        return stop_dicts  # no compact core to anchor on; leave as-is
    kept: List[Dict] = []
    for s in stop_dicts:
        c = _coord_of(s)
        if c and haversine_km(medoid[0], medoid[1], c[0], c[1]) > OUTLIER_KM:
            continue  # far outlier — drop so the day stays in one part of town
        kept.append(s)
    return kept or stop_dicts


def _backfill_day(
    stop_dicts: List[Dict],
    seen_day: Set[str],
    used_global: Set[str],
    rec_by_id: Dict[str, Dict],
    geo: GeoIndex,
    target: int = MIN_STOPS_PER_DAY,
) -> None:
    """Top up a thin day with unused picks that are NEAR the day's core.

    Reaches ``target`` stops when material exists. Among eligible picks we
    prefer ones within NEARBY_KM of the day's core (keeping the route compact),
    and among those the highest-ranked; if none are close, we take the nearest
    available so we never re-introduce a far-flung outlier.
    """
    used_slots = {st["slot"] for st in stop_dicts}
    while len(stop_dicts) < target:
        anchor = _medoid_coord(stop_dicts)
        eligible: List[tuple] = []  # (lid, rec, slot, dist_or_None) in rank order
        for lid, rec in rec_by_id.items():
            name_key = _norm_name(rec.get("location_name"))
            if lid in seen_day or lid in used_global:
                continue
            if name_key and (name_key in seen_day or name_key in used_global):
                continue
            if str(rec.get("primary_category") or "").strip().lower() in _EXCLUDE_CATS:
                continue
            cats = [rec.get("primary_category"), *(rec.get("categories") or [])]
            open_slots = [s for s in suitable_slots(cats) if s not in used_slots]
            if not open_slots:
                continue
            coord = geo.get(lid)
            dist = (
                haversine_km(anchor[0], anchor[1], coord.lat, coord.lon)
                if (anchor and coord) else None
            )
            eligible.append((lid, rec, open_slots[0], dist))
        if not eligible:
            break
        within = [e for e in eligible if e[3] is not None and e[3] <= NEARBY_KM]
        with_dist = [e for e in eligible if e[3] is not None]
        if within:
            pick = within[0]                       # nearest tier, highest rank
        elif with_dist:
            pick = min(with_dist, key=lambda e: e[3])  # nearest available
        else:
            pick = eligible[0]                     # no coords anywhere; rank order
        lid, rec, slot, _ = pick
        stop_dicts.append(_make_stop(lid, slot, rec, geo))
        used_slots.add(slot)
        name_key = _norm_name(rec.get("location_name"))
        seen_day.add(lid)
        used_global.add(lid)
        if name_key:
            seen_day.add(name_key)
            used_global.add(name_key)


FILL_DAY_TARGET = 4  # stops per auto-generated extra day


def _build_extra_day(
    day_num: int,
    used_global: Set[str],
    rec_by_id: Dict[str, Dict],
    geo: GeoIndex,
) -> Optional[Dict]:
    """Compose one more day from the best unused picks (meal-aware, routed).

    Used to guarantee the plan reaches the requested number of days even when
    the LLM returns fewer. Returns None when too few locations remain.
    """
    stop_dicts: List[Dict] = []
    seen_day: Set[str] = set()
    _backfill_day(stop_dicts, seen_day, used_global, rec_by_id, geo, target=FILL_DAY_TARGET)
    if len(stop_dicts) < 2:
        return None

    stop_dicts.sort(key=lambda x: SLOT_SORT_KEY.get(x["slot"], 99))
    visit_idxs = [
        i for i, st in enumerate(stop_dicts)
        if st["slot"] in ("morning", "afternoon")
    ]
    if len(visit_idxs) >= 2:
        ordered = _order_visits_by_coords([stop_dicts[i] for i in visit_idxs])
        for j, idx in enumerate(visit_idxs):
            stop_dicts[idx] = ordered[j]

    return {
        "day_number": day_num,
        "theme": f"Day {day_num}: More top picks",
        "stops": stop_dicts,
        "legs": _legs_from_coords(stop_dicts),
        "narrative": "More of your top Chicago picks, grouped for an easy route.",
        "n_stops": len(stop_dicts),
    }


def assemble_itinerary_plan(
    llm_payload: Dict[str, Any],
    rec_by_id: Dict[str, Dict],
    geo: GeoIndex,
    trip_days: int,
    warehouse_by_id: Optional[Dict[str, Dict]] = None,
    warehouse_by_name: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Validate LLM JSON, attach coordinates, fix meal slots, route-order, backfill."""
    warehouse_by_id = warehouse_by_id or {}
    warehouse_by_name = warehouse_by_name or {}
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
        ai_count = 0

        for s in stops_raw:
            if len(stop_dicts) >= MAX_STOPS_PER_DAY:
                break
            resolved_rec, resolved_lid = _resolve_location_record(
                s, rec_by_id, warehouse_by_id, warehouse_by_name,
            )

            if resolved_rec and resolved_lid:
                # A warehouse venue — from Top picks or matched by name/id.
                name_key = _norm_name(resolved_rec.get("location_name"))
                if resolved_lid in seen_day or resolved_lid in used_global:
                    continue
                if name_key and (name_key in seen_day or name_key in used_global):
                    continue
                cats = [
                    resolved_rec.get("primary_category"),
                    *(resolved_rec.get("categories") or []),
                ]
                if str(resolved_rec.get("primary_category") or "").strip().lower() in _EXCLUDE_CATS:
                    continue
                slot = _correct_slot(str(s.get("slot") or "afternoon"), cats)
                stop = _make_stop(
                    resolved_lid, slot, resolved_rec, geo, note=str(s.get("note") or ""),
                )
                if resolved_lid not in rec_by_id:
                    stop["source"] = "ai"
                stop_dicts.append(stop)
                seen_day.add(resolved_lid)
                used_global.add(resolved_lid)
                if name_key:
                    seen_day.add(name_key)
                    used_global.add(name_key)
            else:
                # A real Chicago place the LLM added on its own. Keep a few.
                if ai_count >= MAX_AI_STOPS_PER_DAY:
                    continue
                name_key = _norm_name(s.get("location_name") or s.get("name"))
                if not name_key or name_key in seen_day or name_key in used_global:
                    continue
                cats = [s.get("primary_category") or s.get("category")]
                slot = _correct_slot(str(s.get("slot") or "afternoon"), cats)
                stop = _make_ai_stop(s, slot)
                if stop is None:
                    continue
                stop_dicts.append(stop)
                seen_day.add(name_key)
                used_global.add(name_key)
                ai_count += 1

        # Keep the day geographically coherent (no north -> far-south -> north),
        # then top up near the core so we still show a sensible number of stops.
        stop_dicts = _enforce_day_compactness(stop_dicts)
        _backfill_day(stop_dicts, seen_day, used_global, rec_by_id, geo)

        if not stop_dicts:
            continue

        stop_dicts.sort(key=lambda x: SLOT_SORT_KEY.get(x["slot"], 99))
        visit_idxs = [
            i for i, st in enumerate(stop_dicts)
            if st["slot"] in ("morning", "afternoon")
        ]
        if len(visit_idxs) >= 2:
            visit_stops = [stop_dicts[i] for i in visit_idxs]
            ordered = _order_visits_by_coords(visit_stops)
            for j, idx in enumerate(visit_idxs):
                stop_dicts[idx] = ordered[j]

        legs = _legs_from_coords(stop_dicts)
        days_out.append({
            "day_number": day_num,
            "theme": str(raw_day.get("theme") or f"Day {day_num}"),
            "stops": stop_dicts,
            "legs": legs,
            "narrative": str(raw_day.get("narrative") or "").strip() or None,
            "n_stops": len(stop_dicts),
        })

    # Safety net: guarantee the requested number of days. If the LLM returned
    # fewer (a known failure mode of small models / stale caches), top up with
    # extra days built from the best unused picks so the user always gets the
    # trip length they asked for.
    while len(days_out) < trip_days:
        extra = _build_extra_day(len(days_out) + 1, used_global, rec_by_id, geo)
        if extra is None:
            break  # not enough locations left to form another day
        days_out.append(extra)

    summary = str(llm_payload.get("summary") or "").strip()
    if not summary and days_out:
        summary = f"A {len(days_out)}-day Chicago plan built around your top recommendations."

    pool_stops = sum(
        1 for d in days_out for st in d["stops"] if st.get("source") != "ai"
    )
    ai_stops = sum(
        1 for d in days_out for st in d["stops"] if st.get("source") == "ai"
    )

    return {
        "days": days_out,
        "summary": summary or "AI itinerary",
        "plan_mode": PLAN_AI,
        "feasible": len(days_out) > 0,
        "skip_reason": None if days_out else "llm_empty_plan",
        "stops_from_pool": pool_stops,
        "ai_added_stops": ai_stops,
    }
