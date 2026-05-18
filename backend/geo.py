"""Geographic utilities for itinerary maps and route legs.

Functions
---------
haversine_km(lat1, lon1, lat2, lon2)
    Great-circle distance in km.

travel_minutes(km, mode)
    Estimate travel time. `mode` is one of:
        "walk"  : 5 km/h (Chicago Loop pedestrian speed)
        "drive" : 25 km/h (city driving incl. traffic + parking)
        "transit": 18 km/h (CTA mix, rough average)
    We multiply crow-flies distance by a 1.3 city-grid factor before
    converting to time, because Chicago is a grid and people don't
    actually walk diagonally through buildings.

choose_mode(km)
    Pick a sensible default mode based on distance:
        ≤ 1.2 km   -> walk
        ≤ 8 km     -> transit
        otherwise  -> drive

GeoIndex
    Wraps the locations DataFrame and exposes fast coord lookup.
    Returns (None, None) for ids we have no coords for; callers must
    degrade gracefully (no map, no travel chips) rather than crashing.

order_stops_by_route(stops, geo)
    Greedy nearest-neighbour ordering: start at the highest-scored
    stop, then always visit the closest unvisited stop next. For 2-6
    stops per day this is near-optimal and 100× faster than full TSP.

route_legs(stops, geo)
    For an ordered list of stops, return [{from, to, km, mode, minutes}].
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


CITY_GRID_FACTOR = 1.3
EARTH_R_KM = 6371.0088

# km/h for each travel mode. Conservative city values.
MODE_SPEEDS_KMH = {
    "walk":    5.0,
    "transit": 18.0,
    "drive":   25.0,
}

MODE_THRESHOLDS_KM = (
    (1.2, "walk"),
    (8.0, "transit"),
    # > 8 km falls through to drive
)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


def choose_mode(km: float) -> str:
    for thr, mode in MODE_THRESHOLDS_KM:
        if km <= thr:
            return mode
    return "drive"


def travel_minutes(km: float, mode: Optional[str] = None) -> int:
    """Convert distance to minutes, applying city-grid detour factor."""
    if mode is None:
        mode = choose_mode(km)
    speed = MODE_SPEEDS_KMH.get(mode, MODE_SPEEDS_KMH["walk"])
    effective_km = km * CITY_GRID_FACTOR
    minutes = (effective_km / speed) * 60.0
    return max(1, int(round(minutes)))


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Coord:
    lat: float
    lon: float
    source: str   # "nominatim" or "fallback"


class GeoIndex:
    """Look up coords by location_id. Missing ids -> None."""

    def __init__(self, by_id: Dict[str, Coord]) -> None:
        self._by_id = by_id

    @classmethod
    def from_locations_frame(cls, locations: pd.DataFrame) -> "GeoIndex":
        """Build from a `locations` DataFrame that has lat/lon columns.

        Rows without coords are skipped, so subsequent lookups return None.
        """
        if locations is None or len(locations) == 0:
            return cls({})
        if "lat" not in locations.columns or "lon" not in locations.columns:
            logger.info("GeoIndex: locations frame has no lat/lon; routing disabled.")
            return cls({})

        by_id: Dict[str, Coord] = {}
        for _, row in locations.iterrows():
            lat, lon = row.get("lat"), row.get("lon")
            if pd.isna(lat) or pd.isna(lon):
                continue
            try:
                lat_f, lon_f = float(lat), float(lon)
            except (TypeError, ValueError):
                continue
            if not (-90 <= lat_f <= 90 and -180 <= lon_f <= 180):
                continue
            by_id[str(row["location_id"])] = Coord(
                lat=lat_f, lon=lon_f,
                source=str(row.get("geo_source") or "unknown"),
            )
        logger.info("GeoIndex: indexed %d / %d locations with coords",
                    len(by_id), len(locations))
        return cls(by_id)

    def get(self, location_id: str) -> Optional[Coord]:
        return self._by_id.get(str(location_id))

    def has(self, location_id: str) -> bool:
        return str(location_id) in self._by_id

    @property
    def size(self) -> int:
        return len(self._by_id)


# --------------------------------------------------------------------------- #
def order_stops_by_route(
    stops: List[dict],
    geo: GeoIndex,
) -> List[dict]:
    """Reorder stops within a single day using greedy nearest-neighbour.

    We honour the slot bucket order (morning < afternoon < evening) as a
    hard prior, then within each slot pick the nearest unvisited stop
    next. The starting stop in each slot is the highest-scored stop
    that landed there (already true after the caller's sort).

    Inputs: list of stop dicts that already have location_id + slot.
    Output: same list, reordered. Stops without coords are placed
    last within their slot so they don't break the chain.
    """
    if not stops:
        return stops

    slot_priority = {"morning": 0, "afternoon": 1, "evening": 2}

    # Group by slot preserving original (score) order
    by_slot: Dict[str, List[dict]] = {}
    for s in stops:
        by_slot.setdefault(s.get("slot", "afternoon"), []).append(s)

    ordered: List[dict] = []
    last_coord: Optional[Coord] = None

    for slot in sorted(by_slot.keys(), key=lambda x: slot_priority.get(x, 99)):
        bucket = list(by_slot[slot])
        has_coord = [s for s in bucket if geo.has(s["location_id"])]
        no_coord = [s for s in bucket if not geo.has(s["location_id"])]

        # Start the slot from either the previous slot's last coord
        # (for cross-slot continuity) or the highest-scored stop.
        chain: List[dict] = []
        remaining = list(has_coord)
        if remaining:
            if last_coord is None:
                # take the first (highest score) as anchor
                anchor = remaining.pop(0)
            else:
                # start with whichever remaining stop is nearest to last_coord
                anchor = min(remaining, key=lambda s: _km(geo, last_coord, s))
                remaining.remove(anchor)
            chain.append(anchor)
            last_coord = geo.get(anchor["location_id"])

            while remaining:
                nxt = min(remaining, key=lambda s: _km(geo, last_coord, s))
                remaining.remove(nxt)
                chain.append(nxt)
                last_coord = geo.get(nxt["location_id"])

        ordered.extend(chain)
        ordered.extend(no_coord)   # tail any geo-missing stops within slot

    return ordered


def _km(geo: GeoIndex, anchor: Optional[Coord], stop: dict) -> float:
    c = geo.get(stop["location_id"])
    if c is None or anchor is None:
        return 9999.0
    return haversine_km(anchor.lat, anchor.lon, c.lat, c.lon)


def route_legs(stops: List[dict], geo: GeoIndex) -> List[dict]:
    """Build leg-by-leg travel info for the ordered stops.

    Each leg connects stop[i] -> stop[i+1] and gets:
        from_id, to_id, from_name, to_name, km, mode, minutes
    Stops without coords break the chain; those legs are emitted with
    km=None / minutes=None / mode='unknown' so the UI can still show
    them while flagging the gap.
    """
    legs: List[dict] = []
    for i in range(len(stops) - 1):
        a, b = stops[i], stops[i + 1]
        ca, cb = geo.get(a["location_id"]), geo.get(b["location_id"])
        if ca is None or cb is None or ca.source == "fallback" or cb.source == "fallback":
            legs.append({
                "from_id":   a["location_id"],
                "to_id":     b["location_id"],
                "from_name": a["location_name"],
                "to_name":   b["location_name"],
                "km":        None,
                "mode":      "unknown",
                "minutes":   None,
            })
            continue
        km = haversine_km(ca.lat, ca.lon, cb.lat, cb.lon)
        mode = choose_mode(km)
        legs.append({
            "from_id":   a["location_id"],
            "to_id":     b["location_id"],
            "from_name": a["location_name"],
            "to_name":   b["location_name"],
            "km":        round(km, 2),
            "mode":      mode,
            "minutes":   travel_minutes(km, mode),
        })
    return legs
