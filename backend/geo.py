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
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional

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
