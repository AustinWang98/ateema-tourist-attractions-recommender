"""CLI: geocode every location in location_dim using OpenStreetMap Nominatim.

Why this exists: the BQ location_dim only carries (id, name). We need
(lat, lon) for two features:
  * geographic routing inside an itinerary (TSP-style ordering)
  * the day-by-day map on the frontend

We use Nominatim because it's free and needs no API key. To stay polite
we throttle to 1 request/sec (their public usage policy) and we cache
every successful lookup to `data/geo_cache.json` so subsequent runs are
~instant. The cache is keyed by lower-cased location name so re-runs
do not duplicate requests when the dim file grows.

Usage:
    python -m backend.geocode                # geocode any missing names
    python -m backend.geocode --force        # re-geocode everything
    python -m backend.geocode --limit 50     # only do 50 (for testing)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

try:
    import httpx
except ImportError as exc:  # noqa: BLE001
    print(f"ERROR: missing httpx — run pip install -r requirements.txt ({exc})", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("geocode")


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "ChicagoDoes-Capstone/1.0 (uchicago.edu student project)"
DEFAULT_CITY_HINT = "Chicago, IL, USA"
RATE_LIMIT_SECS = 1.05   # be a little over 1s to be polite
TIMEOUT_SECS = 10
# Chicago Loop center used as a last-resort fallback so itinerary code
# never blows up on a missing coord. We tag these rows clearly so the
# UI can disclose them.
CHICAGO_LOOP = (41.8781, -87.6298)


def _normalise(name: str) -> str:
    return name.strip().lower()


def load_cache(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cache at %s is corrupt (%s) — starting empty.", path, exc)
        return {}


def save_cache(path: Path, cache: Dict[str, Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def geocode_one(client: httpx.Client, name: str) -> Optional[Dict]:
    """Hit Nominatim once for a single name. Returns dict with lat/lon
    + display info, or None if nothing matched."""
    params = {
        "q": f"{name}, {DEFAULT_CITY_HINT}",
        "format": "json",
        "addressdetails": 0,
        "limit": 1,
    }
    try:
        r = client.get(NOMINATIM_URL, params=params, timeout=TIMEOUT_SECS)
        r.raise_for_status()
        results = r.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Nominatim error for %s: %s", name, exc)
        return None
    if not results:
        return None
    top = results[0]
    try:
        lat = float(top["lat"])
        lon = float(top["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    return {
        "lat": lat,
        "lon": lon,
        "display_name": top.get("display_name"),
        "source": "nominatim",
    }


def export_csv(cache: Dict[str, Dict], dim_path: Path, csv_path: Path) -> int:
    """Join cache into location_dim and write a CSV the loader can read."""
    dim = pd.read_csv(dim_path)
    rows = []
    fallbacks = 0
    for _, row in dim.iterrows():
        loc_id = str(row["location_id"]).strip()
        name = str(row["location_name"]).strip()
        entry = cache.get(_normalise(name))
        if entry:
            rows.append({"location_id": loc_id, "location_name": name,
                         "lat": entry["lat"], "lon": entry["lon"],
                         "geo_source": entry.get("source", "nominatim")})
        else:
            rows.append({"location_id": loc_id, "location_name": name,
                         "lat": CHICAGO_LOOP[0], "lon": CHICAGO_LOOP[1],
                         "geo_source": "fallback"})
            fallbacks += 1
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("Exported %d rows to %s (%d fallback to Loop center)",
                len(rows), csv_path, fallbacks)
    return fallbacks


def main() -> int:
    ap = argparse.ArgumentParser(description="Geocode ChicagoDoes locations.")
    ap.add_argument("--dim", default="data/location_dim.csv")
    ap.add_argument("--cache", default="data/geo_cache.json")
    ap.add_argument("--out", default="data/locations_geo.csv")
    ap.add_argument("--force", action="store_true",
                    help="Re-geocode names already in the cache.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit how many new names to geocode in this run.")
    args = ap.parse_args()

    dim_path = Path(args.dim)
    cache_path = Path(args.cache)
    out_path = Path(args.out)
    if not dim_path.exists():
        print(f"ERROR: {dim_path} not found", file=sys.stderr)
        return 1

    cache = {} if args.force else load_cache(cache_path)
    dim = pd.read_csv(dim_path)
    names = dim["location_name"].dropna().astype(str).str.strip().tolist()
    pending = [n for n in names if _normalise(n) not in cache]
    if args.limit:
        pending = pending[: args.limit]

    print(f"Total locations: {len(names)}")
    print(f"Already cached:  {len(cache)}")
    print(f"To geocode:      {len(pending)}")
    if pending:
        est_secs = int(len(pending) * RATE_LIMIT_SECS)
        print(f"ETA: ~{est_secs // 60} min {est_secs % 60} sec at {RATE_LIMIT_SECS}s/req")
    print()

    if pending:
        with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
            for i, name in enumerate(pending, 1):
                hit = geocode_one(client, name)
                if hit:
                    cache[_normalise(name)] = hit
                    status = f"✓ ({hit['lat']:.4f}, {hit['lon']:.4f})"
                else:
                    cache[_normalise(name)] = None  # mark as tried
                    status = "✗ no match"
                print(f"  [{i:>3d}/{len(pending)}] {name[:50]:<50s} {status}")
                if i % 20 == 0:
                    save_cache(cache_path, cache)
                time.sleep(RATE_LIMIT_SECS)
        # final save (and drop None entries we just inserted — keep only hits)
        cache_clean = {k: v for k, v in cache.items() if v is not None}
        save_cache(cache_path, cache_clean)

    fallbacks = export_csv({k: v for k, v in cache.items() if v}, dim_path, out_path)
    print(f"\nDone. {len(cache) - fallbacks} geocoded, {fallbacks} fell back to Loop center.")
    print(f"  Cache:  {cache_path}")
    print(f"  Export: {out_path}")
    print("\nRestart the server (or hit POST /api/refresh) to use the new coords.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
