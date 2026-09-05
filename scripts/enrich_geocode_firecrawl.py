#!/usr/bin/env python3
"""Enrich locations_geo.csv using Firecrawl search + Nominatim geocoding.

Targets rows that still use the Loop fallback (41.8781, -87.6298) or
geo_source=fallback. For each venue:

  1. Firecrawl web search (description often has street address)
  2. Optional scrape of top official / Choose Chicago / Maps URL
  3. Extract lat/lon from Google Maps links or geocode street address
  4. Write geo_cache.json and re-export locations_geo.csv

Usage:
    python scripts/enrich_geocode_firecrawl.py --limit 5
    python scripts/enrich_geocode_firecrawl.py
    python scripts/enrich_geocode_firecrawl.py --all
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.geocode import (  # noqa: E402
    CHICAGO_LOOP,
    DEFAULT_CITY_HINT,
    NOMINATIM_URL,
    USER_AGENT,
    _normalise,
    export_csv,
    load_cache,
    save_cache,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s :: %(message)s",
)
logger = logging.getLogger("enrich_geocode")

LOOP_LAT, LOOP_LON = CHICAGO_LOOP
FIRECRAWL_DIR = ROOT / ".firecrawl"
PROGRESS_PATH = FIRECRAWL_DIR / "geocode_progress.json"

# Rough Chicago city bounds (reject bad geocodes)
LAT_MIN, LAT_MAX = 41.64, 42.08
LON_MIN, LON_MAX = -87.95, -87.52

MAP_AT_RE = re.compile(r"@(-?\d{1,3}\.\d{4,}),(-?\d{1,3}\.\d{4,})")
MAP_3D_RE = re.compile(r"!3d(-?\d{1,3}\.\d{4,})!4d(-?\d{1,3}\.\d{4,})")
# "875 N. Michigan Avenue" / "1300 S Lake Shore Dr, Chicago, IL 60605"
STREET_SUFFIX = (
    r"Street|St\.?|Avenue|Ave\.?|Boulevard|Blvd\.?|Road|Rd\.?|"
    r"Drive|Dr\.?|Way|Place|Plaza|Court|Ct\.?|Lane|Ln\.?|Highway|Hwy\.?"
)
STREET_RE = re.compile(
    rf"(?:Located at|Address[:\s]+|at\s+)?"
    rf"(\d{{1,5}}\s+(?:[NSEW]\.?\s+)?[\w\s\.'\-]+?(?:{STREET_SUFFIX})"
    rf"(?:\s*,?\s*Chicago,?\s*IL(?:\s+\d{{5}})?)?)",
    re.IGNORECASE,
)
CHICAGO_ADDR_RE = re.compile(
    r"(\d{1,5}\s+[\w\s\.'\-]+?,?\s*Chicago,?\s*IL(?:\s+\d{5})?)",
    re.IGNORECASE,
)
# "875 N Michigan Avenue" without requiring trailing ", Chicago"
NUM_STREET_RE = re.compile(
    rf"(\d{{1,5}}\s+[NSEW]\.?\s+[\w\s\.'\-]{{2,40}}?(?:{STREET_SUFFIX})\.?)",
    re.IGNORECASE,
)
MICHIGAN_RE = re.compile(
    r"(\d{1,5}\s+[NSEW]\.?\s+Michigan(?:\s+Ave(?:nue)?\.?)?)",
    re.IGNORECASE,
)

# High-confidence street addresses when Firecrawl snippets are incomplete.
# Big Bus official hop-on stops (bigbustours.com/en/chicago/find-a-bus-stop-chicago/)
BIG_BUS_STOPS: Dict[int, str] = {
    1: "98 E Wacker Drive, Chicago, IL",
    2: "319 W Jackson Blvd, Chicago, IL",
    3: "17 S Michigan Ave, Chicago, IL",
    4: "800 S Michigan Ave, Chicago, IL",
    5: "500 E Solidarity Drive, Chicago, IL",
    6: "425 E McFetridge Drive, Chicago, IL",
    7: "441 N Columbus Drive, Chicago, IL",
    8: "600 E Grand Ave, Chicago, IL",
    9: "163 E Pearson St, Chicago, IL",
    10: "150 E Chestnut St, Chicago, IL",
    11: "614 N Clark Street, Chicago, IL",
}

KNOWN_ADDRESSES: Dict[str, str] = {
    "360 chicago observation deck": "875 N Michigan Ave, Chicago, IL 60611",
    "21c museum hotel": "55 E Ontario St, Chicago, IL 60611",
    "900 n. michigan shops": "900 N Michigan Ave, Chicago, IL 60611",
    "skydeck chicago": "233 S Wacker Dr, Chicago, IL 60606",
    "willis tower": "233 S Wacker Dr, Chicago, IL 60606",
    "navy pier": "600 E Grand Ave, Chicago, IL 60611",
    "adler planetarium": "1300 S DuSable Lake Shore Dr, Chicago, IL 60605",
    "shedd aquarium": "1200 S DuSable Lake Shore Dr, Chicago, IL 60605",
    "field museum": "1400 S DuSable Lake Shore Dr, Chicago, IL 60605",
    "art institute of chicago": "111 S Michigan Ave, Chicago, IL 60603",
    "millennium park": "201 E Randolph St, Chicago, IL 60601",
    "goodman theater center": "170 N Dearborn St, Chicago, IL 60601",
    "griffin museum of science & industry": "5700 S DuSable Lake Shore Dr, Chicago, IL 60637",
    "lizzie mcneil's": "230 N Michigan Ave, Chicago, IL 60601",
    "origin of great chicago fire of 1871": "558 W DeKoven St, Chicago, IL 60607",
    "raising cane's chicken fingers - loyola": "6554 N Sheridan Rd, Chicago, IL 60626",
    "raising canes chicken fingers - loyola": "6554 N Sheridan Rd, Chicago, IL 60626",
    "raising cane's chicken fingers - lincoln park": "2376 N Lincoln Ave, Chicago, IL 60614",
    "raising cane's chicken fingers - south loop": "564 W Taylor St, Chicago, IL 60607",
    "raising cane's chicken fingers - west loop": "820 W Randolph St, Chicago, IL 60607",
    "raising cane's chicken fingers - wrigleyville": "3700 N Clark St, Chicago, IL 60613",
    "route 66 league": "78 E Washington St, Chicago, IL 60602",
    "route 66 in illinois": "78 E Washington St, Chicago, IL 60602",
    "chicago style by katie lukes": "900 N Michigan Ave, Chicago, IL 60611",
    # Generic hotel names — pin to downtown Mag Mile / Loop properties (not O'Hare suburbs)
    "hyatt centric mag mile": "633 N Saint Clair St, Chicago, IL 60611",
    "water tower": "806 N Michigan Ave, Chicago, IL 60611",
    "omni": "676 N Michigan Ave, Chicago, IL 60611",
    "marriott": "540 N Michigan Ave, Chicago, IL 60611",
    "renaissance": "1 W Upper Wacker Dr, Chicago, IL 60601",
    "springhill suites": "410 N Dearborn St, Chicago, IL 60654",
    # Suburban outlet mall (official list includes it; not Loop)
    "fashion outlets of chicago": "5220 Fashion Outlets Way, Rosemont, IL 60018",
}


def _in_chicago(lat: float, lon: float) -> bool:
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX


def _needs_enrichment(row: pd.Series, *, all_rows: bool) -> bool:
    if all_rows:
        return True
    src = str(row.get("geo_source", "")).strip().lower()
    lat = float(row["lat"])
    lon = float(row["lon"])
    if src == "fallback":
        return True
    if abs(lat - LOOP_LAT) < 1e-4 and abs(lon - LOOP_LON) < 1e-4:
        return True
    return False


def _extract_coords_from_text(text: str) -> List[Tuple[float, float]]:
    found: List[Tuple[float, float]] = []
    for m in MAP_AT_RE.finditer(text):
        lat, lon = float(m.group(1)), float(m.group(2))
        if _in_chicago(lat, lon):
            found.append((lat, lon))
    for m in MAP_3D_RE.finditer(text):
        lat, lon = float(m.group(1)), float(m.group(2))
        if _in_chicago(lat, lon):
            found.append((lat, lon))
    return found


def _valid_address(addr: str) -> bool:
    if not addr or len(addr) > 70:
        return False
    if "\n" in addr or "more info" in addr.lower():
        return False
    if " is " in addr.lower() or "observation deck is" in addr.lower():
        return False
    if not re.search(r"\d", addr):
        return False
    low = addr.lower()
    if any(bad in low for bad in ("ticket", "tilt", "optional", "last entry", "feet above")):
        return False
    return True


def _score_address(addr: str) -> int:
    score = 0
    if re.search(r"\b[NSEW]\.", addr, re.I):
        score += 3
    if re.search(STREET_SUFFIX, addr, re.I):
        score += 2
    if "michigan" in addr.lower():
        score += 1
    if "chicago" in addr.lower():
        score += 1
    return score


def _extract_addresses(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.replace("\n", " "))
    candidates: List[str] = []
    for pat in (MICHIGAN_RE, NUM_STREET_RE, STREET_RE, CHICAGO_ADDR_RE):
        for m in pat.finditer(text):
            addr = m.group(1).strip().rstrip(".,;")
            if _valid_address(addr) and addr not in candidates:
                candidates.append(addr)
    candidates.sort(key=_score_address, reverse=True)
    out: List[str] = []
    for a in candidates:
        if "chicago" not in a.lower():
            a = f"{a}, Chicago, IL"
        out.append(a)
    return out[:5]


def _firecrawl_search(query: str, *, scrape: bool = False) -> Dict[str, Any]:
    FIRECRAWL_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-]+", "_", query[:60]).strip("_")
    out = FIRECRAWL_DIR / f"search_{safe}.json"
    cmd = [
        "firecrawl", "search", query,
        "--limit", "5",
        "-o", str(out),
        "--json",
    ]
    if scrape:
        cmd.insert(3, "--scrape")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
    except subprocess.CalledProcessError as exc:
        logger.warning("Firecrawl search failed: %s", (exc.stderr or exc.stdout or "")[:300])
        return {}
    except FileNotFoundError:
        logger.error("firecrawl CLI not found — run: npx -y firecrawl-cli@latest init --all")
        return {}
    if not out.exists():
        return {}
    try:
        return json.loads(out.read_text())
    except json.JSONDecodeError:
        return {}


def _firecrawl_scrape(url: str) -> str:
    FIRECRAWL_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-]+", "_", url.split("/")[2])[:40]
    out = FIRECRAWL_DIR / f"scrape_{safe}.md"
    cmd = ["firecrawl", "scrape", url, "-o", str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=90)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    return out.read_text() if out.exists() else ""


def _search_query(name: str) -> str:
    n = name.strip()
    if "big bus" in n.lower():
        return f'"{n}" Chicago IL bus stop address OR google maps'
    return f'"{n}" Chicago Illinois street address OR site:choosechicago.com'


def _collect_text_from_search(payload: Dict[str, Any]) -> str:
    chunks: List[str] = []
    data = payload.get("data") or {}
    for item in data.get("web") or []:
        for key in ("url", "title", "description", "markdown"):
            val = item.get(key)
            if val:
                chunks.append(str(val))
    return "\n".join(chunks)


def _pick_scrape_url(payload: Dict[str, Any]) -> Optional[str]:
    data = payload.get("data") or {}
    priority = (
        "choosechicago.com/listing",
        "choosechicago.com",
        "maps.google",
        "google.com/maps",
        "loopchicago.com",
    )
    for needle in priority:
        for item in data.get("web") or []:
            url = str(item.get("url") or "")
            if needle in url.lower():
                return url
    for item in data.get("web") or []:
        url = str(item.get("url") or "")
        if url and "facebook.com" not in url.lower() and "instagram.com" not in url.lower():
            return url
    return None


def _nominatim(client: httpx.Client, query: str) -> Optional[Dict[str, Any]]:
    params = {
        "q": f"{query}, {DEFAULT_CITY_HINT}" if "chicago" not in query.lower() else query,
        "format": "json",
        "limit": 1,
        "countrycodes": "us",
    }
    try:
        r = client.get(NOMINATIM_URL, params=params, timeout=15)
        r.raise_for_status()
        hits = r.json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Nominatim error for %r: %s", query, exc)
        return None
    if not hits:
        return None
    top = hits[0]
    lat, lon = float(top["lat"]), float(top["lon"])
    if not _in_chicago(lat, lon):
        return None
    return {
        "lat": lat,
        "lon": lon,
        "display_name": top.get("display_name", query),
        "source": "firecrawl+nominatim",
    }


def _big_bus_address(name: str) -> Optional[str]:
    m = re.search(r"stop\s+(\d+)\s*:", name, re.IGNORECASE)
    if not m:
        m = re.search(r"stop\s+(\d+)\b", name, re.IGNORECASE)
    if m:
        return BIG_BUS_STOPS.get(int(m.group(1)))
    return None


def enrich_one(
    client: httpx.Client,
    name: str,
    *,
    force_scrape: bool = False,
) -> Optional[Dict[str, Any]]:
    key = _normalise(name)
    bus_addr = _big_bus_address(name)
    if bus_addr:
        hit = _nominatim(client, bus_addr)
        if hit:
            hit["source"] = "bigbus+nominatim"
            return hit

    known = KNOWN_ADDRESSES.get(key)
    if known:
        hit = _nominatim(client, known)
        if hit:
            hit["source"] = "known_address+nominatim"
            return hit

    query = _search_query(name)
    payload = _firecrawl_search(query, scrape=False)
    blob = _collect_text_from_search(payload)

    coords = _extract_coords_from_text(blob)
    if coords:
        lat, lon = coords[0]
        return {
            "lat": lat,
            "lon": lon,
            "display_name": f"{name} (from maps link)",
            "source": "firecrawl+maps",
        }

    addresses = _extract_addresses(blob)
    for addr in addresses:
        hit = _nominatim(client, addr)
        if hit:
            hit["display_name"] = f"{addr} → {hit['display_name'][:80]}"
            return hit
        time.sleep(1.05)

    scrape_url = _pick_scrape_url(payload)
    if scrape_url and (force_scrape or not addresses):
        md = _firecrawl_scrape(scrape_url)
        blob2 = blob + "\n" + md
        coords = _extract_coords_from_text(blob2)
        if coords:
            lat, lon = coords[0]
            return {
                "lat": lat,
                "lon": lon,
                "display_name": f"{name} (from {scrape_url[:50]})",
                "source": "firecrawl+maps",
            }
        for addr in _extract_addresses(blob2):
            hit = _nominatim(client, addr)
            if hit:
                return hit
            time.sleep(1.05)

    # Improved name-only Nominatim (sometimes works after Firecrawl context)
    hit = _nominatim(client, name)
    if hit and hit.get("source"):
        hit["source"] = "firecrawl+nominatim_name"
    return hit


def load_progress() -> Dict[str, Any]:
    if PROGRESS_PATH.exists():
        try:
            return json.loads(PROGRESS_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return {"done": {}, "failed": []}


def save_progress(progress: Dict[str, Any]) -> None:
    FIRECRAWL_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(progress, indent=2, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser(description="Enrich geo via Firecrawl + Nominatim")
    ap.add_argument("--geo-csv", default=str(ROOT / "data" / "locations_geo.csv"))
    ap.add_argument("--dim", default=str(ROOT / "data" / "location_dim.csv"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "geo_cache.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "locations_geo.csv"))
    ap.add_argument("--all", action="store_true", help="Re-enrich every row (uses many credits)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="Re-run even if progress says done")
    args = ap.parse_args()

    geo_path = Path(args.geo_csv)
    cache_path = Path(args.cache)
    dim_path = Path(args.dim)
    out_path = Path(args.out)

    geo_df = pd.read_csv(geo_path)
    targets = geo_df[geo_df.apply(lambda r: _needs_enrichment(r, all_rows=args.all), axis=1)]
    if args.limit:
        targets = targets.head(args.limit)

    logger.info(
        "Enriching %d / %d locations (all=%s)",
        len(targets), len(geo_df), args.all,
    )
    if targets.empty:
        logger.info("Nothing to enrich.")
        return 0

    cache = load_cache(cache_path)
    progress = load_progress()
    done_map: Dict[str, Dict] = progress.get("done") or {}
    failed: List[str] = list(progress.get("failed") or [])

    ok = skip = fail = 0
    with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
        for i, (_, row) in enumerate(targets.iterrows(), 1):
            name = str(row["location_name"]).strip()
            key = _normalise(name)
            if not args.force and key in done_map:
                cache[key] = done_map[key]
                skip += 1
                continue

            logger.info("[%d/%d] %s", i, len(targets), name)
            hit = enrich_one(client, name)
            time.sleep(1.05)  # Nominatim politeness between venues

            if hit:
                cache[key] = hit
                done_map[key] = hit
                if name in failed:
                    failed.remove(name)
                ok += 1
                logger.info("  ✓ %.5f, %.5f (%s)", hit["lat"], hit["lon"], hit.get("source"))
            else:
                if name not in failed:
                    failed.append(name)
                fail += 1
                logger.warning("  ✗ no coordinates found")

            if i % 5 == 0:
                save_cache(cache_path, {k: v for k, v in cache.items() if v})
                progress = {"done": done_map, "failed": failed}
                save_progress(progress)

    save_cache(cache_path, {k: v for k, v in cache.items() if v})
    save_progress({"done": done_map, "failed": failed})

    fallbacks = export_csv({k: v for k, v in cache.items() if v}, dim_path, out_path)
    logger.info(
        "Finished: %d updated, %d skipped (cached progress), %d failed, %d CSV fallbacks remain",
        ok, skip, fail, fallbacks,
    )
    if failed:
        logger.info("Failed venues (%d): %s", len(failed), ", ".join(failed[:8]) + ("…" if len(failed) > 8 else ""))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
