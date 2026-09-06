"""Generate the public demo event fixture without using visitor records.

The production recommender reads current behavioral data from BigQuery.  This
script creates a small, deterministic dataset solely so a clean clone can boot,
demonstrate session/trending signals, and run smoke tests without credentials.

Every identifier, session, timestamp, and interaction in the output is
synthetic.  Location IDs and names come from the public ChicagoDoes catalog.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "location_dim.csv"
DEFAULT_CATEGORY_MAPPING = PROJECT_ROOT / "warehouse" / "seeds" / "category_mapping.csv"
DEFAULT_CATEGORY_BRIDGE = PROJECT_ROOT / "warehouse" / "seeds" / "location_category_bridge.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "demo_events.csv"
SEED = 20260813

META_CATEGORIES = {"HOT SPOTS", "Favorites"}

KEYWORDS = {
    "Museums": ("gallery", "museum"),
    "Theaters and Music Venues": (
        "music", "opera", "symphony", "theater", "theatre",
    ),
    "Restaurants": (
        "bakery", "bbq", "cafe", "coffee", "diner", "food", "grill",
        "kitchen", "pizza", "restaurant", "steak", "sushi", "taco",
    ),
    "Hotels": ("hostel", "hotel", "inn", "resort"),
    "Bars": (
        "bar", "brewery", "club", "lounge", "pub", "tavern",
    ),
    "Parks": (
        "beach", "conservatory", "garden", "harbor", "lake", "park",
        "pier", "riverwalk", "trail", "zoo",
    ),
    "Shops": (
        "boutique", "mall", "market", "outlet", "shop", "store",
    ),
    "Sports Venues": (
        "arena", "bike", "bowling", "field", "fitness", "golf", "stadium",
    ),
}

FIELDNAMES = (
    "event_date",
    "event_timestamp",
    "event_time",
    "event_name",
    "action_id",
    "interaction_weight",
    "user_key",
    "session_key",
    "location_id",
    "location_name",
    "location_category_name",
    "primary_category",
    "num_categories",
    "is_hot_spot_location",
    "is_favorite_location",
    "previous_location_id",
    "page_location",
    "engagement_time_msec",
    "engagement_time_msec_capped",
)


def infer_category(name: str) -> str:
    """Assign a broad demo category from a location name."""
    lowered = name.casefold()
    for category, words in KEYWORDS.items():
        if any(word in lowered for word in words):
            return category
    return "Attractions"


def load_taxonomy(mapping_path: Path, bridge_path: Path) -> dict[str, list[str]]:
    """Load the public place-to-category mapping without visitor data."""
    with mapping_path.open(newline="", encoding="utf-8") as handle:
        category_names = {
            row["category_id"].strip(): row["category"].strip()
            for row in csv.DictReader(handle)
        }

    categories_by_location: defaultdict[str, list[str]] = defaultdict(list)
    with bridge_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            location_id = row["location_id"].strip()
            category = category_names.get(row["category_id"].strip())
            if category and category not in categories_by_location[location_id]:
                categories_by_location[location_id].append(category)
    return dict(categories_by_location)


def load_catalog(
    path: Path,
    mapping_path: Path = DEFAULT_CATEGORY_MAPPING,
    bridge_path: Path = DEFAULT_CATEGORY_BRIDGE,
) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"location_id", "location_name"}.issubset(rows[0]):
        raise ValueError(f"Catalog must contain location_id and location_name: {path}")

    taxonomy = load_taxonomy(mapping_path, bridge_path)
    for row in rows:
        categories = taxonomy.get(row["location_id"], [])
        if not categories:
            categories = [infer_category(row["location_name"])]
        primary = next(
            (category for category in categories if category not in META_CATEGORIES),
            "Attractions",
        )
        row["categories"] = categories
        row["primary_category"] = primary
        row["is_hot_spot_location"] = int("HOT SPOTS" in categories)
        row["is_favorite_location"] = int("Favorites" in categories)
    return rows


def build_events(catalog: list[dict[str, object]]) -> list[dict[str, object]]:
    """Create stable multi-stop demo sessions across broad preferences."""
    rng = random.Random(SEED)
    preference_categories = sorted({str(row["primary_category"]) for row in catalog})
    by_category = {
        category: [row for row in catalog if row["primary_category"] == category]
        for category in preference_categories
    }
    broad_pool = catalog[: min(120, len(catalog))]
    start = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)
    events: list[dict[str, object]] = []

    for user_number in range(1, 81):
        user_key = f"synthetic-user-{user_number:03d}"
        preference = preference_categories[(user_number - 1) % len(preference_categories)]
        preferred = by_category.get(preference) or broad_pool

        for session_number in range(1, 4):
            session_key = f"{user_key}-session-{session_number:02d}"
            session_start = start + timedelta(
                days=(user_number * 3 + session_number * 7) % 31,
                minutes=user_number * 11 + session_number * 37,
            )
            choices: list[dict[str, object]] = []
            while len(choices) < 4:
                pool = preferred if rng.random() < 0.65 else broad_pool
                candidate = rng.choice(pool)
                if candidate not in choices:
                    choices.append(candidate)

            previous_id = ""
            for stop_number, location in enumerate(choices):
                base_time = session_start + timedelta(minutes=stop_number * 18)
                category = str(location["primary_category"])
                categories = [str(value) for value in location["categories"]]
                common = {
                    "user_key": user_key,
                    "session_key": session_key,
                    "location_id": location["location_id"],
                    "location_name": location["location_name"],
                    "location_category_name": "; ".join(categories),
                    "primary_category": category,
                    "num_categories": len(categories),
                    "is_hot_spot_location": location["is_hot_spot_location"],
                    "is_favorite_location": location["is_favorite_location"],
                    "previous_location_id": previous_id,
                    "page_location": f"https://demo.invalid/places/{location['location_id']}",
                }

                for offset_seconds, event_name, action_id, weight, engagement in (
                    (0, "map-user-action", "marker_click", 1.5, 1600),
                    (45, "page_view", "detail_view", 1.0, 4200),
                ):
                    event_time = base_time + timedelta(seconds=offset_seconds)
                    events.append({
                        "event_date": event_time.strftime("%Y%m%d"),
                        "event_timestamp": int(event_time.timestamp() * 1_000_000),
                        "event_time": event_time.isoformat(),
                        "event_name": event_name,
                        "action_id": action_id,
                        "interaction_weight": weight,
                        "engagement_time_msec": engagement,
                        "engagement_time_msec_capped": engagement,
                        **common,
                    })
                previous_id = location["location_id"]

    return sorted(events, key=lambda row: (str(row["event_time"]), str(row["user_key"])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ChicagoDoes demo events")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--category-mapping", type=Path, default=DEFAULT_CATEGORY_MAPPING)
    parser.add_argument("--category-bridge", type=Path, default=DEFAULT_CATEGORY_BRIDGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    catalog = load_catalog(args.catalog, args.category_mapping, args.category_bridge)
    events = build_events(catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(events)

    print(
        f"Wrote {len(events)} synthetic events for "
        f"{len({row['user_key'] for row in events})} demo users to {args.output}"
    )


if __name__ == "__main__":
    main()
