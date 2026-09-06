"""Privacy and reproducibility checks for the checked-in demo fixture."""
from __future__ import annotations

import csv
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from scripts.generate_demo_events import FIELDNAMES, load_taxonomy


ROOT = Path(__file__).resolve().parents[1]
DEMO_EVENTS = ROOT / "data" / "demo_events.csv"
CATALOG = ROOT / "data" / "location_dim.csv"
CATEGORY_MAPPING = ROOT / "warehouse" / "seeds" / "category_mapping.csv"
CATEGORY_BRIDGE = ROOT / "warehouse" / "seeds" / "location_category_bridge.csv"


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def test_demo_events_are_synthetic_and_minimal() -> None:
    fields, rows = _read_csv(DEMO_EVENTS)

    assert fields == list(FIELDNAMES)
    assert len(rows) == 1920
    assert len({row["user_key"] for row in rows}) == 80
    assert all(re.fullmatch(r"synthetic-user-\d{3}", row["user_key"]) for row in rows)
    assert all(
        re.fullmatch(re.escape(row["user_key"]) + r"-session-\d{2}", row["session_key"])
        for row in rows
    )
    assert all(row["page_location"].startswith("https://demo.invalid/") for row in rows)

    forbidden_visitor_fields = {
        "user_pseudo_id",
        "user_id",
        "cid",
        "device_category",
        "mobile_brand",
        "mobile_model",
        "device_language",
        "browser_version",
        "city",
        "country",
        "region",
    }
    assert forbidden_visitor_fields.isdisjoint(rows[0])

    serialized = "\n".join(",".join(row.values()) for row in rows)
    assert not re.search(r"[^@\s]+@[^@\s]+\.[^@\s]+", serialized)
    assert not re.search(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)", serialized)
    assert not re.search(r"(?i)(api[_-]?key|password|bearer|sk-[a-z0-9])", serialized)


def test_demo_events_match_the_public_catalog_and_taxonomy() -> None:
    _, rows = _read_csv(DEMO_EVENTS)
    _, catalog_rows = _read_csv(CATALOG)
    catalog = {row["location_id"]: row["location_name"] for row in catalog_rows}
    taxonomy = load_taxonomy(CATEGORY_MAPPING, CATEGORY_BRIDGE)

    assert len(catalog) == 350
    assert len({row["location_id"] for row in rows}) >= 200
    assert all(catalog.get(row["location_id"]) == row["location_name"] for row in rows)

    for row in rows:
        categories = row["location_category_name"].split("; ")
        if row["location_id"] in taxonomy:
            assert categories == taxonomy[row["location_id"]]
        assert int(row["num_categories"]) == len(categories)
        assert int(row["is_hot_spot_location"]) == int("HOT SPOTS" in categories)

    assert len({row["primary_category"] for row in rows}) >= 10
    assert {row["is_hot_spot_location"] for row in rows} == {"0", "1"}


def test_demo_events_have_valid_temporal_and_session_structure() -> None:
    _, rows = _read_csv(DEMO_EVENTS)
    sessions: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        parsed = datetime.fromisoformat(row["event_time"])
        assert row["event_date"] == parsed.strftime("%Y%m%d")
        assert int(row["event_timestamp"]) == int(parsed.timestamp() * 1_000_000)
        assert float(row["interaction_weight"]) > 0
        assert int(row["engagement_time_msec"]) >= 0
        assert int(row["engagement_time_msec_capped"]) <= int(row["engagement_time_msec"])
        sessions[row["session_key"]].append(row)

    assert len(sessions) == 240
    assert Counter(row["event_name"] for row in rows) == {
        "map-user-action": 960,
        "page_view": 960,
    }
    for session_rows in sessions.values():
        ordered = sorted(session_rows, key=lambda row: row["event_timestamp"])
        assert len(ordered) == 8
        stops = [ordered[index] for index in range(0, len(ordered), 2)]
        assert stops[0]["previous_location_id"] == ""
        for previous, current in zip(stops, stops[1:]):
            assert current["previous_location_id"] == previous["location_id"]


def test_generator_reproduces_the_checked_in_fixture(tmp_path: Path) -> None:
    regenerated = tmp_path / "demo_events.csv"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_demo_events.py"), "--output", str(regenerated)],
        cwd=ROOT,
        check=True,
    )
    assert regenerated.read_bytes() == DEMO_EVENTS.read_bytes()
