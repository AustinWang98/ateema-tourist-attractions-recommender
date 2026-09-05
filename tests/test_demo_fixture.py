"""Privacy and reproducibility checks for the checked-in demo fixture."""
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMO_EVENTS = ROOT / "data" / "demo_events.csv"


def test_demo_events_are_synthetic_and_minimal() -> None:
    with DEMO_EVENTS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1920
    assert {row["user_key"] for row in rows}
    assert all(row["user_key"].startswith("synthetic-user-") for row in rows)
    assert all(row["session_key"].startswith(row["user_key"]) for row in rows)
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
