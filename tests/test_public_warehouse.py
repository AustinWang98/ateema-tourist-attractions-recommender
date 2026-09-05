"""Checks for the public BigQuery template and repository data boundary."""
from __future__ import annotations

from pathlib import Path

from warehouse.run_pipeline import _target_sql


ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = ROOT / "warehouse" / "sql"


def test_all_warehouse_stages_are_public_and_parameterized() -> None:
    sql_files = sorted(SQL_DIR.glob("*.sql"))
    assert [path.name for path in sql_files] == [
        "01_location_dim.sql",
        "02_location_category_bridge_final.sql",
        "03_location_category_dim.sql",
        "04_user_location_category_events.sql",
        "05_user_location_full_features.sql",
        "06_candidate_user_location_table.sql",
    ]

    for path in sql_files:
        source = path.read_text(encoding="utf-8")
        assert "__PROJECT_ID__.__DATASET_ID__" in source
        rendered = _target_sql(path, "public-demo-project", "analytics_demo")
        assert "__PROJECT_ID__" not in rendered
        assert "__DATASET_ID__" not in rendered
        assert "`public-demo-project.analytics_demo." in rendered


def test_private_data_is_not_in_public_tree() -> None:
    forbidden = [
        ROOT / "data" / "events.csv",
        ROOT / "HANDOFF.md",
    ]
    assert all(not path.exists() for path in forbidden)


def test_complete_academic_deliverables_are_public() -> None:
    required = [
        ROOT / "docs" / "paper" / "ChicagoDoes_Capstone_Paper.docx",
        ROOT / "docs" / "paper" / "ChicagoDoes_Capstone_Paper.pdf",
        ROOT / "docs" / "paper" / "main.tex",
        ROOT / "docs" / "presentation" / "ChicagoDoes_Final_Presentation.pptx",
        ROOT / "docs" / "presentation" / "build_presentation.py",
        ROOT / "data" / "weight_search_results.csv",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in required)
