"""Fetch warehouse tables from BigQuery.

Authentication: this module relies on **Application Default Credentials**.
Run `gcloud auth application-default login` once on the host machine.
For service accounts, set GOOGLE_APPLICATION_CREDENTIALS to the JSON path.

Configuration (env vars, read by callers):
    BQ_PROJECT          GCP project id, e.g. ateema-capstone
    BQ_DATASET          dataset name, e.g. analytics_459092297
    BQ_TABLE_FEATURES   default: user_location_full_features
    BQ_TABLE_LOCATION_DIM  default: location_dim
    BQ_TABLE_EVENTS     default: user_location_category_events

Public entry points:
    fetch_table()         - one table -> DataFrame
    fetch_all_tables()    - all three -> dict of DataFrames
    list_dataset_tables() - debugging helper, returns table names
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BQConfig:
    project: str
    dataset: str
    table_features: str = "user_location_full_features"
    table_location_dim: str = "location_dim"
    table_events: str = "user_location_category_events"
    # Optional: cap rows pulled per table (None = all). Useful for fast iteration.
    row_limit: Optional[int] = None

    def qualified(self, table: str) -> str:
        return f"`{self.project}.{self.dataset}.{table}`"


# --------------------------------------------------------------------------- #
# Connection helper
# --------------------------------------------------------------------------- #
def _get_client(project: str):
    """Return a BigQuery client, raising a friendly error if auth is missing."""
    try:
        from google.cloud import bigquery  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "google-cloud-bigquery is not installed. Run "
            "`pip install -r requirements.txt` to add the BigQuery deps."
        ) from exc

    try:
        return bigquery.Client(project=project)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not initialise BigQuery client. Did you run "
            "`gcloud auth application-default login`?\n"
            f"Underlying error: {exc}"
        ) from exc


# --------------------------------------------------------------------------- #
# Query helpers
# --------------------------------------------------------------------------- #
def list_dataset_tables(cfg: BQConfig) -> List[str]:
    client = _get_client(cfg.project)
    dataset_ref = f"{cfg.project}.{cfg.dataset}"
    tables = list(client.list_tables(dataset_ref))
    return sorted(t.table_id for t in tables)


def fetch_table(cfg: BQConfig, table_name: str, client=None) -> pd.DataFrame:
    """SELECT * FROM `project.dataset.table` -> DataFrame.

    Falls back to legacy `pandas-gbq`-style fetch if the BigQuery
    Storage API is missing (e.g. on cold installs without the optional
    extras).
    """
    client = client or _get_client(cfg.project)
    sql = f"SELECT * FROM {cfg.qualified(table_name)}"
    if cfg.row_limit:
        sql += f" LIMIT {int(cfg.row_limit)}"
    logger.info("BQ query: %s", sql)

    job = client.query(sql)
    try:
        df = job.to_dataframe(create_bqstorage_client=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "BQ Storage API failed (%s); retrying with row-iterator fallback.", exc,
        )
        df = job.to_dataframe(create_bqstorage_client=False)
    logger.info("BQ %s -> %d rows × %d cols", table_name, len(df), len(df.columns))
    return df


def fetch_all_tables(cfg: BQConfig) -> Dict[str, pd.DataFrame]:
    """Pull the three warehouse tables. The events table is optional."""
    client = _get_client(cfg.project)
    available = set(list_dataset_tables(cfg))
    logger.info("Dataset %s.%s has %d tables", cfg.project, cfg.dataset, len(available))

    out: Dict[str, pd.DataFrame] = {}
    for logical, name in (
        ("features", cfg.table_features),
        ("location_dim", cfg.table_location_dim),
    ):
        if name not in available:
            raise RuntimeError(
                f"Required table `{name}` not found in {cfg.project}.{cfg.dataset}. "
                f"Available tables: {sorted(available)}"
            )
        out[logical] = fetch_table(cfg, name, client=client)

    # Events table is optional — trending + slot learning gracefully degrade.
    if cfg.table_events in available:
        out["events"] = fetch_table(cfg, cfg.table_events, client=client)
    else:
        logger.warning(
            "Events table `%s` not found in dataset; trending signal disabled.",
            cfg.table_events,
        )
    return out


# --------------------------------------------------------------------------- #
# Direct WarehouseFrames construction (live mode)
# --------------------------------------------------------------------------- #
def load_warehouse_from_bq(cfg: BQConfig, geo_path: Optional[str] = None):
    """Pull BQ tables and construct WarehouseFrames in memory.

    The CSV-based pipeline in `data_loader.py` does heavy normalisation
    (category splitting, leakage-aware column separation, etc.). To
    reuse all of that, we round-trip through the same normalisation
    functions by writing minimal columns into the format those helpers
    already expect.

    `geo_path` is the local CSV produced by `backend.geocode` — BQ does
    not store coords, so we always merge them in from disk.
    """
    from pathlib import Path

    from ..data_loader import (
        WarehouseFrames,
        _build_interactions,
        _build_locations,
        _build_users,
        _expand_to_official_universe_from_df,
        _normalise_events,
        _split_category_string,
    )

    tables = fetch_all_tables(cfg)
    df = tables["features"].copy()

    required = {"user_key", "location_id", "location_name"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"BQ table `{cfg.table_features}` missing columns: {sorted(missing)}"
        )

    if "location_category_name" in df.columns:
        df["categories"] = df["location_category_name"].apply(_split_category_string)
    else:
        df["categories"] = [[] for _ in range(len(df))]

    interactions = _build_interactions(df)
    locations = _build_locations(df)
    users = _build_users(df)
    geo_df = None
    if geo_path:
        gp = Path(geo_path)
        if gp.exists():
            geo_df = pd.read_csv(gp)
            logger.info("BQ + geo merge: %d coords from %s", len(geo_df), gp)
    locations = _expand_to_official_universe_from_df(
        locations, tables["location_dim"], geo_df=geo_df,
    )

    events = None
    if "events" in tables:
        events = _normalise_events(tables["events"])

    logger.info(
        "BQ warehouse: %d interactions, %d locations (observed=%d), %d users, events=%s",
        len(interactions), len(locations),
        int(locations["observed"].sum()) if "observed" in locations.columns else len(locations),
        len(users), len(events) if events is not None else "-",
    )
    return WarehouseFrames(
        interactions=interactions,
        locations=locations,
        users=users,
        events=events,
    )


# --------------------------------------------------------------------------- #
# CSV export (refresh) helper
# --------------------------------------------------------------------------- #
def export_to_csv(cfg: BQConfig, out_dir: str = "data") -> Dict[str, dict]:
    """Dump BQ tables to local CSV files matching the loader's expected paths.

    Returns a summary dict keyed by logical name.
    """
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    targets = {
        "features":     out / "user_location_features.csv",
        "location_dim": out / "location_dim.csv",
        "events":       out / "events.csv",
    }
    tables = fetch_all_tables(cfg)
    summary: Dict[str, dict] = {}
    for logical, df in tables.items():
        target = targets.get(logical)
        if target is None:
            continue
        df.to_csv(target, index=False)
        summary[logical] = {"rows": len(df), "csv": str(target)}
        logger.info("Wrote %s rows to %s", len(df), target)
    return summary
