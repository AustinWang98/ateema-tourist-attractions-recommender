"""Load and normalise the cleaned user_location_full_features CSV.

This file expects the same columns the user pasted in the project brief
(the cleaned user-event level table). It produces three pandas frames:

* `interactions_df` - one row per (user_key, location_id) observed pair.
* `locations_df`    - one row per location, with global popularity priors
                      and aggregated category lists.
* `users_df`        - one row per user_key, with user-level aggregates.

The transformation is leakage-aware: only `*_all_users` and content fields
are kept on `locations_df`; per-user interaction counts stay inside
`interactions_df`.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


CATEGORY_SPLIT_RE = r"\s*;\s*"


def _split_category_string(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = [p.strip() for p in pd.Series([text]).str.split(CATEGORY_SPLIT_RE, regex=True).iloc[0]]
    return [p for p in parts if p]


@dataclass
class WarehouseFrames:
    interactions: pd.DataFrame
    locations: pd.DataFrame
    users: pd.DataFrame
    events: Optional[pd.DataFrame] = None     # optional, event-level grain
    # Populated by engagement.apply_engagement_policy (app-side quality layer).
    interactions_enriched: Optional[pd.DataFrame] = None
    user_location_effective: Optional[pd.DataFrame] = None
    engagement_report: Optional[dict] = None
    engagement_config: Optional[object] = None

    @property
    def all_categories(self) -> List[str]:
        cats: set[str] = set()
        for row in self.locations["categories"]:
            cats.update(row)
        return sorted(cats)

    @property
    def n_official_locations(self) -> int:
        return int((self.locations.get("source") == "official").sum()) if "source" in self.locations.columns else len(self.locations)

    @property
    def n_observed_locations(self) -> int:
        return int((self.locations.get("observed") == True).sum()) if "observed" in self.locations.columns else len(self.locations)


def load_warehouse(
    csv_path: str | os.PathLike[str] | None = None,
    location_dim_path: str | os.PathLike[str] | None = None,
    events_path: str | os.PathLike[str] | None = None,
    geo_path: str | os.PathLike[str] | None = None,
    source: str = "csv",
    bq_config: Optional["BQConfig"] = None,  # type: ignore[name-defined]  # noqa: F821
) -> WarehouseFrames:
    """Load the warehouse and return normalised frames.

    Parameters
    ----------
    source : "csv" | "bq"
        - "csv" (default): read from local CSV files.
        - "bq": query BigQuery live using `bq_config`.
    csv_path : str or Path
        Path to `user_location_features.csv`. Required when source="csv".
    location_dim_path : str or Path, optional
        Path to `location_dim.csv`. If present, this becomes the
        authoritative location universe (all 350 official locations).
        Otherwise, the universe is the 98 observed locations only.
    events_path : str or Path, optional
        Path to `events.csv`. Enables trending signal and time-of-day
        analysis when present.
    bq_config : BQConfig, optional
        BigQuery configuration; required when source="bq".
    """
    if source == "bq":
        if bq_config is None:
            raise ValueError("source='bq' requires bq_config to be provided.")
        from .sources.bq_source import load_warehouse_from_bq
        return load_warehouse_from_bq(bq_config, geo_path=geo_path)

    if csv_path is None:
        raise ValueError("source='csv' requires csv_path to be provided.")
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data CSV not found at {path.resolve()}. "
            "Export `user_location_full_features` from BigQuery and place it here."
        )

    logger.info("Loading warehouse CSV from %s", path)
    df = pd.read_csv(path, low_memory=False)

    required = {"user_key", "location_id", "location_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    if "location_category_name" in df.columns:
        df["categories"] = df["location_category_name"].apply(_split_category_string)
    else:
        df["categories"] = [[] for _ in range(len(df))]

    interactions = _build_interactions(df)
    locations = _build_locations(df)
    users = _build_users(df)

    if location_dim_path:
        dim_path = Path(location_dim_path)
        if dim_path.exists():
            geo_df = None
            if geo_path:
                gp = Path(geo_path)
                if gp.exists():
                    geo_df = pd.read_csv(gp)
                else:
                    logger.info("geo path %s not found; routing will be limited.", gp)
            locations = _expand_to_official_universe(locations, dim_path, geo_df=geo_df)
        else:
            logger.warning("location_dim path %s not found; skipping universe expansion.", dim_path)

    events: Optional[pd.DataFrame] = None
    if events_path:
        ev_path = Path(events_path)
        if ev_path.exists():
            events = _load_events(ev_path)
        else:
            logger.warning("events path %s not found; trending signal disabled.", ev_path)

    logger.info(
        "Loaded warehouse: %d interactions, %d locations (observed=%d), %d users, events=%s",
        len(interactions), len(locations),
        int(locations["observed"].sum()) if "observed" in locations.columns else len(locations),
        len(users), len(events) if events is not None else "-",
    )
    return WarehouseFrames(interactions=interactions, locations=locations, users=users, events=events)


def _expand_to_official_universe(
    observed_locations: pd.DataFrame,
    dim_path: Path,
    geo_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """CSV wrapper: read `location_dim` from disk then expand."""
    dim = pd.read_csv(dim_path)
    return _expand_to_official_universe_from_df(observed_locations, dim, geo_df=geo_df)


def _expand_to_official_universe_from_df(
    observed_locations: pd.DataFrame,
    dim: pd.DataFrame,
    geo_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Make `location_dim` the authoritative location universe.

    Behaviour:
    * Every official location_id appears in the output exactly once.
    * For observed locations we keep all aggregated fields.
    * For locations we have never observed clicks for, aggregates are
      filled with neutral defaults so the recommender can still rank
      them (they will score on content + popularity = 0, surfaced
      honestly to the UI via the evidence pill).
    * If `geo_df` is provided (location_id, lat, lon, geo_source), we
      left-join it so downstream modules can do routing.
    """
    dim_required = {"location_id", "location_name"}
    if not dim_required.issubset(dim.columns):
        raise ValueError(f"location_dim CSV missing columns: {sorted(dim_required - set(dim.columns))}")
    dim["location_id"] = dim["location_id"].astype(str).str.strip()
    dim["location_name"] = dim["location_name"].astype(str).str.strip()

    obs = observed_locations.copy()
    obs["location_id"] = obs["location_id"].astype(str).str.strip()
    obs_ids = set(obs["location_id"])

    # Authoritative name from location_dim wins on conflict.
    obs = obs.drop(columns=["location_name"], errors="ignore")
    merged = dim.merge(obs, on="location_id", how="left")
    merged["observed"] = merged["location_id"].isin(obs_ids)
    merged["source"] = "official"

    # Fill missing aggregates with neutral defaults.
    fill_numeric_zero = [
        "popularity_raw", "popularity_norm", "engagement_norm",
        "is_hot_spot_location", "is_favorite_location",
        "num_categories", "total_location_score_all_users",
        "avg_location_score_all_users",
        "total_location_interactions_all_users",
        "distinct_users_interacted_location",
        "distinct_sessions_interacted_location",
        "total_marker_clicks_all_users", "total_detail_cta_all_users",
        "total_location_engagement_all_users_msec",
        "avg_location_engagement_all_users_msec",
        "total_location_engagement_all_users_msec_capped",
        "avg_location_engagement_all_users_msec_capped",
    ]
    for col in fill_numeric_zero:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    # Categories: list, default empty.
    if "categories" in merged.columns:
        merged["categories"] = merged["categories"].apply(
            lambda v: v if isinstance(v, list) else []
        )
    else:
        merged["categories"] = [[] for _ in range(len(merged))]

    if "primary_category" not in merged.columns:
        merged["primary_category"] = None

    # Geo join (optional). Caller may have loaded data/locations_geo.csv
    # already and passed it in.
    if geo_df is not None and len(geo_df) > 0:
        geo = geo_df.copy()
        geo["location_id"] = geo["location_id"].astype(str).str.strip()
        merged = merged.drop(columns=["lat", "lon", "geo_source"], errors="ignore")
        merged = merged.merge(
            geo[["location_id", "lat", "lon", "geo_source"]],
            on="location_id", how="left",
        )
        n_with = int(merged["lat"].notna().sum())
        logger.info("Geo: %d / %d locations have coords", n_with, len(merged))

    logger.info(
        "Expanded location universe: %d official, %d observed, %d cold (no clicks yet)",
        len(merged), int(merged["observed"].sum()),
        len(merged) - int(merged["observed"].sum()),
    )
    return merged.reset_index(drop=True)


def _load_events(path: Path) -> pd.DataFrame:
    """CSV wrapper: read events from disk then normalise."""
    logger.info("Loading event-level CSV from %s", path)
    ev = pd.read_csv(path, low_memory=False)
    return _normalise_events(ev)


def _normalise_events(ev: pd.DataFrame) -> pd.DataFrame:
    """Parse event_time / event_timestamp and drop rows we can't time-stamp."""
    ev = ev.copy()
    if "event_time" in ev.columns:
        ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce", utc=True)
    elif "event_timestamp" in ev.columns:
        ev["event_time"] = pd.to_datetime(
            ev["event_timestamp"], unit="us", errors="coerce", utc=True
        )
    if "event_time" in ev.columns:
        ev = ev.dropna(subset=["event_time"])
    return ev.reset_index(drop=True)


def _build_interactions(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "user_key",
        "location_id",
        "location_name",
        "primary_category",
        "categories",
        "total_interactions_with_location",
        "total_interaction_score",
        "avg_interaction_score",
        "total_location_engagement_msec_capped",
        "first_interaction_time",
        "last_interaction_time",
    ]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    out = out.dropna(subset=["user_key", "location_id"])
    out = out.drop_duplicates(subset=["user_key", "location_id"], keep="last")
    return out.reset_index(drop=True)


def _build_locations(df: pd.DataFrame) -> pd.DataFrame:
    candidate_cols = {
        "location_id": "first",
        "location_name": "first",
        "primary_category": "first",
        "categories": "first",
        "num_categories": "max",
        "is_hot_spot_location": "max",
        "is_favorite_location": "max",
        "total_location_interactions_all_users": "max",
        "distinct_users_interacted_location": "max",
        "distinct_sessions_interacted_location": "max",
        "total_location_score_all_users": "max",
        "avg_location_score_all_users": "max",
        "total_marker_clicks_all_users": "max",
        "total_detail_cta_all_users": "max",
        "total_location_engagement_all_users_msec_capped": "max",
        "avg_location_engagement_all_users_msec_capped": "max",
    }
    agg = {k: v for k, v in candidate_cols.items() if k in df.columns and k != "location_id"}
    out = (
        df.dropna(subset=["location_id"])
        .groupby("location_id", as_index=False)
        .agg(agg)
    )

    for col in ("is_hot_spot_location", "is_favorite_location"):
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int).clip(0, 1)

    pop_score = out.get(
        "total_location_score_all_users",
        pd.Series([0.0] * len(out)),
    ).fillna(0.0)
    out["popularity_raw"] = pop_score
    out["popularity_norm"] = _min_max_normalise(pop_score)

    eng = out.get(
        "avg_location_engagement_all_users_msec_capped",
        pd.Series([0.0] * len(out)),
    ).fillna(0.0)
    out["engagement_norm"] = _min_max_normalise(eng)

    return out.reset_index(drop=True)


def _build_users(df: pd.DataFrame) -> pd.DataFrame:
    if "user_key" not in df.columns:
        return pd.DataFrame()
    candidate_cols = {
        "total_user_interactions": "max",
        "distinct_locations_interacted": "max",
        "distinct_location_categories_seen": "max",
        "total_sessions": "max",
        "total_user_interaction_score": "max",
        "avg_user_interaction_score": "max",
        "total_user_engagement_msec_capped": "max",
        "first_seen_time": "min",
        "last_seen_time": "max",
        "city": "first",
        "country": "first",
        "device_category": "first",
    }
    agg = {k: v for k, v in candidate_cols.items() if k in df.columns}
    return (
        df.dropna(subset=["user_key"])
        .groupby("user_key", as_index=False)
        .agg(agg)
    )


def _min_max_normalise(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-9:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - lo) / (hi - lo)


def get_user_interactions(
    frames: WarehouseFrames, user_key: str
) -> Optional[pd.DataFrame]:
    """Return engagement-qualified interactions for a user, or None."""
    from .engagement import get_qualified_interactions

    return get_qualified_interactions(frames, user_key)
