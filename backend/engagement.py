"""Engagement-quality layer — app-side only (no BigQuery changes).

Derives which GA4 events and user×location pairs count as *meaningful*
for collaborative signals, profiles, and trending. Global `*_all_users`
location priors in the warehouse are unchanged.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .behavior import EVENT_TYPE_WEIGHTS, MAX_EVENTS_PER_SESSION
from .data_loader import WarehouseFrames

logger = logging.getLogger(__name__)

STRONG_ACTIONS = frozenset({"map-user-action", "detail_cta"})
WEAK_ACTIONS = frozenset({"page_view"})
NOISE_ACTIONS = frozenset({"scroll"})
NAV_ONLY_ACTIONS = frozenset({
    "map-search",
    "view_search_results",
    "session_start",
})

DEFAULT_MIN_PAGE_VIEW_ENGAGEMENT_MSEC = 1000
DEFAULT_MIN_USER_LOC_SCORE = 1.5
DEFAULT_MIN_QUALIFIED_LOCATIONS_RETURNING = 2
DEFAULT_MAX_EVENTS_PER_USER_LOC_PER_DAY = 10
RECOMMENDER_TRAFFIC_MARKERS = (
    "ateema_recommender",
    "utm_source=ateema_recommender",
    "rec_surface=",
    "capstone_recsys",
)


@dataclass(frozen=True)
class EngagementConfig:
    policy: str = "balanced"
    min_page_view_engagement_msec: int = DEFAULT_MIN_PAGE_VIEW_ENGAGEMENT_MSEC
    min_user_loc_score: float = DEFAULT_MIN_USER_LOC_SCORE
    min_qualified_locations_returning: int = DEFAULT_MIN_QUALIFIED_LOCATIONS_RETURNING
    max_events_per_session: int = MAX_EVENTS_PER_SESSION
    max_events_per_user_loc_per_day: int = DEFAULT_MAX_EVENTS_PER_USER_LOC_PER_DAY
    exclude_recommender_traffic: bool = True

    @classmethod
    def from_env(cls) -> "EngagementConfig":
        return cls(
            policy=os.getenv("ENGAGEMENT_POLICY", "balanced").strip().lower(),
            min_page_view_engagement_msec=int(
                os.getenv("MIN_PAGE_VIEW_ENGAGEMENT_MSEC", str(DEFAULT_MIN_PAGE_VIEW_ENGAGEMENT_MSEC))
            ),
            min_user_loc_score=float(
                os.getenv("MIN_USER_LOC_SCORE", str(DEFAULT_MIN_USER_LOC_SCORE))
            ),
            min_qualified_locations_returning=int(
                os.getenv("MIN_QUALIFIED_LOCATIONS_RETURNING", str(DEFAULT_MIN_QUALIFIED_LOCATIONS_RETURNING))
            ),
            exclude_recommender_traffic=os.getenv(
                "EXCLUDE_RECOMMENDER_TRAFFIC", "true"
            ).strip().lower() not in {"0", "false", "no", "off"},
        )


def _engagement_msec(row: pd.Series) -> float:
    for col in ("engagement_time_msec_capped", "engagement_time_msec"):
        if col in row.index and pd.notna(row[col]):
            try:
                return max(0.0, float(row[col]))
            except (TypeError, ValueError):
                pass
    return 0.0


def is_qualified_event(
    event_name: object,
    engagement_msec: float,
    cfg: EngagementConfig,
) -> Tuple[bool, str]:
    """Return (qualified, reject_reason)."""
    name = str(event_name or "").strip()

    if cfg.policy == "permissive":
        if name in NOISE_ACTIONS:
            return False, "scroll_noise"
        if name in NAV_ONLY_ACTIONS:
            return False, "nav_noise"
        if name in STRONG_ACTIONS:
            return True, "strong"
        if name in WEAK_ACTIONS:
            return engagement_msec >= 500, (
                "dwell_soft" if engagement_msec >= 500 else "reject"
            )
        return engagement_msec >= 500, (
            "dwell_soft" if engagement_msec >= 500 else "reject"
        )

    if cfg.policy == "strict":
        if name in NOISE_ACTIONS or name in NAV_ONLY_ACTIONS:
            return False, "scroll_noise" if name in NOISE_ACTIONS else "nav_noise"
        if name in STRONG_ACTIONS:
            return engagement_msec >= 500, (
                "strong" if engagement_msec >= 500 else "reject"
            )
        return engagement_msec >= 2000, (
            "dwell" if engagement_msec >= 2000 else "reject"
        )

    # balanced (default)
    if name in NOISE_ACTIONS:
        return False, "scroll_noise"
    if name in NAV_ONLY_ACTIONS:
        return False, "nav_noise"
    if name in STRONG_ACTIONS:
        return True, "strong"
    if name in WEAK_ACTIONS:
        if engagement_msec >= cfg.min_page_view_engagement_msec:
            return True, "dwell"
        if engagement_msec >= 500:
            return True, "dwell_soft"
        return False, "reject"
    if engagement_msec >= cfg.min_page_view_engagement_msec:
        return True, "dwell"
    return False, "reject"


def event_quality_weight(
    event_name: object,
    engagement_msec: float,
    interaction_weight: float,
    qualified: bool,
) -> float:
    if not qualified:
        return 0.0
    name = str(event_name or "").strip()
    base = max(0.0, float(interaction_weight or 0.5))
    type_w = float(EVENT_TYPE_WEIGHTS.get(name, 1.0))
    dwell_w = 1.0
    if name in WEAK_ACTIONS and engagement_msec < 1000:
        dwell_w = float(np.clip((engagement_msec - 500.0) / 500.0, 0.2, 1.0))
    return base * type_w * dwell_w


def _vectorized_qualify(ev: pd.DataFrame, cfg: EngagementConfig) -> pd.DataFrame:
    names = ev["event_name"].astype(str).str.strip() if "event_name" in ev.columns else pd.Series([""] * len(ev))
    eng = ev["engagement_msec_used"].astype(float)

    is_qualified = np.zeros(len(ev), dtype=bool)
    reason = np.full(len(ev), "reject", dtype=object)

    strong = names.isin(STRONG_ACTIONS)
    is_qualified |= strong
    reason = np.where(strong, "strong", reason)

    noise = names.isin(NOISE_ACTIONS | NAV_ONLY_ACTIONS)
    reason = np.where(names.isin(NOISE_ACTIONS), "scroll_noise", reason)
    reason = np.where(names.isin(NAV_ONLY_ACTIONS), "nav_noise", reason)

    if cfg.policy == "permissive":
        dwell_ok = (~noise & ~strong) & (eng >= 500)
        is_qualified |= dwell_ok
        reason = np.where(dwell_ok & (reason == "reject"), "dwell_soft", reason)
    elif cfg.policy == "strict":
        strong_ok = strong & (eng >= 500)
        is_qualified = strong_ok | ((~noise & ~strong) & (eng >= 2000))
        reason = np.where(strong_ok, "strong", reason)
        reason = np.where((~noise & ~strong) & (eng >= 2000), "dwell", reason)
    else:
        # balanced
        pv = names.isin(WEAK_ACTIONS)
        dwell = (~noise & ~strong) & (eng >= cfg.min_page_view_engagement_msec)
        dwell_soft = (~noise & ~strong) & (eng >= 500) & (eng < cfg.min_page_view_engagement_msec)
        is_qualified |= strong | dwell | dwell_soft
        reason = np.where(dwell, "dwell", reason)
        reason = np.where(dwell_soft & (reason == "reject"), "dwell_soft", reason)

    is_qualified &= ~noise
    ev["is_qualified"] = is_qualified
    ev["qualify_reason"] = reason
    return ev


def enrich_events(events: pd.DataFrame, cfg: EngagementConfig) -> pd.DataFrame:
    """Add is_qualified, qualify_reason, event_quality_weight; drop bot sessions."""
    if events is None or events.empty:
        return events

    ev = events.copy()
    excluded_recommender_traffic = 0
    if cfg.exclude_recommender_traffic:
        source_mask = _recommender_source_mask(ev)
        excluded_recommender_traffic = int(source_mask.sum())
        if excluded_recommender_traffic:
            ev = ev.loc[~source_mask].copy()

    if "session_key" in ev.columns:
        sess_sizes = ev.groupby("session_key").size()
        keep = set(sess_sizes[sess_sizes <= cfg.max_events_per_session].index)
        ev = ev[ev["session_key"].isin(keep)]

    if "engagement_time_msec_capped" in ev.columns:
        ev["engagement_msec_used"] = pd.to_numeric(
            ev["engagement_time_msec_capped"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
    elif "engagement_time_msec" in ev.columns:
        ev["engagement_msec_used"] = pd.to_numeric(
            ev["engagement_time_msec"], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
    else:
        ev["engagement_msec_used"] = 0.0

    ev = _vectorized_qualify(ev, cfg)

    iw = (
        pd.to_numeric(ev["interaction_weight"], errors="coerce").fillna(0.5)
        if "interaction_weight" in ev.columns
        else pd.Series([1.0] * len(ev), index=ev.index)
    )
    names = ev["event_name"].astype(str).str.strip() if "event_name" in ev.columns else pd.Series([""] * len(ev))
    eng = ev["engagement_msec_used"].astype(float)
    qualified = ev["is_qualified"].astype(bool)

    type_w = names.map(lambda n: EVENT_TYPE_WEIGHTS.get(n, 1.0)).astype(float)
    dwell_w = np.ones(len(ev), dtype=float)
    weak = names.isin(WEAK_ACTIONS) & qualified & (eng < 1000)
    dwell_w[weak.to_numpy()] = np.clip((eng[weak] - 500.0) / 500.0, 0.2, 1.0)

    ev["event_quality_weight"] = np.where(
        qualified,
        iw.to_numpy() * type_w.to_numpy() * dwell_w,
        0.0,
    )
    out = ev.reset_index(drop=True)
    out.attrs["excluded_recommender_traffic"] = excluded_recommender_traffic
    return out


def _recommender_source_mask(ev: pd.DataFrame) -> pd.Series:
    """Identify GA4 rows generated by this recommender's outbound links."""
    mask = pd.Series(False, index=ev.index)
    for col in (
        "traffic_source",
        "traffic_medium",
        "traffic_campaign",
        "collected_manual_source",
        "collected_manual_medium",
        "collected_manual_campaign",
        "page_location",
        "page_referrer",
    ):
        if col not in ev.columns:
            continue
        text = ev[col].astype(str).str.lower()
        col_mask = pd.Series(False, index=ev.index)
        for marker in RECOMMENDER_TRAFFIC_MARKERS:
            col_mask |= text.str.contains(marker, regex=False, na=False)
        mask |= col_mask
    return mask


def build_user_location_effective(
    events: pd.DataFrame,
    cfg: EngagementConfig,
) -> pd.DataFrame:
    """Per (user_key, location_id): effective_score and is_qualified."""
    empty = pd.DataFrame(
        columns=["user_key", "location_id", "effective_score", "is_qualified", "n_qualified_events"],
    )
    if events is None or events.empty:
        return empty
    if "event_quality_weight" not in events.columns:
        events = enrich_events(events, cfg)

    need = {"user_key", "location_id"}
    if not need.issubset(events.columns):
        return empty

    ev = events.dropna(subset=["user_key", "location_id"]).copy()
    ev["user_key"] = ev["user_key"].astype(str)
    ev["location_id"] = ev["location_id"].astype(str)

    # Cap per session×location: count strongest event once.
    if "session_key" in ev.columns:
        sess_cap = (
            ev.groupby(["user_key", "location_id", "session_key"])["event_quality_weight"]
            .max()
            .reset_index()
        )
        grouped = sess_cap.groupby(["user_key", "location_id"]).agg(
            effective_score=("event_quality_weight", "sum"),
            n_qualified_events=("event_quality_weight", lambda s: int((s > 0).sum())),
        )
    else:
        grouped = ev.groupby(["user_key", "location_id"]).agg(
            effective_score=("event_quality_weight", "sum"),
            n_qualified_events=("is_qualified", "sum"),
        )

    out = grouped.reset_index()
    out["is_qualified"] = out["effective_score"] >= cfg.min_user_loc_score
    return out


def qualify_interactions(
    interactions: pd.DataFrame,
    user_loc_effective: pd.DataFrame,
    cfg: EngagementConfig,
) -> pd.DataFrame:
    """Merge effective scores onto warehouse interactions; filter qualified rows."""
    if interactions is None or interactions.empty:
        return interactions

    out = interactions.copy()
    if user_loc_effective is not None and not user_loc_effective.empty:
        scores = user_loc_effective[
            ["user_key", "location_id", "effective_score", "is_qualified"]
        ].copy()
        scores["user_key"] = scores["user_key"].astype(str)
        scores["location_id"] = scores["location_id"].astype(str)
        out["user_key"] = out["user_key"].astype(str)
        out["location_id"] = out["location_id"].astype(str)
        out = out.merge(scores, on=["user_key", "location_id"], how="left")
    else:
        out["effective_score"] = np.nan
        out["is_qualified"] = False

    fallback = _fallback_interaction_qualified(out)
    if "is_qualified" in out.columns:
        merged_qual = out["is_qualified"].astype("boolean").fillna(fallback.astype("boolean"))
        out["is_qualified"] = merged_qual.astype(bool)
    else:
        out["is_qualified"] = fallback.astype(bool)
    # Rows with event-derived scores below threshold stay unqualified.
    if "effective_score" in out.columns:
        has_score = out["effective_score"].notna()
        out.loc[has_score, "is_qualified"] = (
            out.loc[has_score, "effective_score"] >= cfg.min_user_loc_score
        )

    if "effective_score" in out.columns:
        out["profile_weight"] = out["effective_score"].fillna(
            pd.to_numeric(out.get("total_interaction_score"), errors="coerce").fillna(1.0)
        )
    else:
        out["profile_weight"] = pd.to_numeric(
            out.get("total_interaction_score"), errors="coerce"
        ).fillna(1.0)

    out.loc[~out["is_qualified"], "profile_weight"] = 0.0
    return out


def _fallback_interaction_qualified(df: pd.DataFrame) -> pd.Series:
    """When events are missing, infer from warehouse aggregates."""
    eng = pd.to_numeric(
        df.get("total_location_engagement_msec_capped"), errors="coerce"
    ).fillna(0.0)
    score = pd.to_numeric(df.get("total_interaction_score"), errors="coerce").fillna(0.0)
    return (eng >= 1000) | (score >= 3.0)


def build_engagement_report(
    events: Optional[pd.DataFrame],
    interactions: pd.DataFrame,
    user_loc_effective: pd.DataFrame,
    interactions_qualified: pd.DataFrame,
    cfg: EngagementConfig,
) -> Dict[str, object]:
    n_events_raw = int(len(events)) if events is not None else 0
    n_qualified_events = 0
    reason_counts: Dict[str, int] = {}
    if events is not None and not events.empty and "is_qualified" in events.columns:
        n_qualified_events = int(events["is_qualified"].sum())
        reason_counts = (
            events["qualify_reason"].value_counts().astype(int).to_dict()
        )

    n_ul_raw = int(len(interactions)) if interactions is not None else 0
    n_ul_qual = (
        int(interactions_qualified["is_qualified"].sum())
        if interactions_qualified is not None and not interactions_qualified.empty
        else 0
    )

    users_returning = 0
    users_total = 0
    users_zero = 0
    if interactions_qualified is not None and not interactions_qualified.empty:
        qual = interactions_qualified[interactions_qualified["is_qualified"]]
        per_user = qual.groupby("user_key")["location_id"].nunique()
        users_total = int(interactions_qualified["user_key"].nunique())
        users_returning = int(
            (per_user >= cfg.min_qualified_locations_returning).sum()
        )
        if not user_loc_effective.empty:
            any_qual = user_loc_effective.groupby("user_key")["is_qualified"].any()
            users_zero = int((~any_qual.reindex(
                interactions_qualified["user_key"].unique(), fill_value=False
            )).sum())

    return {
        "policy": cfg.policy,
        "min_page_view_engagement_msec": cfg.min_page_view_engagement_msec,
        "min_user_loc_score": cfg.min_user_loc_score,
        "min_qualified_locations_returning": cfg.min_qualified_locations_returning,
        "exclude_recommender_traffic": cfg.exclude_recommender_traffic,
        "n_events_excluded_recommender": int(
            getattr(events, "attrs", {}).get("excluded_recommender_traffic", 0)
        ) if events is not None else 0,
        "n_events_raw": n_events_raw,
        "n_events_qualified": n_qualified_events,
        "pct_events_qualified": round(
            100.0 * n_qualified_events / max(n_events_raw, 1), 1
        ),
        "n_user_loc_raw": n_ul_raw,
        "n_user_loc_qualified": n_ul_qual,
        "pct_user_loc_qualified": round(
            100.0 * n_ul_qual / max(n_ul_raw, 1), 1
        ),
        "n_users_returning_under_policy": users_returning,
        "n_users_total": users_total,
        "pct_users_zero_qualified": round(
            100.0 * users_zero / max(users_total, 1), 1
        ),
        "qualify_reason_counts": reason_counts,
    }


def apply_engagement_policy(
    frames: WarehouseFrames,
    cfg: Optional[EngagementConfig] = None,
) -> WarehouseFrames:
    """Enrich frames in place for modelling consumers."""
    cfg = cfg or EngagementConfig.from_env()

    if frames.events is not None and not frames.events.empty:
        frames.events = enrich_events(frames.events, cfg)
        frames.user_location_effective = build_user_location_effective(frames.events, cfg)
    else:
        frames.user_location_effective = pd.DataFrame(
            columns=["user_key", "location_id", "effective_score", "is_qualified"],
        )

    frames.interactions_enriched = qualify_interactions(
        frames.interactions,
        frames.user_location_effective,
        cfg,
    )
    frames.engagement_config = cfg
    frames.engagement_report = build_engagement_report(
        frames.events,
        frames.interactions,
        frames.user_location_effective,
        frames.interactions_enriched,
        cfg,
    )
    logger.info(
        "Engagement policy=%s: %s%% events qualified, %s%% user×loc qualified, "
        "%d returning users (≥%d qualified locs)",
        cfg.policy,
        frames.engagement_report.get("pct_events_qualified"),
        frames.engagement_report.get("pct_user_loc_qualified"),
        frames.engagement_report.get("n_users_returning_under_policy"),
        cfg.min_qualified_locations_returning,
    )
    return frames


def count_qualified_locations(frames: WarehouseFrames, user_key: str) -> int:
    df = get_qualified_interactions(frames, user_key)
    if df is None or df.empty:
        return 0
    return int(len(df))


def is_returning_user(frames: WarehouseFrames, user_key: str) -> bool:
    cfg = getattr(frames, "engagement_config", None) or EngagementConfig.from_env()
    return count_qualified_locations(frames, user_key) >= cfg.min_qualified_locations_returning


def get_qualified_interactions(
    frames: WarehouseFrames,
    user_key: str,
) -> Optional[pd.DataFrame]:
    """Qualified user×location rows for profile / seeds / filter-out."""
    enriched = getattr(frames, "interactions_enriched", None)
    base = enriched if enriched is not None else frames.interactions
    sub = base[base["user_key"].astype(str) == str(user_key)]
    if sub.empty:
        return None
    if "is_qualified" in sub.columns:
        sub = sub[sub["is_qualified"]]
    if sub.empty:
        return None
    return sub.reset_index(drop=True)


def modeling_events(frames: WarehouseFrames) -> Optional[pd.DataFrame]:
    """Events with event_quality_weight > 0 for trends / behavior."""
    ev = frames.events
    if ev is None or ev.empty:
        return ev
    if "event_quality_weight" not in ev.columns:
        return ev
    return ev[ev["event_quality_weight"] > 0].copy()
