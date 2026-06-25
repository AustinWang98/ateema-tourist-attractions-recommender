"""Trending signal derived from `user_location_category_events`.

For every location with at least a few events we split the observation
window into two halves and compute:

    trending_raw[loc] = (recent_weight + 1) / (early_weight + 1)

where `weight` is the sum of `interaction_weight` of events on that
location in that half-window. Using `interaction_weight` (instead of
raw counts) prioritises high-intent actions (`detail_cta` > `marker_click`
> `scroll`), so a location whose CTA clicks are rising scores higher
than one whose scrolls are rising.

The raw ratios are min-max normalised into [0, 1] before exposing
them to the recommender. A small whitelist threshold means cold or
low-volume locations get a 0 trending score rather than a noisy boost.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


MIN_EVENTS_FOR_TRENDING = 3      # ignore locations with <3 events
TRENDING_HIGHLIGHT_THRESHOLD = 0.6  # UI badge cutoff


@dataclass
class TrendingSignal:
    """Per-location trending score in [0, 1]."""
    score_by_location: Dict[str, float]
    cutoff_iso: Optional[str]
    n_events: int

    @classmethod
    def build(cls, events: Optional[pd.DataFrame]) -> "TrendingSignal":
        if events is None or events.empty or "event_time" not in events.columns:
            logger.info("TrendingSignal: no events available.")
            return cls(score_by_location={}, cutoff_iso=None, n_events=0)

        ev = events.dropna(subset=["event_time", "location_id"]).copy()
        if ev.empty:
            return cls(score_by_location={}, cutoff_iso=None, n_events=0)

        ev["location_id"] = ev["location_id"].astype(str)
        if "event_quality_weight" in ev.columns:
            ev["interaction_weight"] = ev["event_quality_weight"].astype(float)
        elif "interaction_weight" not in ev.columns:
            ev["interaction_weight"] = 1.0
        ev["interaction_weight"] = ev["interaction_weight"].fillna(0.5).astype(float)

        # Split the window at the median timestamp -> equal-sized halves
        # in terms of *time*, which is sturdier than a calendar split for
        # short observation periods.
        cutoff = ev["event_time"].median()
        early_mask = ev["event_time"] < cutoff
        late_mask = ~early_mask

        early = ev[early_mask].groupby("location_id")["interaction_weight"].sum()
        late = ev[late_mask].groupby("location_id")["interaction_weight"].sum()
        counts = ev.groupby("location_id")["interaction_weight"].count()

        loc_ids = sorted(set(early.index) | set(late.index))
        rows = []
        for lid in loc_ids:
            n = float(counts.get(lid, 0.0))
            if n < MIN_EVENTS_FOR_TRENDING:
                continue
            e = float(early.get(lid, 0.0))
            l = float(late.get(lid, 0.0))
            raw = (l + 1.0) / (e + 1.0)
            rows.append((lid, raw))

        if not rows:
            return cls(score_by_location={}, cutoff_iso=str(cutoff), n_events=int(len(ev)))

        df = pd.DataFrame(rows, columns=["location_id", "raw"])
        # Compress tails: log to reduce the dominance of single events.
        df["log_raw"] = np.log1p(df["raw"].clip(lower=0.0))
        lo, hi = df["log_raw"].min(), df["log_raw"].max()
        if hi - lo < 1e-9:
            df["score"] = 0.0
        else:
            df["score"] = (df["log_raw"] - lo) / (hi - lo)

        score_by_location = dict(zip(df["location_id"].astype(str), df["score"].astype(float)))
        logger.info(
            "TrendingSignal: %d locations scored (cutoff=%s, n_events=%d)",
            len(score_by_location), str(cutoff), int(len(ev)),
        )
        return cls(score_by_location=score_by_location, cutoff_iso=str(cutoff), n_events=int(len(ev)))

    # ------------------------------------------------------------------ #
    def score_array(self, location_ids: Sequence[str]) -> np.ndarray:
        return np.array(
            [float(self.score_by_location.get(str(lid), 0.0)) for lid in location_ids],
            dtype=float,
        )
