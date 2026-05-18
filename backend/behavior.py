"""Event-level behavioural signals derived from the raw `events.csv`.

Why this module exists
----------------------
The cleaned `user_location_features` table aggregates everything to
the (user, location) grain. That throws away two extremely valuable
signals that only live in the raw events:

1. **Session co-visitation** — two locations clicked inside the *same
   session* are far more related than two locations clicked by the
   same user across different sessions. A session corresponds to
   "actively planning a trip right now", so its pairs reflect intent.

2. **Transition graph** — `previous_location_id` is an explicit
   pointer from one location-view to the next. This is the
   ChicagoDoes equivalent of "users who viewed X also viewed Y"
   and is impossible to reproduce from aggregates.

We also use this module to **validate** the cleaned table against
raw counts and warn when they diverge enough to suggest a stale
or buggy SQL pipeline.

All matrices are sparse and built once at startup. Lookups during
recommendation are O(seed_set_size) per request.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)


# Event types we trust as "real engagement". `scroll` events are noisy
# (auto-fired by the SDK) so we down-weight them; `map-user-action` is
# a deliberate click on a pin so we weight it most.
EVENT_TYPE_WEIGHTS = {
    "map-user-action": 1.5,
    "page_view":       1.0,
    "scroll":          0.3,
}

# Filter sessions with more than this many events as likely bots / QA.
MAX_EVENTS_PER_SESSION = 200


# --------------------------------------------------------------------------- #
# Session co-visitation
# --------------------------------------------------------------------------- #
@dataclass
class SessionCoVisitation:
    """Sparse Jaccard at the SESSION grain (not user grain).

    Built from events grouped by `session_key`. Each session contributes
    one "transaction" of all the distinct locations it touched, weighted
    by event-type weight. Two locations that co-occur in many sessions
    will have a high session-Jaccard, even if they're never co-visited
    by the same user across sessions.
    """
    location_ids: List[str]
    id_to_pos: Dict[str, int]
    jaccard: sparse.csr_matrix       # NxN
    session_counts: np.ndarray       # |sessions that touched i|, length N
    n_sessions: int

    @classmethod
    def build(cls, events: pd.DataFrame, location_ids: Sequence[str]) -> "SessionCoVisitation":
        ids = [str(x) for x in location_ids]
        id_to_pos = {lid: i for i, lid in enumerate(ids)}
        n = len(ids)
        empty = sparse.csr_matrix((n, n), dtype=float)

        if events is None or events.empty or "session_key" not in events.columns:
            logger.info("SessionCoVisitation: no events; returning empty matrix.")
            return cls(ids, id_to_pos, empty, np.zeros(n, dtype=float), 0)

        ev = events.dropna(subset=["session_key", "location_id"]).copy()
        if ev.empty:
            return cls(ids, id_to_pos, empty, np.zeros(n, dtype=float), 0)

        # Bot/QA filter: drop sessions with absurd event counts.
        sess_sizes = ev.groupby("session_key").size()
        keep_sess = set(sess_sizes[sess_sizes <= MAX_EVENTS_PER_SESSION].index)
        ev = ev[ev["session_key"].isin(keep_sess)]

        # Weight by event type — `scroll` is mostly noise.
        if "event_name" in ev.columns:
            ev["w"] = ev["event_name"].map(EVENT_TYPE_WEIGHTS).fillna(1.0)
        else:
            ev["w"] = 1.0
        if "interaction_weight" in ev.columns:
            ev["w"] = ev["w"] * ev["interaction_weight"].fillna(1.0).astype(float)

        # Build sparse session × location incidence matrix.
        sess_keys = ev["session_key"].astype(str).unique().tolist()
        sess_to_row = {s: i for i, s in enumerate(sess_keys)}
        rows: List[int] = []
        cols: List[int] = []
        for _, r in ev.iterrows():
            pos = id_to_pos.get(str(r["location_id"]))
            if pos is None:
                continue
            rows.append(sess_to_row[str(r["session_key"])])
            cols.append(pos)
        if not rows:
            return cls(ids, id_to_pos, empty, np.zeros(n, dtype=float), 0)

        data = np.ones(len(rows), dtype=float)
        sl = sparse.csr_matrix((data, (rows, cols)), shape=(len(sess_keys), n))
        sl.data[:] = 1.0   # collapse repeated (session, loc) pairs

        inter = (sl.T @ sl).toarray()
        sess_counts = np.asarray(inter.diagonal(), dtype=float)
        union = sess_counts[:, None] + sess_counts[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            jacc = np.where(union > 0, inter / union, 0.0)
        np.fill_diagonal(jacc, 0.0)

        logger.info(
            "SessionCoVisitation built: %d sessions × %d locations, %d non-zero pairs",
            len(sess_keys), n, int(np.count_nonzero(jacc)),
        )
        return cls(
            location_ids=ids,
            id_to_pos=id_to_pos,
            jaccard=sparse.csr_matrix(jacc),
            session_counts=sess_counts,
            n_sessions=len(sess_keys),
        )

    def score_against_seed(self, seed_location_ids: Sequence[str]) -> np.ndarray:
        positions = [self.id_to_pos[s] for s in seed_location_ids if s in self.id_to_pos]
        if not positions:
            return np.zeros(self.jaccard.shape[0], dtype=float)
        rows = self.jaccard[positions]
        return np.asarray(rows.mean(axis=0)).ravel()


# --------------------------------------------------------------------------- #
# Transition graph from `previous_location_id`
# --------------------------------------------------------------------------- #
@dataclass
class TransitionGraph:
    """Directed graph: P(next = j | current = i), normalised by row.

    Built from rows where `previous_location_id` is set. The resulting
    matrix is row-stochastic (each row sums to 1 if there were any
    outgoing transitions, else 0). For a seed set S, the score for
    candidate j is `mean_{i in S} matrix[i, j]` — i.e. how likely a
    user who just viewed something in S is to look at j next.
    """
    location_ids: List[str]
    id_to_pos: Dict[str, int]
    transitions: sparse.csr_matrix     # NxN, row-stochastic
    n_transitions: int

    @classmethod
    def build(cls, events: pd.DataFrame, location_ids: Sequence[str]) -> "TransitionGraph":
        ids = [str(x) for x in location_ids]
        id_to_pos = {lid: i for i, lid in enumerate(ids)}
        n = len(ids)
        empty = sparse.csr_matrix((n, n), dtype=float)
        if events is None or events.empty:
            return cls(ids, id_to_pos, empty, 0)
        if "previous_location_id" not in events.columns or "location_id" not in events.columns:
            return cls(ids, id_to_pos, empty, 0)

        ev = events.dropna(subset=["previous_location_id", "location_id"]).copy()
        ev["previous_location_id"] = ev["previous_location_id"].astype(str).str.strip()
        ev["location_id"] = ev["location_id"].astype(str).str.strip()
        ev = ev[ev["previous_location_id"] != ev["location_id"]]
        if ev.empty:
            return cls(ids, id_to_pos, empty, 0)

        # Weight by event type so a deliberate map click matters more.
        if "event_name" in ev.columns:
            ev["w"] = ev["event_name"].map(EVENT_TYPE_WEIGHTS).fillna(1.0)
        else:
            ev["w"] = 1.0

        rows: List[int] = []
        cols: List[int] = []
        weights: List[float] = []
        for _, r in ev.iterrows():
            i = id_to_pos.get(r["previous_location_id"])
            j = id_to_pos.get(r["location_id"])
            if i is None or j is None:
                continue
            rows.append(i)
            cols.append(j)
            weights.append(float(r["w"]))
        if not rows:
            return cls(ids, id_to_pos, empty, 0)

        m = sparse.csr_matrix((weights, (rows, cols)), shape=(n, n))
        # Row-normalise: P(next = j | current = i)
        row_sums = np.asarray(m.sum(axis=1)).ravel()
        row_sums[row_sums < 1e-12] = 1.0
        inv = sparse.diags(1.0 / row_sums)
        m_norm = inv @ m

        logger.info(
            "TransitionGraph built: %d transitions, %d non-zero pairs",
            len(rows), int(m_norm.nnz),
        )
        return cls(
            location_ids=ids,
            id_to_pos=id_to_pos,
            transitions=m_norm.tocsr(),
            n_transitions=len(rows),
        )

    def score_against_seed(self, seed_location_ids: Sequence[str]) -> np.ndarray:
        positions = [self.id_to_pos[s] for s in seed_location_ids if s in self.id_to_pos]
        if not positions:
            return np.zeros(self.transitions.shape[0], dtype=float)
        # For each candidate j, mean P(j | i for i in seed)
        rows = self.transitions[positions]
        return np.asarray(rows.mean(axis=0)).ravel()


# --------------------------------------------------------------------------- #
# Cross-check raw events vs the cleaned features table.
# --------------------------------------------------------------------------- #
def validate_cleaned_vs_raw(
    events: pd.DataFrame,
    interactions: pd.DataFrame,
    warn_threshold: float = 0.5,
) -> Dict[str, dict]:
    """Compare a few aggregates between raw events and the cleaned table.

    Returns a small report dict; also logs warnings when relative
    differences exceed `warn_threshold` (50% by default). This is
    explicitly NOT used to *change* the cleaned table — it's a data
    quality signal we can surface on /api/health.
    """
    report = {
        "n_users_events":   int(events["user_key"].nunique()) if "user_key" in events.columns else 0,
        "n_users_cleaned":  int(interactions["user_key"].nunique()) if "user_key" in interactions.columns else 0,
        "n_locations_events":  int(events["location_id"].nunique()) if "location_id" in events.columns else 0,
        "n_locations_cleaned": int(interactions["location_id"].nunique()) if "location_id" in interactions.columns else 0,
        "n_events_total": int(len(events)),
        "warnings": [],
    }

    # Per-location event count vs cleaned `total_interactions_with_location`
    if (
        not events.empty
        and "location_id" in events.columns
        and "location_id" in interactions.columns
        and "total_interactions_with_location" in interactions.columns
    ):
        raw_per_loc = events.groupby("location_id").size().rename("raw_events")
        agg_per_loc = (
            interactions.groupby("location_id")["total_interactions_with_location"]
            .sum()
            .rename("cleaned_total")
        )
        merged = pd.concat([raw_per_loc, agg_per_loc], axis=1).fillna(0)
        merged["rel_diff"] = (merged["raw_events"] - merged["cleaned_total"]).abs() / (
            merged[["raw_events", "cleaned_total"]].max(axis=1).clip(lower=1)
        )
        # Only flag locations that appear with enough rows on either side.
        big = merged[(merged[["raw_events", "cleaned_total"]].max(axis=1) >= 5)
                     & (merged["rel_diff"] >= warn_threshold)]
        n_warn = len(big)
        report["n_locations_compared"] = int(len(merged))
        report["n_locations_diverge"] = int(n_warn)
        if n_warn:
            preview = big.head(3).to_dict(orient="index")
            report["warnings"].append(
                f"{n_warn} locations diverge >{int(warn_threshold * 100)}% between raw and cleaned"
            )
            logger.warning(
                "Data quality: %d locations have raw vs cleaned counts that diverge "
                ">%d%%. Examples: %s",
                n_warn, int(warn_threshold * 100), preview,
            )

    logger.info(
        "Data quality: users raw=%d / cleaned=%d, locations raw=%d / cleaned=%d, events=%d",
        report["n_users_events"], report["n_users_cleaned"],
        report["n_locations_events"], report["n_locations_cleaned"],
        report["n_events_total"],
    )
    return report
