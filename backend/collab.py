"""Collaborative-filtering signals derived from observed user-location
interactions.

Two complementary signals live here:

* **Item-item co-occurrence (Jaccard)** — `ItemCoVisitation`
    For every pair (a, b) of locations, the fraction of users who
    interacted with both vs. either. Used to score candidate locations
    by their average Jaccard similarity to the locations the user has
    already engaged with (or, for new visitors, to the seed locations
    implied by their selected interests).

* **User-user kNN** — `UserNeighbors`
    Each user is represented by an L2-normalised category-share vector
    (fraction of their interactions per category). For a returning
    user, we take the k nearest neighbours by cosine similarity and
    score candidate locations by the neighbours' interaction scores on
    those locations.

Both signals are **data-only**: they cannot be reproduced by an LLM
without access to the warehouse. They are the project's core
differentiator vs. "just ask ChatGPT".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from .data_loader import WarehouseFrames

logger = logging.getLogger(__name__)


def _qualified_interactions(frames: WarehouseFrames) -> pd.DataFrame:
    enriched = getattr(frames, "interactions_enriched", None)
    df = enriched if enriched is not None else frames.interactions
    if "is_qualified" in df.columns:
        return df[df["is_qualified"]].copy()
    return df


# --------------------------------------------------------------------------- #
# Item-item co-occurrence
# --------------------------------------------------------------------------- #
@dataclass
class ItemCoVisitation:
    """Sparse Jaccard co-occurrence matrix over `location_id`s."""
    location_ids: List[str]
    id_to_pos: Dict[str, int]
    jaccard: sparse.csr_matrix       # NxN, symmetric, zero diagonal
    user_counts: np.ndarray          # |users who interacted with i|, length N

    @classmethod
    def build(cls, frames: WarehouseFrames) -> "ItemCoVisitation":
        """Build the matrix from engagement-qualified interactions."""
        interactions = _qualified_interactions(frames)[["user_key", "location_id"]].dropna()
        location_ids = frames.locations["location_id"].astype(str).tolist()
        id_to_pos = {lid: i for i, lid in enumerate(location_ids)}
        n = len(location_ids)

        # Build the user-location sparse incidence matrix.
        user_keys = interactions["user_key"].astype(str).unique().tolist()
        user_to_row = {u: i for i, u in enumerate(user_keys)}
        rows: List[int] = []
        cols: List[int] = []
        for _, r in interactions.iterrows():
            lid = str(r["location_id"])
            pos = id_to_pos.get(lid)
            if pos is None:
                continue
            rows.append(user_to_row[str(r["user_key"])])
            cols.append(pos)
        if not rows:
            empty = sparse.csr_matrix((n, n), dtype=float)
            return cls(location_ids, id_to_pos, empty, np.zeros(n, dtype=float))

        data = np.ones(len(rows), dtype=float)
        ui = sparse.csr_matrix((data, (rows, cols)), shape=(len(user_keys), n))
        ui.data[:] = 1.0  # collapse repeated (user, loc) pairs

        # Intersection (NxN) = ui.T @ ui ; union = |A| + |B| - |A∩B|
        inter = (ui.T @ ui).toarray()
        user_counts = np.asarray(inter.diagonal(), dtype=float)
        union = user_counts[:, None] + user_counts[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            jacc = np.where(union > 0, inter / union, 0.0)
        np.fill_diagonal(jacc, 0.0)

        logger.info(
            "ItemCoVisitation built: %d locations, %d users, %d non-zero pairs",
            n, len(user_keys), int(np.count_nonzero(jacc)),
        )
        return cls(
            location_ids=location_ids,
            id_to_pos=id_to_pos,
            jaccard=sparse.csr_matrix(jacc),
            user_counts=user_counts,
        )

    # ------------------------------------------------------------------ #
    def score_against_seed(self, seed_location_ids: Sequence[str]) -> np.ndarray:
        """Average Jaccard similarity from each location to the seed set.

        Returns an array of length N (zeros if no seed could be matched).
        """
        positions = [self.id_to_pos[s] for s in seed_location_ids if s in self.id_to_pos]
        if not positions:
            return np.zeros(self.jaccard.shape[0], dtype=float)
        # Average rows of the Jaccard matrix corresponding to seeds.
        rows = self.jaccard[positions]
        return np.asarray(rows.mean(axis=0)).ravel()


# --------------------------------------------------------------------------- #
# User-user kNN over category-share vectors
# --------------------------------------------------------------------------- #
@dataclass
class UserNeighbors:
    """User × category profile + nearest-neighbour lookup."""
    user_keys: List[str]
    key_to_row: Dict[str, int]
    categories: List[str]
    user_cat_matrix: np.ndarray       # (n_users, n_cats), L2-normalised
    user_loc_score: pd.DataFrame      # (user_key, location_id, score) long form

    @classmethod
    def build(cls, frames: WarehouseFrames) -> "UserNeighbors":
        # Long form per-interaction: explode the `categories` list and
        # share `total_interaction_score` evenly across all categories
        # that location belongs to. Robust to the multi-category rows
        # ("Attractions; Parks") common in the warehouse.
        base = _qualified_interactions(frames)
        cols = ["user_key", "location_id", "categories", "total_interaction_score"]
        for extra in ("profile_weight", "effective_score"):
            if extra in base.columns and extra not in cols:
                cols.append(extra)
        df = base[cols].copy()
        if "profile_weight" in df.columns:
            df["total_interaction_score"] = df["profile_weight"].fillna(0.0).astype(float)
        elif "effective_score" in df.columns:
            df["total_interaction_score"] = df["effective_score"].fillna(0.0).astype(float)
        else:
            df["total_interaction_score"] = df["total_interaction_score"].fillna(0.0).astype(float)
        df["n_cats"] = df["categories"].apply(lambda xs: max(len(xs or []), 1))
        df["per_cat_score"] = df["total_interaction_score"] / df["n_cats"]

        exploded = df.explode("categories").dropna(subset=["categories"])
        exploded["categories"] = exploded["categories"].astype(str)

        if exploded.empty:
            empty = np.zeros((0, 0), dtype=float)
            return cls([], {}, [], empty, df.head(0))

        pivot = (
            exploded.groupby(["user_key", "categories"])["per_cat_score"]
            .sum()
            .unstack(fill_value=0.0)
        )
        user_keys = [str(u) for u in pivot.index.tolist()]
        categories = list(pivot.columns)
        mat = pivot.to_numpy(dtype=float)

        # L2-normalise each row (zero rows stay zero).
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        mat = mat / norms

        logger.info(
            "UserNeighbors built: %d users × %d categories", len(user_keys), len(categories)
        )
        return cls(
            user_keys=user_keys,
            key_to_row={u: i for i, u in enumerate(user_keys)},
            categories=categories,
            user_cat_matrix=mat,
            user_loc_score=df[["user_key", "location_id", "total_interaction_score"]],
        )

    # ------------------------------------------------------------------ #
    def neighbours_for(
        self,
        user_key: str,
        k: int = 5,
        exclude_self: bool = True,
    ) -> List[Tuple[str, float]]:
        if not self.user_keys:
            return []
        row = self.key_to_row.get(str(user_key))
        if row is None:
            return []
        q = self.user_cat_matrix[row : row + 1]
        if not np.any(q):
            return []
        sims = cosine_similarity(q, self.user_cat_matrix).ravel()
        order = np.argsort(-sims)
        out: List[Tuple[str, float]] = []
        for j in order:
            if exclude_self and j == row:
                continue
            if sims[j] <= 0.0:
                break
            out.append((self.user_keys[j], float(sims[j])))
            if len(out) >= k:
                break
        return out

    def neighbours_from_categories(
        self,
        interests: Sequence[str],
        weights: Optional[Dict[str, float]] = None,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Find the K real users whose category profile is closest to a
        synthetic interest vector. Used for new visitors who have no
        user_key but did fill the form.

        `interests` is a list of category strings (e.g. ["Parks", "Bars"]).
        `weights` lets the caller emphasise some categories more than
        others; missing keys default to 1.0.
        """
        if not self.user_keys or not interests:
            return []
        weights = weights or {}
        q = np.zeros(len(self.categories), dtype=float)
        cat_to_pos = {c: i for i, c in enumerate(self.categories)}
        for cat in interests:
            pos = cat_to_pos.get(cat)
            if pos is not None:
                q[pos] += float(weights.get(cat, 1.0))
        if not q.any():
            return []
        nrm = np.linalg.norm(q)
        if nrm < 1e-12:
            return []
        q = q / nrm
        sims = self.user_cat_matrix @ q
        order = np.argsort(-sims)
        out: List[Tuple[str, float]] = []
        for j in order:
            if sims[j] <= 0:
                break
            out.append((self.user_keys[j], float(sims[j])))
            if len(out) >= k:
                break
        return out

    def score_locations_for(
        self,
        user_key: str,
        location_ids: Sequence[str],
        k: int = 5,
    ) -> np.ndarray:
        """For each target location, mean (neighbour_sim × neighbour_score).

        Returns an array aligned with `location_ids` (zeros where no
        neighbour engaged with that target).
        """
        loc_ids = [str(lid) for lid in location_ids]
        out = np.zeros(len(loc_ids), dtype=float)
        nbrs = self.neighbours_for(user_key, k=k)
        if not nbrs:
            return out

        nbr_keys = [k for k, _ in nbrs]
        sims = np.array([s for _, s in nbrs], dtype=float)
        # Pull all neighbour interactions in one go.
        sub = self.user_loc_score[self.user_loc_score["user_key"].isin(nbr_keys)]
        if sub.empty:
            return out
        per_loc = sub.groupby("location_id").apply(
            lambda g: float(np.sum([
                sims[nbr_keys.index(str(uk))] * float(score or 0.0)
                for uk, score in zip(g["user_key"].astype(str), g["total_interaction_score"])
            ]))
        )
        lookup = per_loc.to_dict()
        for i, lid in enumerate(loc_ids):
            out[i] = float(lookup.get(lid, 0.0))

        # Normalise to 0..1 so it blends cleanly with cosine sims.
        if out.max() > 0:
            out = out / out.max()
        return out
