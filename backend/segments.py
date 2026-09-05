"""Behavioural archetypes for ChicagoDoes users.

We cluster the observed users on their **category-share vector** (the
fraction of their interactions in each category, weighted by
`total_interaction_score`). Each cluster is given a human-readable
archetype label by looking at which categories dominate its centroid.

For a new visitor (no behaviour yet) we project their form-derived
interests into the same vector space and assign them to the nearest
centroid. This gives us a real, data-grounded persona we can display:

    "You look like an Outdoor Explorer — most similar to 23 past
    ChicagoDoes users."

It also lets us seed `ItemCoVisitation` for new visitors: the seed
locations are the most popular locations among users in the same
archetype.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .collab import UserNeighbors
from .data_loader import WarehouseFrames

logger = logging.getLogger(__name__)


# Human-readable archetype names, ordered by category they over-index on.
# We pick the dominant category in each cluster centroid and look it up
# here. Any category not in this dict gets a generic "Explorer" label.
ARCHETYPE_BY_CATEGORY: Dict[str, str] = {
    "Attractions":  "Sightseeing Explorer",
    "Parks":        "Outdoor Wanderer",
    "Museums":      "Cultural Enthusiast",
    "Bars":         "Nightlife Seeker",
    "Restaurants":  "Foodie Wanderer",
    "Hotels":       "Comfort Traveler",
    "Shops":        "Retail Hunter",
    "Tours":        "Tour-First Visitor",
    "HOT SPOTS":    "Trend Hunter",
    "Favorites":    "Loyal Fan",
}


@dataclass
class ArchetypeAssignment:
    archetype: str
    cluster_id: int
    confidence: float                  # cosine sim to its centroid (0..1)
    cluster_size: int                  # how many real users live in it
    top_categories: List[str]          # top 3 categories of this cluster
    seed_location_ids: List[str]       # most popular locations for this cluster


class UserSegmenter:
    def __init__(self, neighbours: UserNeighbors, frames: WarehouseFrames, k: int = 4) -> None:
        self.neighbours = neighbours
        self.frames = frames
        self.categories = neighbours.categories
        self.k = max(1, min(k, max(len(neighbours.user_keys) // 5, 1)))

        if not neighbours.user_keys:
            logger.warning("UserSegmenter: no users to cluster.")
            self.kmeans: Optional[KMeans] = None
            self.user_to_cluster: Dict[str, int] = {}
            self.cluster_meta: Dict[int, Dict] = {}
            return

        # Drop users with all-zero category vectors before clustering.
        X = neighbours.user_cat_matrix
        keep_mask = X.sum(axis=1) > 1e-12
        X_fit = X[keep_mask]
        if len(X_fit) < self.k:
            self.k = max(1, len(X_fit))
        self.kmeans = KMeans(n_clusters=self.k, n_init="auto", random_state=42)
        labels_fit = self.kmeans.fit_predict(X_fit)

        full_labels = np.full(len(neighbours.user_keys), -1, dtype=int)
        full_labels[keep_mask] = labels_fit
        self.user_to_cluster = {
            u: int(c) for u, c in zip(neighbours.user_keys, full_labels) if c >= 0
        }

        # Pre-compute cluster metadata: top categories + seed locations.
        self.cluster_meta = self._compute_cluster_meta(full_labels)
        logger.info(
            "UserSegmenter: %d users into %d clusters (sizes=%s)",
            len(self.user_to_cluster), self.k,
            [self.cluster_meta[c]["size"] for c in sorted(self.cluster_meta)],
        )

    # ------------------------------------------------------------------ #
    def assign_returning(self, user_key: str) -> Optional[ArchetypeAssignment]:
        cid = self.user_to_cluster.get(str(user_key))
        if cid is None or self.kmeans is None:
            return None
        row = self.neighbours.key_to_row[str(user_key)]
        vec = self.neighbours.user_cat_matrix[row]
        return self._build_assignment(cid, vec)

    def assign_from_categories(
        self,
        weights_by_category: Dict[str, float],
    ) -> Optional[ArchetypeAssignment]:
        """Project a synthetic preference vector and assign the nearest cluster."""
        if self.kmeans is None or not self.categories:
            return None
        vec = np.zeros(len(self.categories), dtype=float)
        for cat, w in weights_by_category.items():
            if cat in self.categories:
                vec[self.categories.index(cat)] = max(float(w), 0.0)
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            return None
        vec = vec / norm
        cid = int(self.kmeans.predict(vec.reshape(1, -1))[0])
        return self._build_assignment(cid, vec)

    # ------------------------------------------------------------------ #
    def _build_assignment(self, cid: int, vec: np.ndarray) -> ArchetypeAssignment:
        meta = self.cluster_meta.get(cid, {})
        centroid = self.kmeans.cluster_centers_[cid] if self.kmeans else vec
        cn = np.linalg.norm(centroid)
        vn = np.linalg.norm(vec)
        conf = float(np.dot(centroid, vec) / (cn * vn)) if cn * vn > 1e-12 else 0.0
        top_cats = list(meta.get("top_categories", []))
        archetype = self._label_for(top_cats)
        return ArchetypeAssignment(
            archetype=archetype,
            cluster_id=cid,
            confidence=max(0.0, min(1.0, conf)),
            cluster_size=int(meta.get("size", 0)),
            top_categories=top_cats[:3],
            seed_location_ids=list(meta.get("seed_location_ids", [])),
        )

    def _compute_cluster_meta(self, labels: np.ndarray) -> Dict[int, Dict]:
        meta: Dict[int, Dict] = {}
        if self.kmeans is None:
            return meta

        for cid in range(self.k):
            members = [self.neighbours.user_keys[i] for i, c in enumerate(labels) if c == cid]
            centroid = self.kmeans.cluster_centers_[cid]
            # Top categories by centroid weight
            top_idx = np.argsort(-centroid)[:5]
            top_cats = [self.categories[i] for i in top_idx if centroid[i] > 1e-6]

            # Seed locations: most popular locations among this cluster's members
            seeds: List[str] = []
            if members:
                enriched = getattr(self.frames, "interactions_enriched", None)
                interactions = enriched if enriched is not None else self.frames.interactions
                sub = interactions[interactions["user_key"].isin(members)]
                if "is_qualified" in sub.columns:
                    sub = sub[sub["is_qualified"]]
                if not sub.empty:
                    pop = (
                        sub.groupby("location_id")["total_interaction_score"]
                        .sum()
                        .sort_values(ascending=False)
                    )
                    seeds = [str(lid) for lid in pop.head(5).index.tolist()]

            meta[cid] = {
                "size": len(members),
                "top_categories": top_cats,
                "seed_location_ids": seeds,
            }
        return meta

    @staticmethod
    def _label_for(top_categories: Sequence[str]) -> str:
        for cat in top_categories:
            if cat in ARCHETYPE_BY_CATEGORY:
                return ARCHETYPE_BY_CATEGORY[cat]
        return "Curious Explorer"
