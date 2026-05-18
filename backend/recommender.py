"""Content-based recommender for ChicagoDoes locations.

Design (aligned with SKILL.md leakage guidance):

* Location vector = TF-IDF over category tokens, concatenated with
  normalised popularity priors derived from `*_all_users` global fields
  and the hot-spot / favourite flags. None of these encode the specific
  user's interaction with the location, so they are safe priors.

* User profile vector:
    - Returning user: weighted average of vectors of locations they
      interacted with, weighted by `total_interaction_score`.
    - New user: synthesised from the form (selected categories,
      traveler type, vibe, optional free-text keywords). We project the
      synthetic profile into the same TF-IDF + popularity space.
    - If both signals exist they are blended (`PROFILE_BLEND_ALPHA`).

* Ranking = `SCORE_BLEND_ALPHA * cosine_sim + (1 - SCORE_BLEND_ALPHA) * popularity_norm`,
  filtered by user-selected categories and `avoid_categories`.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import norm as sparse_norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .behavior import SessionCoVisitation, TransitionGraph, validate_cleaned_vs_raw
from .collab import ItemCoVisitation, UserNeighbors
from .data_loader import WarehouseFrames, get_user_interactions
from .engagement import is_returning_user, modeling_events
from .segments import ArchetypeAssignment, UserSegmenter
from .trends import TRENDING_HIGHLIGHT_THRESHOLD, TrendingSignal

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Scoring weights
# --------------------------------------------------------------------------- #
# Two sets of weights: one for cold-start visitors (no behavioural data)
# and one for returning users (we have observed clicks). The user-collab
# term only fires for returning users.
WEIGHTS_NEW = {
    "sim":          0.40,
    "popularity":   0.20,
    "item_collab":  0.10,
    "user_collab":  0.00,
    "trending":     0.10,
    "session":      0.20,   # session co-visit + transition graph (raw events)
}
WEIGHTS_RETURNING = {
    "sim":          0.22,
    "popularity":   0.15,
    "item_collab":  0.18,
    "user_collab":  0.20,
    "trending":     0.05,
    "session":      0.20,   # session co-visit + transition graph (raw events)
}
# Internal blend INSIDE the "session" term: 60% session-Jaccard, 40% transition.
SESSION_JACCARD_WEIGHT = 0.6
TRANSITION_WEIGHT      = 0.4
PROFILE_BLEND_ALPHA = 0.6       # behavioural vs. form weight when blending
HOT_SPOT_BOOST = 0.05
FAVORITE_BOOST = 0.03

# MMR (Maximal Marginal Relevance) diversity. 1.0 = pure relevance (no
# diversity), 0.0 = pure novelty. 0.7 means "70% of the picking signal
# is final_score, 30% is content distance from already-picked items".
MMR_LAMBDA = 0.7
# How many top scoring candidates to consider before MMR re-ranks them.
# Larger pool = more diversity headroom, slightly slower.
MMR_CANDIDATE_POOL = 60

VIBE_TO_CATEGORIES: Dict[str, Dict[str, float]] = {
    "chill":       {"Parks": 1.0, "Attractions": 0.5, "Shops": 0.3},
    "adventurous": {"Attractions": 1.0, "Tours": 0.8, "HOT SPOTS": 0.6, "Parks": 0.3},
    "foodie":      {"Restaurants": 1.0, "Bars": 0.5, "HOT SPOTS": 0.3},
    "nightlife":   {"Bars": 1.0, "Restaurants": 0.6, "HOT SPOTS": 0.7},
    "cultural":    {"Museums": 1.0, "Attractions": 0.7, "Tours": 0.4},
    "outdoorsy":   {"Parks": 1.0, "Attractions": 0.5, "Tours": 0.3},
}

TRAVELER_TO_CATEGORIES: Dict[str, Dict[str, float]] = {
    "solo":     {"Attractions": 0.5, "Bars": 0.4, "Museums": 0.4},
    "couple":   {"Restaurants": 0.6, "Bars": 0.5, "Attractions": 0.5},
    "family":   {"Parks": 1.0, "Attractions": 0.7, "Museums": 0.5},
    "group":    {"Bars": 0.7, "Restaurants": 0.7, "Attractions": 0.5, "HOT SPOTS": 0.5},
    "business": {"Restaurants": 0.5, "Bars": 0.4, "Hotels": 0.6},
}

FREE_TEXT_KEYWORD_HINTS: Dict[str, Sequence[str]] = {
    "park":         ("Parks",),
    "outdoor":      ("Parks",),
    "garden":       ("Parks", "Attractions"),
    "museum":       ("Museums",),
    "art":          ("Museums", "Attractions"),
    "history":      ("Museums", "Attractions"),
    "food":         ("Restaurants",),
    "dinner":       ("Restaurants",),
    "lunch":        ("Restaurants",),
    "pizza":        ("Restaurants",),
    "deep dish":    ("Restaurants",),
    "bar":          ("Bars",),
    "cocktail":     ("Bars",),
    "beer":         ("Bars",),
    "jazz":         ("Bars", "HOT SPOTS"),
    "music":        ("Bars", "HOT SPOTS", "Attractions"),
    "tour":         ("Tours", "Attractions"),
    "bus":          ("Tours",),
    "shopping":     ("Shops",),
    "shop":         ("Shops",),
    "hotel":        ("Hotels",),
    "stay":         ("Hotels",),
    "hot spot":     ("HOT SPOTS",),
    "trendy":       ("HOT SPOTS",),
    "instagram":    ("HOT SPOTS", "Attractions"),
}


@dataclass
class _LocationIndex:
    location_ids: List[str]
    id_to_pos: Dict[str, int]
    tfidf_matrix: sparse.csr_matrix
    tfidf_vectorizer: TfidfVectorizer
    popularity: np.ndarray            # normalised popularity prior, shape (N,)
    engagement: np.ndarray            # normalised engagement prior, shape (N,)
    is_hot_spot: np.ndarray           # 0/1
    is_favorite: np.ndarray           # 0/1


class ContentRecommender:
    """Hybrid recommender: content + popularity + item-collab + user-collab.

    The content term keeps cold-start visitors covered. The two
    collaborative terms are the project's data moat — they require the
    real user click matrix and cannot be reproduced by an LLM alone.
    """

    def __init__(self, frames: WarehouseFrames) -> None:
        self.frames = frames
        self._index = self._build_location_index(frames.locations)
        self.coviz = ItemCoVisitation.build(frames)
        self.neighbours = UserNeighbors.build(frames)
        # k=6 gives meaningful separation for ~500+ users without splintering.
        self.segmenter = UserSegmenter(self.neighbours, frames, k=6)
        ev_model = modeling_events(frames)
        self.trends = TrendingSignal.build(ev_model)
        self._trending = self.trends.score_array(self._index.location_ids)
        # Session co-visit + transitions use engagement-qualified events only.
        self.session_coviz = SessionCoVisitation.build(ev_model, self._index.location_ids)
        self.transitions = TransitionGraph.build(ev_model, self._index.location_ids)
        # Validate the cleaned table against the raw events. Logs
        # warnings if anything diverges; report stays available for /api/health.
        self.data_quality = validate_cleaned_vs_raw(
            events=frames.events if frames.events is not None else pd.DataFrame(),
            interactions=frames.interactions,
        )
        logger.info(
            "ContentRecommender built: %d locations, TF-IDF dim=%d, "
            "%d coviz pairs, %d clusters, %d trending scores",
            len(self._index.location_ids),
            self._index.tfidf_matrix.shape[1],
            int(self.coviz.jaccard.nnz),
            self.segmenter.k,
            int((self._trending > 0).sum()),
        )

    # ------------------------------------------------------------------ #
    # Index construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_location_index(locations: pd.DataFrame) -> _LocationIndex:
        if locations.empty:
            raise ValueError("Cannot build recommender: locations dataframe is empty.")

        # TF-IDF corpus: comma-joined categories per location, lowercased,
        # with multi-word categories collapsed into single tokens so
        # "HOT SPOTS" stays a single concept.
        corpus = locations["categories"].apply(_categories_to_corpus_text).tolist()
        vectorizer = TfidfVectorizer(
            token_pattern=r"[a-z0-9_]+",
            lowercase=True,
            norm="l2",
        )
        tfidf = vectorizer.fit_transform(corpus)

        popularity = locations.get("popularity_norm", pd.Series([0.0] * len(locations))).to_numpy(dtype=float)
        engagement = locations.get("engagement_norm", pd.Series([0.0] * len(locations))).to_numpy(dtype=float)
        hot = locations.get("is_hot_spot_location", pd.Series([0] * len(locations))).to_numpy(dtype=int)
        fav = locations.get("is_favorite_location", pd.Series([0] * len(locations))).to_numpy(dtype=int)

        location_ids = locations["location_id"].astype(str).tolist()
        id_to_pos = {lid: i for i, lid in enumerate(location_ids)}
        return _LocationIndex(
            location_ids=location_ids,
            id_to_pos=id_to_pos,
            tfidf_matrix=tfidf,
            tfidf_vectorizer=vectorizer,
            popularity=popularity,
            engagement=engagement,
            is_hot_spot=hot,
            is_favorite=fav,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def recommend(
        self,
        user_key: Optional[str] = None,
        interests: Optional[Sequence[str]] = None,
        traveler_type: Optional[str] = None,
        vibe: Optional[str] = None,
        avoid_categories: Optional[Sequence[str]] = None,
        free_text: Optional[str] = None,
        top_k: int = 15,
    ) -> Tuple[List[Dict], List[str], bool, Optional[ArchetypeAssignment]]:
        """Hybrid scoring + archetype assignment.

        Returns (results, inferred_interests, is_returning, archetype).
        Each result dict carries the four scoring components and a
        compact `evidence` block so the UI can show the data behind it.
        """
        interests = list(interests or [])
        avoid_categories = list(avoid_categories or [])

        behavioural_vec, has_behaviour = self._build_behavioural_profile(user_key)
        is_returning = bool(
            has_behaviour
            and user_key
            and is_returning_user(self.frames, user_key)
        )
        form_vec, inferred = self._build_form_profile(interests, traveler_type, vibe, free_text)

        profile = self._blend_profiles(
            behavioural_vec, form_vec, self._index.tfidf_matrix.shape[1]
        )

        # --- term 1: content similarity ---------------------------------- #
        if profile.nnz == 0:
            sims = np.zeros(self._index.tfidf_matrix.shape[0], dtype=float)
        else:
            sims = cosine_similarity(profile, self._index.tfidf_matrix).ravel()

        # --- term 2: popularity (leakage-safe global prior) -------------- #
        pop = self._index.popularity

        # --- term 3: item-item collab via seed set ----------------------- #
        seed_ids = self._collect_seed_ids(user_key, is_returning, interests, inferred)
        item_collab = self.coviz.score_against_seed(seed_ids)
        if item_collab.max() > 0:
            item_collab = item_collab / item_collab.max()  # normalise to [0, 1]

        # --- term 4: user-user kNN (returning users only) ---------------- #
        if is_returning and user_key:
            user_collab = self.neighbours.score_locations_for(
                user_key=user_key,
                location_ids=self._index.location_ids,
                k=5,
            )
        else:
            user_collab = np.zeros_like(sims)

        # --- archetype (D3) ---------------------------------------------- #
        archetype = self._assign_archetype(user_key, is_returning, interests)

        # --- term 5: trending (event-time derived) ----------------------- #
        trending = self._trending

        # --- term 6: session co-visit + transition graph (raw events) ---- #
        # Blends two raw-event signals:
        #   - session-Jaccard: locations co-clicked inside ONE trip-planning
        #     session (much sharper than user-grain Jaccard)
        #   - transition graph: P(next = j | previous = i) from
        #     `previous_location_id`, i.e. "users who viewed i then went to j"
        sess_score = self.session_coviz.score_against_seed(seed_ids)
        trans_score = self.transitions.score_against_seed(seed_ids)
        if sess_score.max() > 0:
            sess_score = sess_score / sess_score.max()
        if trans_score.max() > 0:
            trans_score = trans_score / trans_score.max()
        session_collab = SESSION_JACCARD_WEIGHT * sess_score + TRANSITION_WEIGHT * trans_score

        # --- combine ------------------------------------------------------ #
        weights = WEIGHTS_RETURNING if is_returning else WEIGHTS_NEW
        final = (
            weights["sim"]         * sims
            + weights["popularity"] * pop
            + weights["item_collab"]* item_collab
            + weights["user_collab"]* user_collab
            + weights["trending"]   * trending
            + weights["session"]    * session_collab
            + HOT_SPOT_BOOST        * self._index.is_hot_spot
            + FAVORITE_BOOST        * self._index.is_favorite
        )

        locations = self.frames.locations
        mask = self._build_filter_mask(locations, interests, avoid_categories)
        final_masked = np.where(mask, final, -np.inf)

        if is_returning:
            interacted = get_user_interactions(self.frames, user_key)  # type: ignore[arg-type]
            if interacted is not None:
                for lid in interacted["location_id"].astype(str):
                    pos = self._index.id_to_pos.get(lid)
                    if pos is not None:
                        final_masked[pos] = -np.inf

        # MMR (Maximal Marginal Relevance) selection: balance final_score
        # against content similarity to already-picked items. This avoids
        # showing 10 picks that are all the same category, which the raw
        # score-sort tends to do once a category aligns with user intent.
        order = self._mmr_select(
            final_masked,
            top_k=top_k,
            mmr_lambda=MMR_LAMBDA,
            candidate_pool=MMR_CANDIDATE_POOL,
        )

        results: List[Dict] = []
        for pos in order:
            if not np.isfinite(final_masked[pos]):
                break
            if len(results) >= top_k:
                break
            row = locations.iloc[pos]
            results.append({
                "location_id": str(row["location_id"]),
                "location_name": str(row.get("location_name", "")),
                "primary_category": _safe_str(row.get("primary_category")),
                "categories": list(row.get("categories") or []),
                "session_collab_score": float(session_collab[pos]),
                "is_hot_spot": bool(self._index.is_hot_spot[pos]),
                "is_trending": bool(trending[pos] >= TRENDING_HIGHLIGHT_THRESHOLD),
                "popularity_score": float(pop[pos]),
                "similarity_score": float(sims[pos]),
                "item_collab_score": float(item_collab[pos]),
                "user_collab_score": float(user_collab[pos]),
                "trending_score": float(trending[pos]),
                "final_score": float(final_masked[pos]),
                "evidence": self._build_evidence(row, trending[pos]),
                "reason": self._explain(
                    row, inferred, sims[pos], pop[pos], item_collab[pos], trending[pos],
                ),
            })

        return results, inferred, is_returning, archetype

    def _mmr_select(
        self,
        scores: np.ndarray,
        top_k: int,
        mmr_lambda: float = 0.7,
        candidate_pool: int = 60,
    ) -> np.ndarray:
        """Greedy MMR selection over a fixed candidate pool.

        Step 1: rank all positions by `scores` and take the top
        `candidate_pool` (cheap argsort, ignores -inf).
        Step 2: greedily pick the position that maximises
            mmr_lambda * score - (1-mmr_lambda) * max_sim_to_already_picked
        until we have `top_k` items.

        We compute "similarity to already picked" as the cosine between
        L2-normalised TF-IDF rows. The TF-IDF matrix from sklearn is
        already L2-normalised, so cosine = dot product.
        """
        valid_mask = np.isfinite(scores)
        if not valid_mask.any():
            return np.array([], dtype=int)

        # Step 1: cheap top-pool
        pool_size = min(candidate_pool, int(valid_mask.sum()))
        # argpartition gives unordered top-k indices, then sort that slice
        cand_idx = np.argpartition(-scores, pool_size - 1)[:pool_size]
        cand_idx = cand_idx[np.argsort(-scores[cand_idx])]

        # Step 2: greedy MMR
        mat = self._index.tfidf_matrix
        picked: List[int] = []
        max_sim = np.zeros(len(cand_idx), dtype=float)  # max sim to any picked

        for _ in range(min(top_k, len(cand_idx))):
            best_local = -1
            best_value = -np.inf
            for j, pos in enumerate(cand_idx):
                if pos in picked:
                    continue
                relevance = scores[pos]
                if not np.isfinite(relevance):
                    continue
                # MMR score
                value = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim[j]
                if value > best_value:
                    best_value = value
                    best_local = j
            if best_local < 0:
                break
            chosen_pos = int(cand_idx[best_local])
            picked.append(chosen_pos)
            # Update max_sim for every remaining candidate vs the new pick
            chosen_vec = mat[chosen_pos]
            sims_to_chosen = (mat[cand_idx] @ chosen_vec.T).toarray().ravel()
            max_sim = np.maximum(max_sim, sims_to_chosen)

        return np.array(picked, dtype=int)

    def location_lookup(self, location_id: str) -> Optional[pd.Series]:
        pos = self._index.id_to_pos.get(str(location_id))
        if pos is None:
            return None
        return self.frames.locations.iloc[pos]

    # ------------------------------------------------------------------ #
    # Profile builders
    # ------------------------------------------------------------------ #
    def _build_behavioural_profile(
        self, user_key: Optional[str]
    ) -> Tuple[Optional[sparse.csr_matrix], bool]:
        if not user_key:
            return None, False
        interactions = get_user_interactions(self.frames, user_key)
        if interactions is None or interactions.empty:
            return None, False

        rows: List[sparse.csr_matrix] = []
        weights: List[float] = []
        for _, ir in interactions.iterrows():
            pos = self._index.id_to_pos.get(str(ir["location_id"]))
            if pos is None:
                continue
            w = float(
                ir.get("profile_weight")
                or ir.get("effective_score")
                or ir.get("total_interaction_score")
                or 1.0
            )
            rows.append(self._index.tfidf_matrix[pos])
            weights.append(max(w, 0.1))

        if not rows:
            return None, True

        weights_arr = np.array(weights, dtype=float)
        weights_arr /= weights_arr.sum()
        stacked = sparse.vstack(rows)
        profile = stacked.multiply(weights_arr.reshape(-1, 1)).sum(axis=0)
        profile = sparse.csr_matrix(profile)
        profile = _l2_normalise(profile)
        return profile, True

    def _build_form_profile(
        self,
        interests: Sequence[str],
        traveler_type: Optional[str],
        vibe: Optional[str],
        free_text: Optional[str],
    ) -> Tuple[Optional[sparse.csr_matrix], List[str]]:
        category_weights: Dict[str, float] = {}

        for cat in interests:
            category_weights[cat] = category_weights.get(cat, 0.0) + 1.0

        if vibe and vibe.lower() in VIBE_TO_CATEGORIES:
            for cat, w in VIBE_TO_CATEGORIES[vibe.lower()].items():
                category_weights[cat] = category_weights.get(cat, 0.0) + w

        if traveler_type and traveler_type.lower() in TRAVELER_TO_CATEGORIES:
            for cat, w in TRAVELER_TO_CATEGORIES[traveler_type.lower()].items():
                category_weights[cat] = category_weights.get(cat, 0.0) + w

        if free_text:
            text = free_text.lower()
            for kw, cats in FREE_TEXT_KEYWORD_HINTS.items():
                if kw in text:
                    for cat in cats:
                        category_weights[cat] = category_weights.get(cat, 0.0) + 0.5

        if not category_weights:
            return None, []

        # Project the synthetic profile through the same TF-IDF vectorizer
        # so it lives in exactly the same vector space as location vectors.
        pseudo_doc_tokens: List[str] = []
        for cat, w in category_weights.items():
            token = _category_to_token(cat)
            repeat = max(int(round(w * 3)), 1)
            pseudo_doc_tokens.extend([token] * repeat)
        pseudo_doc = " ".join(pseudo_doc_tokens)
        vec = self._index.tfidf_vectorizer.transform([pseudo_doc])
        if vec.nnz == 0:
            return None, sorted(category_weights, key=category_weights.get, reverse=True)

        vec = _l2_normalise(vec)
        inferred = sorted(category_weights, key=lambda k: category_weights[k], reverse=True)
        return vec, inferred

    @staticmethod
    def _blend_profiles(
        behavioural: Optional[sparse.csr_matrix],
        form: Optional[sparse.csr_matrix],
        vocab_size: int,
    ) -> sparse.csr_matrix:
        if behavioural is not None and form is not None:
            blended = PROFILE_BLEND_ALPHA * behavioural + (1.0 - PROFILE_BLEND_ALPHA) * form
            return _l2_normalise(sparse.csr_matrix(blended))
        if behavioural is not None:
            return behavioural
        if form is not None:
            return form
        # No signal at all: zero vector with correct shape. recommend()
        # detects this and ranks purely by popularity.
        return sparse.csr_matrix((1, vocab_size), dtype=float)

    # ------------------------------------------------------------------ #
    # Filtering / explanations
    # ------------------------------------------------------------------ #
    def _build_filter_mask(
        self,
        locations: pd.DataFrame,
        interests: Sequence[str],
        avoid_categories: Sequence[str],
    ) -> np.ndarray:
        n = len(locations)
        if not interests and not avoid_categories:
            return np.ones(n, dtype=bool)

        interests_set = {c.lower() for c in interests}
        avoid_set = {c.lower() for c in avoid_categories}

        mask = np.ones(n, dtype=bool)
        for i, cats in enumerate(locations["categories"]):
            lc = {str(c).lower() for c in (cats or [])}
            if avoid_set and (lc & avoid_set):
                mask[i] = False
                continue
            if interests_set and not (lc & interests_set):
                mask[i] = False
        return mask

    # ------------------------------------------------------------------ #
    # New: seeds, archetype, evidence, richer explain
    # ------------------------------------------------------------------ #
    def _collect_seed_ids(
        self,
        user_key: Optional[str],
        is_returning: bool,
        interests: Sequence[str],
        inferred: Sequence[str],
    ) -> List[str]:
        """Locations that anchor the item-item collab signal.

        * Returning user → their actually-clicked locations.
        * New visitor    → the most popular locations among users in
                            the same behavioural archetype, restricted
                            to their selected interests when possible.
        * Otherwise      → the most popular locations overall.
        """
        if is_returning and user_key:
            interacted = get_user_interactions(self.frames, user_key)
            if interacted is not None and not interacted.empty:
                return [str(x) for x in interacted["location_id"].astype(str).tolist()]

        archetype = self._assign_archetype(user_key, is_returning, interests)
        if archetype and archetype.seed_location_ids:
            return archetype.seed_location_ids

        # Last resort: globally popular locations matching the interests.
        loc = self.frames.locations
        if interests:
            interests_lower = {c.lower() for c in interests}
            mask = loc["categories"].apply(
                lambda cs: bool({str(c).lower() for c in (cs or [])} & interests_lower)
            )
            sub = loc[mask]
        else:
            sub = loc
        if "popularity_raw" in sub.columns and not sub.empty:
            sub = sub.sort_values("popularity_raw", ascending=False)
        return [str(x) for x in sub["location_id"].astype(str).head(5).tolist()]

    def _assign_archetype(
        self,
        user_key: Optional[str],
        is_returning: bool,
        interests: Sequence[str],
    ) -> Optional[ArchetypeAssignment]:
        if is_returning and user_key:
            assignment = self.segmenter.assign_returning(user_key)
            if assignment:
                return assignment
        if interests:
            weights = {c: 1.0 for c in interests}
            return self.segmenter.assign_from_categories(weights)
        return None

    @staticmethod
    def _build_evidence(
        row: pd.Series, trending_score: float = 0.0
    ) -> Dict[str, float | int | str | bool]:
        """Compact, human-readable evidence pulled from `*_all_users` fields."""
        n_users = int(row.get("distinct_users_interacted_location") or 0)
        n_interactions = int(row.get("total_location_interactions_all_users") or 0)
        n_sessions = int(row.get("distinct_sessions_interacted_location") or 0)
        avg_eng_msec = float(row.get("avg_location_engagement_all_users_msec_capped") or 0.0)
        avg_eng_sec = avg_eng_msec / 1000.0

        if n_users == 0:
            summary = "No prior engagement on ChicagoDoes — content match only."
        else:
            summary = (
                f"{n_users} ChicagoDoes user{'s' if n_users != 1 else ''} engaged here · "
                f"{n_interactions} interactions · "
                f"~{avg_eng_sec:.1f}s avg dwell"
            )
        if trending_score >= TRENDING_HIGHLIGHT_THRESHOLD:
            summary += " · 🔥 trending"
        return {
            "n_users_engaged": n_users,
            "n_interactions": n_interactions,
            "n_sessions": n_sessions,
            "avg_engagement_sec": round(avg_eng_sec, 2),
            "is_trending": bool(trending_score >= TRENDING_HIGHLIGHT_THRESHOLD),
            "summary": summary,
        }

    @staticmethod
    def _explain(
        row: pd.Series,
        inferred: Sequence[str],
        sim: float,
        pop: float,
        item_collab: float,
        trending: float,
    ) -> str:
        # The frontend already shows category tags above this reason line,
        # so listing them here just creates "Museums · matches your interest
        # in Museums; ..." duplication. We keep the "matches your interests"
        # phrase but drop the category list when it would just echo the tags.
        cats = list(row.get("categories") or [])
        primary = row.get("primary_category")
        inferred_set = {c.lower() for c in (inferred or [])}
        overlap_non_primary = [
            c for c in cats
            if c.lower() in inferred_set and c != primary
        ]

        bits: List[str] = []
        if overlap_non_primary:
            bits.append(f"also covers your interest in {overlap_non_primary[0]}")
        elif inferred_set and any(c.lower() in inferred_set for c in cats):
            bits.append("matches your selected interests")

        if item_collab > 0.4:
            bits.append("co-visited with your seed locations by other users")
        if trending >= TRENDING_HIGHLIGHT_THRESHOLD:
            bits.append("engagement rising in the recent window")
        if row.get("is_hot_spot_location") and not (trending >= TRENDING_HIGHLIGHT_THRESHOLD):
            # avoid stacking HOT SPOT + trending; they convey similar info
            bits.append("currently a HOT SPOT")
        if pop > 0.6:
            bits.append("popular with other ChicagoDoes users")
        if not bits:
            bits.append("good content-based match")
        return "; ".join(bits)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _category_to_token(cat: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", cat.lower()).strip("_") or "unknown"


def _categories_to_corpus_text(cats: Sequence[str]) -> str:
    return " ".join(_category_to_token(c) for c in (cats or []))


def _l2_normalise(mat: sparse.csr_matrix) -> sparse.csr_matrix:
    n = float(sparse_norm(mat))
    if n < 1e-12:
        return mat
    return mat.multiply(1.0 / n).tocsr()


def _safe_str(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    s = str(v).strip()
    return s or None
