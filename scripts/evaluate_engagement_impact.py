#!/usr/bin/env python3
"""Compare recommendations before vs after engagement filtering (offline)."""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("ENGAGEMENT_POLICY", "balanced")

import pandas as pd

from backend.data_loader import load_warehouse
from backend.engagement import EngagementConfig, apply_engagement_policy
from backend.recommender import ContentRecommender


def _synthetic_frames_from_events(data: Path):
    """Build a minimal warehouse when features CSV is not cached locally."""
    from backend.data_loader import (
        WarehouseFrames,
        _expand_to_official_universe,
        _normalise_events,
        _split_category_string,
    )

    ev = _normalise_events(pd.read_csv(data / "events.csv", low_memory=False))
    dim = pd.read_csv(data / "location_dim.csv")
    dim["location_id"] = dim["location_id"].astype(str)

    rows = []
    for _, r in ev.dropna(subset=["user_key", "location_id"]).iterrows():
        cats = _split_category_string(r.get("location_category_name", ""))
        rows.append({
            "user_key": str(r["user_key"]),
            "location_id": str(r["location_id"]),
            "location_name": str(r.get("location_name", "")),
            "primary_category": r.get("primary_category") or (cats[0] if cats else None),
            "categories": cats,
            "total_interactions_with_location": 1,
            "total_interaction_score": float(r.get("interaction_weight") or 1.0),
            "total_location_engagement_msec_capped": float(
                r.get("engagement_time_msec_capped") or 0
            ),
        })
    df = pd.DataFrame(rows)
    from backend.data_loader import _build_interactions, _build_locations, _build_users

    interactions = _build_interactions(df)
    locations = _build_locations(df)
    users = (
        df.groupby("user_key", as_index=False)
        .agg(total_user_interaction_score=("total_interaction_score", "sum"))
    )
    geo = data / "locations_geo.csv"
    locations = _expand_to_official_universe(
        locations, data / "location_dim.csv",
        geo_df=pd.read_csv(geo) if geo.exists() else None,
    )
    return WarehouseFrames(
        interactions=interactions, locations=locations, users=users, events=ev,
    )


def _load_frames():
    data = ROOT / "data"
    for name in ("user_location_full_features.csv", "user_location_features.csv"):
        feat = data / name
        if feat.exists():
            return load_warehouse(
                csv_path=feat,
                location_dim_path=data / "location_dim.csv",
                events_path=data / "events.csv",
                geo_path=data / "locations_geo.csv" if (data / "locations_geo.csv").exists() else None,
            )
    print("No features CSV — synthesizing interactions from events.csv")
    return _synthetic_frames_from_events(data)


def _topk(rec: ContentRecommender, user_key: str, interests: list, k: int = 10):
    results, _, is_ret, _ = rec.recommend(
        user_key=user_key,
        interests=interests,
        top_k=k,
    )
    return [r["location_id"] for r in results], is_ret


def main() -> None:
    frames = _load_frames()
    if frames.events is None or frames.events.empty:
        print("No events.csv — run with cached data that includes events.")
        sys.exit(1)

    # Baseline: strip engagement enrichment
    baseline_frames = copy.deepcopy(frames)
    baseline_frames.events = frames.events.copy()
    for attr in (
        "interactions_enriched",
        "user_location_effective",
        "engagement_report",
        "engagement_config",
    ):
        setattr(baseline_frames, attr, None)
    # Restore raw events without quality columns
    for col in ("is_qualified", "qualify_reason", "event_quality_weight", "engagement_msec_used"):
        if col in baseline_frames.events.columns:
            baseline_frames.events = baseline_frames.events.drop(columns=[col])

    rec_before = ContentRecommender(baseline_frames)

    apply_engagement_policy(frames)
    rec_after = ContentRecommender(frames)

    report = frames.engagement_report or {}
    print("=== Engagement report ===")
    for k, v in report.items():
        if k != "qualify_reason_counts":
            print(f"  {k}: {v}")
    print("  qualify_reason_counts:", report.get("qualify_reason_counts"))

    print("\n=== Model graph size ===")
    print(
        f"  coviz nnz: {rec_before.coviz.jaccard.nnz} -> {rec_after.coviz.jaccard.nnz}"
    )
    print(
        f"  session pairs: {rec_before.session_coviz.jaccard.nnz} -> "
        f"{rec_after.session_coviz.jaccard.nnz}"
    )
    print(
        f"  transitions: {rec_before.transitions.transitions.nnz} -> "
        f"{rec_after.transitions.transitions.nnz}"
    )
    trending_before = int((rec_before._trending > 0).sum())
    trending_after = int((rec_after._trending > 0).sum())
    print(f"  trending locations: {trending_before} -> {trending_after}")

    # Pick test users: top by raw interaction count
    users = (
        frames.interactions.groupby("user_key")
        .size()
        .sort_values(ascending=False)
        .head(8)
        .index.tolist()
    )

    print("\n=== Per-user recommendation diff (Bars + Restaurants) ===")
    interests = ["Bars", "Restaurants"]
    changed_users = 0
    returning_flip = 0
    for uid in users:
        before_ids, ret_b = _topk(rec_before, uid, interests)
        after_ids, ret_a = _topk(rec_after, uid, interests)
        overlap = len(set(before_ids) & set(after_ids))
        jacc = overlap / max(len(set(before_ids) | set(after_ids)), 1)
        if before_ids != after_ids:
            changed_users += 1
        if ret_b != ret_a:
            returning_flip += 1
        print(
            f"  {uid[:8]}… ret {ret_b}->{ret_a} "
            f"top10 overlap {overlap}/10 jacc={jacc:.2f}"
        )

    print(f"\nUsers with different top-10: {changed_users}/{len(users)}")
    print(f"Returning flag flips: {returning_flip}/{len(users)}")

    # Cold-start control (no user)
    b0, _ = _topk(rec_before, None, ["Attractions", "Parks"])
    a0, _ = _topk(rec_after, None, ["Attractions", "Parks"])
    print(f"\nCold-start top-5 unchanged: {b0[:5] == a0[:5]} (content+pop still dominate)")


if __name__ == "__main__":
    main()
