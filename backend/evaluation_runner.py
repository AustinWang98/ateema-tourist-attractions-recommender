"""
ChicagoDoes 离线评估 runner —— leakage-safe, 复用真实的 ContentRecommender。

核心思路 (leakage discipline):
  1. 读 events.csv, 按 event_time 做 leave-last-N holdout: 每个用户最近 N 个
     location 留作 test, 其余作 train。
  2. 【只用 train events】重建 WarehouseFrames —— popularity priors / collab /
     K-Means / 用户 behavioural profile 全部不含 held-out 交互。
  3. 用 train frames 实例化真实的 ContentRecommender, 对每个 test 用户跑
     recommend(), 看 held-out 的 location 有没有进 top-K。
  4. 同一份 recommend() 候选结果, 按不同 component score 重排, 得到
     hybrid / hybrid_no_mmr / content_only / collab_only / popularity 五条排序,
     彼此 apples-to-apples 对比 (同一候选池, 只差排序信号)。

跑法 (项目根目录, 已 conda activate ateema):
    python -m backend.evaluation_runner
可选参数:
    python -m backend.evaluation_runner --n-holdout 1 --min-history 3 --k 5 10 20

依赖: backend/evaluation.py (前面给的 metric 模块) 必须和本文件同在 backend/ 下。
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# --- 复用项目里的真实模块 ---
from .data_loader import (
    WarehouseFrames,
    _normalise_events,
    _build_locations_from_events,
    _expand_to_official_universe_from_df,
    _build_interactions_from_events,
    _build_users_from_interactions,
)
from .recommender import ContentRecommender

# --- 复用前面给的 metric 模块 ---
from .evaluation import (
    leave_last_n_split,
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    average_precision_at_k,
)


# 五种排序: 名字 -> 从一组 result dict 里抽取有序 location_id 列表的函数。
# "hybrid" 就是 recommend() 原样输出 (已含 MMR), 是真实产品行为。
# 其余四条是对同一候选池按单一信号重排, 作为 baselines / ablations。
RANKERS = {
    "hybrid":        lambda res: [r["location_id"] for r in res],
    "hybrid_no_mmr": lambda res: [r["location_id"] for r in sorted(res, key=lambda r: r["final_score"],     reverse=True)],
    "content_only":  lambda res: [r["location_id"] for r in sorted(res, key=lambda r: r["similarity_score"], reverse=True)],
    "collab_only":   lambda res: [r["location_id"] for r in sorted(res, key=lambda r: r["item_collab_score"],reverse=True)],
    "popularity":    lambda res: [r["location_id"] for r in sorted(res, key=lambda r: r["popularity_score"], reverse=True)],
}


def build_train_frames(
    train_events: pd.DataFrame,
    dim_path: Path,
    geo_path: Path | None = None,
) -> WarehouseFrames:
    """只用 train events 重建 frames —— 镜像 load_public_demo_warehouse 的逻辑。

    location universe 仍扩到 official dim (和生产一致), 这样候选集完整,
    held-out 的 location 不会因为只剩它一个用户访问而从宇宙里消失。
    """
    observed = _build_locations_from_events(train_events)
    dim = pd.read_csv(dim_path)
    geo_df = pd.read_csv(geo_path) if (geo_path and Path(geo_path).exists()) else None
    locations = _expand_to_official_universe_from_df(observed, dim, geo_df=geo_df)
    interactions = _build_interactions_from_events(train_events)
    users = _build_users_from_interactions(interactions)
    return WarehouseFrames(
        interactions=interactions,
        locations=locations,
        users=users,
        events=train_events,
    )


def evaluate(
    rec: ContentRecommender,
    universe: set,
    test_positives: Dict[str, set],
    k_values=(5, 10, 20),
) -> Dict[str, Dict[str, np.ndarray]]:
    """对每个 test 用户跑 recommend(), 收集五种排序的 per-user 指标数组。

    Returns: {model_name: {metric_name: np.array(每个用户一个值)}}
    """
    n_loc = len(universe)
    metric_fns = {
        "P":    precision_at_k,
        "R":    recall_at_k,
        "HR":   hit_rate_at_k,
        "NDCG": ndcg_at_k,
        "MAP":  average_precision_at_k,
    }
    # 初始化收集器
    per_user = {
        m: {f"{abbr}@{k}": [] for abbr in metric_fns for k in k_values}
        for m in RANKERS
    }

    n_eval = n_returning = dropped_no_truth = dropped_empty = 0

    for raw_user, pos in test_positives.items():
        user = str(raw_user)
        relevant = {str(x) for x in pos} & universe   # 只保留宇宙里存在的 ground truth
        if not relevant:
            dropped_no_truth += 1
            continue

        results, _inferred, is_returning, _arch = rec.recommend(
            user_key=user, top_k=n_loc
        )
        if not results:
            dropped_empty += 1
            continue

        n_eval += 1
        if is_returning:
            n_returning += 1

        for model_name, ranker in RANKERS.items():
            ranking = ranker(results)
            for abbr, fn in metric_fns.items():
                for k in k_values:
                    per_user[model_name][f"{abbr}@{k}"].append(
                        fn(ranking, relevant, k)
                    )

    # 转成 np.array
    out = {
        m: {metric: np.asarray(vals, dtype=float) for metric, vals in d.items()}
        for m, d in per_user.items()
    }
    out["_meta"] = {
        "n_evaluated": n_eval,
        "n_returning": n_returning,
        "n_new": n_eval - n_returning,
        "dropped_no_truth_in_universe": dropped_no_truth,
        "dropped_empty_reco": dropped_empty,
    }
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05):
    """对一个 per-user 指标数组做 bootstrap 95% 置信区间。"""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(0)
    boot = np.array([
        rng.choice(values, size=values.size, replace=True).mean()
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser(description="ChicagoDoes leakage-safe offline eval")
    parser.add_argument("--events", default="data/private/events.csv")
    parser.add_argument("--dim",    default="data/location_dim.csv")
    parser.add_argument("--geo",    default="data/locations_geo.csv")
    parser.add_argument("--n-holdout", type=int, default=1)
    parser.add_argument("--min-history", type=int, default=3,
                        help="参与评估所需的最少 distinct location 数")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    args = parser.parse_args()

    k_values = tuple(args.k)

    # --- 1. 读 events 并解析时间 ---
    print(f"[1/4] 读取 events: {args.events}")
    raw = pd.read_csv(args.events, low_memory=False)
    events = _normalise_events(raw)               # 解析 event_time, 丢掉无时间戳的行
    events["user_key"] = events["user_key"].astype(str).str.strip()
    events["location_id"] = events["location_id"].astype(str).str.strip()
    events = events[(events["user_key"] != "") & (events["location_id"] != "")]
    print(f"      有效事件 {len(events)}, 用户 {events['user_key'].nunique()}, "
          f"location {events['location_id'].nunique()}")
    print(f"      时间范围 {events['event_time'].min()} → {events['event_time'].max()}")

    # --- 2. leave-last-N 时间切分 ---
    print(f"[2/4] leave-last-{args.n_holdout} split (min_history={args.min_history})")
    train_events, test_positives = leave_last_n_split(
        events,
        user_col="user_key",
        item_col="location_id",
        time_col="event_time",
        n_holdout=args.n_holdout,
        min_history=args.min_history,
    )
    print(f"      可评估用户 {len(test_positives)}, train 事件 {len(train_events)}")
    if len(test_positives) < 30:
        print("      ⚠️  可评估用户 < 30, 指标仅作趋势参考, 不要当强结论。")

    # --- 3. 用 train 重建 frames + 实例化真实 recommender ---
    print("[3/4] 用 train events 重建 frames 并构建 ContentRecommender ...")
    try:
        train_frames = build_train_frames(
            train_events, Path(args.dim), Path(args.geo)
        )
        rec = ContentRecommender(train_frames)
    except Exception:
        print("\n❌ 构建 recommender 失败, 完整 traceback 如下 (把它发我):\n")
        traceback.print_exc()
        sys.exit(1)

    universe = set(train_frames.locations["location_id"].astype(str))
    print(f"      location universe = {len(universe)}")

    # --- 4. 评估 ---
    print("[4/4] 逐用户评估中 (调用真实 recommend) ...")
    res = evaluate(rec, universe, test_positives, k_values=k_values)
    meta = res.pop("_meta")

    # 均值对比表
    table = pd.DataFrame({
        model: {metric: arr.mean() for metric, arr in metrics.items()}
        for model, metrics in res.items()
    }).T
    # 列排序: 按 metric/k
    ordered_cols = [f"{abbr}@{k}" for abbr in ("P", "R", "HR", "NDCG", "MAP") for k in k_values]
    table = table[[c for c in ordered_cols if c in table.columns]]

    print("\n" + "=" * 70)
    print(f"评估用户数: {meta['n_evaluated']}  "
          f"(returning={meta['n_returning']}, new={meta['n_new']})")
    print(f"丢弃: ground-truth 不在宇宙={meta['dropped_no_truth_in_universe']}, "
          f"无推荐结果={meta['dropped_empty_reco']}")
    print("=" * 70)
    print("\n各模型均值对比 (越高越好):\n")
    print(table.round(4).to_string())

    # headline 指标 + bootstrap 95% CI (看模型间区间是否重叠)
    head_k = k_values[len(k_values) // 2]
    print(f"\n--- 置信区间检查 (NDCG@{head_k}, Recall@{head_k}, bootstrap 95% CI) ---")
    print("如果各模型 CI 大面积重叠, 说明当前样本量下差异不显著, 论文要诚实写明。\n")
    for metric in (f"NDCG@{head_k}", f"R@{head_k}"):
        print(f"  {metric}:")
        for model in res:
            arr = res[model][metric]
            lo, hi = bootstrap_ci(arr)
            print(f"    {model:14s} mean={arr.mean():.4f}  95%CI=[{lo:.4f}, {hi:.4f}]")
        print()

    # 存一份 CSV, 方便贴进论文 / 后面接 MLflow
    out_path = Path("data") / "offline_eval_results.csv"
    table.round(6).to_csv(out_path)
    print(f"对比表已存: {out_path.resolve()}")


if __name__ == "__main__":
    main()
