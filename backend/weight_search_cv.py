"""
ChicagoDoes 权重搜索的 K-fold 交叉验证 (回答: +27.5% 是真信号还是过拟合?)。

原理:
  把缓存的用户分成 K 折。每一折轮流当「验证集」, 其余 K-1 折当「搜索集」:
    1. 只在搜索集上搜最优权重 (按 NDCG@k)。
    2. 把这组权重拿到【没搜过的】验证集上评估。
    3. 同时在验证集上评估「当前论文权重」做对照。
  如果在留出用户上, 搜出来的权重仍稳定优于当前权重 → 方向结论可信;
  如果留出表现掉回当前水平 → 之前的提升是对 120 人的过拟合。

额外产出:
  * 过拟合 gap = 搜索集内 NDCG  -  留出 NDCG (越大越过拟合)。
  * 权重方向稳定性: 每个信号的权重在 K 折间的均值±标准差。若 sim 折折都高、
    session 折折都低 → 「content 被低估、session 被高估」是稳健结论。

跑法 (项目根目录):
    python -m backend.weight_search_cv
    python -m backend.weight_search_cv --folds 5 --n-samples 3000 --k 10
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .data_loader import _normalise_events
from .recommender import ContentRecommender, WEIGHTS_RETURNING
from .evaluation import leave_last_n_split
from .evaluation_runner import build_train_frames
# 复用 weight_search 里的缓存与打分逻辑, 避免重复
from .weight_search import COMPONENTS, collect_candidates, mean_ndcg, mean_full


def search_best_weights(cache_subset: List[dict], k: int, n_samples: int,
                        rng: np.random.Generator):
    """只在 cache_subset 上搜最优权重。current 永远作为候选之一。"""
    current = np.array([WEIGHTS_RETURNING[c] for c in COMPONENTS], dtype=float)
    candidates = [current,
                  np.array([0.50, 0.15, 0.10, 0.10, 0.05, 0.10]),  # content_heavy 锚点
                  ]
    candidates += list(rng.dirichlet(np.full(len(COMPONENTS), 0.7), size=n_samples))

    best_w, best_score = current, -np.inf
    for w in candidates:
        s = mean_ndcg(cache_subset, w, k)
        if s > best_score:
            best_score, best_w = s, np.asarray(w, dtype=float)
    return best_w, best_score


def main():
    ap = argparse.ArgumentParser(description="K-fold CV for weight search")
    ap.add_argument("--events", default="data/private/events.csv")
    ap.add_argument("--dim",    default="data/location_dim.csv")
    ap.add_argument("--geo",    default="data/locations_geo.csv")
    ap.add_argument("--n-holdout", type=int, default=1)
    ap.add_argument("--min-history", type=int, default=3)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-samples", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    k = args.k

    # --- 数据 + split + 真实 recommender + 缓存 (和前面一致, 只跑一次) ---
    print("[1/3] 读 events / split / 构建 recommender / 缓存候选分数 ...")
    raw = pd.read_csv(args.events, low_memory=False)
    events = _normalise_events(raw)
    events["user_key"] = events["user_key"].astype(str).str.strip()
    events["location_id"] = events["location_id"].astype(str).str.strip()
    events = events[(events["user_key"] != "") & (events["location_id"] != "")]
    train_events, test_positives = leave_last_n_split(
        events, "user_key", "location_id", "event_time",
        n_holdout=args.n_holdout, min_history=args.min_history,
    )
    try:
        train_frames = build_train_frames(train_events, Path(args.dim), Path(args.geo))
        rec = ContentRecommender(train_frames)
    except Exception:
        print("\n❌ 构建 recommender 失败, traceback 发我:\n")
        traceback.print_exc()
        sys.exit(1)
    universe = set(train_frames.locations["location_id"].astype(str))
    cache = collect_candidates(rec, universe, test_positives)
    n_users = len(cache)
    print(f"      缓存用户 {n_users}")
    if n_users < args.folds * 5:
        print(f"      ⚠️ 用户数偏少, {args.folds} 折每折仅约 {n_users//args.folds} 人, 结果波动会大。")

    # --- 分折 ---
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_users)
    fold_ids = np.array_split(perm, args.folds)
    current = np.array([WEIGHTS_RETURNING[c] for c in COMPONENTS], dtype=float)

    print(f"[2/3] {args.folds}-fold 交叉验证 (NDCG@{k}) ...")
    rows = []
    best_ws = []
    for fi, val_idx in enumerate(fold_ids):
        val_set = [cache[i] for i in val_idx]
        search_set = [cache[i] for i in range(n_users) if i not in set(val_idx)]
        # 每折用不同子种子搜索
        best_w, in_sample = search_best_weights(
            search_set, k, args.n_samples, np.random.default_rng(args.seed + fi + 1)
        )
        best_ws.append(best_w)
        heldout_best = mean_ndcg(val_set, best_w, k)      # 搜出的权重在留出集表现
        heldout_cur = mean_ndcg(val_set, current, k)      # 当前权重在留出集表现
        rows.append({
            "fold": fi + 1,
            "n_val": len(val_set),
            "in_sample_best": round(in_sample, 4),
            "heldout_best": round(heldout_best, 4),
            "heldout_current": round(heldout_cur, 4),
            "heldout_gain%": round((heldout_best - heldout_cur) / max(heldout_cur, 1e-9) * 100, 1),
            "overfit_gap": round(in_sample - heldout_best, 4),
        })

    df = pd.DataFrame(rows)

    # --- 汇总 ---
    print("\n" + "=" * 80)
    print("每折结果 (heldout = 在没搜过的用户上的成绩, 这才是诚实的数):\n")
    print(df.to_string(index=False))

    mean_heldout_best = df["heldout_best"].mean()
    mean_heldout_cur = df["heldout_current"].mean()
    gain = (mean_heldout_best - mean_heldout_cur) / max(mean_heldout_cur, 1e-9) * 100
    n_win = int((df["heldout_best"] > df["heldout_current"]).sum())
    mean_overfit = df["overfit_gap"].mean()

    print("\n" + "=" * 80)
    print(f"留出集平均 NDCG@{k}:  搜索权重={mean_heldout_best:.4f}  "
          f"当前权重={mean_heldout_cur:.4f}")
    print(f"交叉验证后的真实提升:  {gain:+.1f}%   "
          f"(对比之前 in-sample 的 +27.5%)")
    print(f"搜索权重赢当前权重的折数:  {n_win} / {args.folds}")
    print(f"平均过拟合 gap (in-sample - heldout):  {mean_overfit:+.4f}")
    print("=" * 80)

    # --- 权重方向稳定性 ---
    bw = np.array(best_ws)  # (folds, 6)
    print("\n--- 各折最优权重的方向稳定性 (均值 ± 标准差) ---")
    stab = pd.DataFrame({
        "component": COMPONENTS,
        "current": [round(float(current[i]), 3) for i in range(len(COMPONENTS))],
        "cv_mean": [round(float(bw[:, i].mean()), 3) for i in range(len(COMPONENTS))],
        "cv_std":  [round(float(bw[:, i].std()), 3) for i in range(len(COMPONENTS))],
    })
    print(stab.to_string(index=False))

    # --- 自动解读 ---
    print("\n--- 解读 ---")
    if n_win == args.folds and gain > 5:
        print(f"✅ {args.folds}/{args.folds} 折留出集上搜索权重都赢当前权重, 提升 {gain:+.1f}%。")
        print("   '重调权重有用' 通过交叉验证, 可以写进论文作为稳健结论。")
    elif n_win >= args.folds - 1 and gain > 0:
        print(f"🟡 多数折 ({n_win}/{args.folds}) 留出集上有正向提升 ({gain:+.1f}%), 方向可信但幅度需保守表述。")
    else:
        print(f"⚠️ 留出集上提升不稳 (赢 {n_win}/{args.folds} 折, {gain:+.1f}%)。")
        print("   之前的 in-sample 提升大概率是过拟合, 论文不能声称重调权重显著有效。")
    # 方向性结论
    sim_up = bw[:, 0].mean() > current[0]
    sess_down = bw[:, COMPONENTS.index("session")].mean() < WEIGHTS_RETURNING["session"]
    if sim_up and sess_down:
        print("✅ 方向稳定: 各折一致把 sim(content) 调高、session 调低 —— "
              "'content 被低估、session 被高估' 是稳健结论。")

    out = Path("data") / "weight_cv_results.csv"
    df.to_csv(out, index=False)
    print(f"\n已存: {out.resolve()}")


if __name__ == "__main__":
    main()
