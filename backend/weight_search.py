"""
ChicagoDoes 权重搜索 + MMR 权衡量化 (leakage-safe, CSV 输出)。

为什么快: 六个 component score 对每个 (用户, 地点) 是固定的, 权重只在最后做
加权求和。所以只跑一次真实 recommend() 把分数缓存下来, 之后试任意权重组合
都只是「重新加权 + 重排」, 试几千组也只要几十秒。

三件事:
  A. 跑一次真实 ContentRecommender, 缓存每个 test 用户所有候选地点的六个分数。
  B. 随机/锚点搜索权重, 按 NDCG@10 找最优 (这一步不含 MMR, 纯排序质量)。
  C. 用最优权重量化 MMR 的权衡: 关掉 vs 打开 MMR, 看 NDCG@10 降多少、
     intra-list diversity (列表多样性) 升多少 —— 给 MMR 一个公平的判决。

跑法 (项目根目录, 已 conda activate ateema):
    python -m backend.weight_search
    python -m backend.weight_search --n-samples 4000 --k 10

依赖: backend/ 下要有 evaluation.py 和 evaluation_runner.py。
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .data_loader import _normalise_events
from .recommender import ContentRecommender, WEIGHTS_RETURNING, MMR_LAMBDA
from .evaluation import leave_last_n_split
from .evaluation_runner import build_train_frames


# 六个信号的固定顺序 (和 WEIGHTS_RETURNING 的 key 对齐)
COMPONENTS = ["sim", "popularity", "item_collab", "user_collab", "trending", "session"]
# result dict 里对应的字段名
RESULT_KEYS = {
    "sim":          "similarity_score",
    "popularity":   "popularity_score",
    "item_collab":  "item_collab_score",
    "user_collab":  "user_collab_score",
    "trending":     "trending_score",
    "session":      "session_collab_score",
}


def collect_candidates(rec, universe, test_positives) -> List[dict]:
    """跑一次真实 recommend(), 缓存每个用户所有候选的六个分数 + 常数偏移。

    offset = final_score - Σ(WEIGHTS_RETURNING · components)
           = hot/fav 加成 - feedback 惩罚 (与权重无关的常数项, 保留以忠于生产打分)
    """
    n_loc = len(universe)
    w_cur = np.array([WEIGHTS_RETURNING[c] for c in COMPONENTS], dtype=float)
    cache: List[dict] = []

    for raw_user, pos in test_positives.items():
        user = str(raw_user)
        relevant = {str(x) for x in pos} & universe
        if not relevant:
            continue
        results, _inf, is_returning, _arch = rec.recommend(user_key=user, top_k=n_loc)
        if not results:
            continue

        ids = np.array([r["location_id"] for r in results])
        comps = np.array(
            [[float(r[RESULT_KEYS[c]]) for c in COMPONENTS] for r in results],
            dtype=float,
        )  # shape (n_cand, 6)
        final = np.array([float(r["final_score"]) for r in results], dtype=float)
        offset = final - comps @ w_cur
        rel_mask = np.array([rid in relevant for rid in ids], dtype=bool)

        cache.append({
            "user": user,
            "ids": ids,
            "comps": comps,
            "offset": offset,
            "rel_mask": rel_mask,
            "n_rel": int(rel_mask.sum()),
            "is_returning": bool(is_returning),
        })
    return cache


def _ndcg_at_k(scores: np.ndarray, rel_mask: np.ndarray, n_rel: int, k: int) -> float:
    if n_rel == 0:
        return 0.0
    k = min(k, len(scores))
    top = np.argsort(-scores)[:k]
    hits = rel_mask[top].astype(float)
    discounts = 1.0 / np.log2(np.arange(2, 2 + k))
    dcg = float((hits * discounts).sum())
    idcg = float(discounts[:min(n_rel, k)].sum())
    return dcg / idcg if idcg > 0 else 0.0


def _full_metrics(scores: np.ndarray, rel_mask: np.ndarray, n_rel: int, k: int) -> dict:
    k = min(k, len(scores))
    top = np.argsort(-scores)[:k]
    hits = rel_mask[top].astype(float)
    h = float(hits.sum())
    discounts = 1.0 / np.log2(np.arange(2, 2 + k))
    dcg = float((hits * discounts).sum())
    idcg = float(discounts[:min(n_rel, k)].sum())
    return {
        f"P@{k}":    h / k,
        f"R@{k}":    h / n_rel if n_rel else 0.0,
        f"HR@{k}":   1.0 if h > 0 else 0.0,
        f"NDCG@{k}": dcg / idcg if idcg > 0 else 0.0,
    }


def mean_ndcg(cache: List[dict], w: np.ndarray, k: int) -> float:
    vals = []
    for u in cache:
        scores = u["comps"] @ w + u["offset"]
        vals.append(_ndcg_at_k(scores, u["rel_mask"], u["n_rel"], k))
    return float(np.mean(vals)) if vals else 0.0


def mean_full(cache: List[dict], w: np.ndarray, k: int) -> dict:
    rows = [
        _full_metrics(u["comps"] @ w + u["offset"], u["rel_mask"], u["n_rel"], k)
        for u in cache
    ]
    return pd.DataFrame(rows).mean().to_dict()


# ------------------------------------------------------------------ #
# MMR + diversity (Phase C)
# ------------------------------------------------------------------ #
def _tfidf_row(rec, location_id):
    pos = rec._index.id_to_pos.get(str(location_id))
    return rec._index.tfidf_matrix[pos] if pos is not None else None


def mmr_order(rec, ids: np.ndarray, scores: np.ndarray,
              lam: float, pool: int, top_k: int) -> List[str]:
    """贪心 MMR: 镜像 recommender._mmr_select 的逻辑, 返回 top_k 个 id。"""
    pool_idx = np.argsort(-scores)[:pool]
    pool_ids = [ids[i] for i in pool_idx]
    pool_scores = scores[pool_idx]
    rows = [_tfidf_row(rec, cid) for cid in pool_ids]

    selected: List[int] = []
    chosen_vecs = []
    remaining = list(range(len(pool_ids)))
    while remaining and len(selected) < top_k:
        best_i, best_val = None, -np.inf
        for i in remaining:
            rel = pool_scores[i]
            if chosen_vecs and rows[i] is not None:
                max_sim = max(
                    float((rows[i] @ v.T).toarray().ravel()[0]) for v in chosen_vecs
                )
            else:
                max_sim = 0.0
            val = lam * rel - (1 - lam) * max_sim
            if val > best_val:
                best_val, best_i = val, i
        selected.append(best_i)
        remaining.remove(best_i)
        if rows[best_i] is not None:
            chosen_vecs.append(rows[best_i])
    return [pool_ids[i] for i in selected]


def intra_list_diversity(rec, id_list: List[str], k: int) -> float:
    rows = [_tfidf_row(rec, cid) for cid in id_list[:k]]
    rows = [r for r in rows if r is not None]
    if len(rows) < 2:
        return 0.0
    sims = []
    for a in range(len(rows)):
        for b in range(a + 1, len(rows)):
            sims.append(float((rows[a] @ rows[b].T).toarray().ravel()[0]))
    return 1.0 - sum(sims) / len(sims)


def mmr_verdict(cache: List[dict], rec, w: np.ndarray, k: int,
                lam: float, pool: int = 60) -> dict:
    """用最优权重比较 关MMR vs 开MMR 的 NDCG@k 和 diversity@k。"""
    ndcg_no, ndcg_mmr, ild_no, ild_mmr = [], [], [], []
    for u in cache:
        scores = u["comps"] @ w + u["offset"]
        # 关 MMR: 纯按分数排
        order = np.argsort(-scores)
        ids_no = [u["ids"][i] for i in order[:k]]
        ndcg_no.append(_ndcg_at_k(scores, u["rel_mask"], u["n_rel"], k))
        ild_no.append(intra_list_diversity(rec, ids_no, k))
        # 开 MMR
        ids_mmr = mmr_order(rec, u["ids"], scores, lam=lam, pool=pool, top_k=k)
        rel = set(u["ids"][u["rel_mask"]])
        hits = np.array([cid in rel for cid in ids_mmr], dtype=float)
        disc = 1.0 / np.log2(np.arange(2, 2 + len(ids_mmr)))
        idcg = disc[:min(u["n_rel"], len(ids_mmr))].sum()
        ndcg_mmr.append(float((hits * disc).sum() / idcg) if idcg > 0 else 0.0)
        ild_mmr.append(intra_list_diversity(rec, ids_mmr, k))
    return {
        "ndcg_no_mmr": float(np.mean(ndcg_no)),
        "ndcg_with_mmr": float(np.mean(ndcg_mmr)),
        "diversity_no_mmr": float(np.mean(ild_no)),
        "diversity_with_mmr": float(np.mean(ild_mmr)),
    }


def main():
    ap = argparse.ArgumentParser(description="Weight search + MMR trade-off (NDCG@K)")
    ap.add_argument("--events", default="data/private/events.csv")
    ap.add_argument("--dim",    default="data/location_dim.csv")
    ap.add_argument("--geo",    default="data/locations_geo.csv")
    ap.add_argument("--n-holdout", type=int, default=1)
    ap.add_argument("--min-history", type=int, default=3)
    ap.add_argument("--k", type=int, default=10, help="优化目标 NDCG@k")
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    k = args.k

    # --- 数据 + split + 真实 recommender (和 runner 一致) ---
    print("[1/4] 读 events / split / 构建真实 recommender ...")
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

    # --- A. 缓存候选分数 ---
    print("[2/4] 缓存每个用户候选的六个 component score ...")
    cache = collect_candidates(rec, universe, test_positives)
    print(f"      缓存用户 {len(cache)} (returning={sum(u['is_returning'] for u in cache)})")

    # --- B. 权重搜索 ---
    print(f"[3/4] 搜索权重 (NDCG@{k}, {args.n_samples} 随机样本 + 锚点) ...")
    anchors = {
        "current(论文)":   np.array([WEIGHTS_RETURNING[c] for c in COMPONENTS]),
        "content_heavy":   np.array([0.50, 0.15, 0.10, 0.10, 0.05, 0.10]),
        "content_only_ish":np.array([0.85, 0.05, 0.00, 0.00, 0.05, 0.05]),
        "sim+session":     np.array([0.45, 0.10, 0.10, 0.05, 0.05, 0.25]),
    }
    rng = np.random.default_rng(args.seed)
    records = []
    for name, w in anchors.items():
        records.append({"config": name, "ndcg": mean_ndcg(cache, w, k),
                        **{c: round(float(w[i]), 3) for i, c in enumerate(COMPONENTS)}})
    # 随机样本 (Dirichlet, 偏向探索角落以覆盖 content-heavy)
    samples = rng.dirichlet(np.full(len(COMPONENTS), 0.7), size=args.n_samples)
    for w in samples:
        records.append({"config": "random", "ndcg": mean_ndcg(cache, w, k),
                        **{c: round(float(w[i]), 3) for i, c in enumerate(COMPONENTS)}})

    df = pd.DataFrame(records).sort_values("ndcg", ascending=False).reset_index(drop=True)
    best_row = df.iloc[0]
    best_w = np.array([best_row[c] for c in COMPONENTS], dtype=float)

    # 当前权重的成绩 (用于对比基线)
    cur_w = np.array([WEIGHTS_RETURNING[c] for c in COMPONENTS])
    cur_ndcg = mean_ndcg(cache, cur_w, k)

    print("\n" + "=" * 78)
    print(f"当前论文权重 NDCG@{k} = {cur_ndcg:.4f}")
    print(f"搜索到的最优   NDCG@{k} = {best_row['ndcg']:.4f}  "
          f"(提升 {(best_row['ndcg']-cur_ndcg)/max(cur_ndcg,1e-9)*100:+.1f}%)")
    print("=" * 78)
    print("\n--- Top 8 权重配置 (按 NDCG 降序) ---")
    show = df.head(8)[["config", "ndcg"] + COMPONENTS]
    print(show.round(4).to_string(index=False))
    print("\n--- 锚点配置成绩 ---")
    print(df[df["config"].isin(anchors)].round(4).to_string(index=False))

    # 最优权重的完整指标
    best_full = mean_full(cache, best_w, k)
    print(f"\n--- 最优权重的完整指标 @{k} ---")
    print({m: round(v, 4) for m, v in best_full.items()})

    # --- C. MMR 判决 ---
    print(f"\n[4/4] 用最优权重量化 MMR (λ={MMR_LAMBDA}) 的权衡 ...")
    verdict = mmr_verdict(cache, rec, best_w, k, lam=MMR_LAMBDA)
    nd_drop = (verdict["ndcg_with_mmr"] - verdict["ndcg_no_mmr"]) / max(verdict["ndcg_no_mmr"], 1e-9) * 100
    dv_gain = (verdict["diversity_with_mmr"] - verdict["diversity_no_mmr"]) / max(verdict["diversity_no_mmr"], 1e-9) * 100
    print("=" * 78)
    print(f"  关 MMR:  NDCG@{k}={verdict['ndcg_no_mmr']:.4f}   diversity@{k}={verdict['diversity_no_mmr']:.4f}")
    print(f"  开 MMR:  NDCG@{k}={verdict['ndcg_with_mmr']:.4f}   diversity@{k}={verdict['diversity_with_mmr']:.4f}")
    print(f"  → MMR 用 {nd_drop:+.1f}% 的 NDCG 换来 {dv_gain:+.1f}% 的多样性")
    print("=" * 78)

    # --- 存档 (后面 MLflow 直接读) ---
    out_dir = Path("data")
    df.round(6).to_csv(out_dir / "weight_search_results.csv", index=False)
    best_payload = {
        "best_weights_returning": {c: float(best_w[i]) for i, c in enumerate(COMPONENTS)},
        f"ndcg@{k}": float(best_row["ndcg"]),
        f"ndcg@{k}_current": float(cur_ndcg),
        "full_metrics": best_full,
        "mmr_verdict": verdict,
        "n_users": len(cache),
    }
    with open(out_dir / "best_weights.json", "w") as f:
        json.dump(best_payload, f, indent=2, ensure_ascii=False)
    print(f"\n已存: {(out_dir/'weight_search_results.csv').resolve()}")
    print(f"已存: {(out_dir/'best_weights.json').resolve()}")
    print("\n提示: best_weights.json 就是下一步 MLflow 要记录的第一条 run。")


if __name__ == "__main__":
    main()
