"""
ChicagoDoes 离线评估模块 / Offline evaluation harness.

实现论文 §3.6 的方法论 (methodology):
  - leave-last-N 时间分层 holdout split
  - ranking metrics: Precision@K, Recall@K, HitRate@K, NDCG@K, MAP@K
  - list-quality metrics: coverage, intra-list diversity, novelty, category entropy
  - baselines + 一个产出对比表 (comparison table) 的 runner

设计为 model-agnostic: 把你们的 recommender 包进 `Recommender` protocol
(一个 `.recommend(user_key, k)` 方法) 即可接入。

LEAKAGE 关键提醒 (read this):
  调用方负责保证传入 evaluate 的 model 是【只用 train_events 重建】出来的。
  即: popularity priors / collab 矩阵 / K-Means / 用户 behavioral profile
  全部必须排除 test_positives 里那些 held-out 交互, 否则就是 label leakage,
  评估结果会虚高、不可信 (论文 §2.4 / Discussion 反复强调的点)。
"""

from __future__ import annotations
import math
import zlib
from dataclasses import dataclass, field
from typing import Protocol, Sequence, Mapping, Callable, Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Holdout split  (leave-last-N, time-stratified)
# ---------------------------------------------------------------------------
def leave_last_n_split(
    events: pd.DataFrame,
    user_col: str = "user_key",
    item_col: str = "location_id",
    time_col: str = "event_timestamp",
    n_holdout: int = 2,
    min_history: int = 3,
) -> tuple[pd.DataFrame, dict[str, set]]:
    """
    每个用户按时间排序其 qualified interactions, 把【最近的 n_holdout 个不同
    location】留作 test。交互数 < min_history 的用户不参与评估 (history 太少),
    但其行为仍保留在 train 里用于构建全局 aggregates。

    Returns
    -------
    train_events   : DataFrame  (除 held-out 交互外的全部行)
    test_positives : {user_key: set(location_id)}  held-out ground truth
    """
    events = events.sort_values([user_col, time_col])
    train_parts = []
    test_positives: dict[str, set] = {}

    for user, grp in events.groupby(user_col, sort=False):
        # 按最后一次交互时间的先后, 取不同 location 的时间顺序
        ordered_items = grp.drop_duplicates(item_col, keep="last")[item_col].tolist()
        if len(ordered_items) < min_history:
            train_parts.append(grp)            # 留作 aggregates, 不评估
            continue
        held = set(ordered_items[-n_holdout:])  # 最近的 n 个 location
        test_positives[user] = held
        train_parts.append(grp[~grp[item_col].isin(held)])

    train_events = pd.concat(train_parts, ignore_index=True)
    return train_events, test_positives


# ---------------------------------------------------------------------------
# 2. Ranking metrics   (recommended: 有序 list[item], relevant: set[item])
# ---------------------------------------------------------------------------
def precision_at_k(recommended: Sequence, relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for it in recommended[:k] if it in relevant)
    return hits / k


def recall_at_k(recommended: Sequence, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for it in recommended[:k] if it in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: Sequence, relevant: set, k: int) -> float:
    return 1.0 if any(it in relevant for it in recommended[:k]) else 0.0


def ndcg_at_k(recommended: Sequence, relevant: set, k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, it in enumerate(recommended[:k]) if it in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended: Sequence, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    score, hits = 0.0, 0
    for i, it in enumerate(recommended[:k]):
        if it in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)


# ---------------------------------------------------------------------------
# 3. List-quality metrics
# ---------------------------------------------------------------------------
def catalog_coverage(all_reco_lists: Sequence[Sequence], catalog_size: int, k: int) -> float:
    seen: set = set()
    for lst in all_reco_lists:
        seen.update(lst[:k])
    return len(seen) / catalog_size if catalog_size else 0.0


def intra_list_diversity(recommended: Sequence, item_vectors: Mapping, k: int) -> float:
    """1 - 平均两两 cosine 相似度, 基于 top-k 的 TF-IDF 向量。越高越多样。"""
    items = [it for it in recommended[:k] if it in item_vectors]
    if len(items) < 2:
        return 0.0
    sims = []
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            va, vb = item_vectors[items[a]], item_vectors[items[b]]
            denom = np.linalg.norm(va) * np.linalg.norm(vb)
            sims.append(float(va @ vb / denom) if denom else 0.0)
    return 1.0 - (sum(sims) / len(sims))


def novelty(recommended: Sequence, popularity_prob: Mapping, k: int) -> float:
    """平均自信息 -log2(p(item)); 越高越冷门/越新颖。"""
    vals = [
        -math.log2(popularity_prob[it])
        for it in recommended[:k]
        if popularity_prob.get(it, 0.0) > 0
    ]
    return sum(vals) / len(vals) if vals else 0.0


def category_entropy(recommended: Sequence, item_categories: Mapping, k: int) -> float:
    """top-k 列表里 category 分布的 Shannon 熵; 越高越分散。"""
    counts: dict = {}
    for it in recommended[:k]:
        for c in item_categories.get(it, []):
            counts[c] = counts.get(c, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


# ---------------------------------------------------------------------------
# 4. Recommender protocol + baselines
# ---------------------------------------------------------------------------
class Recommender(Protocol):
    name: str
    def recommend(self, user_key: str, k: int) -> list:
        """返回该用户的有序 location_id 列表。"""
        ...


@dataclass
class PopularityBaseline:
    """按 distinct-user 热度排序的 top-N (论文强调用 distinct users 而非 raw clicks)。"""
    ranked_items: list                       # location_ids, 热度降序
    name: str = "popularity"
    def recommend(self, user_key, k):
        return self.ranked_items[:k]


@dataclass
class RandomWithinInterestBaseline:
    """在用户兴趣 category 的候选池里随机抽 (论文 baseline 2)。"""
    user_interest_items: Mapping[str, list]  # user -> 兴趣类目下的候选 items
    fallback: list = field(default_factory=list)
    seed: int = 42
    name: str = "random_interest"
    def recommend(self, user_key, k):
        pool = list(self.user_interest_items.get(user_key) or self.fallback)
        if not pool:
            return []
        offset = zlib.crc32(str(user_key).encode())   # 确定性, 跨 run 可复现
        rng = np.random.default_rng(self.seed + offset)
        idx = rng.permutation(len(pool))[:k]
        return [pool[i] for i in idx]


@dataclass
class FunctionRecommender:
    """
    通用适配器: 把任意打分函数包成 recommender。
    用来做 content-only / collab-only 这类 ablation —— 你只要传入
    一个 score_fn(user_key) -> {location_id: score} 即可。
    """
    score_fn: Callable[[str], Mapping]
    name: str = "custom"
    exclude: Optional[Callable[[str], set]] = None   # 例: 排除已交互的 location
    def recommend(self, user_key, k):
        scores = self.score_fn(user_key)
        drop = self.exclude(user_key) if self.exclude else set()
        ranked = sorted(
            (it for it in scores if it not in drop),
            key=lambda it: scores[it], reverse=True,
        )
        return ranked[:k]


# ---------------------------------------------------------------------------
# 5. Evaluation runner
# ---------------------------------------------------------------------------
def evaluate_model(
    model,
    test_positives: dict,
    k_values: Sequence[int] = (5, 10, 20),
    item_vectors: Optional[Mapping] = None,
    popularity_prob: Optional[Mapping] = None,
    item_categories: Optional[Mapping] = None,
    catalog_size: Optional[int] = None,
) -> dict:
    """对单个 model 在 test_positives 上算所有指标, 返回各指标在用户间的均值。"""
    rows, all_lists = [], []
    max_k = max(k_values)

    for user, relevant in test_positives.items():
        reco = model.recommend(user, max_k)
        all_lists.append(reco)
        row = {}
        for k in k_values:
            row[f"P@{k}"]    = precision_at_k(reco, relevant, k)
            row[f"R@{k}"]    = recall_at_k(reco, relevant, k)
            row[f"HR@{k}"]   = hit_rate_at_k(reco, relevant, k)
            row[f"NDCG@{k}"] = ndcg_at_k(reco, relevant, k)
            row[f"MAP@{k}"]  = average_precision_at_k(reco, relevant, k)
            if item_vectors is not None:
                row[f"ILD@{k}"]    = intra_list_diversity(reco, item_vectors, k)
            if popularity_prob is not None:
                row[f"Nov@{k}"]    = novelty(reco, popularity_prob, k)
            if item_categories is not None:
                row[f"CatEnt@{k}"] = category_entropy(reco, item_categories, k)
        rows.append(row)

    metrics = pd.DataFrame(rows).mean().to_dict() if rows else {}
    if catalog_size:
        for k in k_values:
            metrics[f"Coverage@{k}"] = catalog_coverage(all_lists, catalog_size, k)
    metrics["n_users_evaluated"] = len(test_positives)
    return metrics


def compare_models(models: Sequence, test_positives: dict, **kwargs) -> pd.DataFrame:
    """把多个 model 跑一遍, 返回一张 模型 × 指标 的对比表 (DataFrame)。"""
    out = {m.name: evaluate_model(m, test_positives, **kwargs) for m in models}
    return pd.DataFrame(out).T


if __name__ == "__main__":
    # --- 最小自检 (smoke test), 不依赖你们的数据就能跑 ---
    demo = pd.DataFrame({
        "user_key":        ["u1", "u1", "u1", "u2", "u2", "u2", "u2"],
        "location_id":     ["a",  "b",  "c",  "a",  "d",  "e",  "f"],
        "event_timestamp": [1, 2, 3, 1, 2, 3, 4],
    })
    train, test = leave_last_n_split(demo, n_holdout=1, min_history=3)
    pop = PopularityBaseline(ranked_items=["a", "b", "c", "d", "e", "f"])
    print("test_positives:", test)
    print(compare_models([pop], test, k_values=(3,), catalog_size=6))
