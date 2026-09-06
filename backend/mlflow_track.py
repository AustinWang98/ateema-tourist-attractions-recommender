"""
ChicagoDoes 实验追踪 (MLflow) —— 把权重调优 + CV + MMR 的结果记成可对比的 runs。

记三条对照 run, 在 MLflow UI 里并排就是完整故事:
  1. current_paper        —— 论文原权重 (baseline)
  2. tuned_robust         —— CV 验证的稳健权重 (popularity↑/content↑/session↓)
  3. tuned_robust_mmr_on  —— 同样权重但开 MMR (展示 relevance↔diversity 权衡)

每条 run 记录: 六个权重 + λ (params), NDCG/Recall/Precision/HitRate/diversity (metrics),
并把 best_weights.json / 各 CSV 作为 artifact 附上。tuned_robust 额外记 CV 的诚实指标
(留出集 NDCG、提升幅度、赢几折、过拟合 gap), 让这条 run 自带 "已验证" 的证据。

前置: pip install mlflow

跑法 (项目根目录):
    python -m backend.mlflow_track
查看结果:
    mlflow ui            # 然后浏览器打开 http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import mlflow
except ImportError:
    print("❌ 没装 mlflow。先运行:  pip install mlflow  然后重试。")
    sys.exit(1)

from .data_loader import _normalise_events
from .recommender import ContentRecommender, WEIGHTS_RETURNING, MMR_LAMBDA
from .evaluation import leave_last_n_split
from .evaluation_runner import build_train_frames
from .weight_search import (
    COMPONENTS, collect_candidates, mean_ndcg, mean_full, mmr_verdict,
)
from .weight_search_cv import search_best_weights


def run_cross_validation(cache, current_w, k, folds, n_samples, seed):
    """复现 K-fold CV, 返回 cv_mean 权重 + 留出集汇总指标。"""
    n = len(cache)
    rng = np.random.default_rng(seed)
    fold_ids = np.array_split(rng.permutation(n), folds)
    best_ws, heldout_best, heldout_cur = [], [], []
    for fi, val_idx in enumerate(fold_ids):
        val = [cache[i] for i in val_idx]
        search = [cache[i] for i in range(n) if i not in set(val_idx)]
        bw, _ = search_best_weights(search, k, n_samples, np.random.default_rng(seed + fi + 1))
        best_ws.append(bw)
        heldout_best.append(mean_ndcg(val, bw, k))
        heldout_cur.append(mean_ndcg(val, current_w, k))
    best_ws = np.array(best_ws)
    return {
        "cv_mean_w": best_ws.mean(axis=0),
        "cv_std_w": best_ws.std(axis=0),
        "heldout_best": float(np.mean(heldout_best)),
        "heldout_current": float(np.mean(heldout_cur)),
        "folds_won": int(np.sum(np.array(heldout_best) > np.array(heldout_cur))),
        "folds": folds,
    }


def log_config_run(run_name, cache, rec, weights, k_values, lam, mmr_on,
                   extra_params=None, extra_metrics=None, tags=None, artifacts=None):
    """把一组权重配置评估并记成一条 MLflow run。"""
    with mlflow.start_run(run_name=run_name):
        # params: 六个权重
        params = {f"w_{c}": round(float(weights[i]), 4) for i, c in enumerate(COMPONENTS)}
        params.update({
            "mmr_enabled": mmr_on,
            "mmr_lambda": lam if mmr_on else "n/a",
        })
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)

        # metrics: 一次 mmr_verdict 拿到 关/开 MMR 的 ndcg + diversity
        verdict = mmr_verdict(cache, rec, np.asarray(weights, float), max(k_values), lam=lam)
        metrics = {}
        if mmr_on:
            metrics[f"ndcg@{max(k_values)}"] = verdict["ndcg_with_mmr"]
            metrics[f"diversity@{max(k_values)}"] = verdict["diversity_with_mmr"]
        else:
            for kk in k_values:
                for m, v in mean_full(cache, np.asarray(weights, float), kk).items():
                    metrics[m] = float(v)
            metrics[f"diversity@{max(k_values)}"] = verdict["diversity_no_mmr"]
        if extra_metrics:
            metrics.update(extra_metrics)

        # MLflow 的 metric 名不允许 '@', 统一换成 '_at_' (关键修复)
        metrics = {key.replace("@", "_at_"): float(val) for key, val in metrics.items()}
        mlflow.log_metrics(metrics)

        if tags:
            mlflow.set_tags(tags)
        for a in (artifacts or []):
            if Path(a).exists():
                mlflow.log_artifact(a)
        ndcg_key = f"ndcg_at_{max(k_values)}" if mmr_on else f"NDCG_at_{max(k_values)}"
        print(f"  ✓ logged run '{run_name}' ({'mmr' if mmr_on else 'no-mmr'}) "
              f"{ndcg_key}={metrics.get(ndcg_key, float('nan')):.4f}")


def main():
    ap = argparse.ArgumentParser(description="MLflow tracking for ChicagoDoes weight tuning")
    ap.add_argument("--events", default="data/private/events.csv")
    ap.add_argument("--dim",    default="data/location_dim.csv")
    ap.add_argument("--geo",    default="data/locations_geo.csv")
    ap.add_argument("--n-holdout", type=int, default=1)
    ap.add_argument("--min-history", type=int, default=3)
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--experiment", default="chicagodoes-weight-tuning")
    args = ap.parse_args()
    k_values = tuple(args.k)
    head_k = max(k_values)

    # --- 一次性: 数据 + split + recommender + 缓存 ---
    print("[1/3] 读 events / split / 构建 recommender / 缓存候选分数 ...")
    raw = pd.read_csv(args.events, low_memory=False)
    events = _normalise_events(raw)
    events["user_key"] = events["user_key"].astype(str).str.strip()
    events["location_id"] = events["location_id"].astype(str).str.strip()
    events = events[(events["user_key"] != "") & (events["location_id"] != "")]
    window = f"{events['event_time'].min()} → {events['event_time'].max()}"
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
    current_w = np.array([WEIGHTS_RETURNING[c] for c in COMPONENTS], dtype=float)
    print(f"      缓存用户 {n_users}")

    # --- CV 推导稳健权重 + 诚实指标 ---
    print(f"[2/3] {args.folds}-fold CV 推导稳健权重 ...")
    cv = run_cross_validation(cache, current_w, head_k, args.folds, args.n_samples, args.seed)
    robust_w = cv["cv_mean_w"]
    # in-sample best (全量搜) 用于过拟合 gap 参考
    insample_best_w, insample_best = search_best_weights(
        cache, head_k, args.n_samples, np.random.default_rng(args.seed + 99)
    )
    overfit_gap = insample_best - cv["heldout_best"]
    cv_gain = (cv["heldout_best"] - cv["heldout_current"]) / max(cv["heldout_current"], 1e-9) * 100
    print(f"      留出集 NDCG@{head_k}: 稳健={cv['heldout_best']:.4f} vs 当前={cv['heldout_current']:.4f} "
          f"({cv_gain:+.1f}%, 赢 {cv['folds_won']}/{cv['folds']} 折)")

    # --- 写入 best_weights 的稳健版, 供生产/MLflow artifact ---
    out_dir = Path("data")
    robust_payload = {
        "robust_weights_returning": {c: float(robust_w[i]) for i, c in enumerate(COMPONENTS)},
        "cv_std": {c: float(cv["cv_std_w"][i]) for i, c in enumerate(COMPONENTS)},
        f"cv_heldout_ndcg@{head_k}": cv["heldout_best"],
        f"cv_heldout_ndcg@{head_k}_current": cv["heldout_current"],
        "cv_gain_pct": cv_gain,
        "cv_folds_won": cv["folds_won"],
        "cv_folds": cv["folds"],
        "overfit_gap": overfit_gap,
        "n_users": n_users,
        "data_window": window,
    }
    robust_path = out_dir / "robust_weights.json"
    import json
    with open(robust_path, "w") as f:
        json.dump(robust_payload, f, indent=2, ensure_ascii=False)

    # --- 记录 MLflow runs ---
    print(f"[3/3] 记录 MLflow runs (experiment='{args.experiment}') ...")
    mlflow.set_tracking_uri(f"file:{(Path('mlruns')).resolve()}")
    mlflow.set_experiment(args.experiment)

    common_params = {
        "eval_k": head_k, "n_holdout": args.n_holdout,
        "min_history": args.min_history, "n_users_eval": n_users,
        "cv_folds": args.folds,
    }
    common_tags = {"data_window": window, "split": f"leave-last-{args.n_holdout}"}
    artifacts = [str(robust_path)]
    for extra in ("data/best_weights.json", "data/weight_search_results.csv",
                  "data/weight_cv_results.csv", "data/offline_eval_results.csv"):
        if Path(extra).exists():
            artifacts.append(extra)

    # run 1: 论文原权重 (baseline, 未在用户上调过, full-set eval 即诚实)
    log_config_run(
        "current_paper", cache, rec, current_w, k_values, MMR_LAMBDA, mmr_on=False,
        extra_params={**common_params, "source": "paper"},
        tags={**common_tags, "validated": "n/a"},
        artifacts=artifacts,
    )

    # run 2: CV 稳健权重 (附带诚实的 CV 留出指标)
    log_config_run(
        "tuned_robust", cache, rec, robust_w, k_values, MMR_LAMBDA, mmr_on=False,
        extra_params={**common_params, "source": "cv_validated"},
        extra_metrics={
            f"cv_heldout_ndcg@{head_k}": cv["heldout_best"],
            f"cv_heldout_ndcg@{head_k}_baseline": cv["heldout_current"],
            "cv_gain_pct": cv_gain,
            "cv_folds_won": float(cv["folds_won"]),
            "overfit_gap": overfit_gap,
        },
        tags={**common_tags, "validated": "true"},
        artifacts=artifacts,
    )

    # run 3: 稳健权重 + 开 MMR (展示 diversity 权衡)
    log_config_run(
        "tuned_robust_mmr_on", cache, rec, robust_w, k_values, MMR_LAMBDA, mmr_on=True,
        extra_params={**common_params, "source": "cv_validated"},
        tags={**common_tags, "validated": "true"},
        artifacts=artifacts,
    )

    print("\n" + "=" * 70)
    print("✅ 三条 run 已记录。查看:")
    print("   mlflow ui    然后浏览器打开 http://127.0.0.1:5000")
    print(f"   (实验名: {args.experiment})")
    print("=" * 70)
    print(f"稳健权重已存: {robust_path.resolve()}")


if __name__ == "__main__":
    main()
