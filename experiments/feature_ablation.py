"""Feature selection by group ablation: retrain without a group of features and
compare 5-fold GroupKFold CV on the training trips (no student-test data).

    python experiments/feature_ablation.py
Writes results/feature_ablation.csv
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from common import RESULTS, load_table, top1_accuracy
from pathsnap.candidates import FEATURES
from train_models import get_models

GROUPS = {
    "all 18 features": [],
    "- zero-importance (dist_z, heading_ok, dt_prev, dist_rank)": ["dist_z", "heading_ok", "dt_prev", "dist_rank"],
    "- heading (heading_diff, heading_ok)": ["heading_diff", "heading_ok"],
    "- context (ctx_prev, ctx_next, ctx_mean)": ["ctx_prev", "ctx_next", "ctx_mean"],
    "- network (two_way, deg_out_end, deg_in_start, arc_len)": ["two_way", "deg_out_end", "deg_in_start", "arc_len"],
    "- motion (speed, turn_angle, dt_prev)": ["speed", "turn_angle", "dt_prev"],
    "- position on arc (t_along)": ["t_along"],
    "distance only (dist, dist_rank, dist_gap, dist_z, n_cands)":
        [f for f in FEATURES if f not in ("dist", "dist_rank", "dist_gap", "dist_z", "n_cands")],
}


def main():
    train = load_table("train")
    y, g = train["y"].to_numpy(), train["trip"].to_numpy()
    folds = list(GroupKFold(n_splits=5).split(train, y, g))
    model = get_models()["HistGradientBoosting"][1]
    rows = []
    for name, drop in GROUPS.items():
        cols = [f for f in FEATURES if f not in drop]
        X = train[cols].to_numpy()
        t0 = time.time()
        top1, ll = [], []
        for a, b in folds:
            p = clone(model).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
            top1.append(top1_accuracy(train.iloc[b], p))
            ll.append(log_loss(y[b], np.clip(p, 1e-6, 1 - 1e-6)))
        rows.append({"feature_set": name, "n_features": len(cols), "cv_top1": np.mean(top1),
                     "cv_top1_std": np.std(top1), "cv_log_loss": np.mean(ll), "secs": time.time() - t0})
        print(f"{name:62s} n={len(cols):2d}  CV Top-1={np.mean(top1):.4f} (+/-{np.std(top1):.4f})  "
              f"log loss={np.mean(ll):.4f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "feature_ablation.csv"), index=False)


if __name__ == "__main__":
    main()
