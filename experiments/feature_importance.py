"""Which features matter? Permutation importance on held-out GroupKFold folds.

    python experiments/feature_importance.py

For each of the 5 trip-grouped folds: fit HistGradientBoosting (default
configuration) on the other 4 folds, then shuffle one feature at a time in the
held-out fold and measure how much Top-1 accuracy and log loss get worse.
Only training trips are used (no student-test data), so this can inform
feature decisions without leakage.
Writes results/feature_importance.csv and results/figures/feature_importance.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

from common import FIGS, RESULTS, SEED, load_table, top1_accuracy
from pathsnap.candidates import FEATURES
from train_models import get_models

REPEATS = 2


def main():
    train = load_table("train")
    X, y, g = train[FEATURES].to_numpy(), train["y"].to_numpy(), train["trip"].to_numpy()
    model = get_models()["HistGradientBoosting"][1]
    rng = np.random.default_rng(SEED)
    rows = []
    for k, (a, b) in enumerate(GroupKFold(n_splits=5).split(X, y, g)):
        m = clone(model).fit(X[a], y[a])
        held = train.iloc[b]
        Xb = X[b].copy()
        p = m.predict_proba(Xb)[:, 1]
        base_top1, base_ll = top1_accuracy(held, p), log_loss(y[b], p)
        for j, name in enumerate(FEATURES):
            for r in range(REPEATS):
                Xp = Xb.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                pp = m.predict_proba(Xp)[:, 1]
                rows.append({"fold": k, "repeat": r, "feature": name,
                             "top1_drop": base_top1 - top1_accuracy(held, pp),
                             "logloss_increase": log_loss(y[b], pp) - base_ll})
        print(f"fold {k} done (baseline Top-1 {base_top1:.4f})", flush=True)

    df = pd.DataFrame(rows)
    imp = (df.groupby("feature")[["top1_drop", "logloss_increase"]]
             .agg(["mean", "std"]).sort_values(("top1_drop", "mean"), ascending=False))
    imp.columns = ["top1_drop_mean", "top1_drop_std", "logloss_increase_mean", "logloss_increase_std"]
    imp.to_csv(os.path.join(RESULTS, "feature_importance.csv"))
    print(imp.round(4).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    order = imp.index[::-1]
    ax[0].barh(order, imp.loc[order, "top1_drop_mean"], xerr=imp.loc[order, "top1_drop_std"], color="#2a9d8f")
    ax[0].set_xlabel("drop in Top-1 accuracy when shuffled")
    ax[1].barh(order, imp.loc[order, "logloss_increase_mean"], xerr=imp.loc[order, "logloss_increase_std"],
               color="#e76f51")
    ax[1].set_xlabel("increase in log loss when shuffled")
    fig.suptitle("Permutation importance (HistGradientBoosting, held-out GroupKFold folds, training trips)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "feature_importance.png"), dpi=130)
    print("saved feature_importance.png")


if __name__ == "__main__":
    main()
