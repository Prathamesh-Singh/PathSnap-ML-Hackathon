"""Validation curve for one model: vary its most important hyperparameter and
compare in-sample (training) score with 5-fold GroupKFold CV score.

    python experiments/sweep_model.py --model KNN --member "Your Name"

A training score far above the CV score = overfitting; both low = underfitting.
Writes results/sweeps/<model>.csv and results/figures/sweep_<model>.png
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from common import FIGS, RESULTS, load_table, top1_accuracy
from pathsnap.candidates import FEATURES
from train_models import get_models, scores

# hyperparameter to sweep for each model, and the values to try
SWEEPS = {
    "LogisticRegression": ("logisticregression__C", [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]),
    "GaussianNB": ("var_smoothing", [1e-12, 1e-9, 1e-6, 1e-3, 1e-1]),
    "KNN": ("kneighborsclassifier__n_neighbors", [1, 3, 7, 15, 31, 63, 127]),
    "RandomForest": ("min_samples_leaf", [1, 2, 5, 10, 25, 50, 100]),
    "HistGradientBoosting": ("max_iter", [25, 50, 100, 200, 400, 800]),
    "MLP": ("mlpclassifier__alpha", [1e-6, 1e-4, 1e-2, 1e-1, 1.0]),
    "LightGBM": ("num_leaves", [4, 8, 16, 31, 63, 127, 255]),
}


def train_subset(g, tr_idx):
    """Whole trips from the fold's own training part (every observation keeps its full candidate set)."""
    fold_trips = np.unique(g[tr_idx])[:16]
    return tr_idx[np.isin(g[tr_idx], fold_trips)]


def knn_probs(base, X_fit, y_fit, X_query, ks):
    """Distance-weighted KNN probabilities for every k in ks from ONE neighbour search.

    The k nearest neighbours are the first k of the max(ks) nearest, so this gives the same
    numbers as fitting KNeighborsClassifier(n_neighbors=k, weights="distance") once per k,
    including scikit-learn's rule for zero distances (only the exact matches vote), but
    searches the 275k training rows once per fold instead of seven times.
    """
    from sklearn.neighbors import NearestNeighbors
    scaler = clone(base.named_steps["standardscaler"]).fit(X_fit)
    nn = NearestNeighbors(n_neighbors=max(ks), n_jobs=-1).fit(scaler.transform(X_fit))
    dist, idx = nn.kneighbors(scaler.transform(X_query))
    lab = y_fit[idx].astype(float)
    out = {}
    for k in ks:
        d, lk = dist[:, :k], lab[:, :k]
        with np.errstate(divide="ignore"):
            w = 1.0 / d
        zero = d == 0
        has0 = zero.any(axis=1)
        w[has0] = zero[has0]
        out[k] = (w * lk).sum(axis=1) / w.sum(axis=1)
    return out


def knn_sweep(a, train, X, y, g, folds, base, param, values):
    ks = [int(v) for v in values]
    acc = {k: {"cv_top1": [], "cv_f1": [], "tr_top1": [], "tr_f1": [], "secs": 0.0} for k in ks}
    for i, (tr_idx, va_idx) in enumerate(folds):
        t0 = time.time()
        sub = train_subset(g, tr_idx)
        pv = knn_probs(base, X[tr_idx], y[tr_idx], X[va_idx], ks)
        ps = knn_probs(base, X[tr_idx], y[tr_idx], X[sub], ks)
        el = time.time() - t0
        for k in ks:
            s, st = scores(train.iloc[va_idx], pv[k]), scores(train.iloc[sub], ps[k])
            acc[k]["cv_top1"].append(s["top1"]); acc[k]["cv_f1"].append(s["f1_macro"])
            acc[k]["tr_top1"].append(st["top1"]); acc[k]["tr_f1"].append(st["f1_macro"])
            acc[k]["secs"] += el / len(ks)          # one shared search per fold, split evenly
        print(f"fold {i + 1}/5 done ({el:.0f}s)", flush=True)
    rows = []
    for k in ks:
        r = acc[k]
        rows.append({"model": a.model, "param": param, "value": k, "member": a.member,
                     "cv_top1": np.mean(r["cv_top1"]), "cv_top1_std": np.std(r["cv_top1"]),
                     "cv_f1_macro": np.mean(r["cv_f1"]), "train_top1": np.mean(r["tr_top1"]),
                     "train_f1_macro": np.mean(r["tr_f1"]), "secs": r["secs"]})
        print(f"n_neighbors={k}  CV Top-1={rows[-1]['cv_top1']:.4f} (+/-{rows[-1]['cv_top1_std']:.4f})  "
              f"CV F1={rows[-1]['cv_f1_macro']:.4f}  train Top-1={rows[-1]['train_top1']:.4f}  "
              f"train F1={rows[-1]['train_f1_macro']:.4f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SWEEPS))
    ap.add_argument("--member", default="")
    ap.add_argument("--subsample-trips", type=int, default=0,
                    help="use only N training trips (faster on slow laptops); 0 = all 80")
    a = ap.parse_args()

    train = load_table("train")
    if a.subsample_trips:
        keep = sorted(train.trip.unique())[: a.subsample_trips]
        train = train[train.trip.isin(keep)].reset_index(drop=True)
    X, y, g = train[FEATURES].to_numpy(), train["y"].to_numpy(), train["trip"].to_numpy()
    folds = list(GroupKFold(n_splits=5).split(X, y, g))
    param, values = SWEEPS[a.model]
    base = get_models()[a.model][1]

    if a.model == "KNN":
        rows = knn_sweep(a, train, X, y, g, folds, base, param, values)
    else:
        rows = []
    for v in (values if a.model != "KNN" else []):
        model = clone(base).set_params(**{param: v})
        t0 = time.time()
        cv_top1, cv_f1, tr_top1, tr_f1 = [], [], [], []
        for tr_idx, va_idx in folds:
            m = clone(model).fit(X[tr_idx], y[tr_idx])
            pv = m.predict_proba(X[va_idx])[:, 1]
            s = scores(train.iloc[va_idx], pv)
            cv_top1.append(s["top1"])
            cv_f1.append(s["f1_macro"])
            # training score on whole trips from the fold's own training part
            # (whole trips, so every observation keeps its full candidate set)
            fold_trips = np.unique(g[tr_idx])[:16]
            sub = tr_idx[np.isin(g[tr_idx], fold_trips)]
            ps = m.predict_proba(X[sub])[:, 1]
            st = scores(train.iloc[sub], ps)
            tr_top1.append(st["top1"])
            tr_f1.append(st["f1_macro"])
        rows.append({"model": a.model, "param": param, "value": v, "member": a.member,
                     "cv_top1": np.mean(cv_top1), "cv_top1_std": np.std(cv_top1),
                     "cv_f1_macro": np.mean(cv_f1), "train_top1": np.mean(tr_top1),
                     "train_f1_macro": np.mean(tr_f1), "secs": time.time() - t0})
        r = rows[-1]
        print(f"{param.split('__')[-1]}={v:g}  CV Top-1={r['cv_top1']:.4f} (+/-{r['cv_top1_std']:.4f})  "
              f"CV F1={r['cv_f1_macro']:.4f}  train Top-1={r['train_top1']:.4f}  train F1={r['train_f1_macro']:.4f}"
              f"  ({r['secs']:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(RESULTS, "sweeps"), exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "sweeps", f"{a.model}.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(values))
    ax.plot(x, df.train_f1_macro, "o-", label="training (in-sample) macro F1", color="#e76f51")
    ax.errorbar(x, df.cv_f1_macro, fmt="o-", label="5-fold GroupKFold CV macro F1", color="#2a9d8f")
    ax.plot(x, df.cv_top1, "s--", label="CV Top-1 accuracy", color="#457b9d")
    ax.set_xticks(x, [f"{v:g}" for v in values])
    ax.set_xlabel(param.split("__")[-1])
    ax.set_ylabel("score")
    ax.set_title(f"{a.model}: validation curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, f"sweep_{a.model}.png"), dpi=130)
    print("saved", os.path.join(FIGS, f"sweep_{a.model}.png"))
    best = df.loc[df.cv_top1.idxmax()]
    print(f"best {param} = {best.value} (CV Top-1 {best.cv_top1:.4f})")


if __name__ == "__main__":
    main()
