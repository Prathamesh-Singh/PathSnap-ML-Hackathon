"""Step 3: train and compare candidate classifiers.

    python experiments/train_models.py                      # all models
    python experiments/train_models.py --models KNN --member "Your Name"   # one model

Each model scores (GPS observation, candidate arc) pairs: probability that
the candidate is the reference arc for that observation.

* 5-fold GroupKFold on the 80 training trips (groups = trips, so no trip is
  split between fitting and validation)
* refit on all training trips, evaluate on the 10 student-testing trips
* out-of-fold probabilities are saved so the HMM decoder can be tuned on
  training trips without using in-sample (over-confident) scores
"""
import argparse
import glob
import json
import os
import platform
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (ARTIFACTS, CACHE, RESULTS, SEED, argmax_arcs, continuity,
                    load_table, save_json, top1_accuracy, SPLITS)
from pathsnap.candidates import FEATURES
from pathsnap.io_utils import load_case


def get_models():
    models = {
        "LogisticRegression": ("Linear / probabilistic",
            make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
            "StandardScaler, C=1.0, L2, lbfgs"),
        "GaussianNB": ("Linear / probabilistic", GaussianNB(), "var_smoothing=1e-9"),
        "KNN": ("Instance-based",
            make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=31, weights="distance", n_jobs=-1)),
            "StandardScaler, k=31, distance-weighted, euclidean"),
        "RandomForest": ("Ensemble (bagging)",
            RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
            "200 trees, min_samples_leaf=5, max_features=sqrt"),
        "HistGradientBoosting": ("Boosting",
            HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300, max_leaf_nodes=31, random_state=SEED),
            "lr=0.1, 300 iters, 31 leaves"),
        "MLP": ("Neural network",
            make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                          batch_size=512, learning_rate_init=1e-3, max_iter=200, early_stopping=True,
                          n_iter_no_change=10, random_state=SEED)),
            "StandardScaler, layers (64,32), relu, adam lr=1e-3, early stopping"),
    }
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = ("Boosting",
            LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1),
            "400 trees, lr=0.05, 63 leaves, bagging 0.8, feature fraction 0.8")
    except ImportError:
        print("lightgbm not installed - skipping")
    return models


def scores(df, prob):
    y = df["y"].to_numpy()
    pred = (prob >= 0.5).astype(int)
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro"),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "top1": top1_accuracy(df, prob),
    }


def mean_continuity(df, prob, split):
    seqs = argmax_arcs(df, prob)
    vals = []
    for trip, seq in seqs.items():
        arcs = load_case(os.path.join(SPLITS[split], trip))["arcs"]
        vals.append(continuity(seq, arcs[:, 0], arcs[:, 1]))
    return float(np.mean(vals))


def library_versions():
    """Library versions affect floating-point results (especially the MLP), so record them."""
    import sys
    import scipy
    import sklearn
    v = {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
         "scikit-learn": sklearn.__version__, "pandas": pd.__version__}
    try:
        import lightgbm
        v["lightgbm"] = lightgbm.__version__
    except ImportError:
        pass
    return v


def collect():
    """Merge every results/models/<model>.json (one per trained model, possibly
    produced by different team members) into the comparison tables."""
    folder = os.path.join(RESULTS, "models")
    rows, histories = [], {}
    for f in sorted(glob.glob(os.path.join(folder, "*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        if r.get("history"):
            histories[r["model"]] = r["history"]
        rows.append({k: v for k, v in r.items() if k != "history"})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "model_comparison.csv"), index=False)
    save_json(histories, os.path.join(RESULTS, "training_histories.json"))
    return df


def main():
    all_models = get_models()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(all_models), choices=list(all_models),
                    help="which models to train (default: all)")
    ap.add_argument("--member", default="", help="team member who ran this (recorded in the results)")
    ap.add_argument("--collect", action="store_true", help="only merge results/models/*.json")
    a = ap.parse_args()
    if a.collect:
        print(collect()[["model", "member", "cv_f1_macro", "cv_top1", "test_top1"]].round(4).to_string())
        return

    train, test = load_table("train"), load_table("student_test")
    Xtr, ytr, gtr = train[FEATURES].to_numpy(), train["y"].to_numpy(), train["trip"].to_numpy()
    Xte = test[FEATURES].to_numpy()
    folds = list(GroupKFold(n_splits=5).split(Xtr, ytr, gtr))

    for d in (os.path.join(ARTIFACTS, "models"), os.path.join(CACHE, "oof"), os.path.join(RESULTS, "models")):
        os.makedirs(d, exist_ok=True)
    for name in a.models:
        family, model, config = all_models[name]
        print(f"=== {name}", flush=True)
        oof = np.zeros(len(train))
        fold_scores, fit_t = [], []
        for k, (tr_idx, va_idx) in enumerate(folds):
            m = clone(model)
            t0 = time.time()
            m.fit(Xtr[tr_idx], ytr[tr_idx])
            fit_t.append(time.time() - t0)
            oof[va_idx] = m.predict_proba(Xtr[va_idx])[:, 1]
            fold_scores.append(scores(train.iloc[va_idx], oof[va_idx]))
            print(f"  fold {k}: f1={fold_scores[-1]['f1_macro']:.4f} top1={fold_scores[-1]['top1']:.4f}"
                  f"  ({fit_t[-1]:.1f}s)", flush=True)
        np.save(os.path.join(CACHE, "oof", f"{name}.npy"), oof)

        m = clone(model)
        t0 = time.time()
        m.fit(Xtr, ytr)
        fit_all = time.time() - t0
        t0 = time.time()
        pte = m.predict_proba(Xte)[:, 1]
        infer = time.time() - t0
        np.save(os.path.join(CACHE, "oof", f"{name}_student_test.npy"), pte)
        joblib.dump(m, os.path.join(ARTIFACTS, "models", f"{name}.joblib"))
        # in-sample scores of the final fit: compare with CV to judge over/underfitting
        train_s = scores(train, m.predict_proba(Xtr)[:, 1])

        ts = scores(test, pte)
        cm = confusion_matrix(test["y"], (pte >= 0.5).astype(int)).tolist()
        cv = pd.DataFrame(fold_scores)
        row = {
            "model": name, "family": family, "config": config,
            "params": json.dumps({k: str(v) for k, v in (m[-1] if hasattr(m, "steps") else m).get_params().items()}),
            "member": a.member, "machine": platform.node(), "cpu": platform.processor(),
            "seed": SEED, "run_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "versions": json.dumps(library_versions()),
            "cv_f1_macro": cv.f1_macro.mean(), "cv_f1_macro_std": cv.f1_macro.std(),
            "cv_top1": cv.top1.mean(), "cv_log_loss": cv.log_loss.mean(),
            "cv_accuracy": cv.accuracy.mean(), "cv_precision_macro": cv.precision_macro.mean(),
            "cv_recall_macro": cv.recall_macro.mean(),
            "cv_fit_secs_per_fold": float(np.mean(fit_t)),
            "train_insample_f1_macro": train_s["f1_macro"], "train_insample_top1": train_s["top1"],
            "train_insample_log_loss": train_s["log_loss"],
            **{f"test_{k}": v for k, v in ts.items()},
            "test_continuity": mean_continuity(test, pte, "student_test"),
            "fit_secs_full": fit_all, "infer_secs_test": infer,
            "infer_ms_per_1k_rows": 1000 * infer / len(test) * 1000,
            "test_confusion_matrix": json.dumps(cm),
        }
        for k, fs in enumerate(fold_scores):
            row[f"fold{k}_f1_macro"] = fs["f1_macro"]
            row[f"fold{k}_top1"] = fs["top1"]
        # epoch-wise history only exists for models that train in epochs (the MLP)
        est = m[-1] if hasattr(m, "steps") else m
        if hasattr(est, "loss_curve_"):
            row["history"] = {"train_loss": list(map(float, est.loss_curve_)),
                              "val_accuracy": list(map(float, getattr(est, "validation_scores_", None) or []))}
        save_json(row, os.path.join(RESULTS, "models", f"{name}.json"))

        print(f"--- {name} summary")
        for k in ("train_insample_f1_macro", "train_insample_top1", "cv_f1_macro", "cv_top1", "cv_log_loss", "test_accuracy", "test_precision_macro",
                  "test_recall_macro", "test_f1_macro", "test_log_loss", "test_top1", "test_continuity",
                  "fit_secs_full", "infer_secs_test"):
            print(f"  {k:22s} {row[k]:.4f}")
        print(f"  confusion matrix (rows=true other/ref, cols=pred) {cm}", flush=True)

    df = collect()
    print()
    print("All models collected so far:")
    print(df[["model", "member", "cv_f1_macro", "cv_top1", "cv_log_loss", "test_f1_macro", "test_top1",
              "test_log_loss", "test_continuity", "infer_secs_test"]].round(4).to_string())


if __name__ == "__main__":
    main()
