"""Bias / variance (under- / overfitting) table for all models.

    python experiments/bias_variance.py

For every saved model (each fitted on all 80 training trips) it measures the in-sample
score on the training trips and puts it next to the 5-fold GroupKFold CV score and the
student-test score already in results/models/*.json.

    bias     ~ training error         = 1 - training Top-1   (high -> underfitting)
    variance ~ train-to-CV drop       = training Top-1 - CV Top-1   (high -> overfitting)

KNN is scored in-sample on 16 whole training trips only (a neighbour search over all
344k rows is slow and the answer is the same: every row finds itself at distance 0).
Writes results/bias_variance.csv. Evaluator-test numbers are added for information only.
"""
import json
import os

import joblib
import pandas as pd

from common import RESULTS, ROOT, load_table
from pathsnap.candidates import FEATURES
from train_models import scores

MODELS = ["LogisticRegression", "GaussianNB", "KNN", "RandomForest", "HistGradientBoosting", "MLP", "LightGBM"]


def main():
    train = load_table("train")
    knn_trips = sorted(train.trip.unique())[:16]
    ev_path = os.path.join(RESULTS, "evaluator_test", "classifiers.csv")
    ev = pd.read_csv(ev_path).set_index("model") if os.path.exists(ev_path) else None
    bayes = json.load(open(os.path.join(RESULTS, "bayes_opt_summary.json")))

    rows = []
    for name in MODELS + ["HistGradientBoosting (Bayesian-tuned, final)"]:
        if name.endswith("final)"):
            path = os.path.join(ROOT, "artifacts", "final_model.joblib")
            info = {"cv_top1": bayes["tuned"]["cv_top1"], "cv_log_loss": bayes["tuned"]["cv_log_loss"],
                    "test_top1": bayes["tuned"]["test_top1"], "test_log_loss": bayes["tuned"]["test_log_loss"]}
        else:
            path = os.path.join(ROOT, "artifacts", "models", f"{name}.joblib")
            info = json.load(open(os.path.join(RESULTS, "models", f"{name}.json")))
        data = train[train.trip.isin(knn_trips)] if name == "KNN" else train
        p = joblib.load(path).predict_proba(data[FEATURES].to_numpy())[:, 1]
        s = scores(data, p)
        r = {"model": name,
             "train_top1": s["top1"], "cv_top1": info["cv_top1"], "student_test_top1": info["test_top1"],
             "train_log_loss": s["log_loss"], "cv_log_loss": info["cv_log_loss"],
             "student_test_log_loss": info["test_log_loss"]}
        r["bias_train_error"] = 1 - r["train_top1"]
        r["variance_train_minus_cv"] = r["train_top1"] - r["cv_top1"]
        r["cv_minus_student_test"] = r["cv_top1"] - r["student_test_top1"]
        if ev is not None and name in ev.index:
            r["evaluator_top1_post_selection"] = ev.loc[name, "top1"]
        rows.append(r)
        print(f"{name:45s} train {r['train_top1']:.4f}  CV {r['cv_top1']:.4f}  test {r['student_test_top1']:.4f}"
              f"  bias {r['bias_train_error']:.4f}  variance {r['variance_train_minus_cv']:+.4f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "bias_variance.csv"), index=False)


if __name__ == "__main__":
    main()
