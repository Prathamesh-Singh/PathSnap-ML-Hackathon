"""Step 4: Bayesian hyperparameter optimisation (Optuna TPE) for the chosen model.

    python experiments/bayes_opt.py --model LightGBM --trials 40

Objective: maximise mean observation-level Top-1 accuracy over 5-fold
GroupKFold on the 80 training trips (groups = trips). Student-testing trips
are only used afterwards to compare the tuned and untuned model.
Writes results/bayes_opt_trials.csv, results/bayes_opt_summary.json and the
refitted model to artifacts/final_model.joblib.
"""
import argparse
import os
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from common import ARTIFACTS, CACHE, RESULTS, SEED, load_table, save_json, top1_accuracy
from pathsnap.candidates import FEATURES
from train_models import get_models, scores, mean_continuity


def make_lightgbm(trial):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 900, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 300, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        random_state=SEED, verbose=-1,
    )


def make_hgb(trial):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_iter=trial.suggest_int("max_iter", 100, 800, step=50),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 300, log=True),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        random_state=SEED,
    )


def make_rf(trial):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50, log=True),
        max_features=trial.suggest_float("max_features", 0.2, 1.0),
        max_depth=trial.suggest_int("max_depth", 8, 40),
        n_jobs=-1, random_state=SEED,
    )


SPACES = {"LightGBM": make_lightgbm, "HistGradientBoosting": make_hgb, "RandomForest": make_rf}


def cv_top1(model, train, X, y, folds, trial=None):
    vals, oof = [], np.zeros(len(y))
    for k, (a, b) in enumerate(folds):
        m = clone(model).fit(X[a], y[a])
        oof[b] = m.predict_proba(X[b])[:, 1]
        vals.append(top1_accuracy(train.iloc[b], oof[b]))
        if trial is not None:
            trial.report(float(np.mean(vals)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(vals)), oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LightGBM", choices=list(SPACES))
    ap.add_argument("--trials", type=int, default=40)
    a = ap.parse_args()

    train, test = load_table("train"), load_table("student_test")
    X, y, g = train[FEATURES].to_numpy(), train["y"].to_numpy(), train["trip"].to_numpy()
    Xte = test[FEATURES].to_numpy()
    folds = list(GroupKFold(n_splits=5).split(X, y, g))

    def objective(trial):
        t0 = time.time()
        score, _ = cv_top1(SPACES[a.model](trial), train, X, y, folds, trial)
        trial.set_user_attr("secs", time.time() - t0)
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    t0 = time.time()
    study.optimize(objective, n_trials=a.trials, show_progress_bar=False)
    search_secs = time.time() - t0
    trials = study.trials_dataframe()
    trials.to_csv(os.path.join(RESULTS, "bayes_opt_trials.csv"), index=False)

    # refit tuned + untuned, compare on identical folds / data
    best_model = SPACES[a.model](optuna.trial.FixedTrial(study.best_params))
    untuned = get_models()[a.model][1]
    summary = {"model": a.model, "objective": "mean 5-fold GroupKFold observation-level Top-1 accuracy",
               "n_trials": len(study.trials),
               "n_pruned": int((trials["state"] == "PRUNED").sum()),
               "search_secs": search_secs, "best_params": study.best_params,
               "best_cv_top1": study.best_value}
    for tag, model in (("untuned", untuned), ("tuned", best_model)):
        cv, oof = cv_top1(model, train, X, y, folds)
        fold_s = [scores(train.iloc[b], oof[b]) for _, b in folds]
        t1 = time.time()
        m = clone(model).fit(X, y)
        fit_s = time.time() - t1
        t1 = time.time()
        pte = m.predict_proba(Xte)[:, 1]
        inf_s = time.time() - t1
        ts = scores(test, pte)
        summary[tag] = {
            "cv_top1": cv, "cv_f1_macro": float(np.mean([f["f1_macro"] for f in fold_s])),
            "cv_log_loss": float(np.mean([f["log_loss"] for f in fold_s])),
            **{f"test_{k}": v for k, v in ts.items()},
            "test_continuity": mean_continuity(test, pte, "student_test"),
            "fit_secs": fit_s, "infer_secs_test": inf_s,
        }
        if tag == "tuned":
            joblib.dump(m, os.path.join(ARTIFACTS, "final_model.joblib"))
            with open(os.path.join(ARTIFACTS, "final_model_name.txt"), "w") as f:
                f.write(f"{a.model} (Bayesian-tuned)\n")
            np.save(os.path.join(CACHE, "oof", "final_oof.npy"), oof)
            np.save(os.path.join(CACHE, "oof", "final_student_test.npy"), pte)
        print(tag, {k: round(v, 4) for k, v in summary[tag].items()}, flush=True)
    save_json(summary, os.path.join(RESULTS, "bayes_opt_summary.json"))
    print("best params", study.best_params)


if __name__ == "__main__":
    main()
