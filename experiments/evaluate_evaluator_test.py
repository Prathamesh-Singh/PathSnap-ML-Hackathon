"""Final, one-off evaluation on the evaluator-testing trips.

    python scripts/freeze_artifacts.py results/evaluator_test/frozen_artifacts.json   (first!)
    python experiments/evaluate_evaluator_test.py --trips-dir data/evaluator_test

Nothing here trains, tunes or selects anything: the saved models (fitted on the
80 training trips only) are loaded and scored, and the final pipeline
(tuned HistGradientBoosting + HMM/Viterbi) is run exactly as the API runs it.
Results go to results/evaluator_test/ and were not used for any decision.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
import pandas as pd

from common import RESULTS, ROOT, continuity, argmax_arcs, load_config, save_json
from pathsnap.candidates import FEATURES, TripData
from pathsnap.io_utils import load_case, list_cases
from pathsnap.metrics import route_scores
from pathsnap.pipeline import PathSnap
from train_models import scores

OUT = os.path.join(RESULTS, "evaluator_test")


def candidate_rows(folder):
    """Features + labels for one trip (labels only used for scoring)."""
    cand = load_config()["candidates"]
    case = load_case(folder)
    td = TripData(case, radius=cand["radius"], k=cand["k"], min_spacing=cand["min_spacing"])
    X, y, obs, pos, ref, miss, n_obs = td.labelled_rows()
    arcs = np.concatenate([c[0] for c in td.cands if c is not None])
    df = pd.DataFrame(X, columns=FEATURES)
    df.insert(0, "trip", case["id"])
    df.insert(1, "obs", obs)
    df.insert(2, "arc", arcs)
    df["y"] = y
    return df, {"trip": case["id"], "n_fixes": len(case["track"]), "n_obs": n_obs, "cand_miss": miss}


def pipeline_trip(folder):
    """Final system on one trip, exactly as the API runs it."""
    snap = PathSnap.load()
    full = load_case(folder)
    case = dict(full, route=None)            # the pipeline never sees the reference route
    t0 = time.time()
    res = snap.predict_case(case, with_classifier_only=True)
    secs = time.time() - t0
    ref_route = full["route"]
    ref_obs = res["_td"].reference_arcs(route=ref_route)
    net = res["_td"].net
    out = []
    for mode, route, obs in (("classifier + HMM (final)", res["route"], res["obs_arcs"]),
                             ("classifier only", res["classifier_only_route"], res["classifier_only_obs"])):
        s = route_scores(route, ref_route, res["arc_length"], with_lcs=True)
        r = np.asarray(route)
        s.update(trip=full["id"], mode=mode, seconds=secs if mode.startswith("classifier +") else np.nan,
                 pointwise_top1=float(np.mean(obs == ref_obs)),
                 route_connectivity=float(np.mean(net.dst[r[:-1]] == net.src[r[1:]])) if len(r) > 1 else 1.0,
                 n_route_arcs=len(route))
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips-dir", default=os.path.join(ROOT, "data", "evaluator_test"))
    ap.add_argument("--mlp-file", default="", help="optional: Preksha's own MLP model file")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    folders = list_cases(a.trips_dir)
    print(f"{len(folders)} evaluator trips", flush=True)

    # ---- candidate tables (features + labels for scoring only)
    with ProcessPoolExecutor() as ex:
        parts = list(ex.map(candidate_rows, folders))
    table = pd.concat([p[0] for p in parts], ignore_index=True)
    stats = pd.DataFrame([p[1] for p in parts])
    stats.to_csv(os.path.join(OUT, "candidates_per_trip.csv"), index=False)
    miss_rate = stats.cand_miss.sum() / stats.n_obs.sum()
    print(f"candidate rows {len(table)}, observations {stats.n_obs.sum()}, candidate miss rate {miss_rate:.4f}", flush=True)

    # ---- the 7 saved classifiers + the tuned final model
    arcs_by_trip = {os.path.basename(f): load_case(f)["arcs"] for f in folders}
    models = {m: os.path.join(ROOT, "artifacts", "models", f"{m}.joblib")
              for m in ["LogisticRegression", "GaussianNB", "KNN", "RandomForest",
                        "HistGradientBoosting", "MLP", "LightGBM"]}
    if a.mlp_file:
        models["MLP"] = a.mlp_file
    models["HistGradientBoosting (Bayesian-tuned, final)"] = os.path.join(ROOT, "artifacts", "final_model.joblib")
    X = table[FEATURES].to_numpy()
    rows = []
    for name, path in models.items():
        m = joblib.load(path)
        t0 = time.time()
        p = m.predict_proba(X)[:, 1]
        infer = time.time() - t0
        s = scores(table, p)
        seqs = argmax_arcs(table, p)
        cont = np.mean([continuity(seq, arcs_by_trip[t][:, 0], arcs_by_trip[t][:, 1]) for t, seq in seqs.items()])
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(table["y"], (p >= 0.5).astype(int)).tolist()
        rows.append({"model": name, "model_file": os.path.relpath(path, ROOT).replace(os.sep, "/"),
                     **s, "continuity": float(cont), "infer_secs": infer, "confusion_matrix": json.dumps(cm)})
        print(f"{name:45s} Top-1={s['top1']:.4f}  macroF1={s['f1_macro']:.4f}  logloss={s['log_loss']:.4f}  ({infer:.1f}s)", flush=True)
    clf = pd.DataFrame(rows)
    clf.to_csv(os.path.join(OUT, "classifiers.csv"), index=False)

    # ---- final pipeline (route level)
    with ProcessPoolExecutor() as ex:
        per = [r for rs in ex.map(pipeline_trip, folders) for r in rs]
    per = pd.DataFrame(per)
    per.to_csv(os.path.join(OUT, "pipeline_per_trip.csv"), index=False)
    summ = per.groupby("mode")[["f1", "precision", "recall", "rmf", "arc_acc", "pointwise_top1",
                                "route_connectivity"]].mean()
    summ.to_csv(os.path.join(OUT, "pipeline_summary.csv"))
    print(per[per["mode"].str.startswith("classifier +")][["trip", "f1", "precision", "recall", "rmf",
                                                          "pointwise_top1", "seconds"]].round(4).to_string(index=False))
    print(summ.round(4).to_string())
    save_json({"trips": len(folders), "observations": int(stats.n_obs.sum()), "candidate_rows": len(table),
               "candidate_miss_rate": float(miss_rate),
               "pipeline_mean": summ.to_dict(orient="index")}, os.path.join(OUT, "summary.json"))


if __name__ == "__main__":
    main()
