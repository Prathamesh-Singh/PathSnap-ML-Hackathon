"""Step 5: HMM / Viterbi route reconstruction - decoder tuning and comparison.

    python experiments/hmm_eval.py

1. Tune the decoder (emission weight, transition scale beta, U-turn penalty,
   backward tolerance) on the 80 training trips, using the final classifier's
   OUT-OF-FOLD probabilities (in-sample probabilities would be over-confident).
2. Compare on the student-testing trips:
     classifier-only     per-observation argmax, stitched with shortest paths
     geometric HMM       Gaussian distance emissions (no classifier) - baseline
     classifier + HMM    classifier log-probabilities as emissions + Viterbi
3. Write artifacts/final_config.json used by the API.
"""
import itertools
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd

from common import ARTIFACTS, CACHE, RESULTS, SPLITS, load_config, load_table, save_json
from pathsnap.candidates import FEATURES
from pathsnap.geo import LocalProjection
from pathsnap.io_utils import load_case
from pathsnap.metrics import route_scores
from pathsnap.network import RoadNetwork
from pathsnap.pipeline import PathSnap

GEO_BASELINE = {"sigma": 14.0, "beta": 10.0, "back_tol": 40.0, "uturn_penalty": 4.0}  # from tune.py

DEFAULT_DECODER = {"emission_weight": 1.0, "beta": 10.0, "uturn_penalty": 4.0, "back_tol": 40.0}
GRID = {
    "emission_weight": [0.5, 1.0, 2.0, 3.0, 5.0],
    "beta": [5.0, 10.0, 20.0, 40.0],
    "uturn_penalty": [0.0, 4.0, 8.0, 15.0],
    "back_tol": [20.0, 40.0, 60.0],
}


def light_trip(split, tid):
    """Rebuild what the decoder needs from the cached candidates."""
    case = load_case(os.path.join(SPLITS[split], tid))
    with open(os.path.join(CACHE, split, f"{tid}.pkl"), "rb") as f:
        c = pickle.load(f)
    proj = LocalProjection(*case["track"][:, :2].mean(axis=0))
    net = RoadNetwork(proj(case["nodes"]), case["arcs"])
    pts = proj(case["track"][:, :2])[c["keep"]]
    return case, SimpleNamespace(net=net, pts=pts, cands=c["cands"], ref=c["ref"])


def split_probs(table, prob):
    """trip -> list (per observation) of candidate probability arrays."""
    out = {}
    df = table[["trip", "obs"]].copy()
    df["p"] = prob
    for tid, g in df.groupby("trip", sort=False):
        per = {o: gg["p"].to_numpy() for o, gg in g.groupby("obs", sort=False)}
        out[tid] = per
    return out


def run_trip(args):
    split, tid, per_obs, cfg, mode = args
    case, td = light_trip(split, tid)
    snap = PathSnap(None, cfg)
    t0 = time.time()
    if mode == "geo":
        sig = GEO_BASELINE["sigma"]
        emis = [None if c is None else -0.5 * (c[2] / sig) ** 2 for c in td.cands]
        chains = snap.hmm.decode(td.net, td.pts, td.cands, emis)
        route = snap.hmm.stitch(td.net, chains)
        obs = np.full(len(td.pts), -1)
        for ch in chains:
            for i, a, _ in ch:
                obs[i] = a
        obs = snap._fill_unmatched(td, obs, route)
        n_chains = len(chains)
    else:
        probs = [per_obs.get(i) if c is not None else None for i, c in enumerate(td.cands)]
        if mode == "clf":
            route, obs = snap.classifier_only(td, probs)
            n_chains = 1
        else:
            emis = [None if p is None else snap.emission_weight * np.log(np.maximum(p, snap.prob_floor))
                    for p in probs]
            chains = snap.hmm.decode(td.net, td.pts, td.cands, emis)
            route = snap.hmm.stitch(td.net, chains)
            obs = np.full(len(td.pts), -1)
            for ch in chains:
                for i, a, _ in ch:
                    obs[i] = a
            obs = snap._fill_unmatched(td, obs, route)
            n_chains = len(chains)
    secs = time.time() - t0
    s = route_scores(route, case["route"], td.net.length, with_lcs=False)
    r = np.asarray(route)
    s.update(
        trip=tid, mode=mode, decode_secs=secs, n_chains=n_chains,
        pointwise_top1=float(np.mean(obs == td.ref)),
        # share of consecutive output arcs that are physically connected
        route_connectivity=float(np.mean(td.net.dst[r[:-1]] == td.net.src[r[1:]])) if len(r) > 1 else 1.0,
        # share of consecutive observation assignments that are the same or adjacent arcs
        obs_continuity=float(np.mean((obs[:-1] == obs[1:]) | (td.net.dst[obs[:-1]] == td.net.src[obs[1:]]))),
    )
    return s


def evaluate(split, probs_by_trip, cfg, mode, trips):
    jobs = [(split, t, probs_by_trip.get(t, {}) if probs_by_trip else {}, cfg, mode) for t in trips]
    with ProcessPoolExecutor() as ex:
        return pd.DataFrame(list(ex.map(run_trip, jobs)))


def cfg_with(dec):
    base = load_config()
    return {"candidates": base["candidates"], "decoder": dict(dec)}


def main():
    train, test = load_table("train"), load_table("student_test")
    oof = np.load(os.path.join(CACHE, "oof", "final_oof.npy"))
    pte = np.load(os.path.join(CACHE, "oof", "final_student_test.npy"))
    ptr_by_trip, pte_by_trip = split_probs(train, oof), split_probs(test, pte)
    tr_trips, te_trips = sorted(train.trip.unique()), sorted(test.trip.unique())

    # ---- 1. decoder tuning on training trips (coordinate descent, 2 rounds)
    log, seen = [], {}

    def score(dec):
        key = tuple(sorted(dec.items()))
        if key not in seen:
            t0 = time.time()
            r = evaluate("train", ptr_by_trip, cfg_with(dec), "hmm", tr_trips)
            seen[key] = r.f1.mean()
            log.append({**dec, "train_route_f1": r.f1.mean(), "train_rmf": r.rmf.mean(),
                        "train_pointwise_top1": r.pointwise_top1.mean(), "secs": time.time() - t0})
            print(log[-1], flush=True)
            pd.DataFrame(log).to_csv(os.path.join(RESULTS, "decoder_tuning_log.csv"), index=False)
        return seen[key]

    best = dict(DEFAULT_DECODER)
    best_f1 = score(best)
    for _ in range(2):
        for name, values in GRID.items():
            for v in values:
                cand = dict(best, **{name: v})
                f = score(cand)
                if f > best_f1 + 1e-5:
                    best, best_f1 = cand, f
    print("best decoder", best, best_f1)

    # ---- 2. comparison on train (OOF) and student-testing trips
    geo_cfg = cfg_with({k: v for k, v in GEO_BASELINE.items() if k != "sigma"})
    rows = []
    for split, trips, pb in (("train", tr_trips, ptr_by_trip), ("student_test", te_trips, pte_by_trip)):
        for mode, cfg in (("clf", cfg_with(best)), ("geo", geo_cfg), ("hmm", cfg_with(best))):
            r = evaluate(split, pb, cfg, mode, trips)
            r["split"] = split
            rows.append(r)
            print(split, mode, r[["f1", "precision", "recall", "rmf", "pointwise_top1",
                                  "route_connectivity", "obs_continuity"]].mean().round(4).to_dict(), flush=True)
    per_trip = pd.concat(rows, ignore_index=True)
    per_trip.to_csv(os.path.join(RESULTS, "hmm_comparison_per_trip.csv"), index=False)
    summary = per_trip.groupby(["split", "mode"])[
        ["f1", "precision", "recall", "rmf", "pointwise_top1", "route_connectivity",
         "obs_continuity", "n_chains", "decode_secs"]].mean()
    summary.to_csv(os.path.join(RESULTS, "hmm_comparison_summary.csv"))
    print(summary.round(4).to_string())

    # ---- 3. final config for the API
    cfg = cfg_with(best)
    cfg.update({"model_file": "final_model.joblib",
                "model_name": open(os.path.join(ARTIFACTS, "final_model_name.txt")).read().strip(),
                "features": FEATURES})
    save_json(cfg, os.path.join(ARTIFACTS, "final_config.json"))
    print("wrote artifacts/final_config.json")


if __name__ == "__main__":
    main()
