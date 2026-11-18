"""Step 2: generate candidates, features and reference labels for every trip.

    python experiments/build_dataset.py

Writes cache/<split>/<trip>.pkl (per-trip candidates, features, reference
arcs) and cache/<split>_candidates.pkl (one table with every candidate row).
"""
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import SPLITS, CACHE, RESULTS, load_config, trip_cache_path, save_json
from pathsnap.candidates import TripData, FEATURES
from pathsnap.io_utils import load_case, list_cases


def process(args):
    split, folder, cfg = args
    t0 = time.time()
    case = load_case(folder)
    td = TripData(case, radius=cfg["radius"], k=cfg["k"], min_spacing=cfg["min_spacing"])
    X, y, obs, pos, ref, miss, n_obs = td.labelled_rows()
    arcs = np.concatenate([c[0] for c in td.cands if c is not None])
    trip = {
        "id": case["id"], "keep": td.keep, "cands": td.cands, "features": td.features,
        "ref": ref, "miss": miss, "n_obs": n_obs,
    }
    os.makedirs(os.path.dirname(trip_cache_path(split, case["id"])), exist_ok=True)
    with open(trip_cache_path(split, case["id"]), "wb") as f:
        pickle.dump(trip, f)
    df = pd.DataFrame(X, columns=FEATURES)
    df.insert(0, "trip", case["id"])
    df.insert(1, "obs", obs)
    df.insert(2, "cand", pos)
    df.insert(3, "arc", arcs)
    df["y"] = y
    return df, {"trip": case["id"], "n_fixes": len(case["track"]), "n_obs": n_obs,
                "n_rows": len(df), "cand_miss": miss, "secs": time.time() - t0}


def main():
    cfg = load_config()["candidates"]
    summary = {}
    for split, folder in SPLITS.items():
        t0 = time.time()
        jobs = [(split, f, cfg) for f in list_cases(folder)]
        with ProcessPoolExecutor() as ex:
            out = list(ex.map(process, jobs))
        table = pd.concat([o[0] for o in out], ignore_index=True)
        table.to_pickle(os.path.join(CACHE, f"{split}_candidates.pkl"))
        stats = pd.DataFrame([o[1] for o in out])
        stats.to_csv(os.path.join(RESULTS, f"candidates_{split}.csv"), index=False)
        summary[split] = {
            "trips": len(out), "observations": int(stats.n_obs.sum()), "rows": len(table),
            "positive_rate": float(table.y.mean()),
            "candidate_miss_rate": float(stats.cand_miss.sum() / stats.n_obs.sum()),
            "build_secs": time.time() - t0,
        }
        print(split, summary[split], flush=True)
    save_json({"config": cfg, **summary}, os.path.join(RESULTS, "candidate_summary.json"))


if __name__ == "__main__":
    main()
