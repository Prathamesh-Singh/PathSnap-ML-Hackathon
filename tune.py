"""Tune the HMM parameters on the training folder (coordinate descent on mean F1).

    python tune.py "data/Student folder/train" --out configs/geometric_hmm_params.json
"""
import argparse
import json
import time

from evaluate import evaluate
from pathsnap.io_utils import list_cases
from pathsnap.matcher import Params

GRID = {
    "sigma": [5.0, 6.5, 8.0, 10.0, 14.0],
    "beta": [2.0, 4.0, 6.0, 10.0, 20.0, 40.0],
    "radius": [40.0, 60.0, 90.0],
    "k": [6, 8, 12],
    "min_spacing": [10.0, 20.0, 35.0],
    "back_tol": [10.0, 20.0, 40.0],
    "uturn_penalty": [0.0, 4.0, 8.0, 15.0, 30.0],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_dir")
    ap.add_argument("--out", default="configs/geometric_hmm_params.json")
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()

    folders = list_cases(a.train_dir)
    best = Params().as_dict()
    seen = {}

    def score(params):
        key = json.dumps(params, sort_keys=True)
        if key not in seen:
            t0 = time.time()
            s, _ = evaluate(folders, params, with_lcs=False, verbose=False)
            seen[key] = s
            changed = {k: v for k, v in params.items() if v != Params().as_dict()[k]}
            print(f"F1={s['f1']:.4f} RMF={s['rmf']:.4f} P={s['precision']:.4f} "
                  f"R={s['recall']:.4f}  {changed}  ({time.time() - t0:.0f}s)", flush=True)
        return seen[key]["f1"]

    best_f1 = score(best)
    for r in range(a.rounds):
        print(f"--- round {r + 1}", flush=True)
        for name, values in GRID.items():
            for v in values:
                cand = dict(best, **{name: v})
                f1 = score(cand)
                if f1 > best_f1 + 1e-5:
                    best, best_f1 = cand, f1
            print(f"best so far F1={best_f1:.4f}: {name}={best[name]}", flush=True)

    with open(a.out, "w") as f:
        json.dump(best, f, indent=2)
    print("saved", a.out, best)


if __name__ == "__main__":
    main()
