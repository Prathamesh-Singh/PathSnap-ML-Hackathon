"""Predict routes for every trip folder in a directory.

    python predict.py "data/Student folder/student_test" --out predictions/student_test

Each trip folder needs <id>.nodes, <id>.arcs and <id>.track (a .route file,
if present, is ignored). Writes predictions/<...>/<id>.route (one arc id per
line, same format as the given .route files) and a combined routes.csv.

By default the final pipeline (classifier + HMM, artifacts/) is used;
--geometric uses the distance-only HMM baseline instead.
"""
import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

from pathsnap.io_utils import load_case, list_cases, write_route
from pathsnap.matcher import HMMMatcher, Params


def _predict(args):
    folder, params, geometric = args
    case = load_case(folder)
    case["route"] = None  # never use the reference route for prediction
    t0 = time.time()
    if geometric:
        route = HMMMatcher(Params(**params)).match(case)
    else:
        from pathsnap.pipeline import PathSnap
        route = PathSnap.load().predict_case(case)["route"]
    return case["id"], route, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", help="folder of trip folders (or a single trip folder)")
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--params", default="configs/geometric_hmm_params.json")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--geometric", action="store_true", help="distance-only HMM baseline")
    a = ap.parse_args()

    params = Params().as_dict()
    if os.path.exists(a.params):
        with open(a.params) as f:
            params.update(json.load(f))

    cid = os.path.basename(os.path.normpath(a.input_dir))
    if os.path.exists(os.path.join(a.input_dir, cid + ".track")):
        folders = [a.input_dir]
    else:
        folders = list_cases(a.input_dir)

    os.makedirs(a.out, exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(_predict, [(f, params, a.geometric) for f in folders]))

    with open(os.path.join(a.out, "routes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "route"])
        for cid, route, secs in results:
            write_route(os.path.join(a.out, f"{cid}.route"), route)
            w.writerow([cid, " ".join(map(str, route))])
            print(f"{cid}: {len(route)} arcs ({secs:.1f}s)")
    print(f"wrote {len(results)} routes to {a.out}")


if __name__ == "__main__":
    main()
