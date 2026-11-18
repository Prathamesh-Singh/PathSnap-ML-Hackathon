"""Run the matcher on folders that have ground-truth routes and report scores.

    python evaluate.py "data/Student folder/student_test"
    python evaluate.py "data/Student folder/train" --sigma 8 --beta 6
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from pathsnap.io_utils import load_case, list_cases
from pathsnap.matcher import HMMMatcher, Params
from pathsnap.metrics import route_scores


def run_one(args):
    folder, params, with_lcs = args
    case = load_case(folder)
    t0 = time.time()
    m = HMMMatcher(Params(**params))
    net, pts = m.prepare(case)
    route = m.stitch(net, m.viterbi(net, pts))
    s = route_scores(route, case["route"], net.length, with_lcs=with_lcs)
    s.update(id=case["id"], secs=time.time() - t0, n_pred=len(route), n_true=len(case["route"]))
    return s


def evaluate(folders, params, workers=None, with_lcs=True, verbose=True):
    jobs = [(f, params, with_lcs) for f in folders]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(run_one, jobs))
    if verbose:
        for r in res:
            print(f"{r['id']}  F1={r['f1']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}  "
                  f"RMF={r['rmf']:.4f}" + (f"  arcAcc={r['arc_acc']:.4f}" if with_lcs else "")
                  + f"  ({r['secs']:.1f}s)")
    keys = ["f1", "precision", "recall", "rmf"] + (["arc_acc"] if with_lcs else [])
    summary = {k: float(np.mean([r[k] for r in res])) for k in keys}
    return summary, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir")
    ap.add_argument("--params", help="json file with tuned parameters")
    ap.add_argument("--workers", type=int, default=None)
    for name, val in Params().as_dict().items():
        ap.add_argument(f"--{name}", type=type(val), default=None)
    a = ap.parse_args()

    params = Params().as_dict()
    if a.params:
        with open(a.params) as f:
            params.update(json.load(f))
    for name in params:
        if getattr(a, name) is not None:
            params[name] = getattr(a, name)

    summary, _ = evaluate(list_cases(a.split_dir), params, a.workers)
    print("MEAN", {k: round(v, 4) for k, v in summary.items()})


if __name__ == "__main__":
    main()
