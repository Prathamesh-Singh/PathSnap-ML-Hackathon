"""Plot one trip: road network, raw GPS, true route and predicted route.

    python visualize.py "data/Student folder/student_test/00000052" --out figures/00000052.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pathsnap.io_utils import load_case
from pathsnap.matcher import HMMMatcher, Params
from pathsnap.metrics import route_scores


def draw_arcs(ax, net, arcs, **kw):
    if len(arcs) == 0:
        return
    arcs = np.asarray(arcs)
    xs = np.stack([net.A[arcs, 0], net.B[arcs, 0], np.full(len(arcs), np.nan)], 1).ravel()
    ys = np.stack([net.A[arcs, 1], net.B[arcs, 1], np.full(len(arcs), np.nan)], 1).ravel()
    ax.plot(xs, ys, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--params", default="configs/geometric_hmm_params.json")
    a = ap.parse_args()

    params = Params().as_dict()
    if os.path.exists(a.params):
        with open(a.params) as f:
            params.update(json.load(f))
    case = load_case(a.case_dir)
    m = HMMMatcher(Params(**params))
    net, pts = m.prepare(case)
    route = m.stitch(net, m.viterbi(net, pts))

    pad = 150
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    inside = np.where(((net.A >= lo) & (net.A <= hi)).all(1))[0]

    fig, ax = plt.subplots(figsize=(10, 10))
    draw_arcs(ax, net, inside, color="#d0d0d0", lw=0.6)
    title = f"trip {case['id']}"
    if case["route"] is not None:
        draw_arcs(ax, net, case["route"], color="#2a9d8f", lw=5, alpha=0.5, label="true route")
        s = route_scores(route, case["route"], net.length, with_lcs=False)
        title += f"   F1={s['f1']:.3f}  RMF={s['rmf']:.3f}"
    draw_arcs(ax, net, route, color="#e76f51", lw=1.6, label="predicted route")
    ax.plot(pts[:, 0], pts[:, 1], ".", ms=2, color="#264653", label="GPS fixes")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend(loc="best")
    out = a.out or f"figures/{case['id']}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
