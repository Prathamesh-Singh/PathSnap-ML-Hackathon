"""Route comparison metrics.

All length-based scores treat the route as a multiset of directed arcs,
weighted by arc length (metres):

  precision / recall / F1  - share of predicted / true length that is correct
  RMF (route mismatch fraction, Newson & Krumm 2009)
        = (length wrongly added + length wrongly missed) / true length
        0 is perfect, lower is better.
  arc accuracy             - LCS of the two arc sequences / max(len) (order aware)
"""
from collections import Counter
import numpy as np


def _lcs_len(a, b):
    # O(n*m) DP with two rows; routes are at most a few thousand arcs
    if not a or not b:
        return 0
    b_arr = np.asarray(b)
    prev = np.zeros(len(b) + 1, dtype=np.int32)
    for x in a:
        cur = np.zeros_like(prev)
        eq = b_arr == x
        for j in range(1, len(b) + 1):
            cur[j] = prev[j - 1] + 1 if eq[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return int(prev[-1])


def route_scores(pred, truth, arc_len, with_lcs=True):
    pred = [int(a) for a in pred]
    truth = [int(a) for a in truth]
    cp, ct = Counter(pred), Counter(truth)
    inter = cp & ct
    L = lambda c: float(sum(arc_len[a] * n for a, n in c.items()))
    lp, lt, li = L(cp), L(ct), L(inter)
    prec = li / lp if lp else 0.0
    rec = li / lt if lt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    rmf = ((lp - li) + (lt - li)) / lt if lt else 0.0
    out = {"precision": prec, "recall": rec, "f1": f1, "rmf": rmf}
    if with_lcs:
        out["arc_acc"] = _lcs_len(pred, truth) / max(len(pred), len(truth))
    return out
