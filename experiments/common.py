"""Shared paths, config and metric helpers for the experiment scripts."""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data", "Student folder")
SPLITS = {"train": os.path.join(DATA, "train"), "student_test": os.path.join(DATA, "student_test")}
RESULTS = os.path.join(ROOT, "results")
CACHE = os.path.join(ROOT, "cache")
ARTIFACTS = os.path.join(ROOT, "artifacts")
FIGS = os.path.join(RESULTS, "figures")
SEED = 42

for d in (RESULTS, CACHE, ARTIFACTS, FIGS):
    os.makedirs(d, exist_ok=True)


def load_config():
    with open(os.path.join(ROOT, "configs", "pipeline.json")) as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def trip_cache_path(split, trip_id):
    return os.path.join(CACHE, split, f"{trip_id}.pkl")


def load_trip_cache(split, trip_id):
    with open(trip_cache_path(split, trip_id), "rb") as f:
        return pickle.load(f)


def load_table(split):
    """Candidate table for a split: one row per (observation, candidate)."""
    return pd.read_pickle(os.path.join(CACHE, f"{split}_candidates.pkl"))


# --------------------------------------------------------------------------- #
# metrics on the candidate table
def top1_accuracy(df, prob):
    """Observation-level Top-1: does the highest-scoring candidate equal the
    reference arc? Observations whose reference arc is missing from the
    candidate set count as wrong (that is the candidate miss rate)."""
    tmp = df[["trip", "obs", "y"]].copy()
    tmp["p"] = prob
    best = tmp.loc[tmp.groupby(["trip", "obs"], sort=False)["p"].idxmax()]
    return float(best["y"].mean())


def argmax_arcs(df, prob):
    tmp = df[["trip", "obs", "arc"]].copy()
    tmp["p"] = prob
    best = tmp.loc[tmp.groupby(["trip", "obs"], sort=False)["p"].idxmax()]
    return {t: g.sort_values("obs")["arc"].to_numpy() for t, g in best.groupby("trip")}


def continuity(arc_seq, src, dst):
    """Share of consecutive predictions that are the same arc or directly
    connected (end node of one = start node of the next)."""
    a, b = arc_seq[:-1], arc_seq[1:]
    if len(a) == 0:
        return 1.0
    ok = (a == b) | (dst[a] == src[b])
    return float(ok.mean())
