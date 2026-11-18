"""Reading and writing the dataset files.

Folder layout (one folder per trip, e.g. train/00000003/):
    <id>.nodes  : one line per node      -> lon  lat
    <id>.arcs   : one line per directed arc -> from_node  to_node   (arc id = line number, 0-based)
    <id>.track  : one line per GPS fix   -> lon  lat  time(s)
    <id>.route  : one line per arc id    -> the ground-truth sequence of arcs driven
"""
import os
import numpy as np


def _load(path, dtype=float):
    # np.loadtxt copes with the \r\n line endings in the .track files
    return np.loadtxt(path, dtype=dtype, ndmin=2)


def load_case(folder):
    cid = os.path.basename(os.path.normpath(folder))
    p = lambda ext: os.path.join(folder, f"{cid}.{ext}")
    case = {
        "id": cid,
        "nodes": _load(p("nodes")),
        "arcs": _load(p("arcs"), dtype=np.int64),
        "track": _load(p("track")),
        "route": None,
    }
    if os.path.exists(p("route")):
        case["route"] = np.loadtxt(p("route"), dtype=np.int64, ndmin=1)
    return case


def write_route(path, route):
    with open(path, "w", newline="\n") as f:
        for a in route:
            f.write(f"{int(a)}\n")


def list_cases(split_dir):
    return sorted(
        os.path.join(split_dir, d)
        for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    )
