"""PathSnap API (runs on Laptop 1).

    uvicorn api.app:app --host 0.0.0.0 --port 8000

Environment variables:
    PATHSNAP_TRIPS_DIR   folder holding the evaluation trip folders
                         (default: data/Student folder/student_test)

Endpoints:
    GET  /health              liveness + which model is loaded
    GET  /trips               trip ids available for evaluation (no labels)
    POST /predict/{trip_id}   run the full pipeline on that trip and score it

The reference route of a trip is read only *after* prediction, inside
`score_against_reference`, and only metrics leave the server - the reference
geometry is never sent to the UI and never passed to the model. The response
does include the trip's road network (an input of the task) so the UI can draw
the roads even when no internet map tiles are available.
"""
import logging
import os
import sys
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pathsnap.io_utils import load_case, list_cases  # noqa: E402
from pathsnap.metrics import route_scores  # noqa: E402
from pathsnap.pipeline import PathSnap  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

TRIPS_DIR = os.environ.get("PATHSNAP_TRIPS_DIR", os.path.join(ROOT, "data", "Student folder", "student_test"))

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

snap = PathSnap.load()
log.info("model loaded: %s | trips dir: %s", snap.describe(), TRIPS_DIR)


def _trip_folder(trip_id):
    if not trip_id.isdigit():
        raise HTTPException(400, "trip id must be numeric")
    folder = os.path.join(TRIPS_DIR, trip_id)
    if not os.path.isdir(folder):
        raise HTTPException(404, f"trip {trip_id} not found")
    return folder


def _inputs_only(folder):
    """Load the trip WITHOUT its reference route - this is all the model sees."""
    case = load_case(folder)
    case["route"] = None
    return case


def score_against_reference(folder, case, result):
    """Server-side evaluation against the hidden reference route."""
    ref_route = load_case(folder)["route"]
    if ref_route is None:
        return None
    s = route_scores(result["route"], ref_route, result["arc_length"], with_lcs=True)
    ref_obs = result["_td"].reference_arcs(route=np.asarray(ref_route))
    pred_obs = result["obs_arcs"]
    valid = pred_obs >= 0
    s["pointwise_top1"] = float(np.mean(pred_obs == ref_obs))
    s["pointwise_top1_matched_only"] = float(np.mean(pred_obs[valid] == ref_obs[valid])) if valid.any() else 0.0
    return s


def _network_segments(nodes, arcs):
    """All road segments of the trip's map as [[lat, lon], [lat, lon]], two-way roads drawn once."""
    seen, out = set(), []
    for u, v in arcs:
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        out.append([[round(float(nodes[u][1]), 6), round(float(nodes[u][0]), 6)],
                    [round(float(nodes[v][1]), 6), round(float(nodes[v][0]), 6)]])
    return out


@app.get("/health")
def health():
    return {"status": "ok", "model": snap.describe(), "trips_dir": TRIPS_DIR}


@app.get("/trips")
def trips():
    ids = [os.path.basename(f) for f in list_cases(TRIPS_DIR)]
    log.info("GET /trips -> %d trips", len(ids))
    return {"trips": ids}


@app.post("/predict/{trip_id}")
def predict(trip_id: str):
    folder = _trip_folder(trip_id)
    log.info("POST /predict/%s  received", trip_id)
    t0 = time.time()
    case = _inputs_only(folder)
    result = snap.predict_case(case)
    t_pred = time.time() - t0
    metrics = score_against_reference(folder, case, result)
    nodes, arcs = case["nodes"], case["arcs"]
    route = result["route"]
    track = case["track"]
    step = max(1, len(track) // 2500)
    resp = {
        "trip_id": trip_id,
        "n_gps_fixes": int(len(track)),
        "n_observations": int(len(result["obs_arcs"])),
        "n_route_arcs": int(len(route)),
        "route_arc_ids": [int(a) for a in route],
        # predicted route geometry as [lat, lon] pairs per arc
        "route_latlon": [[[float(nodes[u][1]), float(nodes[u][0])], [float(nodes[v][1]), float(nodes[v][0])]]
                         for u, v in arcs[route]],
        "gps_latlon": [[float(y), float(x)] for x, y in track[::step, :2]],
        # the trip's road network (model input, not the answer), drawn as a background layer
        "network_latlon": _network_segments(nodes, arcs),
        "inference_seconds": round(t_pred, 3),
        "metrics": metrics,
    }
    if metrics:
        log.info("POST /predict/%s  done in %.2fs | arcs=%d | F1=%.4f P=%.4f R=%.4f RMF=%.4f top1=%.4f",
                 trip_id, t_pred, len(route), metrics["f1"], metrics["precision"], metrics["recall"],
                 metrics["rmf"], metrics["pointwise_top1"])
    else:
        log.info("POST /predict/%s  done in %.2fs | arcs=%d | no reference available", trip_id, t_pred, len(route))
    return resp
