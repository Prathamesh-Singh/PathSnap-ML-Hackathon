"""End-to-end inference: trip files -> candidates -> classifier -> HMM -> route.

Used by the API, the Folium maps and the final evaluation script, so all of
them run exactly the same code path.
"""
import json
import os
import time

import joblib
import numpy as np

from .candidates import TripData
from .geo import project_point_on_segments
from .matcher import HMMMatcher, Params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(ROOT, "artifacts")


class PathSnap:
    def __init__(self, model, config):
        self.model = model
        self.cfg = config
        dec = dict(config["decoder"])
        self.emission_weight = dec.pop("emission_weight", 1.0)
        self.prob_floor = dec.pop("prob_floor", 1e-4)
        cand = config["candidates"]
        self.hmm = HMMMatcher(Params(radius=cand["radius"], k=cand["k"],
                                     min_spacing=cand["min_spacing"], **dec))

    @classmethod
    def load(cls, artifacts_dir=ARTIFACTS):
        with open(os.path.join(artifacts_dir, "final_config.json")) as f:
            cfg = json.load(f)
        model = joblib.load(os.path.join(artifacts_dir, cfg["model_file"]))
        return cls(model, cfg)

    def describe(self):
        return f"{self.cfg.get('model_name', type(self.model).__name__)} + HMM/Viterbi"

    # ------------------------------------------------------------------ #
    def trip_data(self, case):
        cand = self.cfg["candidates"]
        case = dict(case, route=None)  # the model never sees the reference route
        return TripData(case, radius=cand["radius"], k=cand["k"], min_spacing=cand["min_spacing"])

    def candidate_probs(self, td):
        rows = [X for X in td.features if X is not None]
        if not rows:
            return [None] * len(td.features)
        p = self.model.predict_proba(np.vstack(rows))[:, 1]
        out, i = [], 0
        for X in td.features:
            if X is None:
                out.append(None)
            else:
                out.append(p[i:i + len(X)])
                i += len(X)
        return out

    def decode(self, td, probs):
        emis = [None if p is None else self.emission_weight * np.log(np.maximum(p, self.prob_floor))
                for p in probs]
        chains = self.hmm.decode(td.net, td.pts, td.cands, emis)
        route = self.hmm.stitch(td.net, chains)
        obs_arcs = np.full(len(td.pts), -1, dtype=np.int64)
        for ch in chains:
            for i, a, _ in ch:
                obs_arcs[i] = a
        return route, self._fill_unmatched(td, obs_arcs, route)

    @staticmethod
    def _fill_unmatched(td, obs_arcs, route):
        """Observations skipped as outliers (or without candidates) get the
        nearest arc of the final route, so every observation has a segment."""
        miss = np.where(obs_arcs < 0)[0]
        if len(miss) and len(route):
            r = np.asarray(route)
            A, B = td.net.A[r], td.net.B[r]
            for i in miss:
                d, _ = project_point_on_segments(td.pts[i][None], A, B)
                obs_arcs[i] = r[int(np.argmin(d))]
        return obs_arcs

    def classifier_only(self, td, probs):
        """Independent per-observation argmax, stitched with shortest paths."""
        chain = []
        obs = np.full(len(td.pts), -1, dtype=np.int64)
        for i, (c, p) in enumerate(zip(td.cands, probs)):
            if c is None:
                continue
            j = int(np.argmax(p))
            obs[i] = c[0][j]
            chain.append((c[0][j], c[1][j]))
        route = self.hmm.stitch(td.net, [chain]) if chain else []
        return route, obs

    def predict_case(self, case, with_classifier_only=False):
        t0 = time.time()
        td = self.trip_data(case)
        t_feat = time.time() - t0
        probs = self.candidate_probs(td)
        t_clf = time.time() - t0 - t_feat
        route, obs_arcs = self.decode(td, probs)
        out = {
            "route": route, "obs_arcs": obs_arcs, "arc_length": td.net.length,
            "timing": {"features_s": t_feat, "classifier_s": t_clf,
                       "decode_s": time.time() - t0 - t_feat - t_clf},
            "_td": td, "_probs": probs,
        }
        if with_classifier_only:
            out["classifier_only_route"], out["classifier_only_obs"] = self.classifier_only(td, probs)
        self._last_td = td
        return out

    def reference_per_observation(self, case, ref_route):
        """Server-side helper: reference arc per observation of the last
        predicted trip (monotone alignment of fixes to the reference route)."""
        td = getattr(self, "_last_td", None)
        if td is None or td.case["id"] != case["id"]:
            td = self.trip_data(case)
        return td.reference_arcs(route=np.asarray(ref_route))
