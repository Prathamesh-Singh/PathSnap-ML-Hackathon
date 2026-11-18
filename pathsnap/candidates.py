"""Candidate generation, feature engineering and reference labels.

For every kept GPS observation we collect the nearby directed arcs (the
candidates) and describe each (observation, candidate) pair with a feature
vector. A classifier then scores how likely each candidate is the road the
vehicle was really on, and the HMM uses those scores as emissions.

Reference labels: the .route file only lists arcs in driving order, it does
not say which arc each GPS fix belongs to. We recover that with a monotone
alignment (dynamic programming): fixes are assigned to route arcs so that the
assignment never goes backwards in the route and the total (capped) GPS-to-arc
distance is minimal.
"""
import numpy as np

from .geo import LocalProjection, project_point_on_segments
from .network import RoadNetwork

FEATURES = [
    "dist",            # perpendicular distance fix -> arc (m)
    "dist_rank",       # 0 = closest candidate
    "dist_gap",        # dist - closest candidate's dist
    "dist_z",          # dist / sigma-ish scale, squared (Gaussian log-lik shape)
    "n_cands",         # size of the candidate set
    "heading_diff",    # angle between GPS movement and arc direction (deg, 0..180)
    "heading_ok",      # 1 if the vehicle moved enough for heading to mean something
    "speed",           # m/s estimated from neighbouring fixes
    "t_along",         # projected position along the arc (0..1)
    "arc_len",         # m
    "two_way",         # 1 if the reverse arc exists
    "deg_out_end",     # number of arcs leaving the arc's end node
    "deg_in_start",    # number of arcs entering the arc's start node
    "ctx_prev",        # distance of the previous kept fix to this arc
    "ctx_next",        # distance of the next kept fix to this arc
    "ctx_mean",        # mean distance of fixes i-2..i+2 to this arc
    "turn_angle",      # GPS turning angle at this fix (deg)
    "dt_prev",         # seconds since previous kept fix
]


def downsample(pts, min_spacing):
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) >= min_spacing:
            keep.append(i)
    if keep[-1] != len(pts) - 1:
        keep.append(len(pts) - 1)
    return np.array(keep)


def _angle(u, v):
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    c = np.einsum("...i,...i->...", u, v) / np.maximum(nu * nv, 1e-9)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


class TripData:
    """Everything derived from one trip that the classifier and decoder need."""

    def __init__(self, case, radius=60.0, k=8, min_spacing=20.0, heading_win=3):
        track = case["track"]
        self.case = case
        self.proj = LocalProjection(*track[:, :2].mean(axis=0))
        self.net = RoadNetwork(self.proj(case["nodes"]), case["arcs"])
        self.raw = self.proj(track[:, :2])
        self.time = track[:, 2]
        self.keep = downsample(self.raw, min_spacing)
        self.pts = self.raw[self.keep]
        self.radius, self.k = radius, k

        net = self.net
        self.deg_out = np.bincount(net.src, minlength=len(net.xy))
        self.deg_in = np.bincount(net.dst, minlength=len(net.xy))

        # movement vector & speed around each kept fix, from the raw 1 Hz track
        n = len(self.raw)
        lo = np.clip(self.keep - heading_win, 0, n - 1)
        hi = np.clip(self.keep + heading_win, 0, n - 1)
        self.move = self.raw[hi] - self.raw[lo]
        dt = np.maximum(self.time[hi] - self.time[lo], 1e-6)
        self.speed = np.linalg.norm(self.move, axis=1) / dt

        # turning angle of the kept polyline
        self.turn = np.zeros(len(self.pts))
        if len(self.pts) > 2:
            self.turn[1:-1] = _angle(self.pts[1:-1] - self.pts[:-2], self.pts[2:] - self.pts[1:-1])
        self.dt_prev = np.r_[0.0, np.diff(self.time[self.keep])]

        # candidate sets (None when nothing is inside the radius)
        self.cands = [net.candidates(p, radius, k) for p in self.pts]
        self.features = [self._features(i) for i in range(len(self.pts))]

    # ------------------------------------------------------------------ #
    def _seg_dist(self, i, arcs):
        if i < 0 or i >= len(self.pts):
            return np.full(len(arcs), np.nan)
        d, _ = project_point_on_segments(self.pts[i][None, :], self.net.A[arcs], self.net.B[arcs])
        return d

    def _features(self, i):
        c = self.cands[i]
        if c is None:
            return None
        arcs, t, d = c
        net = self.net
        m = len(arcs)
        arc_vec = net.B[arcs] - net.A[arcs]
        heading_ok = float(np.linalg.norm(self.move[i]) > 3.0)
        hd = _angle(np.repeat(self.move[i][None], m, 0), arc_vec) if heading_ok else np.full(m, 90.0)
        prev, nxt = self._seg_dist(i - 1, arcs), self._seg_dist(i + 1, arcs)
        win = np.vstack([self._seg_dist(j, arcs) for j in range(i - 2, i + 3)])
        X = np.column_stack([
            d,
            np.arange(m),                     # candidates come sorted by distance
            d - d.min(),
            (d / 10.0) ** 2,
            np.full(m, m),
            hd,
            np.full(m, heading_ok),
            np.full(m, self.speed[i]),
            t,
            net.length[arcs],
            (net.rev[arcs] >= 0).astype(float),
            self.deg_out[net.dst[arcs]],
            self.deg_in[net.src[arcs]],
            np.where(np.isnan(prev), d, prev),
            np.where(np.isnan(nxt), d, nxt),
            np.nanmean(win, axis=0),
            np.full(m, self.turn[i]),
            np.full(m, self.dt_prev[i]),
        ])
        return X

    # ------------------------------------------------------------------ #
    def reference_arcs(self, route=None, cap=80.0):
        """Monotone alignment of kept fixes to the reference route.
        Returns one arc id per kept fix."""
        if route is None:
            route = self.case["route"]
        if route is None:
            return None
        A, B = self.net.A[route], self.net.B[route]
        R = len(route)
        back = np.zeros((len(self.pts), R), dtype=np.int32)
        prev = None
        for i, p in enumerate(self.pts):
            cost, _ = project_point_on_segments(p[None, :], A, B)
            cost = np.minimum(cost, cap)
            if prev is None:
                cur = cost
                back[i] = np.arange(R)
            else:
                # best previous position j' <= j is a running minimum;
                # track where each running minimum was set for backtracking
                run_min = np.minimum.accumulate(prev)
                is_new = np.r_[True, prev[1:] < run_min[:-1]]
                back[i] = np.maximum.accumulate(np.where(is_new, np.arange(R), 0))
                cur = cost + run_min
            prev = cur
        j = int(np.argmin(prev))
        out = np.empty(len(self.pts), dtype=np.int64)
        for i in range(len(self.pts) - 1, -1, -1):
            out[i] = route[j]
            j = back[i, j] if i > 0 else j
        return out

    def labelled_rows(self):
        """(X, y, obs_index, cand_index) for every candidate with a label.
        Also returns per-observation info used for the candidate miss rate."""
        ref = self.reference_arcs()
        Xs, ys, obs, pos = [], [], [], []
        miss = 0
        n_obs = 0
        for i, X in enumerate(self.features):
            if X is None:
                if ref is not None:
                    miss += 1
                    n_obs += 1
                continue
            arcs = self.cands[i][0]
            n_obs += 1
            y = (arcs == ref[i]).astype(int) if ref is not None else np.full(len(arcs), -1)
            if ref is not None and y.sum() == 0:
                miss += 1
            Xs.append(X)
            ys.append(y)
            obs.append(np.full(len(arcs), i))
            pos.append(np.arange(len(arcs)))
        return (np.vstack(Xs), np.concatenate(ys), np.concatenate(obs),
                np.concatenate(pos), ref, miss, n_obs)
