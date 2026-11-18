"""HMM map matching (Newson & Krumm style).

Hidden states  = candidate positions on road arcs near each GPS fix
Emission       = Gaussian on the GPS-to-road distance (sigma)
Transition     = Exponential on |driving distance - straight-line distance| (beta)
Decoding       = Viterbi, then the shortest paths between the chosen
                 positions are stitched together into one arc sequence.
"""
from dataclasses import dataclass, asdict
import numpy as np

from .geo import LocalProjection
from .network import RoadNetwork


@dataclass
class Params:
    sigma: float = 8.0          # GPS noise (m)
    beta: float = 6.0           # transition scale (m)
    radius: float = 60.0        # candidate search radius (m)
    k: int = 8                  # max candidate arcs per fix
    min_spacing: float = 20.0   # drop fixes closer than this to the previous kept fix (m)
    back_tol: float = 20.0      # allowed backward jitter along the same arc (m)
    limit_mult: float = 2.0     # Dijkstra bound = limit_mult * gc + limit_add
    limit_add: float = 400.0
    uturn_penalty: float = 4.0  # log-penalty for turning onto the reverse arc
    max_outlier_skip: int = 135  # most kept fixes we may jump over as outliers
    max_skip_dist: float = 2500.0  # ...as long as the jump is shorter than this (m)
    max_backtrack: int = 8      # states we may step back when the decoder hits a dead end
    bridge_mult: float = 3.0    # between chains, only bridge with paths shorter than
    bridge_add: float = 1000.0  #   bridge_mult * straight distance + bridge_add

    def as_dict(self):
        return asdict(self)


class HMMMatcher:
    def __init__(self, params=None):
        self.p = params or Params()

    # ------------------------------------------------------------------ #
    def prepare(self, case):
        track = case["track"]
        proj = LocalProjection(*track[:, :2].mean(axis=0))
        net = RoadNetwork(proj(case["nodes"]), case["arcs"])
        pts = proj(track[:, :2])
        return net, pts

    def _downsample(self, pts):
        keep = [0]
        for i in range(1, len(pts)):
            if np.linalg.norm(pts[i] - pts[keep[-1]]) >= self.p.min_spacing:
                keep.append(i)
        if keep[-1] != len(pts) - 1:
            keep.append(len(pts) - 1)
        return keep

    # ------------------------------------------------------------------ #
    def _transition(self, net, c1, c2, gc):
        """Log transition matrix between candidate sets c1 -> c2."""
        p = self.p
        a1, t1, _ = c1
        a2, t2, _ = c2
        limit = p.limit_mult * gc + p.limit_add
        T = np.full((len(a1), len(a2)), -np.inf)
        L = net.length
        for i in range(len(a1)):
            u = int(net.dst[a1[i]])
            rest = (1.0 - t1[i]) * L[a1[i]]
            dist = None
            for j in range(len(a2)):
                if a1[i] == a2[j] and (t1[i] - t2[j]) * L[a1[i]] <= p.back_tol:
                    d = max(0.0, (t2[j] - t1[i]) * L[a1[i]])
                else:
                    if dist is None:
                        dist, pred, first = net.shortest_from(u, limit)
                    v = int(net.src[a2[j]])
                    sp = dist.get(v)
                    if sp is None or sp > limit:
                        continue
                    d = rest + sp + t2[j] * L[a2[j]]
                    # real U-turns are rare: penalise turning back onto the
                    # reverse arc at either end of the connecting path
                    r1 = net.rev[a1[i]]
                    if a2[j] == r1 or (v != u and first[v] == r1):
                        d_pen = p.uturn_penalty
                    elif v != u and pred[v] == net.rev[a2[j]]:
                        d_pen = p.uturn_penalty
                    else:
                        d_pen = 0.0
                    T[i, j] = -abs(d - gc) / p.beta - d_pen
                    continue
                T[i, j] = -abs(d - gc) / p.beta
        return T

    def _emission(self, c):
        return -0.5 * (c[2] / self.p.sigma) ** 2

    # ------------------------------------------------------------------ #
    def viterbi(self, net, pts):
        """Geometric HMM: Gaussian emissions on GPS-to-arc distance.
        Returns a list of chains; each chain is a list of (arc, t)."""
        p = self.p
        idx = self._downsample(pts)
        obs_pts = pts[idx]
        cands = [net.candidates(q, p.radius, p.k) for q in obs_pts]
        emis = [None if c is None else self._emission(c) for c in cands]
        chains = self.decode(net, obs_pts, cands, emis)
        return [[(a, t) for _, a, t in ch] for ch in chains]

    def decode(self, net, obs_pts, obs_cands, obs_emis):
        """Viterbi over observations.

        obs_cands[i] : (arcs, t, dist) candidate set of observation i, or None
        obs_emis[i]  : log emission score for each candidate of observation i
        Returns a list of chains; each chain is a list of (obs index, arc, t).
        """
        p = self.p
        keep = [i for i, c in enumerate(obs_cands) if c is not None]
        if not keep:  # nothing near the road at all: snap first/last fix to nearest arcs
            keep = sorted({0, len(obs_pts) - 1})
            obs_cands = list(obs_cands)
            obs_emis = list(obs_emis)
            for i in keep:
                obs_cands[i] = net.nearest_candidates(obs_pts[i], p.k)
                obs_emis[i] = self._emission(obs_cands[i])
        cands = [obs_cands[i] for i in keep]
        emis = [np.asarray(obs_emis[i], dtype=float) for i in keep]
        used_pts = [obs_pts[i] for i in keep]

        chains = []
        states = [0]                        # indices into cands, current chain
        scores = [emis[0]]                  # Viterbi column for each state
        back = [None]
        n = 1
        while n < len(cands):
            # Normal case: step from the last state to fix n. If no candidate of
            # n is reachable, try skipping fixes ahead (outliers / road missing
            # from the map) and stepping back a few states (the decoder may have
            # wandered onto a dead end), cheapest combinations first. Only if
            # all of that fails do we start a new chain.
            ok = False
            for depth, skip in self._recovery_order(len(states) - 1):
                base = len(states) - 1 - depth
                m = n + skip
                if m >= len(cands):
                    continue
                if skip and np.linalg.norm(used_pts[m] - used_pts[n]) > p.max_skip_dist:
                    continue
                prev, score = states[base], scores[base]
                gc = np.linalg.norm(used_pts[m] - used_pts[prev])
                T = self._transition(net, cands[prev], cands[m], gc)
                tot = score[:, None] + T
                best = np.argmax(tot, axis=0)
                new = tot[best, np.arange(tot.shape[1])] + emis[m]
                if np.isfinite(new).any():
                    del states[base + 1:], scores[base + 1:], back[base + 1:]
                    states.append(m)
                    scores.append(new - new[np.isfinite(new)].max())
                    back.append(best)
                    n, ok = m + 1, True
                    break
            if not ok:
                chains.append(self._backtrack(cands, states, back, scores[-1], keep))
                states, scores, back = [n], [emis[n]], [None]
                n += 1
        chains.append(self._backtrack(cands, states, back, scores[-1], keep))
        return chains

    SKIPS = (0, 1, 2, 3, 5, 8, 12, 18, 27, 40, 60, 90, 135)

    def _recovery_order(self, max_depth):
        """(depth, skip) pairs to try, ordered roughly by how drastic they are."""
        depths = range(min(self.p.max_backtrack, max_depth) + 1)
        skips = [s for s in self.SKIPS if s <= self.p.max_outlier_skip] or [0]
        pairs = [(d, s) for d in depths for s in skips]
        pairs.sort(key=lambda ds: (max(ds[0], skips.index(ds[1])), ds[1], ds[0]))
        return pairs

    @staticmethod
    def _backtrack(cands, states, back, score, keep):
        j = int(np.argmax(score))
        out = []
        for pos in range(len(states) - 1, -1, -1):
            c = cands[states[pos]]
            out.append((keep[states[pos]], int(c[0][j]), float(c[1][j])))
            if back[pos] is not None:
                j = int(back[pos][j])
        return out[::-1]

    # ------------------------------------------------------------------ #
    def stitch(self, net, chains):
        p = self.p
        route = []

        def push(a):
            if not route or route[-1] != a:
                route.append(a)

        def where(a, t):
            return net.A[a] + t * (net.B[a] - net.A[a])

        prev = None
        for chain in chains:
            for k, state in enumerate(chain):
                a, t = state[-2], state[-1]  # accepts (arc, t) or (obs, arc, t)
                if prev is None:
                    push(a)
                else:
                    pa, pt = prev
                    if pa == a and (pt - t) * net.length[a] <= p.back_tol:
                        pass  # still on the same arc
                    else:
                        if k == 0:
                            # crossing a break between chains: only bridge it
                            # if the road path is a plausible length
                            gc = np.linalg.norm(where(a, t) - where(pa, pt))
                            lim = p.bridge_mult * gc + p.bridge_add
                            gap = net.path_arcs(int(net.dst[pa]), int(net.src[a]), lim, max_len=lim)
                        else:
                            gap = net.path_arcs(int(net.dst[pa]), int(net.src[a]))
                        for g in gap or []:
                            push(g)
                        push(a)
                prev = (a, t)
        return route

    def match(self, case):
        net, pts = self.prepare(case)
        chains = self.viterbi(net, pts)
        return self.stitch(net, chains)
