import heapq
import numpy as np
from scipy.spatial import cKDTree

from .geo import project_point_on_segments


class RoadNetwork:
    """Directed road graph in local metric coordinates.

    Every arc is a straight segment between two nodes. A spatial index is built
    from points sampled along each arc so we can quickly find the arcs near a
    GPS fix.
    """

    def __init__(self, xy_nodes, arcs, sample_step=15.0):
        self.xy = xy_nodes
        self.src = arcs[:, 0]
        self.dst = arcs[:, 1]
        self.A = xy_nodes[self.src]
        self.B = xy_nodes[self.dst]
        self.length = np.linalg.norm(self.B - self.A, axis=1)
        self.n_arcs = len(arcs)

        # reverse twin of each arc (-1 if the road is one-way)
        lookup = {(int(u), int(v)): a for a, (u, v) in enumerate(arcs)}
        self.rev = np.array([lookup.get((int(v), int(u)), -1) for u, v in arcs])

        # adjacency: node -> list of (arc id, next node, length)
        self.out = [[] for _ in range(len(xy_nodes))]
        for a, (u, v) in enumerate(arcs):
            self.out[u].append((a, int(v), float(self.length[a])))

        # spatial index over sample points along the arcs
        n = np.maximum(1, np.ceil(self.length / sample_step).astype(int))
        arc_of = np.repeat(np.arange(self.n_arcs), n + 1)
        offs = np.concatenate([np.arange(k + 1) / k for k in n])
        pts = self.A[arc_of] + (self.B - self.A)[arc_of] * offs[:, None]
        self.sample_step = sample_step
        self._tree = cKDTree(pts)
        self._arc_of = arc_of

        self._cache = {}

    # ------------------------------------------------------------------ #
    def candidates(self, p, radius, k):
        """Arcs within `radius` metres of p: arrays (arc, t, dist), best k."""
        idx = self._tree.query_ball_point(p, radius + self.sample_step)
        if not idx:
            return None
        arcs = np.unique(self._arc_of[idx])
        d, t = project_point_on_segments(p[None, :], self.A[arcs], self.B[arcs])
        keep = d <= radius
        arcs, d, t = arcs[keep], d[keep], t[keep]
        if len(arcs) == 0:
            return None
        order = np.argsort(d)[:k]
        return arcs[order], t[order], d[order]

    def nearest_candidates(self, p, k):
        """Fallback when nothing is inside the search radius."""
        _, idx = self._tree.query(p, k=min(50, len(self._arc_of)))
        arcs = np.unique(self._arc_of[np.atleast_1d(idx)])
        d, t = project_point_on_segments(p[None, :], self.A[arcs], self.B[arcs])
        order = np.argsort(d)[:k]
        return arcs[order], t[order], d[order]

    # ------------------------------------------------------------------ #
    def shortest_from(self, source, limit):
        """Bounded Dijkstra from a node. Results are cached per source and
        reused while the cached search radius is large enough.

        Returns dist[node], pred[node] (last arc into node) and first[node]
        (first arc leaving the source on the way to node)."""
        hit = self._cache.get(source)
        if hit is not None and hit[0] >= limit:
            return hit[1], hit[2], hit[3]
        dist = {source: 0.0}
        pred = {}
        first = {}
        heap = [(0.0, source)]
        out = self.out
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf):
                continue
            if d > limit:
                break
            fu = first.get(u)
            for a, v, w in out[u]:
                nd = d + w
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    pred[v] = a
                    first[v] = a if fu is None else fu
                    heapq.heappush(heap, (nd, v))
        self._cache[source] = (limit, dist, pred, first)
        if len(self._cache) > 4000:  # keep memory in check on long tracks
            for key in list(self._cache)[:2000]:
                del self._cache[key]
        return dist, pred, first

    def path_arcs(self, source, target, limit=np.inf, max_len=np.inf):
        """Arc sequence of the shortest path source -> target (nodes), or None
        if there is no path no longer than max_len."""
        if source == target:
            return []
        hit = self._cache.get(source)
        if hit is not None and hit[1].get(target, np.inf) <= hit[0]:
            dist, pred = hit[1], hit[2]  # already settled in an earlier search
        else:
            dist, pred, _ = self.shortest_from(source, limit)
        if target not in dist or dist[target] > min(max_len, self._cache[source][0]):
            return None
        seq = []
        v = target
        while v != source:
            a = pred[v]
            seq.append(a)
            v = int(self.src[a])
        return seq[::-1]
