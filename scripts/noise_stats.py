import sys, numpy as np
sys.path.insert(0, '.')
from pathsnap.io_utils import load_case, list_cases
from pathsnap.geo import LocalProjection, project_point_on_segments
allq = []
for d in list_cases(sys.argv[1]):
    c = load_case(d)
    tr = c['track']; proj = LocalProjection(*tr[:, :2].mean(0))
    P = proj(c['nodes']); T = proj(tr[:, :2])
    r = c['route']; A = P[c['arcs'][r, 0]]; B = P[c['arcs'][r, 1]]
    ds = np.array([project_point_on_segments(t[None], A, B)[0].min() for t in T])
    step = np.linalg.norm(np.diff(T, axis=0), axis=1)
    L = np.linalg.norm(B - A, axis=1)
    # start/end: distance from first/last fix to first/last arc
    d0 = project_point_on_segments(T[:1], A[:1], B[:1])[0][0]; d1 = project_point_on_segments(T[-1:], A[-1:], B[-1:])[0][0]
    q = np.percentile(ds, [50, 90, 99, 100]); allq.append(ds)
    print(c['id'], 'pts', len(T), 'routeLen %.0f' % L.sum(), 'dev p50/p90/p99/max', np.round(q, 1), 'step med %.1f' % np.median(step), 'd_first %.0f d_last %.0f' % (d0, d1), 'arcLen med %.0f' % np.median(L))
a = np.concatenate(allq); print('ALL', np.round(np.percentile(a, [50, 75, 90, 95, 99, 99.9]), 1), 'MAD sigma est', 1.4826*np.median(a))
