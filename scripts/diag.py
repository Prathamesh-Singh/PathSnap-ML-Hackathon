import sys; sys.path.insert(0, '.')
import numpy as np
from pathsnap.io_utils import load_case
from pathsnap.matcher import HMMMatcher, Params
c = load_case(sys.argv[1]); m = HMMMatcher(Params())
net, pts = m.prepare(c); chains = m.viterbi(net, pts); r = m.stitch(net, chains)
print('chains', len(chains), [len(x) for x in chains])
truth = list(c['route']); ts = set(truth); ps = set(r)
extra = [i for i, a in enumerate(r) if a not in ts]
# group consecutive extras
grp = []
for i in extra:
    if grp and i == grp[-1][-1] + 1: grp[-1].append(i)
    else: grp.append([i])
for g in grp[:25]:
    arcs = [r[i] for i in g]; L = net.length[arcs].sum()
    rev = [any(net.src[a]==net.dst[t] and net.dst[a]==net.src[t] for t in truth) for a in arcs]
    print('pos', g[0], 'n', len(g), 'len %.0f' % L, 'reverse-of-true', sum(rev), 'arcs', arcs[:6], 'lens', np.round(net.length[arcs][:6]))
missing = [a for a in truth if a not in ps]
print('missing', len(missing), 'len %.0f' % net.length[missing].sum())
print('len pred', len(r), 'len truth', len(truth))
from pathsnap.geo import project_point_on_segments
for g in grp:
    if len(g) < 10: continue
    arcs = [r[i] for i in g]
    mid = (net.A[arcs] + net.B[arcs]) / 2
    dmin = [np.min(np.linalg.norm(pts - q, axis=1)) for q in mid]
    print('group at', g[0], 'arc->nearest GPS dist p50 %.0f max %.0f' % (np.median(dmin), np.max(dmin)))
ch0_last = chains[0][-1]; ch1_first = chains[1][0] if len(chains) > 1 else None
print('chain0 end', ch0_last, 'chain1 start', ch1_first)
if ch1_first:
    print('gap path len', len(net.path_arcs(int(net.dst[ch0_last[0]]), int(net.src[ch1_first[0]])) or []))
    print('is chain0 end arc in truth', ch0_last[0] in ts, 'chain1 start in truth', ch1_first[0] in ts, 'idx in truth', truth.index(ch0_last[0]) if ch0_last[0] in ts else None, truth.index(ch1_first[0]) if ch1_first[0] in ts else None)
