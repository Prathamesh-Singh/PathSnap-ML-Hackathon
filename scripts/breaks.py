import sys; sys.path.insert(0, '.')
import numpy as np
from pathsnap.io_utils import load_case
from pathsnap.matcher import HMMMatcher, Params
from pathsnap.geo import project_point_on_segments
c = load_case(sys.argv[1]); m = HMMMatcher(Params())
net, pts = m.prepare(c)
truth = c['route']; A = net.A[truth]; B = net.B[truth]
idx = m._downsample(pts)
tm = c['track'][:, 2]
# deviation from true route for kept points + time gaps
dev = np.array([project_point_on_segments(pts[i][None], A, B)[0].min() for i in idx])
dt = np.diff(tm[idx])
print('kept', len(idx), 'dev>60:', (dev > 60).sum(), 'longest run dev>60:', max((len(s) for s in ''.join('1' if d > 60 else '0' for d in dev).split('0')), default=0))
print('time gaps>30s at kept idx:', [(int(idx[i]), float(dt[i]), round(float(np.linalg.norm(pts[idx[i+1]]-pts[idx[i]])))) for i in np.where(dt > 30)[0]][:15])
bad = np.where(dev > 60)[0]
print('first bad kept positions', bad[:40])
print('dev around them', np.round(dev[bad[:40]]))
