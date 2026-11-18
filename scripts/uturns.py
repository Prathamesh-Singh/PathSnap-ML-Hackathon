import sys; sys.path.insert(0, '.')
from pathsnap.io_utils import load_case, list_cases
tot = 0; n = 0; arcs = 0
for split in ['train', 'student_test']:
    for d in list_cases(f'data/Student folder/{split}'):
        c = load_case(d); A = c['arcs']; r = c['route']
        u = sum(1 for x, y in zip(r[:-1], r[1:]) if A[x, 0] == A[y, 1] and A[x, 1] == A[y, 0])
        if u: print(split, c['id'], 'uturns', u, 'of', len(r))
        tot += u; arcs += len(r); n += 1
print('total uturns', tot, 'routes', n, 'arcs', arcs)
