import os, re, sys, glob
import numpy as np
root = sys.argv[1]
num = re.compile(r'^[-+0-9.eE\t ]*$')
bad = 0
for f in glob.glob(os.path.join(root, '*', '*', '*')):
    with open(f, 'rb') as fh:
        raw = fh.read()
    if b'\r' in raw: print('CRLF', f)
    for i, line in enumerate(raw.decode('utf-8', 'replace').split('\n')):
        if not num.match(line):
            bad += 1
            if bad < 30: print('ODD', f, i, repr(line[:200]))
print('odd lines total', bad)
for split in ['train', 'student_test']:
    dirs = sorted(glob.glob(os.path.join(root, split, '*')))
    st = []
    for d in dirs:
        n = os.path.basename(d)
        tr = np.loadtxt(os.path.join(d, n + '.track'), ndmin=2)
        rt = np.loadtxt(os.path.join(d, n + '.route'), dtype=int, ndmin=1)
        ar = np.loadtxt(os.path.join(d, n + '.arcs'), dtype=int, ndmin=2)
        nd = np.loadtxt(os.path.join(d, n + '.nodes'), ndmin=2)
        dt = np.diff(tr[:, 2])
        # route continuity check
        cont = np.mean(ar[rt[:-1], 1] == ar[rt[1:], 0]) if len(rt) > 1 else 1
        st.append((n, len(tr), len(rt), len(ar), len(nd), dt.min(), np.median(dt), dt.max(), cont, rt.max() < len(ar)))
    print(split, len(dirs))
    for s in st[:12]: print('  ', s)
    print('  route continuity mean', np.mean([s[8] for s in st]))
