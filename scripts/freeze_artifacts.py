"""Record SHA-256 fingerprints of every model/config file (run before evaluator-test evaluation)."""
import glob
import hashlib
import json
import os
import sys
import time

out = sys.argv[1]
files = sorted(glob.glob("artifacts/*.json") + glob.glob("artifacts/*.joblib")
               + glob.glob("artifacts/models/*.joblib") + ["configs/pipeline.json"])
h = {f.replace(os.sep, "/"): hashlib.sha256(open(f, "rb").read()).hexdigest() for f in files}
rec = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
       "note": "Fingerprints of every model/config file taken BEFORE the first run on the evaluator-testing "
               "trips. Nothing was retrained, tuned or re-selected afterwards.",
       "sha256": h}
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(rec, open(out, "w"), indent=2)
for k, v in h.items():
    print(v[:16], k)
