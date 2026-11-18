"""One-off: store the first full training run (results/model_comparison.csv, written
before per-model JSON files existed) as results/models/<model>.json so it merges
with the team's runs. Existing JSON files from team members are not overwritten."""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
df = pd.read_csv(os.path.join(RES, "model_comparison.csv"))
hist_path = os.path.join(RES, "training_histories.json")
hist = json.load(open(hist_path)) if os.path.exists(hist_path) else {}
os.makedirs(os.path.join(RES, "models"), exist_ok=True)
for _, r in df.iterrows():
    out = os.path.join(RES, "models", f"{r['model']}.json")
    row = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.to_dict().items()}
    if os.path.exists(out) and json.load(open(out)).get("member"):
        print("keep", out)
        continue
    row["member"] = "reference run (E S Sukruth, all models)"
    if r["model"] in hist:
        row["history"] = hist[r["model"]]
    json.dump(row, open(out, "w"), indent=2, default=float)
    print("wrote", out)
