"""Extend the decoder search below the original grid's lowest emission weight
(the coordinate descent in hmm_eval.py picked the boundary value 0.5).
Training trips + out-of-fold probabilities only."""
import os

import numpy as np
import pandas as pd

from hmm_eval import CACHE, RESULTS, cfg_with, evaluate, load_table, split_probs

if __name__ == "__main__":
    train = load_table("train")
    oof = np.load(os.path.join(CACHE, "oof", "final_oof.npy"))
    pb, trips = split_probs(train, oof), sorted(train.trip.unique())
    log = pd.read_csv(os.path.join(RESULTS, "decoder_tuning_log.csv"))
    rows = []
    for w in (0.25, 0.1):
        dec = {"emission_weight": w, "beta": 10.0, "uturn_penalty": 4.0, "back_tol": 40.0}
        r = evaluate("train", pb, cfg_with(dec), "hmm", trips)
        rows.append({**dec, "train_route_f1": r.f1.mean(), "train_rmf": r.rmf.mean(),
                     "train_pointwise_top1": r.pointwise_top1.mean(), "secs": np.nan})
        print(rows[-1], flush=True)
    pd.concat([log, pd.DataFrame(rows)], ignore_index=True).to_csv(
        os.path.join(RESULTS, "decoder_tuning_log.csv"), index=False)
    print("best so far (grid):", log.loc[log.train_route_f1.idxmax()].to_dict())
