# Evaluator-testing trips — final evaluation (post-selection)

These results were computed **once, after** model selection, Bayesian optimisation and decoder tuning were
finished. Every model/config file was fingerprinted (SHA-256) before the run (`frozen_artifacts.json`) and
verified unchanged afterwards. **Nothing was trained, tuned or selected using these trips**; the 7
classifiers are the saved models fitted on the 80 training trips.

| File | Contents |
|---|---|
| `classifiers.csv` | candidate-level metrics of the 7 saved classifiers + the tuned final model |
| `pipeline_per_trip.csv`, `pipeline_summary.csv` | route-level results of the final system vs classifier-only |
| `candidates_per_trip.csv`, `summary.json` | observation counts, candidate miss rate |
| `api_log.txt` | live API run on these 10 trips (same numbers as the offline run) |
| `run_log.txt` | console output of `experiments/evaluate_evaluator_test.py` |
| `frozen_artifacts.json` | fingerprints taken before the run |
| `MLP_Preksha_B.joblib` | Preksha B's MLP (trained on Colab, scikit-learn 1.6.1) used for the MLP row |

## Update (same day): MLP row
The MLP row in `classifiers.csv` now uses the team's local MLP run (`artifacts/models/MLP.joblib`, trained on the
80 training trips only, seed 42, the same environment as all other models). Retraining it reproduced the file
fingerprinted in `frozen_artifacts.json` byte for byte, and all ten fingerprints were re-verified as unchanged.
The earlier run used Preksha B's Colab model (scikit-learn 1.6.1); its result record is kept in
`MLP_Preksha_B_colab_run.json`. Only scoring was repeated; nothing was trained, tuned or selected on these trips.
