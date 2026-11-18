# PathSnap — map matching & route reconstruction

UE24CS352A Machine Learning hackathon, PES University (RR campus).

Given a noisy GPS track and the road network around it, PathSnap predicts the
sequence of road arcs the vehicle actually drove. A classifier scores the
candidate road segments near every GPS observation, and an HMM decoded with
Viterbi turns those scores into one connected route.

```
trip files ──► candidate generation ──► feature engineering ──► classifier (P(candidate is the true road))
                                                                         │
            route (arc ids) ◄── shortest-path stitching ◄── Viterbi ◄────┘ (emissions)
                                                     ▲
                               transitions: road-network distance vs. straight-line distance
```

## Repository layout

| Folder | What is in it |
|---|---|
| `pathsnap/` | the library: file IO, projection, road graph + Dijkstra, candidate features & labels, HMM/Viterbi decoder, metrics, inference pipeline |
| `experiments/` | numbered steps of the ML workflow (EDA → dataset → models → Bayesian optimisation → HMM → plots → Folium maps) |
| `configs/` | candidate-generation settings |
| `artifacts/` | final tuned model (`final_model.joblib`) and `final_config.json` used by the API |
| `api/` | FastAPI backend (Laptop 1) |
| `ui/` | evaluator web UI (Laptop 2) |
| `results/` | every CSV / JSON log, EDA outputs, figures and Folium maps produced by the scripts |
| `tune.py`, `evaluate.py`, `predict.py`, `visualize.py` | geometric-HMM baseline: tuning, evaluation, batch prediction, quick plots |

## Setup

Python 3.11+ (developed on 3.14, Windows 11).

```bash
pip install -r requirements.txt
```

Put the dataset so that these folders exist (the data is not committed):

```
data/Student folder/train/<trip>/<trip>.{nodes,arcs,track,route}
data/Student folder/student_test/<trip>/...
```

## Reproducing the experiments

Run from the repository root, in this order:

```bash
python experiments/eda.py                                   # results/eda/
python experiments/build_dataset.py                        # cache/ (candidate tables)
python experiments/feature_importance.py                    # results/feature_importance.csv (permutation importance)
python experiments/feature_ablation.py                       # results/feature_ablation.csv (feature-group ablation)
python experiments/train_models.py                          # results/model_comparison.csv
python experiments/bayes_opt.py --model HistGradientBoosting --trials 30   # results/bayes_opt_*.{csv,json}, artifacts/final_model.joblib
python experiments/hmm_eval.py                              # results/hmm_comparison_*.csv, artifacts/final_config.json
python experiments/plots.py                                 # results/figures/
python experiments/folium_maps.py --split student_test --trips 00000052 00000056
python experiments/folium_maps.py --split train --trips 00000003 --eda
```

Geometric HMM baseline (no classifier):

```bash
python tune.py "data/Student folder/train" --out params.json
python evaluate.py "data/Student folder/student_test" --params params.json
```

All random seeds are fixed to 42 (`experiments/common.py`).

## Deployment (two laptops)

**Laptop 1 — API.** Copy the evaluation trips into a folder (one sub-folder per
trip, same file layout as the dataset), then:

```bash
set PATHSNAP_TRIPS_DIR=D:\path\to\evaluation_trips      (Windows cmd)
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Every request is logged in this console (trip id, time taken, route metrics).
Find the laptop's LAN address with `ipconfig` and allow port 8000 through the
firewall if Windows asks.

**Laptop 2 — UI.**

```bash
python ui/serve.py
```

Open http://localhost:5500, enter `http://<laptop-1-ip>:8000`, press
*Load trips*, click a trip. The map uses OpenStreetMap tiles when internet is
available and falls back to a plain drawing when it is not.

API endpoints: `GET /health`, `GET /trips`, `POST /predict/{trip_id}`.
The reference route is only read on the server *after* prediction to compute
the metrics; it is never given to the model and never sent to the UI.

## Results

All numbers below come from the files in `results/` (seed 42).

**Candidate classifiers** — 5-fold GroupKFold on the 80 training trips, then the 10 student-testing trips
(`results/model_comparison.csv`, one JSON per model in `results/models/`):

| Model | Family | CV macro F1 | CV Top-1 | CV log loss | Test Top-1 | Test macro F1 | Test log loss |
|---|---|---|---|---|---|---|---|
| Logistic Regression | linear | 0.9525 | 0.9525 | 0.0723 | 0.9491 | 0.9561 | 0.0654 |
| Gaussian NB | probabilistic | 0.9094 | 0.9444 | 0.2913 | 0.9463 | 0.9124 | 0.2831 |
| KNN (k=31) | instance-based | 0.9594 | 0.9509 | 0.0688 | 0.9494 | 0.9597 | 0.0679 |
| Random Forest | bagging | 0.9738 | 0.9624 | 0.0384 | 0.9612 | 0.9723 | 0.0387 |
| HistGradientBoosting | boosting | 0.9748 | 0.9648 | 0.0374 | 0.9630 | 0.9723 | 0.0372 |
| MLP (64, 32) | neural network | 0.9710 | 0.9607 | 0.0423 | 0.9601 | 0.9714 | 0.0399 |
| LightGBM | boosting | 0.9760 | 0.9674 | 0.0340 | 0.9536 | 0.9706 | 0.0583 |

All rows were produced with the same code and seed on one environment (Windows 11, Python 3.14,
scikit-learn 1.9.1; LR and GNB were also reproduced exactly on a second laptop). An earlier MLP run on
Google Colab (scikit-learn 1.6.1) gave CV Top-1 0.9618 / test Top-1 0.9588 - neural-network training
differs slightly across library/BLAS versions.

**Validation curves** (`experiments/sweep_model.py`, all 80 training trips, `results/sweeps/`,
`results/figures/sweep_<model>.png`): one key hyperparameter per model, training vs CV score. At each
model's default setting the curve reproduces the CV score in the table above. Bias / variance of every
model: `results/bias_variance.csv` (`experiments/bias_variance.py`).

Selected: **HistGradientBoosting** (reasoning in `results/model_selection.md`).

**Bayesian optimisation** (Optuna TPE, 30 trials, objective = CV Top-1; `results/bayes_opt_summary.json`):
CV Top-1 0.9648 → 0.9670, student-test Top-1 0.9630 → 0.9657, student-test log loss 0.0372 → 0.0349.

**Route reconstruction** on the student-testing trips (`results/hmm_comparison_summary.csv`):

| Method | Pointwise Top-1 | Route F1 | Route mismatch fraction |
|---|---|---|---|
| classifier only (argmax per observation, stitched) | 0.961 | 0.762 | 1.191 |
| geometric HMM (distance emissions, no classifier) | 0.965 | 0.994 | 0.013 |
| **classifier + HMM (final)** | **0.986** | 0.993 | 0.015 |

The HMM is what makes the route usable; the classifier mainly improves which road each GPS
observation is assigned to. At route level the two HMM variants are on par.

**Features**: heading difference dominates (removing it drops CV Top-1 from 0.965 to 0.799;
`results/feature_ablation.csv`, `results/feature_importance.csv`).

**Evaluator-testing trips** (run once, after everything was frozen - see `results/evaluator_test/README.md`):
route F1 0.9955, observation Top-1 0.9893 for the final system. These trips were not used for any
selection, tuning or threshold.

**Report**: `report/build_report.py` builds the technical report from the result files.

## Team

| Name | SRN | Section |
|---|---|---|
| Diya R Gowda | PES1UG24CS159 | C |
| E S Sukruth | PES1UG24CS160 | C |
| Epari Subhransi | PES1UG24CS161 | C |
| Prathamesh Kumar Singh | PES1UG24CS341 | F |
| Preksha B | PES1UG24CS342 | F |

## AI assistance

Parts of the code in this repository were written with the help of an AI
coding assistant. All experiments were run by the team on the
provided data, and every number in `results/` comes from those runs.
