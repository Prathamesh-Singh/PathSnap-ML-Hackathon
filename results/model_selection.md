# Model selection (recorded before Bayesian optimisation)

Primary metric: observation-level Top-1 accuracy (does the highest-scoring candidate equal the
reference arc?). It is what the HMM needs from the classifier: the right road ranked first.
Secondary: log loss, because the HMM uses the probabilities themselves as emissions.

| model | CV macro F1 | CV Top-1 | CV log loss | student-test Top-1 | student-test log loss | fit s/fold |
|---|---|---|---|---|---|---|
| LightGBM | 0.9760 | 0.9674 | 0.0340 | 0.9536 | 0.0583 | 5.0 |
| HistGradientBoosting | 0.9748 | 0.9648 | 0.0374 | 0.9630 | 0.0372 | 2.3 |
| RandomForest | 0.9738 | 0.9624 | 0.0384 | 0.9612 | 0.0387 | - |
| MLP | 0.9710 | 0.9607 | 0.0423 | 0.9601 | 0.0399 | - |

Chosen: **HistGradientBoosting**.
- CV Top-1 is 0.0026 below LightGBM, well inside the fold-to-fold spread (macro-F1 std ~0.009).
- On the student-testing trips it is >= LightGBM on 9 of 10 trips and has the best Top-1 and
  log loss of all models; LightGBM's larger default trees (63 leaves, 400 trees) generalise worse.
- Half the training time of LightGBM, fast inference (0.05 s for 38.8k candidates).
Bayesian optimisation is run on HistGradientBoosting only, objective = CV Top-1 on training trips.

Note (added later): the MLP row above is the reference run. Preksha B's rerun on Google Colab
(different library versions) gave CV Top-1 0.9618 / test Top-1 0.9588; the selection is unchanged.

## Why HistGradientBoosting generalises better than LightGBM (checked after selection)
Both are histogram-based gradient-boosted trees; the difference is regularisation, not the algorithm.

| | HGB untuned | HGB tuned (final) | LightGBM |
|---|---|---|---|
| max trees allowed | 300 | 650 | 400 |
| trees actually built | 86 | 206 | 400 |
| max leaves per tree | 31 | 46 | 63 |
| min samples per leaf | 20 | 136 | 20 |
| early stopping | auto (on: >10k rows, 10 % random hold-out, patience 10) | on | off |

LightGBM used its full capacity and overfit (Top-1: train 0.981 / CV 0.967 / student-test 0.954, test log loss
0.058). HGB's automatic early stopping and smaller leaves kept it close to its CV score on unseen trips
(0.963). Caveat: HGB's early-stopping hold-out is a random row split, not trip-grouped.
A LightGBM with early stopping and stronger regularisation would likely close the gap.
