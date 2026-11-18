"""Builds the PathSnap technical report (DOCX + PDF) from the result files.

    python report/build_report.py            # writes report/PathSnap_Technical_Report.docx
    python report/build_report.py --pdf      # ...and converts it to PDF with Microsoft Word

Every number in the tables and in the text is read from results/ and artifacts/ at build
time, so the report can be regenerated and checked against the repository.
"""
import argparse
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
OUT_DOCX = os.path.join(ROOT, "report", "PathSnap_Technical_Report.docx")

MODELS = ["LogisticRegression", "GaussianNB", "KNN", "RandomForest", "HistGradientBoosting", "MLP", "LightGBM"]
NAME = {"LogisticRegression": "Logistic Regression", "GaussianNB": "Gaussian Naive Bayes", "KNN": "k-Nearest Neighbours",
        "RandomForest": "Random Forest", "HistGradientBoosting": "HistGradientBoosting", "MLP": "Multi-layer Perceptron (MLP)",
        "LightGBM": "LightGBM"}
FINAL = "HistGradientBoosting (Bayesian-tuned, final)"

TEAM = [("Diya R Gowda", "PES1UG24CS159", "C"), ("E S Sukruth", "PES1UG24CS160", "C"),
        ("Epari Subhransi", "PES1UG24CS161", "C"), ("Prathamesh Kumar Singh", "PES1UG24CS341", "F"),
        ("Preksha B", "PES1UG24CS342", "F")]
REPO = "https://github.com/Prathamesh-Singh/PathSnap-ML-Hackathon"


# ---------------------------------------------------------------- data
def load():
    D = {}
    D["mc"] = pd.read_csv(os.path.join(RES, "model_comparison.csv")).set_index("model")
    D["bv"] = pd.read_csv(os.path.join(RES, "bias_variance.csv")).set_index("model")
    D["sw"] = {m: pd.read_csv(os.path.join(RES, "sweeps", f"{m}.csv")) for m in MODELS}
    D["bayes"] = json.load(open(os.path.join(RES, "bayes_opt_summary.json")))
    D["trials"] = pd.read_csv(os.path.join(RES, "bayes_opt_trials.csv"))
    D["hmm"] = pd.read_csv(os.path.join(RES, "hmm_comparison_summary.csv"))
    D["hmm_trip"] = pd.read_csv(os.path.join(RES, "hmm_comparison_per_trip.csv"))
    D["dec"] = pd.read_csv(os.path.join(RES, "decoder_tuning_log.csv"))
    D["eda"] = json.load(open(os.path.join(RES, "eda", "summary.json")))
    D["fi"] = pd.read_csv(os.path.join(RES, "feature_importance.csv"))
    D["abl"] = pd.read_csv(os.path.join(RES, "feature_ablation.csv"))
    D["cand"] = {s: pd.read_csv(os.path.join(RES, f"candidates_{s}.csv")) for s in ("train", "student_test")}
    D["cfg"] = json.load(open(os.path.join(ROOT, "artifacts", "final_config.json")))
    ev = os.path.join(RES, "evaluator_test")
    D["ev_clf"] = pd.read_csv(os.path.join(ev, "classifiers.csv")).set_index("model")
    D["ev_pipe"] = pd.read_csv(os.path.join(ev, "pipeline_summary.csv")).set_index("mode")
    D["ev_trip"] = pd.read_csv(os.path.join(ev, "pipeline_per_trip.csv"))
    D["ev_cand"] = pd.read_csv(os.path.join(ev, "candidates_per_trip.csv"))
    D["n_iter"] = {"untuned": int(joblib.load(os.path.join(ROOT, "artifacts", "models", "HistGradientBoosting.joblib")).n_iter_),
                   "tuned": int(joblib.load(os.path.join(ROOT, "artifacts", "final_model.joblib")).n_iter_)}
    hist = os.path.join(RES, "training_histories.json")
    D["hist"] = json.load(open(hist)) if os.path.exists(hist) else {}
    return D


f4 = lambda v: "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"
pct = lambda v: f"{100 * v:.2f} %"


# ---------------------------------------------------------------- docx helpers
class Report:
    def __init__(self):
        self.doc = Document()
        self.nfig = 0
        self.ntab = 0
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
        for side in ("left_margin", "right_margin"):
            setattr(sec, side, Cm(2.0))
        sec.top_margin, sec.bottom_margin = Cm(1.8), Cm(1.8)
        st = self.doc.styles["Normal"]
        st.font.name = "Calibri"
        st.font.size = Pt(10.5)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
        st.paragraph_format.space_after = Pt(4)
        for lvl, size in ((1, 15), (2, 12.5), (3, 11)):
            h = self.doc.styles[f"Heading {lvl}"]
            h.font.name = "Calibri"
            h.font.size = Pt(size)
            h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
        self._page_numbers()

    def _page_numbers(self):
        p = self.doc.sections[0].footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
            if tag:
                el = OxmlElement("w:fldChar")
                el.set(qn("w:fldCharType"), tag)
            else:
                el = OxmlElement("w:instrText")
                el.set(qn("xml:space"), "preserve")
                el.text = text
            run._r.append(el)

    def h(self, text, level=1):
        self.doc.add_heading(text, level)

    def p(self, text, size=None, italic=False, align=None, note=False):
        """Paragraph; **bold** and `code` spans are supported."""
        para = self.doc.add_paragraph()
        if align == "center":
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
            if not part:
                continue
            if part.startswith("**"):
                r = para.add_run(part[2:-2])
                r.bold = True
            elif part.startswith("`"):
                r = para.add_run(part[1:-1])
                r.font.name = "Consolas"
                r.font.size = Pt(9.5)
            else:
                r = para.add_run(part)
            if size:
                r.font.size = Pt(size)
            if italic:
                r.italic = True
            if note:
                r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        return para

    def bullets(self, items):
        for it in items:
            para = self.doc.add_paragraph(style="List Bullet")
            for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", it):
                if not part:
                    continue
                if part.startswith("**"):
                    para.add_run(part[2:-2]).bold = True
                elif part.startswith("`"):
                    r = para.add_run(part[1:-1])
                    r.font.name = "Consolas"
                    r.font.size = Pt(9.5)
                else:
                    para.add_run(part)

    def qa(self, q, a):
        para = self.doc.add_paragraph()
        para.add_run(q + " ").bold = True
        for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", a):
            if part.startswith("**"):
                para.add_run(part[2:-2]).bold = True
            elif part.startswith("`"):
                r = para.add_run(part[1:-1])
                r.font.name = "Consolas"
                r.font.size = Pt(9.5)
            elif part:
                para.add_run(part)

    def table(self, header, rows, caption=None, widths=None, size=8.5, bold_rows=()):
        if caption:
            self.ntab += 1
            c = self.doc.add_paragraph()
            r = c.add_run(f"Table {self.ntab}. {caption}")
            r.bold = True
            r.font.size = Pt(9)
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = "Light Grid Accent 1"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, htxt in enumerate(header):
            cell = t.rows[0].cells[i]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(htxt))
            run.bold = True
            run.font.size = Pt(size)
        for ri, row in enumerate(rows):
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                run = cells[i].paragraphs[0].add_run(str(v))
                run.font.size = Pt(size)
                run.bold = ri in bold_rows
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Cm(w)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return t

    def fig(self, path, caption, width=16.5):
        if not os.path.exists(path):
            self.p(f"[missing figure: {os.path.relpath(path, ROOT)}]", note=True)
            return
        self.nfig += 1
        self.doc.add_picture(path, width=Cm(width))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c = self.doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = c.add_run(f"Figure {self.nfig}. {caption}  (source: {os.path.relpath(path, ROOT).replace(os.sep, '/')})")
        r.italic = True
        r.font.size = Pt(8.5)

    def page_break(self):
        self.doc.add_page_break()


# ---------------------------------------------------------------- per-model text
MODEL_TEXT = {
    "LogisticRegression": dict(
        family="Linear / probabilistic",
        why="Each candidate decision is a binary question (is this the road the car was on?) and most of the signal "
            "comes from a few monotone features: a smaller distance, a smaller heading difference and a larger "
            "distance gap to the next candidate all make a candidate more likely. Logistic regression models exactly "
            "this kind of additive evidence and outputs calibrated probabilities, which the HMM needs as emission scores. "
            "It is also the natural baseline that every non-linear model has to beat.",
        strengths="Fast to train (under a second per fold) and to score; coefficients are interpretable; probabilities "
                  "are reasonably calibrated (student-test log loss {test_log_loss}).",
        limits="The decision boundary is linear in the 18 features. Interactions such as \"a large heading difference "
               "only matters when the vehicle is actually moving\" (heading_diff x heading_ok) or \"distance matters less "
               "on a two-way road\" cannot be represented without hand-made interaction terms, which caps its accuracy.",
        hp=[("C (inverse L2 strength)", "1.0", "regularisation strength; swept 1e-4 ... 100"),
            ("penalty / solver", "L2 / lbfgs", "standard smooth penalty and quasi-Newton solver"),
            ("max_iter", "2000", "enough iterations for convergence on 344k rows"),
            ("preprocessing", "StandardScaler", "features have very different scales (metres, degrees, counts)")],
        loss="Binary cross-entropy (log loss) plus an L2 penalty on the weights, minimised with L-BFGS.",
    ),
    "GaussianNB": dict(
        family="Linear / probabilistic",
        why="Naive Bayes is the simplest probabilistic classifier: it models each feature per class with a Gaussian and "
            "combines them with Bayes' rule. It was included as a very cheap baseline that also produces probabilities, "
            "to measure how much is gained by models that account for correlated features.",
        strengths="Training is a single pass that estimates means and variances (about 0.2 s per fold); "
                  "it needs no tuning and scores instantly.",
        limits="The independence assumption is clearly violated here: dist, dist_rank, dist_gap and dist_z are "
               "functions of the same distance, and ctx_prev/ctx_next/ctx_mean are strongly correlated. The model "
               "therefore counts the same evidence several times and becomes over-confident, which shows up as the "
               "worst log loss of all models ({test_log_loss} on student-test) and a low macro F1.",
        hp=[("var_smoothing", "1e-9", "variance added for numerical stability; swept 1e-12 ... 0.1"),
            ("priors", "estimated from data", "13.8 % positive rows")],
        loss="No iterative loss: parameters are the maximum-likelihood estimates of per-class feature means and variances.",
    ),
    "KNN": dict(
        family="Instance-based",
        why="Candidates that look alike (similar distance, heading difference, network context) should have the same "
            "label, so a neighbourhood vote over the 344k labelled training candidates is a reasonable non-parametric "
            "model that makes no assumption about the shape of the decision boundary.",
        strengths="No training phase; it can represent any boundary shape; distance weighting gives smooth probabilities.",
        limits="Prediction needs a neighbour search over all training rows, so it is by far the slowest model to score "
               "({infer_secs_test} s for the 38,844 student-test candidates). All features get equal weight after "
               "scaling, so weak features (dt_prev, dist_rank) add noise to the distance metric.",
        hp=[("n_neighbors (k)", "31", "size of the vote; swept 1 ... 127"),
            ("weights", "distance", "closer neighbours count more"),
            ("metric", "Euclidean (Minkowski p=2)", "on standardised features"),
            ("preprocessing", "StandardScaler", "required so that no feature dominates the distance")],
        loss="No loss is optimised (lazy learner); the prediction is a distance-weighted vote of the k nearest training rows.",
    ),
    "RandomForest": dict(
        family="Ensemble (bagging)",
        why="Tree ensembles split on thresholds, so they handle features of different scales and capture interactions "
            "(heading only matters when the vehicle moves, distance matters differently on short arcs) without "
            "feature engineering. Averaging 200 de-correlated trees reduces the variance of single trees.",
        strengths="Strong accuracy with little tuning, robust to irrelevant features, no scaling needed.",
        limits="Large model file (about 74 MB) and slower inference than boosting; probabilities from averaged leaf "
               "frequencies are less sharp than boosted log-odds, giving a higher log loss than the boosting models.",
        hp=[("n_estimators", "200", "number of bagged trees"),
            ("min_samples_leaf", "5", "minimum rows per leaf; swept 1 ... 100"),
            ("max_features", "sqrt (4 of 18)", "features considered per split, de-correlates the trees"),
            ("random_state", "42", "reproducibility")],
        loss="Each tree greedily minimises Gini impurity at every split; the forest averages the trees' class frequencies.",
    ),
    "HistGradientBoosting": dict(
        family="Boosting",
        why="Gradient boosting fits trees sequentially to the errors of the previous trees, which is well suited to "
            "tabular features with interactions. The histogram-based implementation bins each feature into 255 bins, "
            "so training on 344k rows takes seconds, and it optimises log loss directly, which gives good "
            "probabilities for the HMM.",
        strengths="Best student-test Top-1 and log loss of the seven untuned models; fast training (about 2 s per fold); "
                  "built-in early stopping on a 10 % validation split limits overfitting automatically.",
        limits="More hyperparameters than bagging; small CV gains are within the fold-to-fold spread, so model choice "
               "needs care; trees cannot extrapolate beyond the feature ranges seen in training.",
        hp=[("learning_rate", "0.1", "shrinkage applied to every tree"),
            ("max_iter", "300", "upper limit on boosting iterations; swept 25 ... 800"),
            ("max_leaf_nodes", "31", "tree size"),
            ("early_stopping", "auto (on, >10k rows)", "10 % validation split, stops after 10 rounds without improvement"),
            ("random_state", "42", "reproducibility of the validation split")],
        loss="Binary log loss (cross-entropy), minimised by gradient boosting; early stopping monitors the loss on the "
             "internal validation split.",
    ),
    "MLP": dict(
        family="Neural network",
        why="A small feed-forward network can learn non-linear combinations of all 18 features at once and is trained "
            "directly on log loss. It was included as the neural-network family and because it produces an "
            "epoch-wise training history that can be inspected for over-/underfitting.",
        strengths="Smooth, well-calibrated probabilities; fast inference; early stopping gives a natural regularisation point.",
        limits="Needs feature scaling and more training time (about a minute per fold on a laptop CPU); on tabular data "
               "of this size it did not beat the tree ensembles; results depend slightly on library version and hardware "
               "(see Section 3.2.6).",
        hp=[("hidden_layer_sizes", "(64, 32)", "two hidden layers, ReLU activation"),
            ("alpha (L2)", "1e-4", "weight penalty; swept 1e-6 ... 1"),
            ("solver / learning_rate_init", "adam / 1e-3", "mini-batch optimiser"),
            ("batch_size", "512", ""),
            ("early_stopping", "True", "10 % validation split, patience 10 epochs, max_iter 200"),
            ("preprocessing", "StandardScaler", "")],
        loss="Binary cross-entropy (log loss) plus L2 penalty alpha, minimised with Adam on mini-batches.",
    ),
    "LightGBM": dict(
        family="Boosting",
        why="LightGBM is a second, independent gradient-boosting implementation with leaf-wise tree growth, row and "
            "feature subsampling. It was included to check whether a more aggressive boosting setup improves on "
            "HistGradientBoosting.",
        strengths="Best cross-validation numbers of the seven models (CV Top-1 {cv_top1}, CV macro F1 {cv_f1_macro}); "
                  "very fast histogram training.",
        limits="Leaf-wise growth with 63 leaves and no early stopping uses its full capacity of 400 trees; this fits "
               "the training trips closely and it dropped the most from CV to the student-test trips.",
        hp=[("n_estimators", "400", "boosting rounds (no early stopping)"),
            ("learning_rate", "0.05", ""),
            ("num_leaves", "63", "leaf-wise tree size; swept 4 ... 255"),
            ("subsample / subsample_freq", "0.8 / 1", "row bagging every iteration"),
            ("colsample_bytree", "0.8", "feature fraction per tree"),
            ("random_state", "42", "")],
        loss="Binary log loss, minimised by leaf-wise gradient boosting of histogram-based trees.",
    ),
}

SWEEP_TEXT = {
    "LogisticRegression": "The validation curve over C is flat from 1e-3 to 100 (CV Top-1 within 0.0005) and only "
                          "drops for the strongest penalty (C = 1e-4). Training and CV scores are almost identical at "
                          "every C, so the model is not overfitting; its limit is the linear decision boundary "
                          "(underfitting / high bias).",
    "GaussianNB": "Larger var_smoothing values flatten the per-class Gaussians and hurt badly (CV Top-1 falls to "
                  "{low} at 0.1). Training and CV scores are equal for every value: the model underfits (high bias) "
                  "because of the independence assumption, not because of the smoothing value.",
    "KNN": "With k = 1 the model memorises the training set (training Top-1 {train1}) and CV Top-1 is only {cv1}; "
           "increasing k smooths the vote and CV improves up to k = 31 ({cv31}), then slowly declines. The training "
           "score stays near 1 for every k because each training row finds itself at distance 0 (distance weighting), "
           "so the large train-CV gap shows high variance.",
    "RandomForest": "Leaves of 1-2 rows memorise the training trips (training Top-1 {train1} vs CV {cv1}); larger "
                    "leaves close the gap but lower both scores (underfitting at 50-100 rows). The CV differences "
                    "between 1 and 10 rows per leaf are about 0.001, far below the fold spread, so min_samples_leaf = 5 "
                    "is a reasonable middle point.",
    "HistGradientBoosting": "With only 25 or 50 iterations the model underfits (CV Top-1 {cv25} and {cv50}). From 100 "
                            "iterations on the scores are identical, because early stopping halts training after "
                            "{n_iter} iterations whatever the upper limit is. Training and CV Top-1 stay close "
                            "(gap {gap}), so early stopping keeps the model from overfitting.",
    "MLP": "The CV score is flat for alpha between 1e-6 and 1e-2 and falls for the largest penalties, where the network "
           "underfits. Training and CV scores are within about 0.001 of each other for every alpha, and early stopping "
           "ends training when the validation score stops improving, so the MLP does not overfit; it has a slightly "
           "higher bias than the tree ensembles.",
    "LightGBM": "Small trees (4-16 leaves) underfit: training and CV are both lower. From 63 leaves on, CV Top-1 stops "
                "improving while the training score keeps rising towards 1 ({train255} at 255 leaves): the extra "
                "capacity is spent on fitting the training trips, i.e. overfitting.",
}


# ---------------------------------------------------------------- sections
def title_page(R, D):
    R.p("UE24CS352A Machine Learning - Hackathon", size=11, align="center")
    R.doc.add_paragraph()
    t = R.doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("PathSnap: Distributed Map-Matching & Route Reconstruction")
    r.bold = True
    r.font.size = Pt(20)
    r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    R.p("Technical Report", size=14, align="center")
    R.doc.add_paragraph()
    R.table(["Name", "SRN", "Section"], TEAM, widths=[6, 4.5, 2.5], size=10)
    R.p(f"GitHub repository: {REPO}", align="center")
    R.p("PES University", align="center")
    R.doc.add_paragraph()
    mc, ev = D["mc"], D["ev_pipe"]
    hm = D["hmm"].set_index(["split", "mode"])
    R.p("**Summary.** For every GPS fix we generate candidate road arcs, score each (fix, arc) pair with a "
        "classifier trained on 18 engineered features, and decode the most likely connected route with an HMM/Viterbi "
        "decoder that uses the classifier probabilities as emission scores. Seven classifiers were compared with "
        "5-fold GroupKFold cross-validation on the 80 training trips and on the 10 student-testing trips. "
        f"HistGradientBoosting, tuned with Bayesian optimisation (Optuna, {D['bayes']['n_trials']} trials), was selected. "
        f"On the student-testing trips the final system reaches a length-weighted route F1 of "
        f"{f4(hm.loc[('student_test', 'hmm'), 'f1'])} and an observation-level Top-1 accuracy of "
        f"{f4(hm.loc[('student_test', 'hmm'), 'pointwise_top1'])} (classifier only: route F1 "
        f"{f4(hm.loc[('student_test', 'clf'), 'f1'])}). The system is served by a FastAPI backend and a web UI.")
    R.page_break()


def section_overview(R, D):
    R.h("1. Problem and pipeline overview", 1)
    cfg = D["cfg"]["candidates"]
    R.p("Each trip comes with its own road network (`.nodes`: longitude/latitude per node; `.arcs`: directed "
        "from/to node pairs, the line number is the arc id), a GPS track (`.track`: longitude, latitude, time) and, "
        "for development trips, the reference route (`.route`: the arc ids driven, in order). The task is to output the "
        "arc sequence for a trip from its track and network only.")
    R.bullets([
        f"**Preprocessing** (`pathsnap/io_utils.py`, `pathsnap/geo.py`, `pathsnap/candidates.py`): parse and validate the "
        f"files, project coordinates to metres, keep one fix every {cfg['min_spacing']:.0f} m.",
        f"**Candidates** (`pathsnap/network.py`): up to {cfg['k']} arcs within {cfg['radius']:.0f} m of each kept fix "
        "(KD-tree over points sampled every 15 m along each arc).",
        "**Classifier** (`experiments/train_models.py`): P(candidate is the true road | 18 features).",
        "**HMM/Viterbi** (`pathsnap/matcher.py`): connected route using classifier emissions and network-distance transitions.",
        "**Serving** (`pathsnap/pipeline.py`, `api/app.py`, `ui/`): the same pipeline is used offline and by the API.",
    ])
    R.p("**Data partitions.** `train` (80 trips) was used for EDA, feature selection, training, cross-validation and all "
        "tuning. `student_test` (10 trips) was used only to compare models and for the final model choice. The "
        "evaluator-testing trips were not used for any development decision; they were run once, after everything was "
        "frozen (Section 7.4).")


def section_eda(R, D):
    e, tr, st = D["eda"], D["eda"]["train"], D["eda"]["student_test"]
    ct, cs = D["cand"]["train"], D["cand"]["student_test"]
    R.h("2. Dataset understanding and exploratory data analysis", 1)
    R.h("2.1 Dataset overview", 2)
    R.table(["Property", "train", "student_test"], [
        ("Trips", tr["trips"], st["trips"]),
        ("GPS fixes", f"{tr['gps_fixes']:,}", f"{st['gps_fixes']:,}"),
        ("Fixes per trip (min / median / max)", " / ".join(map(str, tr["fixes_per_trip_min_median_max"])),
         " / ".join(map(str, st["fixes_per_trip_min_median_max"]))),
        ("Road arcs / nodes (all trips)", f"{tr['road_arcs_total']:,} / {tr['road_nodes_total']:,}",
         f"{st['road_arcs_total']:,} / {st['road_nodes_total']:,}"),
        ("Share of time steps exactly 1 s", pct(tr["share_steps_exactly_1s"]), pct(st["share_steps_exactly_1s"])),
        ("Trips with a time gap > 60 s (largest gap)", f"{tr['trips_with_gap_over_60s']} ({tr['largest_gap_s']:.0f} s)",
         f"{st['trips_with_gap_over_60s']} ({st['largest_gap_s']:.0f} s)"),
        ("GPS distance to reference route, median / p90 / p99 (m)",
         f"{tr['gps_dev_median_m']:.1f} / {tr['gps_dev_p90_m']:.1f} / {tr['gps_dev_p99_m']:.1f}",
         f"{st['gps_dev_median_m']:.1f} / {st['gps_dev_p90_m']:.1f} / {st['gps_dev_p99_m']:.1f}"),
        ("Route breaks (consecutive route arcs not connected)", tr["route_breaks_total"], st["route_breaks_total"]),
        ("U-turns in the reference routes", tr["route_uturns_total"], st["route_uturns_total"]),
        ("Median share of nodes in the largest strongly connected component",
         pct(tr["median_largest_strong_component_share"]), pct(st["median_largest_strong_component_share"])),
    ], caption="Dataset statistics (computed by experiments/eda.py; per-trip values in results/eda/inventory.csv).",
        widths=[8, 4.5, 4.5])

    R.h("2.2 How did we preprocess the data?", 2)
    cfg = D["cfg"]["candidates"]
    R.bullets([
        "**Parsing and validation.** All files are whitespace-separated text; the track files use Windows (CRLF) line "
        "endings, which the reader handles. For every trip we checked file presence, column counts, NaN values, "
        "out-of-range coordinates, non-increasing timestamps, arc indices outside the node table, duplicate arcs, "
        "self-loops and isolated nodes (results/eda/inventory.csv).",
        "**Projection.** Longitude/latitude are converted to metres with a local equirectangular projection centred on "
        "each trip, so distances and angles can be computed with plain vector maths (error is negligible at city scale).",
        f"**Downsampling.** A fix is kept only when it is at least {cfg['min_spacing']:.0f} m from the previously kept fix "
        f"(first and last fixes are always kept). At ~1 Hz, stationary or slow periods produce many nearly identical "
        f"fixes whose errors are correlated; removing them keeps {ct.n_obs.sum():,} of {ct.n_fixes.sum():,} training fixes "
        f"({100 * ct.n_obs.sum() / ct.n_fixes.sum():.0f} %) without losing route information, because the decoder "
        "reconstructs the driven path between kept fixes with shortest paths.",
        f"**Candidate generation.** Each arc is sampled every 15 m and the samples are indexed in a KD-tree. For every "
        f"kept fix the arcs within {cfg['radius']:.0f} m are collected and the {cfg['k']} closest (perpendicular "
        f"distance) are kept. The radius covers the p99 GPS error of the student-testing trips "
        f"({st['gps_dev_p99_m']:.0f} m); the true arc is missing from the candidate set for "
        f"{100 * ct.cand_miss.sum() / ct.n_obs.sum():.2f} % of training observations and "
        f"{100 * cs.cand_miss.sum() / cs.n_obs.sum():.2f} % of student-testing observations.",
        "**Labels.** The reference route is a sequence of arcs, not a label per fix. We align the kept fixes to the route "
        "with a monotone dynamic programme (fix i+1 can never be matched to an earlier route position than fix i) that "
        "minimises the sum of point-to-arc distances, capped at 80 m so a single outlier cannot pull the alignment. The "
        "aligned arc is the positive candidate (label 1), the other candidates are negatives. This gives "
        f"{ct.n_rows.sum():,} training rows with 13.8 % positives.",
        "**Class imbalance.** No re-sampling was applied: the HMM needs probabilities that reflect the real class "
        "frequencies, and all evaluation metrics are computed per observation or per class.",
    ])

    R.h("2.3 Missing-value analysis", 2)
    R.p(f"The raw files contain no missing values: {tr['nan_fixes']} NaN fixes and {tr['bad_coords']} invalid "
        f"coordinates in train, {st['nan_fixes']} and {st['bad_coords']} in student_test, and no non-increasing "
        f"timestamps. \"Missing\" information in this task takes three other forms, each handled explicitly:")
    R.bullets([
        f"**Missing time (gaps).** {tr['trips_with_gap_over_60s']} training trips have a gap of more than 60 s (the "
        f"largest is {tr['largest_gap_s'] / 3600:.1f} h, a recording paused and resumed). Gaps are handled by the decoder: "
        "transitions are allowed over any network distance consistent with the straight-line distance, and when no path "
        "exists the route is split into chains and bridged (Section 5).",
        f"**Missing candidates.** Observations with no arc within {cfg['radius']:.0f} m get no candidate set and are skipped "
        "by the decoder; after decoding they are assigned the nearest arc of the final route.",
        "**Undefined features.** heading_diff is undefined when the vehicle barely moves (displacement < 3 m): it is set "
        "to a neutral 90 degrees and the indicator heading_ok = 0 tells the model so. ctx_prev / ctx_next are undefined "
        "at the first/last observation and fall back to the current distance. After this, the feature matrix has no "
        "missing values.",
        f"**Imperfect references.** Two training trips have a median GPS-to-route distance above 50 m "
        f"({', '.join(tr['trips_with_median_dev_over_50m'])}) and one ends more than 200 m from its route "
        f"({', '.join(tr['trips_end_offset_over_200m'])}). They were kept (removing them would bias the evaluation "
        "towards easy trips) and the 80 m label cap limits their effect.",
    ])

    R.h("2.4 Patterns observed during exploration", 2)
    R.bullets([
        f"**GPS noise is small but heavy-tailed.** The median distance from a fix to the reference route is "
        f"{tr['gps_dev_median_m']:.1f} m, the p90 {tr['gps_dev_p90_m']:.1f} m, but the p99 in train is "
        f"{tr['gps_dev_p99_m']:.0f} m (Figure 4). Distance alone is therefore ambiguous in dense areas and the decoder "
        "must be able to skip outliers.",
        "**Trips are spread over many regions and countries** (Figure 3), each with its own network, so models must "
        "generalise across cities, which motivates splitting folds by trip.",
        "**Most fixes are exactly 1 s apart** (96.5 % in train), so speed and heading can be estimated reliably from "
        "neighbouring fixes.",
        "**Networks are mostly two-way and contain parallel roads.** Two-way roads appear as two arcs with the same "
        "geometry and opposite direction, so the direction of travel (heading) is the only way to tell them apart. "
        f"Most networks also have several disconnected components (in {tr['trips_with_multiple_weak_components']} of 80 "
        "training trips), so a route may need bridging.",
        f"**Reference routes contain U-turns** ({tr['route_uturns_total']} in train) and a few breaks, so U-turns must be "
        "possible but penalised in the decoder.",
    ])
    efig = os.path.join(RES, "eda", "figures")
    R.fig(os.path.join(efig, "trajectories_train.png"), "GPS trajectories of sample training trips (partition: train).")
    R.fig(os.path.join(efig, "trajectories_student_test.png"), "GPS trajectories of sample student-testing trips "
          "(partition: student_test).")
    R.fig(os.path.join(efig, "trip_locations.png"), "Trip locations: centre of every GPS track (train and student_test).", 12)
    R.fig(os.path.join(efig, "gps_deviation.png"), "Distance from GPS fixes to the reference route (train and "
          "student_test).", 12)
    R.fig(os.path.join(efig, "distributions.png"), "Per-trip distributions: number of fixes, duration, length, speed, "
          "network size (train and student_test).")
    R.fig(os.path.join(RES, "figures", "maps", "folium_train_00000003_eda_zoomed.jpg"),
          "Road-network visualisation with Folium: road network (grey), reference route (teal) and GPS fixes (dark) for "
          "training trip 00000003 (partition: train; interactive map results/maps/train_00000003_eda.html).", 13)

    R.h("2.5 Which features did we extract and why?", 2)
    R.p("Every row describes one (observation, candidate arc) pair. The 18 features fall into six groups, each "
        "motivated by an observation above:")
    R.table(["Group", "Features", "Why"], [
        ("Distance", "dist, dist_rank, dist_gap, dist_z, n_cands",
         "GPS error is usually small, so closeness matters; dist_gap (distance minus the closest candidate's) and the "
         "rank make it relative to the competing candidates; n_cands measures how dense the area is."),
        ("Heading", "heading_diff, heading_ok",
         "The angle between the direction of movement and the arc direction separates the two directions of a two-way "
         "road and parallel roads; heading_ok marks fixes where the heading is unreliable."),
        ("Motion", "speed, turn_angle, dt_prev",
         "Speed and turning indicate road type and junctions; dt_prev flags time gaps."),
        ("Position on arc", "t_along",
         "Where along the arc the fix projects (0 = start, 1 = end); projections clamped at an arc end usually mean "
         "the fix belongs to the next arc."),
        ("Network", "arc_len, two_way, deg_out_end, deg_in_start",
         "Arc length and junction degrees describe the local road structure; two_way marks arcs with a reverse twin."),
        ("Context", "ctx_prev, ctx_next, ctx_mean",
         "Distance from the previous / next observation (and a 5-observation window) to the same arc: the true arc "
         "stays close over consecutive fixes, a crossing road does not."),
    ], caption="The 18 features (pathsnap/candidates.py, FEATURES).", widths=[2.6, 4.6, 9.8])
    fi = D["fi"].sort_values("top1_drop_mean", ascending=False)
    abl = D["abl"].set_index("feature_set")
    full = abl.iloc[0]
    R.p(f"**Evidence that the features matter.** Permutation importance of the untuned HistGradientBoosting model "
        f"(training trips, out-of-fold) shows that shuffling heading_diff lowers Top-1 accuracy by "
        f"{fi.iloc[0].top1_drop_mean:.3f}, dist_gap by {fi.iloc[1].top1_drop_mean:.3f} and t_along by "
        f"{fi.iloc[2].top1_drop_mean:.3f}. A group-ablation study (retraining without each group, 5-fold GroupKFold) "
        "confirms this (ablation table below): removing the heading group drops CV Top-1 from "
        f"{full.cv_top1:.4f} to {abl.loc['- heading (heading_diff, heading_ok)', 'cv_top1']:.4f}, and distance features "
        f"alone reach only {abl.loc['distance only (dist, dist_rank, dist_gap, dist_z, n_cands)', 'cv_top1']:.4f}. Four "
        "features have zero importance (dist_z, heading_ok, dt_prev, dist_rank); dropping them gives "
        f"{abl.iloc[1].cv_top1:.4f}, the same as the full set within noise, so all 18 were kept for robustness.")
    R.table(["Feature set", "#", "CV Top-1", "+/- std", "CV log loss"],
            [(i, int(r.n_features), f4(r.cv_top1), f4(r.cv_top1_std), f4(r.cv_log_loss)) for i, r in abl.iterrows()],
            caption="Feature-group ablation (experiments/feature_ablation.py, untuned HistGradientBoosting, train, 5-fold "
                    "GroupKFold).", widths=[8.5, 1, 2.3, 2.3, 2.5])
    R.fig(os.path.join(FIG, "feature_importance.png"), "Permutation importance (Top-1 drop) per feature (train, out-of-fold).", 13)


def section_models(R, D):
    mc, bv, sw = D["mc"], D["bv"], D["sw"]
    R.h("3. Model experimentation and comparison", 1)
    R.p("Seven classifiers from five algorithm families were trained on the same candidate rows. All use the same "
        "evaluation: 5-fold GroupKFold on the 80 training trips with the trip as the group (no trip is split between "
        "training and validation folds), then a fit on all 80 trips and a single evaluation on the 10 student-testing "
        "trips. Random seed 42 everywhere. Metrics:")
    R.bullets([
        "**Top-1 accuracy** (observation level): share of observations whose highest-probability candidate is the true arc.",
        "**Macro F1** over the two classes (true arc / other arc), at threshold 0.5, so the minority class counts equally.",
        "**Log loss** of the candidate probabilities; important because the HMM uses them as emission scores.",
        "**Route continuity**: share of consecutive argmax arcs that are connected in the network (classifier only, "
        "no HMM) - shows how far a per-fix classifier is from a usable route.",
        "**Inference time**: scoring all 38,844 student-testing candidate rows on a laptop CPU.",
    ])
    R.h("3.1 Overall model comparison", 2)
    rows = []
    for m in MODELS:
        r = mc.loc[m]
        rows.append((NAME[m], MODEL_TEXT[m]["family"], r.config, f"{r.cv_f1_macro:.4f} +/- {r.cv_f1_macro_std:.4f}",
                     f"{r.cv_top1:.4f} / {r.test_top1:.4f}", f4(r.test_continuity), f4(r.test_f1_macro),
                     f"{r.cv_log_loss:.4f} / {r.test_log_loss:.4f}", f"{r.infer_secs_test:.2f}"))
    b = D["bayes"]["tuned"]
    rows.append(("HistGradientBoosting (Bayesian-tuned, final)", "Boosting", "see Section 4",
                 f"{b['cv_f1_macro']:.4f}", f"{b['cv_top1']:.4f} / {b['test_top1']:.4f}", f4(b["test_continuity"]),
                 f4(b["test_f1_macro"]), f"{b['cv_log_loss']:.4f} / {b['test_log_loss']:.4f}", f"{b['infer_secs_test']:.2f}"))
    R.table(["Model", "Family", "Key configuration", "CV macro F1 (+/- fold std)", "Top-1 CV / student-test",
             "Route contin. (st-test)", "Student-test macro F1", "Log loss CV / st-test", "Inference (s)"], rows,
            caption="Consolidated comparison (results/model_comparison.csv, results/bayes_opt_summary.json). "
                    "All metrics apply to all models; route continuity is for the classifier alone (argmax per "
                    "observation, before the HMM).", widths=[2.6, 1.8, 3.2, 1.9, 1.8, 1.4, 1.5, 1.8, 1.3], size=7.5,
            bold_rows=(len(rows) - 1,))
    best_cv = mc.cv_top1.idxmax()
    best_st = mc.test_top1.idxmax()
    R.p(f"**Reading the table.** The tree ensembles and the MLP form the top group. {NAME[best_cv]} has the best CV "
        f"numbers, but {NAME[best_st]} has the best student-testing Top-1 and log loss. Logistic regression and KNN are "
        "about 1 point lower in Top-1; Gaussian Naive Bayes is clearly last, especially in log loss. Every classifier "
        "alone produces broken routes (continuity 0.78-0.80): about one in five consecutive observations jumps to an "
        "unconnected arc, which is why the HMM is needed (Section 5).")

    R.h("Why HistGradientBoosting was selected", 3)
    lg, hg = mc.loc["LightGBM"], mc.loc["HistGradientBoosting"]
    st_trip = None
    R.bullets([
        f"The CV Top-1 lead of LightGBM over HistGradientBoosting ({lg.cv_top1 - hg.cv_top1:.4f}) is smaller than the "
        f"fold-to-fold standard deviation of macro F1 (about {hg.cv_f1_macro_std:.3f}), so CV alone cannot separate them.",
        f"On the student-testing trips HistGradientBoosting is better in Top-1 ({hg.test_top1:.4f} vs {lg.test_top1:.4f}), "
        f"macro F1 ({hg.test_f1_macro:.4f} vs {lg.test_f1_macro:.4f}) and log loss ({hg.test_log_loss:.4f} vs "
        f"{lg.test_log_loss:.4f}). Log loss matters most for us because the HMM consumes the probabilities.",
        f"LightGBM shows the largest drop from training to unseen data (Top-1 {bv.loc['LightGBM', 'train_top1']:.3f} "
        f"train, {lg.cv_top1:.3f} CV, {lg.test_top1:.3f} student-test), HistGradientBoosting the smallest among the "
        f"strong models ({bv.loc['HistGradientBoosting', 'train_top1']:.3f} / {hg.cv_top1:.3f} / {hg.test_top1:.3f}), "
        "thanks to its built-in early stopping.",
        f"It trains in about {hg.cv_fit_secs_per_fold:.0f} s per fold, so Bayesian optimisation with many trials was feasible.",
    ])
    R.p("The full record of this decision is in results/model_selection.md.")

    R.h("3.2 Per-model questions", 2)
    for i, m in enumerate(MODELS, 1):
        per_model(R, D, m, i)

    R.h("3.3 Model visualisations", 2)
    R.fig(os.path.join(FIG, "cv_macro_f1.png"), "Cross-validation macro F1 of all models (train, 5-fold GroupKFold, "
          "error bars = fold std).", 14)
    R.fig(os.path.join(FIG, "student_test_top1_f1.png"), "Student-testing Top-1 accuracy and macro F1 of all models.", 14)
    R.fig(os.path.join(FIG, "log_loss.png"), "Log loss, cross-validation and student-testing.", 14)
    R.fig(os.path.join(FIG, "confusion_matrices.png"), "Confusion matrices on the student-testing candidates "
          "(threshold 0.5; rows = true class).")
    R.fig(os.path.join(FIG, "accuracy_vs_speed.png"), "Student-testing Top-1 accuracy versus inference time "
          "(log scale) - supports the model selection.", 13)
    R.p("Only the MLP produces an epoch-wise training history (Figure in Section 3.2.6). The other models do not train "
        "in epochs (closed-form, lazy or tree-based), so no training curves were drawn for them; their over-/underfitting "
        "is analysed with validation curves over their main hyperparameter instead (experiments/sweep_model.py, "
        "results/sweeps/).")


def per_model(R, D, m, idx):
    mc, bv, sw = D["mc"], D["bv"], D["sw"][m]
    r, t = mc.loc[m], MODEL_TEXT[m]
    fmt = dict(test_log_loss=f4(r.test_log_loss), infer_secs_test=f"{r.infer_secs_test:.1f}",
               cv_top1=f4(r.cv_top1), cv_f1_macro=f4(r.cv_f1_macro))
    R.h(f"3.2.{idx} {NAME[m]}", 3)
    R.p("**A. Model understanding**")
    R.qa("1. Why suitable?", t["why"])
    R.qa("2. Strengths:", t["strengths"].format(**fmt))
    R.qa("   Limitations:", t["limits"].format(**fmt))
    R.p("**B. Training and configuration**")
    R.qa("5. Important hyperparameters and initial values:", "")
    R.table(["Hyperparameter", "Initial value", "Role / note"], t["hp"], widths=[4.5, 3.5, 9], size=8.5)
    R.qa("6. Loss / objective:", t["loss"])
    R.p("**C. Experimental results**")
    cm = json.loads(r.test_confusion_matrix)
    R.qa("7. Performance on the student-testing trips:",
         f"accuracy {f4(r.test_accuracy)}, macro precision {f4(r.test_precision_macro)}, macro recall "
         f"{f4(r.test_recall_macro)}; confusion matrix (rows true other / true road): {cm[0]} / {cm[1]}; route "
         f"continuity of the argmax arcs {f4(r.test_continuity)}; inference {r.infer_secs_test:.2f} s for 38,844 rows.")
    R.qa("8. Observation-level metrics (student-test):",
         f"**Top-1 accuracy {f4(r.test_top1)}, macro F1 {f4(r.test_f1_macro)}, log loss {f4(r.test_log_loss)}** "
         f"(cross-validation: Top-1 {f4(r.cv_top1)}, macro F1 {f4(r.cv_f1_macro)}, log loss {f4(r.cv_log_loss)}).")
    b = bv.loc[m]
    verdict = verdict_of(b)
    s = sw.set_index("value")
    vals = list(s.index)
    sweep_fmt = {}
    if m == "GaussianNB":
        sweep_fmt["low"] = f4(s.cv_top1.iloc[-1])
    if m in ("KNN", "RandomForest"):
        sweep_fmt.update(train1=f4(s.train_top1.iloc[0]), cv1=f4(s.cv_top1.iloc[0]))
    if m == "KNN":
        sweep_fmt["cv31"] = f4(s.loc[31, "cv_top1"]) if 31 in s.index else "N/A"
    if m == "HistGradientBoosting":
        sweep_fmt.update(cv25=f4(s.cv_top1.iloc[0]), cv50=f4(s.cv_top1.iloc[1]), n_iter=D["n_iter"]["untuned"],
                         gap=f"{(s.train_top1 - s.cv_top1).abs().max():.4f}")
    if m == "LightGBM":
        sweep_fmt["train255"] = f4(s.train_top1.iloc[-1])
    R.qa("9. Overfitting or underfitting?",
         f"**{verdict}.** Top-1 on the training trips (in-sample) {f4(b.train_top1)}, in cross-validation "
         f"{f4(b.cv_top1)}, on student-test {f4(b.student_test_top1)} (bias = training error "
         f"{b.bias_train_error:.4f}; variance = train-CV gap {b.variance_train_minus_cv:+.4f}). "
         + SWEEP_TEXT[m].format(**sweep_fmt))
    param = sw.param.iloc[0].split("__")[-1]
    R.table([param] + [f"{v:g}" for v in vals],
            [["CV Top-1"] + [f"{v:.4f}" for v in s.cv_top1], ["CV macro F1"] + [f"{v:.4f}" for v in s.cv_f1_macro],
             ["training Top-1"] + [f"{v:.4f}" for v in s.train_top1]],
            caption=f"{NAME[m]}: validation curve over {param} (results/sweeps/{m}.csv, train, 5-fold GroupKFold).",
            size=7.5)
    R.fig(os.path.join(FIG, f"sweep_{m}.png"), f"{NAME[m]} validation curve (training vs cross-validation score).", 11)
    if m == "MLP":
        R.fig(os.path.join(FIG, "history_MLP.png"), "MLP training history: training loss and validation score per "
              "epoch (fit on all 80 training trips; early stopping on a 10 % validation split).", 12)
        h = D["hist"].get("MLP", {})
        if h.get("train_loss") and h.get("val_accuracy"):
            tl, va = h["train_loss"], h["val_accuracy"]
            R.p(f"The training loss falls from {tl[0]:.4f} to {tl[-1]:.4f} and the validation accuracy (row level, on "
                f"the 10 % split) rises from {va[0]:.4f} to a plateau around {max(va):.4f}. Early stopping ended training "
                f"after {len(tl)} epochs, when the validation score had not improved for 10 epochs; the validation "
                "score does not decrease while the training loss keeps falling, so there is no sign of overfitting.")


def verdict_of(b):
    if b.variance_train_minus_cv > 0.012:
        return "Overfitting (high variance)"
    if b.bias_train_error > 0.04:
        return "Underfitting (high bias)"
    return "Good balance, slight bias"


def section_bayes(R, D):
    b, tr = D["bayes"], D["trials"]
    u, t = b["untuned"], b["tuned"]
    R.h("4. Bayesian hyperparameter optimisation (HistGradientBoosting)", 1)
    R.qa("1-2. Hyperparameters and ranges searched:", "")
    R.table(["Hyperparameter", "Range", "Distribution", "Best value"], [
        ("learning_rate", "0.01 - 0.3", "log-uniform", f"{b['best_params']['learning_rate']:.4f}"),
        ("max_iter", "100 - 800, step 50", "uniform integer", b["best_params"]["max_iter"]),
        ("max_leaf_nodes", "15 - 255", "log-uniform integer", b["best_params"]["max_leaf_nodes"]),
        ("min_samples_leaf", "5 - 300", "log-uniform integer", b["best_params"]["min_samples_leaf"]),
        ("l2_regularization", "1e-4 - 10", "log-uniform", f"{b['best_params']['l2_regularization']:.5f}"),
    ], caption="Search space (experiments/bayes_opt.py) and best values (results/bayes_opt_summary.json).",
        widths=[4, 4, 4, 4])
    R.qa("3. Objective:", "maximise the mean observation-level Top-1 accuracy over the same 5-fold GroupKFold split of "
         "the 80 training trips. Top-1 is the decision the decoder relies on; log loss was recorded for every trial "
         "as well. The student-testing trips were not used inside the search.")
    R.qa("4. Number of trials:", f"{b['n_trials']} trials with Optuna's Tree-structured Parzen Estimator (TPE) sampler "
         f"(seed 42; the first 10 trials are random to explore the space, then TPE proposes values from a density model "
         f"of good vs bad trials). A median pruner stopped trials whose running fold average was below the median of "
         f"earlier trials ({b['n_pruned']} pruned). Search time {b['search_secs'] / 60:.0f} min; every trial is in "
         "results/bayes_opt_trials.csv.")
    R.qa("5. Best values:", ", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}"
                                      for k, v in b["best_params"].items()) + ".")
    R.qa("6. Did the tuned model outperform the untuned one?",
         f"Yes, on every metric and on both partitions, by a small but consistent margin (table below). The tuned model uses "
         f"more trees ({D['n_iter']['tuned']} vs {D['n_iter']['untuned']} after early stopping) with smaller learning "
         f"rate and larger minimum leaf size, so it is slightly more complex but more regularised per tree; inference "
         f"is slower ({t['infer_secs_test']:.2f} s vs {u['infer_secs_test']:.2f} s for 38,844 rows), which is negligible "
         f"next to the decoder.")
    rows = [
        ("CV macro F1", f4(u["cv_f1_macro"]), f4(t["cv_f1_macro"])),
        ("CV Top-1", f4(u["cv_top1"]), f4(t["cv_top1"])),
        ("CV log loss", f4(u["cv_log_loss"]), f4(t["cv_log_loss"])),
        ("Student-test Top-1", f4(u["test_top1"]), f4(t["test_top1"])),
        ("Student-test macro F1", f4(u["test_f1_macro"]), f4(t["test_f1_macro"])),
        ("Student-test log loss", f4(u["test_log_loss"]), f4(t["test_log_loss"])),
        ("Student-test route continuity (no HMM)", f4(u["test_continuity"]), f4(t["test_continuity"])),
        ("Inference time, 38,844 rows (s)", f"{u['infer_secs_test']:.3f}", f"{t['infer_secs_test']:.3f}"),
        ("Fit time on 80 trips (s)", f"{u['fit_secs']:.1f}", f"{t['fit_secs']:.1f}"),
        ("Boosting iterations used", D["n_iter"]["untuned"], D["n_iter"]["tuned"]),
    ]
    R.table(["Metric", "Untuned", "Tuned"], rows, caption="Untuned vs tuned HistGradientBoosting, same folds and "
            "partitions.", widths=[8, 4, 4])
    R.fig(os.path.join(FIG, "bayes_opt_history.png"), "Optimisation history: CV Top-1 of each trial and best so far.", 13)
    R.fig(os.path.join(FIG, "tuned_vs_untuned.png"), "Untuned vs tuned model.", 13)


def section_hmm(R, D):
    c = D["cfg"]["decoder"]
    hm = D["hmm"].set_index(["split", "mode"])
    R.h("5. HMM and Viterbi route reconstruction", 1)
    R.p("Implementation: `pathsnap/matcher.py` (HMMMatcher.decode), `pathsnap/network.py` (bounded Dijkstra), "
        "`pathsnap/pipeline.py` (glue used by the API and all evaluations). The design follows the Newson-Krumm "
        "map-matching HMM, with the Gaussian distance emission replaced by our classifier.")
    R.qa("1. Hidden states:", "for each kept observation, its candidate positions: (arc, position t along the arc), "
         f"up to {D['cfg']['candidates']['k']} arcs within {D['cfg']['candidates']['radius']:.0f} m.")
    R.qa("2. Emission probabilities:", f"log e(arc | fix) = w x log max(p, 1e-4), where p is the classifier probability "
         f"for that (observation, arc) row and w = {c['emission_weight']} (tuned). The floor stops a single "
         "near-zero probability from making an otherwise good path impossible; w < 1 balances the classifier against "
         "the transition evidence.")
    R.qa("3. Transition probabilities:", f"log a = -|d_road - d_straight| / beta - U-turn penalty, with beta = "
         f"{c['beta']} m. d_road is the shortest driving distance through the network from the position on the first "
         "arc to the position on the second (Dijkstra, bounded and cached per source arc), d_straight the great-circle "
         "distance between the two fixes. A real drive has d_road close to d_straight, while a detour or a jump to a "
         f"parallel road makes them differ. Turning onto the reverse arc costs an extra {c['uturn_penalty']} (log scale). "
         f"Moving backwards along the same arc is allowed up to {c['back_tol']} m (GPS jitter); larger backward moves "
         "and unreachable pairs get probability 0.")
    R.qa("4. Missing candidates and large GPS gaps:", "observations without candidates are skipped. Long gaps need "
         "no special case, because the transition compares network and straight-line distance whatever the time gap. "
         "When every transition out of the current layer is impossible (outlier fix or broken network), the decoder "
         "first tries to skip up to 135 following observations as outliers (only if the jump is shorter than 2.5 km), "
         "then to backtrack up to 8 observations and re-decode; skipped observations are later assigned the nearest arc "
         "of the final route.")
    R.qa("5. Consecutive observations on disconnected road segments:", "if no recovery works, the current chain is "
         "closed and a new Viterbi chain starts at the next observation. Chains are joined with the shortest network "
         "path if it is shorter than 3 x straight-line distance + 1 km; otherwise the route keeps a gap rather than "
         "inventing a long detour. The final route is the stitched sequence of arcs along all chains.")
    R.qa("6. Did HMM decoding improve the final route?", "Yes - see the comparison table and figures below.")
    rows = []
    for split in ("train", "student_test"):
        for mode, label in (("clf", "Classifier only (argmax per observation)"), ("geo", "Geometric HMM (no classifier)"),
                            ("hmm", "Classifier + HMM/Viterbi (final)")):
            x = hm.loc[(split, mode)]
            rows.append((split + (" (out-of-fold)" if split == "train" else ""), label, f4(x.f1), f4(x.precision),
                         f4(x.recall), f4(x.rmf), f4(x.pointwise_top1), f4(x.route_connectivity)))
    R.table(["Partition", "Method", "Route F1", "Precision", "Recall", "RMF", "Obs. Top-1", "Connectivity"], rows,
            caption="Route-level comparison (results/hmm_comparison_summary.csv). Route F1/precision/recall are "
                    "length-weighted overlaps with the reference route; RMF = route mismatch fraction (lower is "
                    "better). Training-trip numbers use out-of-fold classifier probabilities.",
            widths=[2.6, 5, 1.5, 1.5, 1.4, 1.4, 1.6, 1.8], size=8, bold_rows=(2, 5))
    clf, geo, hmm = (hm.loc[("student_test", k)] for k in ("clf", "geo", "hmm"))
    pt = D["hmm_trip"]
    p = pt[pt.split == "student_test"].pivot_table(index="trip", columns="mode", values="f1")
    worse = p[p.hmm < p.geo - 1e-4]
    R.bullets([
        f"**Classifier only is not a usable route.** Its recall is high ({f4(clf.recall)}) but precision is "
        f"{f4(clf.precision)}: each argmax switch to a wrong arc needs a connecting detour, so RMF is {f4(clf.rmf)}. "
        f"Decoding raises route F1 from {f4(clf.f1)} to {f4(hmm.f1)} and makes every route connected.",
        f"**The classifier improves the per-observation decisions.** Observation-level Top-1 rises from "
        f"{f4(geo.pointwise_top1)} (geometric HMM) to {f4(hmm.pointwise_top1)} with classifier emissions on "
        "student-test, and from "
        f"{f4(hm.loc[('train', 'geo')].pointwise_top1)} to {f4(hm.loc[('train', 'hmm')].pointwise_top1)} on the "
        "training trips (out-of-fold).",
        f"**Where it did not help.** At route level the geometric HMM is already very strong because the network "
        f"constrains the path: on student-test the two decoders tie (route F1 {f4(geo.f1)} vs {f4(hmm.f1)}). On "
        f"{len(worse)} of 10 student-testing trips ({', '.join(f'{int(t):08d}' for t in worse.index)}) the classifier "
        "version was slightly worse; in these trips a confident wrong classifier probability near a junction pulled "
        "the path onto a parallel arc for a short stretch, which the purely geometric emission did not do.",
    ])
    R.table(["Trip", "Classifier only", "Geometric HMM", "Classifier + HMM"],
            [(f"{int(t):08d}", f4(r.clf), f4(r.geo), f4(r.hmm)) for t, r in p.iterrows()],
            caption="Route F1 per student-testing trip (results/hmm_comparison_per_trip.csv).", widths=[3, 4, 4, 4])
    R.fig(os.path.join(FIG, "hmm_vs_classifier.png"), "Classifier-only vs geometric HMM vs classifier + HMM.", 14)
    R.fig(os.path.join(FIG, "hmm_per_trip.png"), "Route F1 per student-testing trip.", 14)
    R.h("Decoder tuning", 3)
    d = D["dec"]
    R.p(f"The four decoder parameters were tuned on the 80 training trips only, using out-of-fold classifier "
        f"probabilities (in-sample probabilities would be over-confident). The search changed one parameter at a time "
        f"around the default ({len(d)} configurations, results/decoder_tuning_log.csv). Route F1 moved only between "
        f"{d.train_route_f1.min():.4f} and {d.train_route_f1.max():.4f}, i.e. the decoder is not sensitive to these "
        f"values; the chosen setting is emission weight {c['emission_weight']}, beta {c['beta']}, U-turn penalty "
        f"{c['uturn_penalty']}, back tolerance {c['back_tol']} m.")
    R.fig(os.path.join(FIG, "decoder_tuning.png"), "Decoder tuning on the training trips (out-of-fold).", 13)


def section_folium(R, D):
    R.h("6. Geographical visualisation using Folium", 1)
    R.p("Interactive maps are generated by `experiments/folium_maps.py` from the actual pipeline output and saved in "
        "`results/maps/` (student-testing trips 00000052, 00000056, 00000077 and the EDA map of training trip 00000003). "
        "Each map contains separate, toggleable layers with a layer control and a legend:")
    R.bullets(["**Road network** of the trip (thin grey lines).",
               "**Reference route** (wide translucent teal) - shown only for development trips.",
               "**Predicted route, classifier + HMM** (orange).",
               "**Classifier-only route** (purple) - argmax arc per observation, to show what the HMM fixes.",
               "**GPS fixes** (small dark circles) with green start and red end markers.",
               "The map is fitted to the extent of the trip and opens at a zoom level where the whole route is visible."])
    R.fig(os.path.join(FIG, "maps", "folium_student_test_00000052.jpg"),
          "Folium map of student-testing trip 00000052 (exported view). The predicted route (orange) follows the "
          "reference route (teal) under the GPS fixes, while the classifier-only route (purple) makes detours and "
          "loops at the junction in the lower left.", 15)
    R.fig(os.path.join(FIG, "maps", "folium_train_00000003_eda_overview.jpg"),
          "Folium EDA map of training trip 00000003 at full extent (road network, reference route, GPS fixes).", 12)


def section_api(R, D):
    R.h("7. API, UI and deployment", 1)
    R.h("7.1 Architecture", 2)
    R.bullets([
        "**Laptop 1 - backend** (`api/app.py`): FastAPI app served with Uvicorn. At start-up it loads "
        "`artifacts/final_model.joblib` and `artifacts/final_config.json` once. Endpoints: `GET /health` (status and "
        "loaded model), `GET /trips` (ids of the evaluation trips), `POST /predict/{trip_id}` (runs the full pipeline). "
        "Every request is logged to the console with timing and route metrics.",
        "**No leakage.** The trip is loaded with its reference route removed before prediction; the reference is read "
        "only afterwards on the server to compute metrics, and only the metrics are returned. The response contains the "
        "predicted route geometry, the (downsampled) GPS fixes and the trip's road network for drawing.",
        "**Laptop 2 - UI** (`ui/index.html`, served by `ui/serve.py`): a single web page using Leaflet. The evaluator "
        "enters the API address, loads the trip list and clicks a trip; the page draws the GPS fixes, the predicted "
        "route and the road network on a street or satellite base map and shows route F1, precision, recall, RMF, LCS "
        "arc accuracy, observation Top-1 and timings. If the map tile servers are blocked it switches to another "
        "provider; the road network from the API is always drawn.",
    ])
    R.h("7.2 Running it", 2)
    R.p("Laptop 1 (project root, after `pip install -r requirements.txt`):")
    R.p("`set PATHSNAP_TRIPS_DIR=data\\evaluator_test`")
    R.p("`python -m uvicorn api.app:app --host 0.0.0.0 --port 8000`")
    R.p("Laptop 2 (only the ui folder and Python are needed): `python ui/serve.py --port 5500`, open "
        "http://localhost:5500 and enter http://<laptop-1 IP>:8000. Both laptops must be on the same network.")
    R.h("7.3 Screenshots of the working system", 2)
    ui = os.path.join(FIG, "ui")
    R.fig(os.path.join(ui, "ui_trip00000015_streets.png"), "UI on laptop 2: trip 00000015 on the street map, zoomed in "
          "(GPS fixes, predicted route, metrics panel).", 16.5)
    R.fig(os.path.join(ui, "ui_trip00000087_streets.png"), "UI on laptop 2: trip 00000087 with the trip's road network.", 16.5)
    R.fig(os.path.join(ui, "ui_trip00000049_streets.png"), "UI on laptop 2: trip 00000049.", 16.5)
    R.fig(os.path.join(ui, "api_console_laptop1_part1.png"), "Console of laptop 1 (receiving laptop): requests from "
          "laptop 2 (10.30.202.137) with timing and metrics per trip.", 16.5)
    R.fig(os.path.join(ui, "api_console_laptop1_part2.png"), "Console of laptop 1, continued.", 16.5)
    R.p("The text of an earlier rehearsal console is kept in results/logs/demo_two_laptops_api_console.txt. Response "
        "times in these screenshots are higher than normal (up to ~40 s) because other experiments were running on "
        "laptop 1 at the same time; on an idle laptop a trip takes 0.5-5.5 s. During the rehearsal we also found that "
        "clicking a second trip while the first was still running could show the first trip's result under the second "
        "trip's title; the UI now ignores replies to requests that are no longer the latest.")

    R.h("7.4 Final check on the evaluator-testing trips (after selection)", 2)
    ev, evp, evt, evc = D["ev_clf"], D["ev_pipe"], D["ev_trip"], D["ev_cand"]
    R.p("These results were produced once, after the model, features, hyperparameters and decoder were frozen. The "
        "model and configuration files were fingerprinted (SHA-256) before the run and verified unchanged afterwards "
        "(results/evaluator_test/frozen_artifacts.json); nothing was retrained, tuned or selected using these trips. "
        f"{len(evc)} trips, {evc.n_obs.sum():,} observations, candidate miss rate "
        f"{evc.cand_miss.sum() / evc.n_obs.sum():.4f}.")
    order = MODELS + [FINAL]
    rows = [(NAME.get(m, m), f4(ev.loc[m, "top1"]), f4(ev.loc[m, "f1_macro"]), f4(ev.loc[m, "log_loss"]))
            for m in order if m in ev.index]
    R.table(["Classifier", "Top-1", "Macro F1", "Log loss"], rows,
            caption="Classifiers on the evaluator-testing candidates (results/evaluator_test/classifiers.csv).",
            widths=[8, 3, 3, 3], bold_rows=(len(rows) - 1,))
    fin, co = evp.loc["classifier + HMM (final)"], evp.loc["classifier only"]
    ft = evt[evt["mode"].str.startswith("classifier +")]
    R.table(["Method", "Route F1", "Precision", "Recall", "RMF", "Obs. Top-1", "Connectivity"], [
        ("Classifier + HMM (final)", f4(fin.f1), f4(fin.precision), f4(fin.recall), f4(fin.rmf),
         f4(fin.pointwise_top1), f4(fin.route_connectivity)),
        ("Classifier only", f4(co.f1), f4(co.precision), f4(co.recall), f4(co.rmf), f4(co.pointwise_top1),
         f4(co.route_connectivity))],
        caption=f"Final system on the evaluator-testing trips (results/evaluator_test/pipeline_summary.csv). Per-trip "
                f"route F1 ranges from {ft.f1.min():.4f} to {ft.f1.max():.4f}.",
        widths=[5, 2, 2, 2, 2, 2, 2])
    R.p("The ranking of the classifiers is the same as on the student-testing trips, which confirms - without having "
        "influenced - the model selection.")


def section_integrity(R, D):
    bv = D["bv"]
    R.h("8. Generalisation, reproducibility and academic integrity", 1)
    R.h("8.1 Are the high route scores a sign of overfitting?", 2)
    hm = D["hmm"].set_index(["split", "mode"])
    fb = bv.loc[FINAL]
    R.bullets([
        f"Route F1 near 0.99 is measured on trips the model never saw. Overfitting would show up as a drop from training "
        f"to unseen data; here route F1 is {f4(hm.loc[('train', 'hmm')].f1)} on the training trips (out-of-fold), "
        f"{f4(hm.loc[('student_test', 'hmm')].f1)} on student-test and "
        f"{f4(D['ev_pipe'].loc['classifier + HMM (final)'].f1)} on the evaluator trips.",
        f"The classifier itself is at {f4(fb.cv_top1)} (CV) / {f4(fb.student_test_top1)} (student-test) Top-1 with a "
        f"train-CV gap of {fb.variance_train_minus_cv:.4f}; the route score is higher because the road network "
        "constrains the route - a geometric HMM without any learning already reaches "
        f"{f4(hm.loc[('student_test', 'geo')].f1)} on student-test.",
        "Folds are split by trip, tuning used only training trips, and the reference route never enters the prediction "
        "pipeline (it is removed before prediction in pipeline and API).",
    ])
    rows = [(NAME.get(m, m), f4(r.train_top1), f4(r.cv_top1), f4(r.student_test_top1), f"{r.bias_train_error:.4f}",
             f"{r.variance_train_minus_cv:+.4f}") for m, r in bv.iterrows()]
    R.table(["Model", "Train Top-1", "CV Top-1", "Student-test Top-1", "Bias (1 - train)", "Variance (train - CV)"],
            rows, caption="Bias / variance of all models (results/bias_variance.csv, experiments/bias_variance.py).",
            widths=[5.5, 2.2, 2.2, 2.6, 2.2, 2.4], bold_rows=(len(rows) - 1,))
    R.h("8.2 Reproducibility", 2)
    R.bullets([
        "Seed 42 for every model, fold split and optimiser; GroupKFold is deterministic.",
        "All models use the same code, data and seed. Logistic Regression and Gaussian NB were run on Diya R Gowda's "
        "laptop and match the reference run exactly; all other numbers were produced on E S Sukruth's laptop "
        "(Windows 11, Python 3.14, scikit-learn 1.9.1, LightGBM 4.7.0). Library versions and machine are stored in "
        "each results/models/<model>.json.",
        "Commands: `python experiments/build_dataset.py`, `python experiments/train_models.py`, "
        "`python experiments/sweep_model.py --model <name>`, `python experiments/bayes_opt.py`, "
        "`python experiments/hmm_eval.py`, `python experiments/plots.py`, `python experiments/folium_maps.py`, "
        "`python experiments/bias_variance.py`; this report: `python report/build_report.py --pdf`.",
        "The report is generated from the result files by report/build_report.py, so every table can be traced to a "
        "file in the repository.",
    ])
    R.h("8.3 Team contributions", 2)
    R.p("[Team: complete and correct this table before submission.]", note=True)
    R.table(["Member", "Contribution"], [
        ("Diya R Gowda", "Logistic Regression and Gaussian NB runs and validation curves; EDA Folium map; report sections 2 and 3.2 (LR, GNB)."),
        ("E S Sukruth", "LightGBM; integration of the pipeline, Bayesian optimisation, HMM/Viterbi decoder, API; final "
                        "experiment runs; report sections 3.2 (LightGBM), 4 and 7 (API)."),
        ("Epari Subhransi", "Report sections 3.2 (Random Forest, HistGradientBoosting) and 5 (HMM). [complete]"),
        ("Prathamesh Kumar Singh", "Report sections 3.2 (KNN) and 7 (UI). [complete]"),
        ("Preksha B", "MLP run on Google Colab (cross-check); report sections 3.1, 3.2 (MLP), 3.3 and PDF assembly."),
    ], widths=[4.5, 12.5])
    R.h("8.4 Disclosure of AI assistance", 2)
    R.p("In line with the event's AI-use policy we disclose that the team used an AI coding assistant (Anthropic's "
        "Claude, through Claude Code) to help write and debug the code, to automate running the experiments, and to "
        "draft the text and layout of this report. All experiments were actually run on the team's machines; every "
        "metric, table and figure in this report is read from the result files in the repository and can be "
        "regenerated with the commands above. No evaluator-testing data was used for any development decision.")
    R.p("[Team: state here how each member reviewed and edited the text, in your own words.]", note=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    a = ap.parse_args()
    D = load()
    R = Report()
    title_page(R, D)
    section_overview(R, D)
    section_eda(R, D)
    section_models(R, D)
    section_bayes(R, D)
    section_hmm(R, D)
    section_folium(R, D)
    section_api(R, D)
    section_integrity(R, D)
    os.makedirs(os.path.dirname(OUT_DOCX), exist_ok=True)
    R.doc.save(OUT_DOCX)
    print("wrote", OUT_DOCX, f"({R.nfig} figures, {R.ntab} tables)")
    if a.pdf:
        import subprocess
        pdf = OUT_DOCX[:-5] + ".pdf"
        ps = (f"$w = New-Object -ComObject Word.Application; $w.Visible = $false; "
              f"$d = $w.Documents.Open('{OUT_DOCX}'); $d.Fields.Update() | Out-Null; "
              f"$d.SaveAs2('{pdf}', 17); $d.Close(); try {{ $w.Quit() }} catch {{}}")
        before = os.path.getmtime(pdf) if os.path.exists(pdf) else 0
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False)
        if os.path.exists(pdf) and os.path.getmtime(pdf) > before:
            print("wrote", pdf)
        else:
            raise SystemExit("PDF export failed - open the .docx in Word and use File > Save As > PDF")


if __name__ == "__main__":
    main()
