"""Step 6: report figures, all generated from the saved experiment outputs.

    python experiments/plots.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGS, RESULTS


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, name), dpi=130)
    plt.close(fig)
    print("saved", name)


def model_comparison():
    df = pd.read_csv(os.path.join(RESULTS, "model_comparison.csv"))
    order = df.sort_values("cv_top1")["model"]
    df = df.set_index("model").loc[order]
    folds = df[[c for c in df.columns if c.startswith("fold") and c.endswith("f1_macro")]]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(df.index, df.cv_f1_macro, xerr=folds.std(axis=1), color="#457b9d", capsize=3)
    for i, (m, v) in enumerate(df.cv_f1_macro.items()):
        ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8)
    ax.set_xlim(df.cv_f1_macro.min() - 0.03, 1.0)
    ax.set_xlabel("5-fold GroupKFold macro F1 (training trips), error bar = fold std")
    ax.set_title("Cross-validation macro F1 by model")
    savefig(fig, "cv_macro_f1.png")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.test_top1, 0.4, label="Top-1 accuracy", color="#2a9d8f")
    ax.bar(x + 0.2, df.test_f1_macro, 0.4, label="Macro F1", color="#e9c46a")
    ax.set_xticks(x, df.index, rotation=20)
    ax.set_ylim(min(df.test_top1.min(), df.test_f1_macro.min()) - 0.03, 1.0)
    ax.set_title("Student-testing trips: Top-1 accuracy and macro F1")
    ax.legend()
    savefig(fig, "student_test_top1_f1.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, df.cv_log_loss, 0.4, label="CV (train trips)", color="#6a4c93")
    ax.bar(x + 0.2, df.test_log_loss, 0.4, label="student test", color="#f4a261")
    ax.set_xticks(x, df.index, rotation=20)
    ax.set_ylabel("log loss (lower is better)")
    ax.set_title("Probability quality: log loss")
    ax.legend()
    savefig(fig, "log_loss.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(df.infer_ms_per_1k_rows, df.test_top1, s=60, color="#1d3557")
    for m, r in df.iterrows():
        ax.annotate(m, (r.infer_ms_per_1k_rows, r.test_top1), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("inference time per 1k candidate rows (ms, log scale)")
    ax.set_ylabel("student-test Top-1 accuracy")
    ax.set_title("Accuracy vs inference cost")
    savefig(fig, "accuracy_vs_speed.png")

    n = len(df)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.8))
    for ax, (m, r) in zip(axes, df.iterrows()):
        cm = np.array(json.loads(r.test_confusion_matrix))
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}", ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] / cm[i].sum() > 0.5 else "black")
        ax.set_xticks([0, 1], ["other", "ref"])
        ax.set_yticks([0, 1], ["other", "ref"])
        ax.set_title(m, fontsize=9)
        ax.set_xlabel("predicted", fontsize=8)
    axes[0].set_ylabel("true", fontsize=8)
    savefig(fig, "confusion_matrices.png")


def training_history():
    path = os.path.join(RESULTS, "training_histories.json")
    if not os.path.exists(path):
        return
    hist = json.load(open(path))
    for name, h in hist.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(h["train_loss"], label="training loss (log loss)", color="#e76f51")
        ax.set_xlabel("epoch")
        ax.set_ylabel("training loss")
        if h.get("val_accuracy"):
            ax2 = ax.twinx()
            ax2.plot(h["val_accuracy"], label="validation accuracy (10% hold-out)", color="#2a9d8f")
            ax2.set_ylabel("validation accuracy")
            ax2.legend(loc="center right")
        ax.legend(loc="upper right")
        ax.set_title(f"{name}: epoch-wise training history (final fit)")
        savefig(fig, f"history_{name}.png")


def bayes_opt():
    path = os.path.join(RESULTS, "bayes_opt_trials.csv")
    if not os.path.exists(path):
        return
    t = pd.read_csv(path)
    s = json.load(open(os.path.join(RESULTS, "bayes_opt_summary.json")))
    done = t[t.state == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(done.number, done.value, s=18, color="#457b9d", label="completed trial")
    pr = t[t.state == "PRUNED"]
    if len(pr):
        ax.scatter(pr.number, np.full(len(pr), done.value.min()), marker="x", color="#bbbbbb", label="pruned trial")
    ax.plot(done.number, done.value.cummax(), color="#e76f51", label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel("CV Top-1 accuracy")
    ax.set_title(f"Bayesian optimisation (TPE) for {s['model']}")
    ax.legend()
    savefig(fig, "bayes_opt_history.png")

    keys = ["cv_f1_macro", "cv_top1", "test_top1", "test_f1_macro"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [s["untuned"][k] for k in keys], 0.4, label="untuned", color="#adb5bd")
    ax.bar(x + 0.2, [s["tuned"][k] for k in keys], 0.4, label="tuned", color="#2a9d8f")
    ax.set_xticks(x, ["CV macro F1", "CV Top-1", "test Top-1", "test macro F1"])
    lo = min(min(s["untuned"][k], s["tuned"][k]) for k in keys)
    ax.set_ylim(lo - 0.01, 1.0)
    ax.legend()
    ax.set_title(f"{s['model']}: tuned vs untuned")
    savefig(fig, "tuned_vs_untuned.png")


def hmm():
    path = os.path.join(RESULTS, "hmm_comparison_per_trip.csv")
    if not os.path.exists(path):
        return
    d = pd.read_csv(path)
    names = {"clf": "classifier only", "geo": "geometric HMM", "hmm": "classifier + HMM"}
    te = d[d.split == "student_test"]
    keys = ["pointwise_top1", "f1", "precision", "recall", "route_connectivity"]
    labels = ["pointwise Top-1", "route F1", "route precision", "route recall", "route connectivity"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(keys))
    for i, (mode, col) in enumerate(zip(["clf", "geo", "hmm"], ["#adb5bd", "#457b9d", "#e76f51"])):
        vals = te[te["mode"] == mode][keys].mean()
        ax.bar(x + (i - 1) * 0.27, vals, 0.27, label=names[mode], color=col)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower left")
    ax.set_title("Student-testing trips: effect of HMM/Viterbi decoding")
    savefig(fig, "hmm_vs_classifier.png")

    p = te.pivot(index="trip", columns="mode", values="f1")
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(p))
    for i, (mode, col) in enumerate(zip(["clf", "geo", "hmm"], ["#adb5bd", "#457b9d", "#e76f51"])):
        ax.bar(x + (i - 1) * 0.27, p[mode], 0.27, label=names[mode], color=col)
    ax.set_xticks(x, p.index, rotation=30)
    ax.set_ylim(max(0, p.values.min() - 0.05), 1.0)
    ax.set_ylabel("route F1")
    ax.legend()
    ax.set_title("Route F1 per student-testing trip")
    savefig(fig, "hmm_per_trip.png")

    dec = os.path.join(RESULTS, "decoder_tuning_log.csv")
    if os.path.exists(dec):
        g = pd.read_csv(dec)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(np.arange(len(g)), g.train_route_f1, "o-", color="#457b9d", ms=4)
        ax.plot(np.arange(len(g)), g.train_route_f1.cummax(), color="#e76f51", label="best so far")
        ax.set_xlabel("decoder configuration evaluated")
        ax.set_ylabel("train route F1 (OOF probabilities)")
        ax.set_title("Decoder parameter search on training trips")
        ax.legend()
        savefig(fig, "decoder_tuning.png")


if __name__ == "__main__":
    model_comparison()
    training_history()
    bayes_opt()
    hmm()
