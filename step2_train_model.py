"""
STEP 2: Machine-learning diagnostic model (Random Forest)
Run:  python step2_train_model.py
Outputs: metrics printed to terminal, figures in figures/, model + metrics in models/
"""

import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             balanced_accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

from ecg_utils import (CLASS_NAMES, CLASS_SHORT, FIG_DIR, MODEL_DIR, load_mitbih,
                       set_publication_style, time_axis_ms)

RANDOM_STATE = 42   # fixed seed -> reproducible results (important for science!)
TEST_SIZE = 0.20    # hold out 20% of beats the model never sees during training
N_TREES = 200       # number of decision trees in the forest


def train_model(X_train, y_train):
    """Build and fit the Random Forest."""
    model = RandomForestClassifier(
        n_estimators=N_TREES,
        class_weight="balanced_subsample",  # up-weights rare arrhythmias so they aren't ignored
        n_jobs=-1,                          # use every CPU core
        random_state=RANDOM_STATE,
    )
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    print(f"Training finished in {time.perf_counter() - t0:.1f} s")
    return model


def evaluate(model, X_test, y_test):
    """Compute overall, per-class, and clinical-safety metrics."""
    y_pred = model.predict(X_test)
    labels = list(CLASS_NAMES)

    # Overall metrics
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)      # average recall across classes
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average="macro", zero_division=0)  # every class equal
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average="weighted", zero_division=0)  # weighted by size

    # Clinical safety view: collapse to "Normal" vs "ANY arrhythmia"
    true_abn, pred_abn = y_test != 0, y_pred != 0
    fn = int(np.sum(true_abn & ~pred_abn))                 # arrhythmias MISSED (dangerous)
    fp = int(np.sum(~true_abn & pred_abn))                 # false alarms (annoying)
    tp = int(np.sum(true_abn & pred_abn))
    tn = int(np.sum(~true_abn & ~pred_abn))
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0     # = recall for arrhythmia
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    # Print summary
    line = "=" * 64
    print(f"\n{line}\n  DIAGNOSTIC MODEL METRICS  (test set: {len(y_test):,} beats)\n{line}")
    print(f"  Overall accuracy          : {acc:.4f}")
    print(f"  Balanced accuracy         : {bal_acc:.4f}   <- fairer with imbalance")
    print(f"  Precision (macro / wtd)   : {p_mac:.4f} / {p_w:.4f}")
    print(f"  Recall    (macro / wtd)   : {r_mac:.4f} / {r_w:.4f}")
    print(f"  F1-score  (macro / wtd)   : {f_mac:.4f} / {f_w:.4f}")
    print(f"\n  CLINICAL SAFETY (Normal vs. any arrhythmia)")
    print(f"  Sensitivity / recall      : {sensitivity:.4f}")
    print(f"  Specificity               : {specificity:.4f}")
    print(f"  False negatives (missed)  : {fn:,}  <- arrhythmias labeled 'Normal'")
    print(f"  False positives (alarms)  : {fp:,}")
    print(f"\n  PER-CLASS REPORT\n{line}")
    print(classification_report(y_test, y_pred, labels=labels,
                                target_names=[CLASS_NAMES[k] for k in labels],
                                digits=4, zero_division=0))

    metrics = {
        "accuracy": acc, "balanced_accuracy": bal_acc,
        "precision_macro": p_mac, "recall_macro": r_mac, "f1_macro": f_mac,
        "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f_w,
        "binary_sensitivity": sensitivity, "binary_specificity": specificity,
        "binary_false_negatives": fn, "binary_false_positives": fp,
        "per_class": classification_report(y_test, y_pred, labels=labels,
                                           target_names=CLASS_SHORT,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "n_test": int(len(y_test)),
    }
    return y_pred, metrics


def plot_confusion_matrices(y_test, y_pred, save_path):
    """Raw counts (left) + row-normalized recall (right): the 'optimized' view.
    Normalizing by row stops the huge Normal class from hiding errors on rare classes."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, norm, fmt, title in [
        (axes[0], None, "d", "Counts"),
        (axes[1], "true", ".2f", "Row-normalized (diagonal = recall per class)"),
    ]:
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, labels=list(CLASS_NAMES), display_labels=CLASS_SHORT,
            normalize=norm, values_format=fmt, cmap="Blues", colorbar=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.grid(False)
    fig.suptitle("Random Forest Confusion Matrices: AAMI Classes N, S, V, F, Q",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved figure -> {save_path}")


def plot_feature_importance(model, X, y, save_path):
    """Which moments of the heartbeat did the model rely on? Overlay on the normal template."""
    t = time_axis_ms()
    template = X[y == 0].mean(axis=0)

    fig, ax1 = plt.subplots(figsize=(10, 4.2))
    ax1.plot(t, template, color="#1f5fa8", lw=2, label="Average normal beat")
    ax1.set_xlabel("Time after R-peak (ms)")
    ax1.set_ylabel("Normalized amplitude (a.u.)", color="#1f5fa8")

    ax2 = ax1.twinx()                                     # second y-axis for importance
    ax2.bar(t, model.feature_importances_, width=8, color="#e67e22", alpha=0.6,
            label="Feature importance")
    ax2.set_ylabel("Gini importance", color="#e67e22")
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)

    ax1.set_title("Where in the Heartbeat Does the Model Look? (Random Forest Feature Importance)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved figure -> {save_path}")


def main():
    set_publication_style()
    FIG_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    # 1) Load + clean (same function as Step 1)
    X, y = load_mitbih()

    # 2) Stratified split: keeps the same class proportions in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"Train: {len(y_train):,} beats | Test: {len(y_test):,} beats")

    # 3) Train
    model = train_model(X_train, y_train)

    # 4) Evaluate
    y_pred, metrics = evaluate(model, X_test, y_test)

    # 5) Figures
    plot_confusion_matrices(y_test, y_pred, FIG_DIR / "fig3_confusion_matrix.png")
    plot_feature_importance(model, X, y, FIG_DIR / "fig4_feature_importance.png")

    # 6) Save model + metrics for the Streamlit dashboard (Step 3)
    joblib.dump(model, MODEL_DIR / "random_forest.joblib", compress=3)
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model + metrics -> {MODEL_DIR}/")


if __name__ == "__main__":
    main()
