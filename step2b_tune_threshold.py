"""
STEP 2B: Safety tuning -- lowering the alarm threshold
Run:  python step2b_tune_threshold.py               # target 95% arrhythmia sensitivity
      python step2b_tune_threshold.py --target 0.90 # try a different target
      python step2b_tune_threshold.py --min-spec 0.97 # stricter false-alarm limit

Why: Step 2's model defaults to "Normal" when unsure, so it misses arrhythmias.
Here we flag a beat as abnormal whenever P(abnormal) >= t, and pick t so that
sensitivity reaches a target. Crucially, t is chosen on OUT-OF-BAG data (beats
each tree never trained on), NEVER on the test set -- otherwise the test score
would be secretly tuned to the test set and look better than reality.
"""

import argparse
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import train_test_split

from ecg_utils import (CLASS_NAMES, CLASS_SHORT, FIG_DIR, MODEL_DIR, load_mitbih,
                       set_publication_style)
from step2_train_model import N_TREES, RANDOM_STATE, TEST_SIZE   # same settings as Step 2

THRESHOLDS = np.round(np.arange(0.02, 0.99, 0.01), 2)   # candidate thresholds to sweep
LABELS = list(CLASS_NAMES)


def predict_with_threshold(proba, t):
    """Abnormal if P(abnormal) = 1 - P(Normal) >= t; then pick the most likely abnormal class."""
    p_abnormal = 1.0 - proba[:, 0]
    best_abnormal = 1 + np.argmax(proba[:, 1:], axis=1)     # which arrhythmia is most likely
    return np.where(p_abnormal >= t, best_abnormal, 0)


def safety_stats(y_true, y_pred):
    """Collapse to Normal vs ANY arrhythmia and count the clinical outcomes."""
    true_abn, pred_abn = y_true != 0, y_pred != 0
    tp = int(np.sum(true_abn & pred_abn))
    fn = int(np.sum(true_abn & ~pred_abn))                  # missed arrhythmias
    tn = int(np.sum(~true_abn & ~pred_abn))
    fp = int(np.sum(~true_abn & pred_abn))                  # false alarms
    return {"sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp), "fn": fn, "fp": fp}


def choose_threshold(y_val, proba_val, target, min_spec):
    """Highest threshold (= fewest false alarms) that reaches the target sensitivity,
    without letting specificity drop below min_spec (guardrail against alarm fatigue)."""
    curve = [safety_stats(y_val, predict_with_threshold(proba_val, t)) for t in THRESHOLDS]
    sens = np.array([c["sensitivity"] for c in curve])
    spec = np.array([c["specificity"] for c in curve])
    allowed = spec >= min_spec                              # thresholds clinicians could live with
    if not allowed.any():
        raise ValueError(f"No threshold keeps specificity >= {min_spec:.0%}; lower --min-spec.")
    ok = np.flatnonzero(allowed & (sens >= target))         # lower t -> higher sensitivity
    if ok.size:
        return float(THRESHOLDS[ok[-1]]), curve, True       # target reached
    idx = np.flatnonzero(allowed)[0]                        # else: most sensitive allowed t
    return float(THRESHOLDS[idx]), curve, False


def summarize(name, y_true, y_pred):
    """All the numbers we compare before vs after."""
    s = safety_stats(y_true, y_pred)
    s["accuracy"] = accuracy_score(y_true, y_pred)
    s["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    s["per_class_recall"] = recall_score(y_true, y_pred, labels=LABELS, average=None,
                                         zero_division=0).tolist()
    s["name"] = name
    return s


def print_comparison(default, tuned, t_star):
    line = "=" * 66
    print(f"\n{line}\n  BEFORE vs AFTER THRESHOLD TUNING  (test set, t = {t_star:.2f})\n{line}")
    print(f"  {'Metric':<28}{'Default':>12}{'Tuned':>12}{'Change':>12}")
    rows = [("Sensitivity (catch rate)", "sensitivity", "{:.4f}"),
            ("Specificity", "specificity", "{:.4f}"),
            ("Missed arrhythmias (FN)", "fn", "{:,}"),
            ("False alarms (FP)", "fp", "{:,}"),
            ("Overall accuracy", "accuracy", "{:.4f}"),
            ("Balanced accuracy", "balanced_accuracy", "{:.4f}")]
    for label, key, fmt in rows:
        a, b = default[key], tuned[key]
        diff = f"{b - a:+,}" if isinstance(a, int) else f"{b - a:+.4f}"
        print(f"  {label:<28}{fmt.format(a):>12}{fmt.format(b):>12}{diff:>12}")
    print(f"\n  Per-class recall")
    for k in LABELS:
        a, b = default["per_class_recall"][k], tuned["per_class_recall"][k]
        print(f"  {CLASS_NAMES[k]:<28}{a:>12.4f}{b:>12.4f}{b - a:>+12.4f}")
    print(line)


def plot_tuning(curve, t_star, target, min_spec, default, tuned, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Panel A: the trade-off curve (on out-of-bag data, where t was chosen)
    sens = [c["sensitivity"] for c in curve]
    spec = [c["specificity"] for c in curve]
    ax1.plot(THRESHOLDS, sens, color="#c0392b", lw=2, label="Sensitivity (catch arrhythmias)")
    ax1.plot(THRESHOLDS, spec, color="#1f5fa8", lw=2, label="Specificity (avoid false alarms)")
    ax1.axhline(target, color="gray", ls=":", lw=1, label=f"Target sensitivity = {target:.0%}")
    ax1.axhline(min_spec, color="#1f5fa8", ls=":", lw=1, alpha=0.6,
                label=f"Specificity floor = {min_spec:.0%}")
    ax1.axvline(t_star, color="black", ls="--", lw=1.2, label=f"Chosen threshold t = {t_star:.2f}")
    ax1.set_xlabel("Alarm threshold t   (flag beat if P(abnormal) ≥ t)")
    ax1.set_ylabel("Rate (out-of-bag validation)")
    ax1.set_title("A. The Safety Trade-off")
    ax1.set_ylim(0, 1.02)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

    # Panel B: per-class recall before vs after (on the untouched test set)
    x = np.arange(len(LABELS))
    w = 0.38
    ax2.bar(x - w / 2, default["per_class_recall"], w, color="#95a5a6", label="Default")
    ax2.bar(x + w / 2, tuned["per_class_recall"], w, color="#c0392b", label=f"Tuned (t = {t_star:.2f})")
    for xi, v in zip(x + w / 2, tuned["per_class_recall"]):
        ax2.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x, CLASS_SHORT)
    ax2.set_xlabel("AAMI heartbeat class")
    ax2.set_ylabel("Recall (test set)")
    ax2.set_ylim(0, 1.12)
    ax2.set_title(f"B. Missed arrhythmias {default['fn']:,} → {tuned['fn']:,}   |   "
                  f"False alarms {default['fp']:,} → {tuned['fp']:,}")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

    fig.suptitle("Threshold Tuning: Trading False Alarms for Fewer Missed Arrhythmias",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved figure -> {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Step 2B: threshold tuning")
    parser.add_argument("--target", type=float, default=0.95,
                        help="Target arrhythmia sensitivity, between 0 and 1 (default 0.95)")
    parser.add_argument("--min-spec", type=float, default=0.95,
                        help="Never let specificity fall below this (default 0.95)")
    args = parser.parse_args()

    set_publication_style()
    FIG_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    # 1) Same data + same split as Step 2 (same seed -> identical test set)
    X, y = load_mitbih()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    # 2) Same forest as Step 2, plus oob_score=True -> free validation predictions
    model = RandomForestClassifier(n_estimators=N_TREES, class_weight="balanced_subsample",
                                   oob_score=True, n_jobs=-1, random_state=RANDOM_STATE)
    print("Training forest with out-of-bag scoring...")
    model.fit(X_train, y_train)
    assert list(model.classes_) == LABELS                   # column k = class k

    # 3) Pick the threshold on OOB predictions (each beat scored only by trees that never saw it)
    proba_oob = model.oob_decision_function_
    valid = ~np.isnan(proba_oob).any(axis=1)                # drop any beat with no OOB trees
    t_star, curve, reached = choose_threshold(y_train[valid], proba_oob[valid],
                                              args.target, args.min_spec)
    print(f"Chosen threshold t = {t_star:.2f} (target sensitivity {args.target:.0%}, "
          f"specificity floor {args.min_spec:.0%}, on OOB data)")
    if not reached:
        print(f"  NOTE: {args.target:.0%} sensitivity is unreachable without dropping specificity "
              f"below {args.min_spec:.0%}; using the most sensitive threshold that stays above it.")

    # 4) Apply BOTH rules to the untouched test set
    proba_test = model.predict_proba(X_test)
    default = summarize("default", y_test, model.predict(X_test))
    tuned = summarize("tuned", y_test, predict_with_threshold(proba_test, t_star))
    print_comparison(default, tuned, t_star)

    # 5) Figure + saved artifacts for the dashboard
    plot_tuning(curve, t_star, args.target, args.min_spec, default, tuned,
                FIG_DIR / "fig5_threshold_tuning.png")
    joblib.dump(model, MODEL_DIR / "random_forest_tuned.joblib", compress=3)
    with open(MODEL_DIR / "threshold.json", "w") as f:
        json.dump({"threshold": t_star, "target_sensitivity": args.target,
                   "min_specificity": args.min_spec, "target_reached": reached,
                   "test_default": default, "test_tuned": tuned}, f, indent=2)
    print(f"Saved tuned model + threshold -> {MODEL_DIR}/")


if __name__ == "__main__":
    main()
