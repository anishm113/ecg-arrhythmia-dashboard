"""
make_deploy_model.py -- shrink the tuned model so the dashboard can run online.
GitHub's website accepts files up to 25 MB, and the tuned model sits right at that limit.
Run: python make_deploy_model.py
Output: models/random_forest_deploy.joblib (plus a report of any accuracy change)
"""

import json

import joblib
from sklearn.model_selection import train_test_split

from ecg_utils import MODEL_DIR, load_mitbih
from step2_train_model import RANDOM_STATE, TEST_SIZE
from step2b_tune_threshold import predict_with_threshold, safety_stats

SRC = MODEL_DIR / "random_forest_tuned.joblib"
DST = MODEL_DIR / "random_forest_deploy.joblib"
LIMIT_MB = 20                 # comfortable margin under GitHub's 25 MB upload limit
TREE_OPTIONS = [200, 150, 120, 100, 80]


def save(model):
    """Save with xz compression (much stronger than the default) and return size in MB."""
    joblib.dump(model, DST, compress=("xz", 9))
    return DST.stat().st_size / 2**20


def main():
    model = joblib.load(SRC)
    t = json.load(open(MODEL_DIR / "threshold.json"))["threshold"]

    # Out-of-bag predictions were only needed to choose the threshold -> drop them
    for attr in ("oob_decision_function_", "oob_score_"):
        if hasattr(model, attr):
            delattr(model, attr)

    # Same test set as Step 2, to measure any accuracy change honestly
    X, y = load_mitbih(verbose=False)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    full = safety_stats(y_test, predict_with_threshold(model.predict_proba(X_test), t))

    trees = list(model.estimators_)
    for n in [k for k in TREE_OPTIONS if k <= len(trees)]:
        model.estimators_ = trees[:n]        # a forest of fewer trees is still a valid forest
        model.n_estimators = n
        mb = save(model)
        s = safety_stats(y_test, predict_with_threshold(model.predict_proba(X_test), t))
        print(f"{n} trees: {mb:5.1f} MB | sensitivity {s['sensitivity']:.4f} "
              f"(full model {full['sensitivity']:.4f}) | specificity {s['specificity']:.4f} "
              f"(full model {full['specificity']:.4f})")
        if mb <= LIMIT_MB:
            print(f"\nSaved {DST.name}: {n} trees, {mb:.1f} MB. Ready to upload.")
            return
    print(f"\nStill above {LIMIT_MB} MB at {n} trees. Send this output to Claude.")


if __name__ == "__main__":
    main()
