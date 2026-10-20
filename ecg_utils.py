"""
ecg_utils.py
Shared constants and helper functions used by every step of the project.
Keeping these in one place means Step 1, Step 2, and the Streamlit app
(Step 3) all load and label the data in exactly the same way.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to this file, so the scripts work from any directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "mitbih_train.csv"
FIG_DIR = PROJECT_ROOT / "figures"
MODEL_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Dataset facts (Kaggle "ECG Heartbeat Categorization" version of MIT-BIH)
# ---------------------------------------------------------------------------
SAMPLING_RATE_HZ = 125   # beats were resampled to 125 Hz -> 1 sample = 8 ms
N_SAMPLES = 187          # samples per beat (shorter beats are zero-padded)

# AAMI EC57 heartbeat classes used by this dataset
CLASS_NAMES = {
    0: "Normal (N)",
    1: "Supraventricular ectopic (S)",
    2: "Ventricular ectopic (V)",
    3: "Fusion (F)",
    4: "Unclassifiable / paced (Q)",
}
CLASS_SHORT = ["N", "S", "V", "F", "Q"]


def load_mitbih(path=DATA_PATH, verbose=True):
    """Load the CSV, run basic cleaning checks, and return (X, y) as NumPy arrays."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Download mitbih_train.csv from Kaggle "
            "and place it in the data/ folder."
        )

    # The Kaggle file has NO header row -> header=None
    df = pd.read_csv(path, header=None)
    n_raw = len(df)

    # Sanity check: 187 signal columns + 1 label column
    if df.shape[1] != N_SAMPLES + 1:
        raise ValueError(f"Expected {N_SAMPLES + 1} columns, found {df.shape[1]}.")

    # Cleaning 1: drop rows with missing values (corrupted recordings)
    df = df.dropna()
    n_nan = n_raw - len(df)

    # Cleaning 2: drop exact duplicate beats (they would leak between train/test)
    n_before_dup = len(df)
    df = df.drop_duplicates()
    n_dup = n_before_dup - len(df)

    # Split into signal matrix X and label vector y
    X = df.iloc[:, :N_SAMPLES].to_numpy(dtype=np.float32)   # float32 halves memory
    y = df.iloc[:, N_SAMPLES].round().astype(int).to_numpy()  # labels stored as 0.0, 1.0...

    # Cleaning 3: make sure every label is a valid class
    invalid = ~np.isin(y, list(CLASS_NAMES))
    if invalid.any():
        raise ValueError(f"{invalid.sum()} rows have labels outside 0-4.")

    if verbose:
        print(f"Loaded {n_raw:,} beats from {path.name}")
        print(f"  Removed {n_nan:,} rows with missing values")
        print(f"  Removed {n_dup:,} duplicate rows")
        print(f"  Final dataset: {X.shape[0]:,} beats x {X.shape[1]} samples")
        print(f"  Amplitude range: [{X.min():.3f}, {X.max():.3f}] (already normalized to 0-1)")
    return X, y


def time_axis_ms(n=N_SAMPLES):
    """Convert sample indices to milliseconds (sample k occurs at k * 8 ms)."""
    return np.arange(n) * 1000.0 / SAMPLING_RATE_HZ


def signal_length(beat, tol=1e-6):
    """Number of real samples before the trailing zero-padding starts."""
    nonzero = np.flatnonzero(np.abs(beat) > tol)
    return int(nonzero[-1]) + 1 if nonzero.size else 0


def set_publication_style():
    """Clean, journal-style Matplotlib defaults."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,            # print-quality export
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,      # remove chart-junk borders
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.fontsize": 9,
    })
