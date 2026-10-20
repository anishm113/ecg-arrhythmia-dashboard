"""
STEP 1: Data cleaning & biomedical visualization
Run:  python step1_visualize.py                # Normal vs Ventricular (class 2)
      python step1_visualize.py --abnormal 1   # Normal vs Supraventricular
      python step1_visualize.py --abnormal 3   # Normal vs Fusion
      python step1_visualize.py --seed 7       # show a different random set of beats
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from ecg_utils import (CLASS_NAMES, CLASS_SHORT, FIG_DIR, SAMPLING_RATE_HZ, load_mitbih,
                       set_publication_style, signal_length, time_axis_ms)

N_EXAMPLE_BEATS = 8   # thin background lines per panel
MORPH_WINDOW_MS = 600 # first 600 ms = the beat's own shape (QRS + T wave);
                      # after that the NEXT beat arrives at a patient-specific time


def find_medoid(beats, rng, max_n=500, window=None):
    """Return the real beat with the smallest average distance to the others.
    Unlike 'closest to the mean', this lands inside a dense cluster of REAL shapes,
    so it doesn't come out as a bland blur when a class is very variable.
    window = number of samples to compare (shape only, ignoring next-beat timing)."""
    pick = rng.choice(len(beats), size=min(max_n, len(beats)), replace=False)
    sample = beats[pick][:, :window].astype(np.float64)   # compare shape window only
    sq = (sample ** 2).sum(axis=1)
    dist = np.sqrt(np.clip(sq[:, None] + sq[None, :] - 2 * sample @ sample.T, 0, None))
    return beats[pick[np.argmin(dist.mean(axis=1))]]   # return the FULL beat


def plot_normal_vs_abnormal(X, y, abnormal_label=2, seed=42, save_path=None):
    """Side-by-side plot: several real beats per class (thin) + the most typical one (bold)."""
    rng = np.random.default_rng(seed)                         # fixed seed -> same figure every run
    t_all = time_axis_ms()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    colors = ["#1f5fa8", "#c0392b"]                           # blue = normal, red = abnormal

    for ax, label, color in zip(axes, [0, abnormal_label], colors):
        beats = X[y == label]
        if len(beats) == 0:
            raise ValueError(f"No beats found for class {label}.")

        # Thin lines: random real beats -> shows how much shapes vary between patients
        for i, idx in enumerate(rng.choice(len(beats), N_EXAMPLE_BEATS, replace=False)):
            n = signal_length(beats[idx])                     # cut off the zero-padding
            ax.plot(t_all[:n], beats[idx][:n], color=color, lw=0.8, alpha=0.25,
                    label="Other beats from this class" if i == 0 else None)

        # Shade the region where the NEXT heartbeat shows up (timing, not shape)
        ax.axvspan(MORPH_WINDOW_MS, t_all[-1] + 8, color="gray", alpha=0.08, lw=0,
                   label="Next beat arrives here (timing)")

        # Bold line: the medoid (most typical real beat), judged on shape only
        window = int(MORPH_WINDOW_MS * SAMPLING_RATE_HZ / 1000)   # 600 ms -> 75 samples
        medoid = find_medoid(beats, rng, window=window)
        n = signal_length(medoid)
        ax.plot(t_all[:n], medoid[:n], color=color, lw=2.4, label=f"Most typical beat (medoid, first {MORPH_WINDOW_MS} ms)")

        # Each segment in this dataset starts at the R-peak -> label it
        r = int(np.argmax(medoid[:10]))
        ax.annotate("R-peak", xy=(t_all[r], medoid[r]), xytext=(t_all[r] + 60, medoid[r] - 0.05),
                    arrowprops=dict(arrowstyle="->", lw=1), fontsize=10)

        ax.set_title(f"Class {label}: {CLASS_NAMES[label]}\n(n = {len(beats):,} beats)")
        ax.set_xlabel("Time after R-peak (ms)")
        ax.set_ylim(-0.05, 1.12)
        ax.set_xlim(-20, t_all[-1] + 8)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=8.5)

    axes[0].set_ylabel("Normalized amplitude (a.u.)")
    fig.suptitle("MIT-BIH Arrhythmia Database: Normal vs. Arrhythmic Heartbeat Morphology",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Saved figure -> {save_path}")
    plt.close(fig)


def plot_class_distribution(y, save_path=None):
    """Bar chart of beats per class -- reveals the class imbalance we must handle in Step 2."""
    counts = np.bincount(y, minlength=len(CLASS_SHORT))
    pct = 100 * counts / counts.sum()

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(CLASS_SHORT, counts, color=["#1f5fa8"] + ["#c0392b"] * 4)
    for bar, c, p in zip(bars, counts, pct):                  # print count + % on each bar
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{c:,}\n({p:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("AAMI heartbeat class")
    ax.set_ylabel("Number of beats")
    ax.set_title("Class Distribution (severe imbalance toward Normal beats)")
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        print(f"Saved figure -> {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Step 1: ECG visualization")
    parser.add_argument("--abnormal", type=int, default=2, choices=[1, 2, 3, 4],
                        help="Arrhythmia class to compare against Normal (default: 2 = Ventricular)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for which example beats are drawn")
    args = parser.parse_args()

    set_publication_style()
    FIG_DIR.mkdir(exist_ok=True)

    # 1) Load + clean
    X, y = load_mitbih()

    # 2) Print a quick class summary table
    print("\nClass summary:")
    lengths = np.array([signal_length(b) for b in X])
    for k, name in CLASS_NAMES.items():
        mask = y == k
        print(f"  {k}  {name:<30} {mask.sum():>7,} beats   "
              f"avg segment length {lengths[mask].mean() * 8:.0f} ms" if mask.any() else
              f"  {k}  {name:<30}       0 beats")

    # 3) Figures
    plot_class_distribution(y, FIG_DIR / "fig1_class_distribution.png")
    plot_normal_vs_abnormal(X, y, args.abnormal, args.seed,
                            FIG_DIR / f"fig2_normal_vs_class{args.abnormal}.png")


if __name__ == "__main__":
    main()
