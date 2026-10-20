"""
app.py -- interactive dashboard for the ECG arrhythmia detector
Run:  streamlit run app.py
Needs: models/random_forest_tuned.joblib + models/threshold.json (from step2b_tune_threshold.py)
       data/mitbih_train.csv (first launch only, to build models/dashboard_cache.npz)
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import signal_processing as sp
from ecg_utils import (CLASS_NAMES, CLASS_SHORT, DATA_PATH, MODEL_DIR, load_mitbih,
                       signal_length, time_axis_ms)
from step2_train_model import RANDOM_STATE, TEST_SIZE
from step2b_tune_threshold import THRESHOLDS, predict_with_threshold, safety_stats

MODEL_PATH = MODEL_DIR / "random_forest_tuned.joblib"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"
CACHE_PATH = MODEL_DIR / "dashboard_cache.npz"
BEATS_PER_CLASS = 200        # demo beats per class (Fusion has only 128 in the test set)

# Palette: ECG paper + two meaningful colors (alarm = arrhythmia, calm = normal)
PAPER, GRID_MINOR, GRID_MAJOR = "#FFF5F3", "#F6D5D0", "#E8A49B"
INK, ALARM, CALM, MUTED = "#1B2A3A", "#B3261E", "#2E6F5E", "#8C96A0"
LABELS = list(CLASS_NAMES)


# ---------------------------------------------------------------------------
# Loading (cached so Streamlit doesn't reload on every click)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the trained model...")
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_threshold_info():
    with open(THRESHOLD_PATH) as f:
        return json.load(f)


def build_cache(model):
    """One-time: recreate Step 2's test set, score it, and save a small demo file."""
    X, y = load_mitbih(verbose=False)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)  # same split as Step 2
    proba_test = model.predict_proba(X_test).astype(np.float32)
    rng = np.random.default_rng(RANDOM_STATE)
    idx = np.concatenate([                                   # balanced sample for exploring
        rng.choice(np.flatnonzero(y_test == k), min(BEATS_PER_CLASS, np.sum(y_test == k)),
                   replace=False) for k in LABELS])
    np.savez_compressed(CACHE_PATH, y_test=y_test.astype(np.int8), proba_test=proba_test,
                        demo_X=X_test[idx], demo_y=y_test[idx].astype(np.int8),
                        demo_proba=proba_test[idx])


@st.cache_data(show_spinner=False)
def load_cache():
    with np.load(CACHE_PATH) as d:
        return {k: d[k] for k in d.files}


@st.cache_data(show_spinner=False)
def tradeoff_curve(y, proba):
    """Sensitivity + specificity at every threshold (for the trade-off chart)."""
    stats = [safety_stats(y, predict_with_threshold(proba, t)) for t in THRESHOLDS]
    return [s["sensitivity"] for s in stats], [s["specificity"] for s in stats]


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def ecg_paper(ax, t_max):
    """Style an axis like real ECG paper: small box = 40 ms, large box = 200 ms."""
    ax.set_facecolor(PAPER)
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.xaxis.set_minor_locator(MultipleLocator(40))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(which="minor", color=GRID_MINOR, lw=0.5)
    ax.grid(which="major", color=GRID_MAJOR, lw=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", colors=INK, labelsize=8, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, t_max)
    ax.set_ylim(-0.08, 1.08)


def fig_beat(beat, color):
    n = signal_length(beat)
    t = time_axis_ms()[:n]
    fig, ax = plt.subplots(figsize=(9, 3.1))
    ax.plot(t, beat[:n], color=color, lw=1.8)
    ecg_paper(ax, t[-1] + 8)
    ax.set_xlabel("Time after R-peak (ms)", fontsize=9, color=INK)
    ax.set_ylabel("Amplitude (normalized)", fontsize=9, color=INK)
    fig.tight_layout()
    return fig


def fig_probabilities(proba_row):
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    names = [CLASS_NAMES[k] for k in LABELS][::-1]
    vals = proba_row[::-1]
    colors = [ALARM] * 4 + [CALM]                            # reversed: Normal is last
    ax.barh(names, vals, color=colors, height=0.6)
    for yi, v in enumerate(vals):
        ax.text(v + 0.02, yi, f"{v:.0%}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 1.15)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=8, colors=INK, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Model's confidence for each type", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    return fig


def verdict_html(pred, true, p_abn, t):
    flagged = pred != 0
    color = ALARM if flagged else CALM
    title = f"Arrhythmia flagged: {CLASS_NAMES[pred]}" if flagged else "No arrhythmia flagged"
    rule = "at or above" if flagged else "below"
    agree = pred == true
    return f"""
<div style="border-left:6px solid {color};background:{PAPER};padding:0.7rem 1rem;border-radius:4px;color:{INK}">
  <div style="font-size:1.2rem;font-weight:700;color:{color}">{title}</div>
  <div style="margin-top:0.4rem">Chance this beat is abnormal: <b>{p_abn:.0%}</b>, {rule} the {t:.0%} alarm threshold.</div>
  <div>Doctor's label: <b>{CLASS_NAMES[true]}</b></div>
  <div style="margin-top:0.3rem;font-weight:600;color:{CALM if agree else ALARM}">
    {"Model agrees with the doctor" if agree else "Model disagrees with the doctor"}</div>
</div>"""


# ---------------------------------------------------------------------------
# Tab 1: Beat explorer
# ---------------------------------------------------------------------------
def tab_explorer(cache, t):
    X, y, P = cache["demo_X"], cache["demo_y"].astype(int), cache["demo_proba"]
    preds = predict_with_threshold(P, t)

    c1, c2, c3 = st.columns([2.2, 2, 1.3], vertical_alignment="bottom")
    options = ["All types"] + [CLASS_NAMES[k] for k in LABELS]
    choice = c1.selectbox("Heartbeat type (doctor's label)", options)
    only_wrong = c2.toggle("Only beats the model gets wrong")
    new_beat = c3.button("Show another beat", type="primary")

    mask = np.ones(len(y), dtype=bool)
    if choice != "All types":
        mask &= y == options.index(choice) - 1
    if only_wrong:
        mask &= preds != y
    candidates = np.flatnonzero(mask)
    if candidates.size == 0:
        st.info("No beats match at this alarm threshold. Turn off “Only beats the model gets "
                "wrong”, choose another type, or move the threshold.")
        return
    if new_beat or st.session_state.get("beat_idx") not in set(candidates.tolist()):
        st.session_state.beat_idx = int(np.random.default_rng().choice(candidates))

    i = st.session_state.beat_idx
    pred, true, p_abn = int(preds[i]), int(y[i]), 1.0 - float(P[i, 0])
    left, right = st.columns([3, 2])
    with left:
        fig = fig_beat(X[i], ALARM if true != 0 else CALM)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Drawn on ECG paper: each small box is 40 ms, each large box 200 ms, "
                   "the same grid cardiologists read. The beat starts at its R-peak.")
    with right:
        st.markdown(verdict_html(pred, true, p_abn, t), unsafe_allow_html=True)
        fig = fig_probabilities(P[i])
        st.pyplot(fig)
        plt.close(fig)
    st.caption(f"Showing 1 of {candidates.size:,} matching beats from a balanced sample of "
               f"{len(y):,} test beats. The model never saw any of them during training.")


# ---------------------------------------------------------------------------
# Tab 2: Safety trade-off
# ---------------------------------------------------------------------------
def tab_safety(cache, t, t_star):
    y, P = cache["y_test"].astype(int), cache["proba_test"]
    cur = safety_stats(y, predict_with_threshold(P, t))
    base = safety_stats(y, P.argmax(axis=1))                 # untuned: model's top guess

    m = st.columns(4)
    m[0].metric("Arrhythmias caught (sensitivity)", f"{cur['sensitivity']:.1%}",
                f"{(cur['sensitivity'] - base['sensitivity']) * 100:+.1f} pts vs untuned")
    m[1].metric("Normal beats cleared (specificity)", f"{cur['specificity']:.1%}",
                f"{(cur['specificity'] - base['specificity']) * 100:+.1f} pts vs untuned")
    m[2].metric("Missed arrhythmias", f"{cur['fn']:,}",
                f"{cur['fn'] - base['fn']:+,} vs untuned", delta_color="inverse")
    m[3].metric("False alarms", f"{cur['fp']:,}",
                f"{cur['fp'] - base['fp']:+,} vs untuned", delta_color="inverse")
    st.caption(f"Measured on all {len(y):,} test beats. “Untuned” is the model’s plain top guess. "
               "Move the alarm threshold in the sidebar to watch these change.")

    left, right = st.columns(2)
    with left:                                               # trade-off curve
        sens, spec = tradeoff_curve(y, P)
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        ax.plot(THRESHOLDS, sens, color=ALARM, lw=2, label="Arrhythmias caught")
        ax.plot(THRESHOLDS, spec, color=CALM, lw=2, label="Normal beats cleared")
        ax.axvline(t_star, color=MUTED, ls="--", lw=1, label=f"Recommended ({t_star:.2f})")
        ax.axvline(t, color=INK, lw=1.5, label=f"Your threshold ({t:.2f})")
        ax.set_xlabel("Alarm threshold", fontsize=9)
        ax.set_ylabel("Rate on test beats", fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
        ax.set_title("Every threshold trades one error for the other", fontsize=10, loc="left")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:                                              # confusion matrix at threshold t
        cm = confusion_matrix(y, predict_with_threshold(P, t), labels=LABELS)
        rates = cm / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        cmap = LinearSegmentedColormap.from_list("ink", ["#FFFFFF", "#A9BCCF", INK])
        ax.imshow(rates, cmap=cmap, vmin=0, vmax=1)
        for r in range(5):
            for c in range(5):
                ax.text(c, r, f"{cm[r, c]:,}\n{rates[r, c]:.0%}", ha="center", va="center",
                        fontsize=7, color="white" if rates[r, c] > 0.55 else INK)
        ax.set_xticks(range(5), CLASS_SHORT)
        ax.set_yticks(range(5), CLASS_SHORT)
        ax.set_xlabel("Model's call", fontsize=9)
        ax.set_ylabel("Doctor's label", fontsize=9)
        ax.tick_params(labelsize=8, length=0)
        ax.set_title("Where the errors go (diagonal = correct)", fontsize=10, loc="left")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.caption("N normal · S supraventricular · V ventricular · F fusion · Q paced or unclassifiable. "
               "The recommended threshold was chosen on separate out-of-bag data, not these test beats.")


# ---------------------------------------------------------------------------
# Tab 3: Noise lab
# ---------------------------------------------------------------------------
def run_robustness(model, X, y, ns, fset, t):
    """Run every demo beat through the pipeline and score four conditions."""
    rng = np.random.default_rng(0)
    clean_settings = sp.NoiseSettings(powerline_hz=ns.powerline_hz)
    noisy, filtered, clean_filtered, flagged = [], [], [], []
    for beat in X:
        out = sp.run_pipeline(beat, ns, fset, rng)
        noisy.append(out["noisy"])
        filtered.append(out["filtered"])
        flagged.append(out["unreliable"])
        clean_filtered.append(sp.run_pipeline(beat, clean_settings, fset, rng)["filtered"])
    conditions = {"Original beats": X, "Clean beats through your filters": clean_filtered,
                  "With noise, no filters": noisy, "With noise, then your filters": filtered}
    rows = []
    for name, arr in conditions.items():
        pred = predict_with_threshold(model.predict_proba(np.asarray(arr)), t)
        s = safety_stats(y, pred)
        rows.append({"Condition": name, "Correct type": f"{np.mean(pred == y):.1%}",
                     "Arrhythmias caught": f"{s['sensitivity']:.1%}",
                     "Normal beats cleared": f"{s['specificity']:.1%}"})
    return pd.DataFrame(rows), float(np.mean(flagged))


def tab_noise(model, cache, t):
    if "beat_idx" not in st.session_state:
        st.info("Open Beat explorer first to choose a heartbeat.")
        return
    i = st.session_state.beat_idx
    beat, true = cache["demo_X"][i], int(cache["demo_y"][i])
    st.write(f"Real ECG hardware never records a clean signal. Add the noise a hospital monitor "
             f"picks up, then filter it out and see whether the diagnosis survives. "
             f"Working on the beat from Beat explorer (doctor's label: **{CLASS_NAMES[true]}**).")

    noise_col, filter_col = st.columns(2)
    with noise_col:
        st.subheader("1. Add noise")
        mains = st.radio("Power-line frequency where you live", [60, 50], horizontal=True,
                         format_func=lambda f: f"{f} Hz" + (" (Americas)" if f == 60 else " (most other countries)"))
        drift = st.slider("Breathing drift", 0.0, 0.5, 0.25, 0.05,
                          help="Slow wander of the whole signal as the chest rises and falls (0.15–0.4 Hz).")
        hum = st.slider("Power-line hum", 0.0, 0.2, 0.05, 0.01,
                        help="Interference from wall wiring at exactly 60 or 50 Hz.")
        muscle = st.slider("Muscle noise", 0.0, 0.1, 0.03, 0.005,
                           help="Fast random jitter from shivering or tensed muscles. It overlaps the "
                                "ECG's own frequencies, so no filter removes it completely.")
        pop = st.toggle("Loose electrode",
                        help="The electrode lifts off the skin: the signal jumps and the amplifier "
                             "gets stuck at its limit.")
        if st.button("New random noise"):
            st.session_state.noise_seed = st.session_state.get("noise_seed", 0) + 1
    with filter_col:
        st.subheader("2. Filter it out")
        f_base = st.toggle("Baseline removal (removes breathing drift)", value=True,
                           help="Two median filters (200 ms, then 600 ms) estimate the drifting "
                                "baseline, which is subtracted. Published method for MIT-BIH "
                                "(de Chazal et al., 2004). A standard 0.5 Hz high-pass needs about "
                                "2 s of signal, and these beats are about 1 s long.")
        f_notch = st.toggle(f"Notch filter at {mains} Hz (removes power-line hum)", value=True)
        f_low = st.toggle("Low-pass filter (reduces muscle noise)", value=True)
        low_hz = st.slider("Low-pass cutoff (Hz)", 15, 45, 35, 1, disabled=not f_low,
                           help="Lower removes more muscle noise but also blunts the sharp QRS spike.")
        st.caption("Real monitors sample at 250–1,000 Hz. This dataset was resampled to 125 Hz, so "
                   "60 Hz hum sits close to the highest frequency it can represent (62.5 Hz).")

    ns = sp.NoiseSettings(drift, hum, float(mains), muscle, pop)
    fset = sp.FilterSettings(f_base, f_notch, f_low, float(low_hz))
    rng = np.random.default_rng(1000 * st.session_state.get("noise_seed", 0) + i)
    out = sp.run_pipeline(beat, ns, fset, rng)
    signals = [("Original", beat, CALM), ("With noise", out["noisy"], MUTED),
               ("After filters", out["filtered"], INK)]
    probas = model.predict_proba(np.stack([s for _, s, _ in signals]))

    st.subheader("3. Compare")
    plot_col, table_col = st.columns([3, 2])
    with plot_col:
        n = signal_length(beat)
        t_ms = time_axis_ms()[:n]
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 6.2), sharex=True)
        for ax, (name, sig, color) in zip(axes, signals):
            ax.plot(t_ms, sig[:n], color=color, lw=1.6)
            ecg_paper(ax, t_ms[-1] + 8)
            ax.set_title(name, fontsize=9, loc="left", color=INK)
        axes[-1].set_xlabel("Time after R-peak (ms)", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with table_col:
        rows = []
        for (name, _, _), p in zip(signals, probas):
            pred = int(predict_with_threshold(p[None, :], t)[0])
            rejected = out["unreliable"] and name == "After filters"
            rows.append({"Signal": name,
                         "Model's call": "Rejected: unreliable signal" if rejected else CLASS_NAMES[pred],
                         "Chance abnormal": "—" if rejected else f"{1 - p[0]:.0%}",
                         "Matches doctor": "—" if rejected else ("Yes" if pred == true else "No")})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        if out["unreliable"]:
            ms = out["flat_run"] * 1000 // sp.FS
            st.warning(f"Quality check failed: the signal stayed perfectly flat for {ms} ms. That "
                       "happens when an electrode lifts off and the amplifier saturates. Filters "
                       "can’t repair it, so a real monitor shows “Check electrode” instead of a "
                       "diagnosis.")
        else:
            st.caption("Quality check passed: no flat-line or saturation detected.")

    st.subheader("4. Test on every demo beat")
    X, y = cache["demo_X"], cache["demo_y"].astype(int)
    st.write(f"Runs your current noise and filter settings on all {len(y):,} demo beats, "
             "with fresh random noise for each one.")
    if st.button("Run robustness test", type="primary"):
        with st.spinner("Processing every beat..."):
            table, flag_rate = run_robustness(model, X, y, ns, fset, t)
        st.session_state.robust = (table, flag_rate, ns, fset, t)
    if "robust" in st.session_state:
        table, flag_rate, r_ns, r_fset, r_t = st.session_state.robust
        st.dataframe(table, hide_index=True)
        st.caption(
            f"Settings when you ran it: drift {r_ns.baseline_amp:.2f}, hum {r_ns.powerline_amp:.2f} "
            f"at {r_ns.powerline_hz:.0f} Hz, muscle {r_ns.muscle_std:.3f}, loose electrode "
            f"{'on' if r_ns.electrode_pop else 'off'}; threshold {r_t:.2f}. The quality check "
            f"rejected {flag_rate:.0%} of noisy beats (scored anyway above, to show what an "
            "unchecked system would do). “Clean beats through your filters” shows how much the "
            "filters themselves distort a perfect signal.")


# ---------------------------------------------------------------------------
# Tab 4: About
# ---------------------------------------------------------------------------
def tab_about(info, n_test):
    d, tu = info["test_default"], info["test_tuned"]
    st.markdown(f"""
### What this is
A Random Forest trained on 70,043 heartbeats from the MIT-BIH Arrhythmia Database classifies each
beat into one of five AAMI classes. It was tested on {n_test:,} beats it never saw during training.

### Headline result
Tuning the alarm threshold on out-of-bag data cut missed arrhythmias from **{d['fn']:,} to
{tu['fn']:,}**, raising sensitivity from {d['sensitivity']:.1%} to {tu['sensitivity']:.1%} while
specificity stayed at {tu['specificity']:.1%}.

### Limitations
- **Same patients in training and testing.** The Kaggle version of MIT-BIH has no patient IDs, so
  beats from the same 47 people appear on both sides. Accuracy on new patients would be lower.
- **One beat at a time.** Each segment starts at its R-peak, so the model never sees the gap
  before a beat. Supraventricular beats are diagnosed largely from that gap, which is why they
  remain the hardest class.
- **Simulated noise.** The Noise lab adds synthetic noise to already-cleaned beats; real hardware
  noise is messier.
- **Not a medical device.** This is a student research project, not for clinical use.
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="ECG Arrhythmia Detector", page_icon="🫀", layout="wide")

    missing = [p.name for p in (MODEL_PATH, THRESHOLD_PATH) if not p.exists()]
    if missing:
        st.error(f"Missing {', '.join(missing)} in the models folder. Run "
                 "`python step2b_tune_threshold.py` first, then reload this page.")
        st.stop()
    model = load_model()
    info = load_threshold_info()
    if not CACHE_PATH.exists():
        if not DATA_PATH.exists():
            st.error("First launch needs data/mitbih_train.csv to prepare the demo beats. "
                     "Put it in the data folder, then reload this page.")
            st.stop()
        with st.spinner("First launch: preparing the test heartbeats (one time only)..."):
            build_cache(model)
    cache = load_cache()
    t_star = float(info["threshold"])

    with st.sidebar:
        st.header("Alarm threshold")
        st.session_state.setdefault("threshold", t_star)
        t = st.slider("Flag a beat when its chance of being abnormal is at least",
                      0.02, 0.98, step=0.01, key="threshold")
        st.caption(f"Recommended: {t_star:.2f}. Chosen on out-of-bag data to catch at least 95% "
                   "of arrhythmias while clearing at least 95% of normal beats. Lower catches "
                   "more arrhythmias but raises more false alarms.")
        st.button("Reset to recommended",
                  on_click=lambda: st.session_state.update(threshold=t_star))

    st.title("ECG Arrhythmia Detector")
    st.write("Reads a single heartbeat from a real hospital ECG recording and flags it as normal "
             "or one of four arrhythmia types. Trained on the MIT-BIH Arrhythmia Database.")

    tabs = st.tabs(["Beat explorer", "Safety trade-off", "Noise lab", "About"])
    with tabs[0]:
        tab_explorer(cache, t)
    with tabs[1]:
        tab_safety(cache, t, t_star)
    with tabs[2]:
        tab_noise(model, cache, t)
    with tabs[3]:
        tab_about(info, len(cache["y_test"]))


if __name__ == "__main__":
    main()
