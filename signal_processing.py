"""
signal_processing.py
Simulated ECG hardware noise, the filters a BME uses to remove it, and a
signal-quality check that rejects segments no filter can repair.

Every function works on ONE beat: 187 samples at 125 Hz, where trailing
zeros are padding. Noise and filters touch only the real samples.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

from ecg_utils import N_SAMPLES, SAMPLING_RATE_HZ, signal_length

FS = SAMPLING_RATE_HZ      # 125 Hz
MEDIAN_WINDOWS_MS = (200, 600)  # de Chazal et al. 2004: 200 ms removes QRS + P waves,
                                # 600 ms removes T waves -> what remains is the drifting baseline.
                                # (A 0.5 Hz high-pass needs ~2 s of signal; our beats are ~1 s.)
NOTCH_Q = 30               # notch width: 60 Hz / 30 = 2 Hz wide
FLAT_RUN_SAMPLES = 8       # 8 identical samples (64 ms) = amplifier stuck at its rail
MIN_FILTER_LEN = 30        # too-short segments are left unfiltered


@dataclass
class NoiseSettings:
    baseline_amp: float = 0.0     # breathing drift amplitude
    powerline_amp: float = 0.0    # mains hum amplitude
    powerline_hz: float = 60.0    # 60 Hz (Americas) or 50 Hz (most other countries)
    muscle_std: float = 0.0       # muscle (EMG) noise strength
    electrode_pop: bool = False   # loose electrode


@dataclass
class FilterSettings:
    baseline: bool = False
    notch: bool = False
    lowpass: bool = False
    lowpass_hz: float = 35.0


# ---------------------------------------------------------------------------
# 1) Noise: what real hardware adds to the signal
# ---------------------------------------------------------------------------
def add_noise(beat, ns, rng):
    """Return the beat's real samples with simulated noise added."""
    n = signal_length(beat)
    x = beat[:n].astype(np.float64).copy()
    t = np.arange(n) / FS                                    # time in seconds

    if ns.baseline_amp > 0:                                  # breathing: 0.15-0.4 Hz
        f = rng.uniform(0.15, 0.4)
        x += ns.baseline_amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    if ns.powerline_amp > 0:                                 # wall-socket hum
        x += ns.powerline_amp * np.sin(2 * np.pi * ns.powerline_hz * t + rng.uniform(0, 2 * np.pi))
    if ns.muscle_std > 0:                                    # muscle: broadband random noise
        x += rng.normal(0, ns.muscle_std, n)
    if ns.electrode_pop and n > 40:                          # loose electrode (applied last:
        s = int(rng.integers(20, n - 20))                    # a saturated amplifier clips
        x[s:] += 0.5                                         # everything, noise included)
        x[s:s + 15] = x.max() + 0.3                          # 120 ms stuck flat at the rail
    return x


# ---------------------------------------------------------------------------
# 2) Filters: what a BME applies before the signal reaches the model
# ---------------------------------------------------------------------------
def apply_filters(x, fset, mains_hz):
    """Zero-phase filtering (filtfilt runs forward + backward, so waves don't shift in time)."""
    y = x.copy()
    if len(y) < MIN_FILTER_LEN:
        return y
    if fset.baseline:                                        # removes slow breathing drift:
        base = y                                             # estimate the baseline with two
        for ms in MEDIAN_WINDOWS_MS:                         # median filters, then subtract it
            k = int(ms * FS / 1000) | 1                      # window in samples (odd)
            base = median_filter(base, size=k, mode="reflect")
        y = y - base
    if fset.notch:                                           # removes one exact frequency
        b, a = iirnotch(mains_hz, NOTCH_Q, fs=FS)
        y = filtfilt(b, a, y)
    if fset.lowpass:                                         # removes fast jitter above cutoff
        sos = butter(4, fset.lowpass_hz, btype="lowpass", fs=FS, output="sos")
        y = sosfiltfilt(sos, y)
    return y


# ---------------------------------------------------------------------------
# 3) Quality check: catch what filters can't fix
# ---------------------------------------------------------------------------
def longest_flat_run(x, tol=1e-9):
    """Length of the longest stretch of identical samples. Real ECG always wiggles."""
    best = run = 1
    for d in np.abs(np.diff(x)):
        run = run + 1 if d < tol else 1
        best = max(best, run)
    return best


# ---------------------------------------------------------------------------
# 4) Full pipeline: noise -> quality check -> filters -> normalize
# ---------------------------------------------------------------------------
def normalize(x):
    """Rescale to 0-1, like the training data."""
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


def to_model_input(core):
    """Normalize the real samples and re-attach the zero padding the model expects."""
    out = np.zeros(N_SAMPLES, dtype=np.float32)
    out[:len(core)] = normalize(core)
    return out


def run_pipeline(beat, ns, fset, rng):
    noisy = add_noise(beat, ns, rng)
    flat = longest_flat_run(noisy)                           # check BEFORE filtering:
    filtered = apply_filters(noisy, fset, ns.powerline_hz)   # filters would smear the evidence
    return {
        "noisy": to_model_input(noisy),
        "filtered": to_model_input(filtered),
        "flat_run": flat,
        "unreliable": flat >= FLAT_RUN_SAMPLES,
    }
