"""
03_connectivity.py
------------------
Computes functional connectivity matrices from segmented EEG epochs.

For each saved segment (Baseline/T0/T1/T2), computes pairwise
connectivity between all 22 channels across 6 frequency bands
using three methods:

    1. Magnitude Squared Coherence (MSCoh) — matches Vecchio et al.
    2. Weighted Phase Lag Index   (wPLI)   — robust to volume conduction
    3. Amplitude Envelope Correlation (AEC) — captures slower coupling

Output: one (22x22) matrix per (seizure × window × band × method)
saved as .npy files in results/connectivity_matrices/chb01/

Run after 02_segmentation.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy.signal import hilbert
from itertools import combinations
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DATA_PROC   = ROOT / "data"    / "processed"
ANNOT_DIR   = ROOT / "data"    / "annotations"
RESULTS_DIR = ROOT / "results" / "connectivity_matrices"

# ── Config ─────────────────────────────────────────────────────────────────
SUBJECT  = "chb01"
SFREQ    = 256.0      # Hz

# Frequency bands (Vecchio et al. 2016)
FREQ_BANDS = {
    "delta"  : (2,  4),
    "theta"  : (4,  8),
    "alpha1" : (8,  10),
    "alpha2" : (10, 13),
    "beta1"  : (13, 20),
    "beta2"  : (20, 30),
}

WINDOWS  = ["Baseline", "T0", "T1", "T2"]
METHODS  = ["coherence", "wpli", "aec"]


# ══════════════════════════════════════════════════════════════════════════
# CONNECTIVITY METHOD 1 — Magnitude Squared Coherence
# ══════════════════════════════════════════════════════════════════════════

def compute_coherence_matrix(epochs: np.ndarray,
                              sfreq: float,
                              fmin: float,
                              fmax: float) -> np.ndarray:
    """
    Compute average magnitude squared coherence between all channel pairs
    across all epochs, within a frequency band.

    Parameters
    ----------
    epochs : (n_epochs, n_channels, n_samples)
    sfreq  : sampling frequency in Hz
    fmin   : band lower bound in Hz
    fmax   : band upper bound in Hz

    Returns
    -------
    conn_matrix : (n_channels, n_channels) symmetric matrix
                  values between 0 and 1
    """
    n_epochs, n_channels, n_samples = epochs.shape
    conn_matrix = np.zeros((n_channels, n_channels))

    for i, j in combinations(range(n_channels), 2):
        coh_values = []

        for ep in range(n_epochs):
            x = epochs[ep, i, :]
            y = epochs[ep, j, :]

            # Compute coherence using Welch's method
            freqs, Cxy = signal.coherence(
                x, y,
                fs       = sfreq,
                nperseg  = min(256, n_samples),
                noverlap = 128
            )

            # Average coherence within the frequency band
            band_mask = (freqs >= fmin) & (freqs <= fmax)
            if band_mask.sum() > 0:
                coh_values.append(np.mean(Cxy[band_mask]))

        # Average across epochs
        avg_coh = np.mean(coh_values) if coh_values else 0.0
        conn_matrix[i, j] = avg_coh
        conn_matrix[j, i] = avg_coh   # symmetric

    # Diagonal = 1 (perfect self-coherence)
    np.fill_diagonal(conn_matrix, 1.0)
    return conn_matrix


# ══════════════════════════════════════════════════════════════════════════
# CONNECTIVITY METHOD 2 — Weighted Phase Lag Index (wPLI)
# ══════════════════════════════════════════════════════════════════════════

def bandpass_filter(data: np.ndarray,
                    sfreq: float,
                    fmin: float,
                    fmax: float) -> np.ndarray:
    """
    Apply a bandpass FIR filter to a 1D or 2D array.
    data shape: (n_samples,) or (n_channels, n_samples)
    """
    nyq    = sfreq / 2.0
    low    = fmin / nyq
    high   = fmax / nyq
    # Clamp to valid range
    low    = max(low,  0.001)
    high   = min(high, 0.999)
    b, a   = signal.butter(4, [low, high], btype='band')
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        return np.array([signal.filtfilt(b, a, row) for row in data])


def compute_wpli_matrix(epochs: np.ndarray,
                         sfreq: float,
                         fmin: float,
                         fmax: float) -> np.ndarray:
    """
    Compute weighted Phase Lag Index between all channel pairs.

    wPLI measures the consistency of phase differences while
    down-weighting near-zero phase differences (volume conduction).
    This makes it more reliable for scalp EEG than coherence.

    wPLI = |E[Im(C)]| / E[|Im(C)|]
    where C is the cross-spectrum and Im is the imaginary part.

    Parameters
    ----------
    epochs : (n_epochs, n_channels, n_samples)

    Returns
    -------
    conn_matrix : (n_channels, n_channels) symmetric matrix
                  values between 0 and 1
    """
    n_epochs, n_channels, n_samples = epochs.shape
    conn_matrix = np.zeros((n_channels, n_channels))

    for i, j in combinations(range(n_channels), 2):
        imag_parts = []

        for ep in range(n_epochs):
            # Bandpass filter both signals
            x = bandpass_filter(epochs[ep, i, :], sfreq, fmin, fmax)
            y = bandpass_filter(epochs[ep, j, :], sfreq, fmin, fmax)

            # Analytic signal via Hilbert transform
            x_analytic = hilbert(x)
            y_analytic = hilbert(y)

            # Cross-spectrum
            cross_spectrum = x_analytic * np.conj(y_analytic)

            # Imaginary part of cross-spectrum
            imag_cross = np.imag(cross_spectrum)
            imag_parts.append(imag_cross)

        # Stack all epochs: (n_epochs * n_samples,)
        all_imag = np.concatenate(imag_parts)

        # wPLI formula
        numerator   = np.abs(np.mean(all_imag))
        denominator = np.mean(np.abs(all_imag))

        wpli = numerator / denominator if denominator > 1e-10 else 0.0

        conn_matrix[i, j] = wpli
        conn_matrix[j, i] = wpli

    np.fill_diagonal(conn_matrix, 1.0)
    return conn_matrix


# ══════════════════════════════════════════════════════════════════════════
# CONNECTIVITY METHOD 3 — Amplitude Envelope Correlation (AEC)
# ══════════════════════════════════════════════════════════════════════════

def compute_aec_matrix(epochs: np.ndarray,
                        sfreq: float,
                        fmin: float,
                        fmax: float) -> np.ndarray:
    """
    Compute Amplitude Envelope Correlation between all channel pairs.

    AEC measures the correlation between the amplitude envelopes
    of two bandpass-filtered signals. It captures slower coupling
    dynamics complementary to phase-based measures.

    Steps:
        1. Bandpass filter → isolate frequency band
        2. Hilbert transform → get analytic signal
        3. Take absolute value → amplitude envelope
        4. Pearson correlation between envelopes

    Parameters
    ----------
    epochs : (n_epochs, n_channels, n_samples)

    Returns
    -------
    conn_matrix : (n_channels, n_channels) symmetric matrix
                  values between -1 and 1 (usually 0 to 1)
    """
    n_epochs, n_channels, n_samples = epochs.shape
    conn_matrix = np.zeros((n_channels, n_channels))

    for i, j in combinations(range(n_channels), 2):
        correlations = []

        for ep in range(n_epochs):
            # Bandpass filter
            x = bandpass_filter(epochs[ep, i, :], sfreq, fmin, fmax)
            y = bandpass_filter(epochs[ep, j, :], sfreq, fmin, fmax)

            # Amplitude envelopes via Hilbert transform
            env_x = np.abs(hilbert(x))
            env_y = np.abs(hilbert(y))

            # Pearson correlation between envelopes
            if np.std(env_x) > 1e-10 and np.std(env_y) > 1e-10:
                corr = np.corrcoef(env_x, env_y)[0, 1]
                correlations.append(corr)

        avg_corr = np.mean(correlations) if correlations else 0.0
        conn_matrix[i, j] = avg_corr
        conn_matrix[j, i] = avg_corr

    np.fill_diagonal(conn_matrix, 1.0)
    return conn_matrix


# ══════════════════════════════════════════════════════════════════════════
# MAIN COMPUTATION LOOP
# ══════════════════════════════════════════════════════════════════════════

def compute_all_connectivity(subject: str) -> pd.DataFrame:
    """
    For every (seizure × window × band × method) combination,
    compute the connectivity matrix and save it to disk.

    Returns a summary DataFrame of all computed matrices.
    """
    subject_proc_dir   = DATA_PROC   / subject
    subject_result_dir = RESULTS_DIR / subject
    subject_result_dir.mkdir(parents=True, exist_ok=True)

    # Load segmentation summary to know which files exist
    summary_path = ANNOT_DIR / f"{subject}_segments_summary.csv"
    summary_df   = pd.read_csv(summary_path)

    # Get unique seizure IDs
    seizure_ids  = summary_df['seizure_id'].unique()

    print(f"Computing connectivity for {subject}")
    print(f"  Seizures : {len(seizure_ids)}")
    print(f"  Windows  : {WINDOWS}")
    print(f"  Bands    : {list(FREQ_BANDS.keys())}")
    print(f"  Methods  : {METHODS}")

    total = len(seizure_ids) * len(WINDOWS) * len(FREQ_BANDS) * len(METHODS)
    print(f"  Total matrices to compute: {total}")
    print()

    records = []

    # ── Outer loop: seizures ───────────────────────────────────────────────
    for sz_id in seizure_ids:
        print(f"── {sz_id} ──────────────────────────────────────")

        # ── Middle loop: windows ───────────────────────────────────────────
        for window in WINDOWS:
            npy_path = subject_proc_dir / f"{sz_id}_{window}.npy"

            if not npy_path.exists():
                print(f"  [skip] {npy_path.name} not found")
                continue

            # Load epochs: (n_epochs, n_channels, n_samples)
            epochs = np.load(npy_path)
            print(f"  [{window}] loaded {epochs.shape} ...", end=" ")

            # ── Inner loop: frequency bands ────────────────────────────────
            band_results = {}

            for band_name, (fmin, fmax) in FREQ_BANDS.items():

                band_results[band_name] = {}

                for method in METHODS:

                    # Compute connectivity matrix
                    if method == "coherence":
                        matrix = compute_coherence_matrix(
                            epochs, SFREQ, fmin, fmax)

                    elif method == "wpli":
                        matrix = compute_wpli_matrix(
                            epochs, SFREQ, fmin, fmax)

                    elif method == "aec":
                        matrix = compute_aec_matrix(
                            epochs, SFREQ, fmin, fmax)

                    # Save matrix
                    fname     = (f"{sz_id}_{window}_"
                                 f"{band_name}_{method}.npy")
                    save_path = subject_result_dir / fname
                    np.save(save_path, matrix)

                    band_results[band_name][method] = {
                        "mean" : float(np.mean(matrix[
                                    np.triu_indices_from(matrix, k=1)])),
                        "std"  : float(np.std(matrix[
                                    np.triu_indices_from(matrix, k=1)])),
                    }

                    records.append({
                        "subject"   : subject,
                        "seizure_id": sz_id,
                        "window"    : window,
                        "band"      : band_name,
                        "method"    : method,
                        "mean_conn" : band_results[band_name][method]["mean"],
                        "std_conn"  : band_results[band_name][method]["std"],
                        "saved_as"  : fname,
                    })

            print("done")

            # Print a quick summary table for this window
            print(f"       {'':10s} " +
                  " ".join(f"{m:>10s}" for m in METHODS))
            for band_name in FREQ_BANDS:
                row_str = f"       {band_name:10s} "
                for method in METHODS:
                    val = band_results[band_name][method]["mean"]
                    row_str += f"{val:>10.4f} "
                print(row_str)
            print()

    # ── Save results summary ───────────────────────────────────────────────
    results_df   = pd.DataFrame(records)
    summary_path = ANNOT_DIR / f"{subject}_connectivity_summary.csv"
    results_df.to_csv(summary_path, index=False)

    print("=" * 60)
    print(f"Connectivity computation complete!")
    print(f"Matrices saved → {subject_result_dir}")
    print(f"Summary   saved → {summary_path}")
    print(f"Total matrices  : {len(records)}")

    return results_df


# ══════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════

def sanity_check(subject: str) -> None:
    """
    Load one saved matrix and print basic properties.
    A connectivity matrix should be:
    - Symmetric
    - Values between 0 and 1
    - Diagonal = 1
    - Higher values in some bands than others
    """
    subject_result_dir = RESULTS_DIR / subject
    test_file = subject_result_dir / "chb01_sz01_T2_theta_coherence.npy"

    if not test_file.exists():
        print("Sanity check file not found — skipping")
        return

    matrix = np.load(test_file)
    upper  = matrix[np.triu_indices_from(matrix, k=1)]

    print("\n── Sanity Check: chb01_sz01_T2_theta_coherence ──")
    print(f"  Shape         : {matrix.shape}")
    print(f"  Symmetric     : "
          f"{'YES' if np.allclose(matrix, matrix.T) else 'NO'}")
    print(f"  Diagonal mean : {np.mean(np.diag(matrix)):.4f}  (should be 1.0)")
    print(f"  Off-diag mean : {np.mean(upper):.4f}")
    print(f"  Off-diag min  : {np.min(upper):.4f}")
    print(f"  Off-diag max  : {np.max(upper):.4f}")
    print(f"  Values in [0,1]: "
          f"{'YES' if upper.min() >= 0 and upper.max() <= 1 else 'NO'}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print(f"  Connectivity Pipeline — Subject: {SUBJECT}")
    print("=" * 60)

    # Compute all connectivity matrices
    results_df = compute_all_connectivity(SUBJECT)

    # Sanity check one matrix
    sanity_check(SUBJECT)

    print(f"\n── Preview of results ──")
    print(results_df.groupby(['window', 'band', 'method'])[
        'mean_conn'].mean().unstack('method').round(4).to_string())

    print(f"\nNext step: run 04_graph_metrics.py")