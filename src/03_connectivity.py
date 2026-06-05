"""
03_connectivity.py
------------------
Computes functional connectivity matrices from segmented EEG epochs.

Colab-ready version:
- resumes from existing matrix files instead of trusting the CSV only
- mirrors each saved matrix and the summary CSV to Google Drive immediately
- avoids repeated filtering for wPLI/AEC by filtering each band once per window
- uses vectorized coherence across channel pairs/epochs
"""

import os
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import hilbert

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"
ANNOT_DIR = ROOT / "data" / "annotations"
RESULTS_DIR = ROOT / "results" / "connectivity_matrices"

# Canonical Google Drive project folder used by the Colab notebook.
# Override with: export THESIS_DRIVE_ROOT=/content/drive/MyDrive/thesis
DRIVE_ROOT = Path(os.environ.get("THESIS_DRIVE_ROOT", "/content/drive/MyDrive/thesis"))

# ── Config ─────────────────────────────────────────────────────────────────
SUBJECT = "chb01"
SFREQ = 256.0  # Hz

FREQ_BANDS = {
    "delta": (2, 4),
    "theta": (4, 8),
    "alpha1": (8, 10),
    "alpha2": (10, 13),
    "beta1": (13, 20),
    "beta2": (20, 30),
}

WINDOWS = ["Baseline", "T0", "T1", "T2"]
METHODS = ["coherence", "wpli", "aec"]
EPS = 1e-12


def drive_available() -> bool:
    """Return True only when the canonical Drive thesis folder is mounted."""
    return DRIVE_ROOT.exists() and (DRIVE_ROOT / "data").exists()


def mirror_to_drive(local_path: Path) -> None:
    """Copy a local repo file to the same relative path under Google Drive."""
    if not drive_available():
        return
    local_path = Path(local_path)
    try:
        rel = local_path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return
    drive_path = DRIVE_ROOT / rel
    drive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, drive_path)


def restore_from_drive(local_dir: Path) -> None:
    """Restore a local directory from Drive if the matching Drive directory exists."""
    if not drive_available():
        return
    local_dir = Path(local_dir)
    try:
        rel = local_dir.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return
    drive_dir = DRIVE_ROOT / rel
    if drive_dir.exists():
        local_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(drive_dir, local_dir, dirs_exist_ok=True)
        print(f"Restored from Drive: {drive_dir} → {local_dir}")


def save_matrix(path: Path, matrix: np.ndarray) -> None:
    """Save matrix locally, then mirror it to Drive immediately if available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)
    mirror_to_drive(path)


def save_summary(records: list[dict], progress_path: Path) -> pd.DataFrame:
    """Write a de-duplicated connectivity summary and mirror it to Drive."""
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(
            subset=["subject", "seizure_id", "window", "band", "method"],
            keep="last",
        ).sort_values(["seizure_id", "window", "band", "method"])
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(progress_path, index=False)
    mirror_to_drive(progress_path)
    return df


# ══════════════════════════════════════════════════════════════════════════
# Connectivity methods
# ══════════════════════════════════════════════════════════════════════════

def compute_coherence_matrix(epochs: np.ndarray, sfreq: float, fmin: float, fmax: float) -> np.ndarray:
    """
    Compute average magnitude squared coherence between all channel pairs.

    epochs shape: (n_epochs, n_channels, n_samples)
    """
    n_epochs, n_channels, n_samples = epochs.shape
    nperseg = min(256, n_samples)
    noverlap = min(128, nperseg // 2)

    freqs, pxy = signal.csd(
        epochs[:, :, None, :],
        epochs[:, None, :, :],
        fs=sfreq,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=-1,
    )
    _, pxx = signal.welch(
        epochs,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=-1,
    )

    coh = np.abs(pxy) ** 2 / (pxx[:, :, None, :] * pxx[:, None, :, :] + EPS)
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    conn_matrix = coh[..., band_mask].mean(axis=(0, -1))

    conn_matrix = np.nan_to_num(conn_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    conn_matrix = np.clip(conn_matrix, 0.0, 1.0)
    np.fill_diagonal(conn_matrix, 1.0)
    return conn_matrix


def bandpass_epochs(epochs: np.ndarray, sfreq: float, fmin: float, fmax: float) -> np.ndarray:
    """Bandpass-filter a 3D epoch array with a stable Butterworth IIR filter."""
    nyq = sfreq / 2.0
    low = max(fmin / nyq, 0.001)
    high = min(fmax / nyq, 0.999)
    sos = signal.butter(4, [low, high], btype="band", output="sos")
    return signal.sosfiltfilt(sos, epochs, axis=-1)


def compute_wpli_from_analytic(analytic: np.ndarray) -> np.ndarray:
    """Compute wPLI from band-limited analytic signals."""
    _, n_channels, _ = analytic.shape
    conn_matrix = np.zeros((n_channels, n_channels), dtype=float)

    for i, j in combinations(range(n_channels), 2):
        imag_cross = np.imag(analytic[:, i, :] * np.conj(analytic[:, j, :])).ravel()
        numerator = abs(np.mean(imag_cross))
        denominator = np.mean(np.abs(imag_cross))
        val = numerator / denominator if denominator > EPS else 0.0
        conn_matrix[i, j] = val
        conn_matrix[j, i] = val

    np.fill_diagonal(conn_matrix, 1.0)
    return conn_matrix


def compute_aec_from_analytic(analytic: np.ndarray) -> np.ndarray:
    """Compute AEC by averaging per-epoch envelope correlations."""
    envelopes = np.abs(analytic)
    epoch_corrs = []

    for ep in range(envelopes.shape[0]):
        corr = np.corrcoef(envelopes[ep])
        epoch_corrs.append(corr)

    conn_matrix = np.nanmean(epoch_corrs, axis=0)
    conn_matrix = np.nan_to_num(conn_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    conn_matrix = np.clip(conn_matrix, -1.0, 1.0)
    np.fill_diagonal(conn_matrix, 1.0)
    return conn_matrix


# ══════════════════════════════════════════════════════════════════════════
# Main computation loop
# ══════════════════════════════════════════════════════════════════════════

def matrix_filename(sz_id: str, window: str, band_name: str, method: str) -> str:
    return f"{sz_id}_{window}_{band_name}_{method}.npy"


def matrix_stats(matrix: np.ndarray) -> tuple[float, float]:
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    return float(np.mean(upper)), float(np.std(upper))


def compute_all_connectivity(subject: str) -> pd.DataFrame:
    """
    Compute all missing connectivity matrices.

    Resume rule: matrix files are the source of truth. The summary CSV is rebuilt
    from actual matrix files, so partial seizures are not skipped accidentally.
    """
    subject_proc_dir = DATA_PROC / subject
    subject_result_dir = RESULTS_DIR / subject
    progress_path = ANNOT_DIR / f"{subject}_connectivity_summary.csv"

    # Self-contained Colab recovery: restore existing outputs if the notebook has not already done it.
    restore_from_drive(subject_proc_dir)
    restore_from_drive(ANNOT_DIR)
    restore_from_drive(subject_result_dir)

    subject_result_dir.mkdir(parents=True, exist_ok=True)

    summary_path = ANNOT_DIR / f"{subject}_segments_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run 02_segmentation.py first or restore annotations from Drive."
        )

    summary_df = pd.read_csv(summary_path)
    seizure_ids = list(summary_df["seizure_id"].drop_duplicates())

    expected_tasks = []
    for sz_id in seizure_ids:
        for window in WINDOWS:
            npy_path = subject_proc_dir / f"{sz_id}_{window}.npy"
            if not npy_path.exists():
                print(f"[warning] Missing segment file: {npy_path.name}; skipping that window")
                continue
            for band_name, (fmin, fmax) in FREQ_BANDS.items():
                for method in METHODS:
                    expected_tasks.append((sz_id, window, band_name, fmin, fmax, method, npy_path))

    n_existing = sum(
        (subject_result_dir / matrix_filename(sz, win, band, method)).exists()
        for sz, win, band, _, _, method, _ in expected_tasks
    )

    print(f"\nComputing connectivity for {subject}")
    print(f"  Drive mirror : {'ON' if drive_available() else 'OFF'} → {DRIVE_ROOT}")
    print(f"  Expected     : {len(expected_tasks)} matrices")
    print(f"  Existing     : {n_existing} matrices")
    print(f"  Remaining    : {len(expected_tasks) - n_existing} matrices\n")

    records = []
    current_window_key = None
    epochs = None
    analytic_by_band = {}

    for task_idx, (sz_id, window, band_name, fmin, fmax, method, npy_path) in enumerate(expected_tasks, start=1):
        fname = matrix_filename(sz_id, window, band_name, method)
        save_path = subject_result_dir / fname

        # Load a window only once, then reuse it for all bands/methods.
        window_key = (sz_id, window)
        if window_key != current_window_key:
            current_window_key = window_key
            analytic_by_band = {}
            epochs = np.load(npy_path)
            print(f"── {sz_id} {window}: loaded {epochs.shape}")

        if save_path.exists():
            matrix = np.load(save_path)
            status = "loaded"
        else:
            if method == "coherence":
                matrix = compute_coherence_matrix(epochs, SFREQ, fmin, fmax)
            else:
                if band_name not in analytic_by_band:
                    filtered = bandpass_epochs(epochs, SFREQ, fmin, fmax)
                    analytic_by_band[band_name] = hilbert(filtered, axis=-1)
                analytic = analytic_by_band[band_name]

                if method == "wpli":
                    matrix = compute_wpli_from_analytic(analytic)
                elif method == "aec":
                    matrix = compute_aec_from_analytic(analytic)
                else:
                    raise ValueError(f"Unknown method: {method}")

            save_matrix(save_path, matrix)
            status = "computed"

        mean_conn, std_conn = matrix_stats(matrix)
        records.append(
            {
                "subject": subject,
                "seizure_id": sz_id,
                "window": window,
                "band": band_name,
                "method": method,
                "mean_conn": mean_conn,
                "std_conn": std_conn,
                "saved_as": fname,
            }
        )

        # Save a resumable summary after every matrix. It is small and protects Colab progress.
        save_summary(records, progress_path)
        print(
            f"  [{task_idx:03d}/{len(expected_tasks)}] {status:8s} "
            f"{window:8s} {band_name:6s} {method:9s} mean={mean_conn:.4f}"
        )

    final_df = save_summary(records, progress_path)

    print("=" * 60)
    print("Connectivity computation complete!")
    print(f"Matrices saved  → {subject_result_dir}")
    print(f"Summary saved   → {progress_path}")
    print(f"Total rows      : {len(final_df)}")
    print(f"Expected rows   : {len(expected_tasks)}")

    if len(final_df) != len(expected_tasks):
        print("[warning] Summary row count does not match expected task count.")

    return final_df


# ══════════════════════════════════════════════════════════════════════════
# Sanity check
# ══════════════════════════════════════════════════════════════════════════

def sanity_check(subject: str) -> None:
    subject_result_dir = RESULTS_DIR / subject
    files = sorted(subject_result_dir.glob("*.npy"))

    if not files:
        print("Sanity check file not found — skipping")
        return

    test_file = files[0]
    matrix = np.load(test_file)
    upper = matrix[np.triu_indices_from(matrix, k=1)]

    print(f"\n── Sanity Check: {test_file.name} ──")
    print(f"  Shape          : {matrix.shape}")
    print(f"  Symmetric      : {'YES' if np.allclose(matrix, matrix.T) else 'NO'}")
    print(f"  Diagonal mean  : {np.mean(np.diag(matrix)):.4f}")
    print(f"  Off-diag mean  : {np.mean(upper):.4f}")
    print(f"  Off-diag min   : {np.min(upper):.4f}")
    print(f"  Off-diag max   : {np.max(upper):.4f}")
    print(f"  Finite values  : {'YES' if np.isfinite(matrix).all() else 'NO'}")


if __name__ == "__main__":
    print("=" * 60)
    print(f"  Connectivity Pipeline — Subject: {SUBJECT}")
    print("=" * 60)

    results_df = compute_all_connectivity(SUBJECT)
    sanity_check(SUBJECT)

    if not results_df.empty:
        print("\n── Preview of results ──")
        print(
            results_df.groupby(["window", "band", "method"])["mean_conn"]
            .mean()
            .unstack("method")
            .round(4)
            .to_string()
        )

    print("\nNext step: run 04_graph_metrics.py")
