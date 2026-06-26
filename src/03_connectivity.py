"""
03_connectivity.py
------------------
Compute multi-subject functional connectivity matrices from segmented EEG.

The script reads results/segmentation/segments_summary.csv, keeps only rows
where status == "saved", and computes six frequency bands x three connectivity
methods for each saved segment window.

Run examples:
    python src/03_connectivity.py --subject chb02
    python src/03_connectivity.py --all
    python src/03_connectivity.py --subject chb02 --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import hilbert


# Paths
ROOT = Path(__file__).resolve().parent.parent
SEGMENTS_DIR = ROOT / "data" / "segments"
SEGMENTATION_SUMMARY = ROOT / "results" / "segmentation" / "segments_summary.csv"
CONNECTIVITY_MATRIX_DIR = ROOT / "results" / "connectivity_matrices"
CONNECTIVITY_DIR = ROOT / "results" / "connectivity"
CONNECTIVITY_SUMMARY = CONNECTIVITY_DIR / "connectivity_summary.csv"


# Connectivity parameters
SFREQ = 256.0
EXPECTED_SEGMENT_SHAPE = (90, 22, 512)
EPS = 1e-12

FREQ_BANDS = {
    "delta": (2, 4),
    "theta": (4, 8),
    "alpha1": (8, 10),
    "alpha2": (10, 13),
    "beta1": (13, 20),
    "beta2": (20, 30),
}

METHODS = ["coherence", "wpli", "aec"]

SUMMARY_COLUMNS = [
    "subject_id",
    "seizure_id",
    "window",
    "band",
    "method",
    "mean_conn",
    "std_conn",
    "saved_as",
    "status",
    "error_message",
]

REQUIRED_SEGMENT_COLUMNS = {
    "subject_id",
    "seizure_id",
    "window",
    "output_file",
    "status",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute CHB-MIT connectivity matrices from saved segments."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--subject",
        help="Process one subject, for example: --subject chb02",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all subjects with saved segment windows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate matrices even when .npy outputs already exist.",
    )
    return parser.parse_args()


def load_segments_summary(path: Path = SEGMENTATION_SUMMARY) -> pd.DataFrame:
    """Load segmentation summary and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Missing segmentation summary: {path}")

    df = pd.read_csv(path)
    missing = sorted(REQUIRED_SEGMENT_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    df = df.copy()
    df["subject_id"] = df["subject_id"].astype(str)
    df["seizure_id"] = df["seizure_id"].astype(str)
    df["window"] = df["window"].astype(str)
    df["output_file"] = df["output_file"].astype(str)
    df["status"] = df["status"].astype(str)
    return df


def select_saved_segments(summary: pd.DataFrame, subject: str | None) -> pd.DataFrame:
    """Select saved segment rows, optionally for one subject."""
    selected = summary[summary["status"].str.lower() == "saved"].copy()
    if subject is not None:
        selected = selected[selected["subject_id"] == subject].copy()

    return selected.sort_values(["subject_id", "seizure_id", "window"]).reset_index(
        drop=True
    )


def segment_path_for_row(row: pd.Series) -> Path:
    """Return canonical segment path for a segmentation summary row."""
    return (
        SEGMENTS_DIR
        / str(row["subject_id"])
        / f"{row['seizure_id']}_{row['window']}.npy"
    )


def matrix_filename(seizure_id: str, window: str, band: str, method: str) -> str:
    """Return one connectivity matrix filename."""
    return f"{seizure_id}_{window}_{band}_{method}.npy"


def matrix_path(subject_id: str, seizure_id: str, window: str, band: str, method: str) -> Path:
    """Return one connectivity matrix output path."""
    return (
        CONNECTIVITY_MATRIX_DIR
        / subject_id
        / matrix_filename(seizure_id, window, band, method)
    )


def expected_task_count(n_windows: int) -> int:
    """Return expected matrix count for a number of segment windows."""
    return n_windows * len(FREQ_BANDS) * len(METHODS)


def load_segment(row: pd.Series) -> np.ndarray:
    """Load one saved segment array from data/segments."""
    canonical_path = segment_path_for_row(row)
    if not canonical_path.exists():
        raise FileNotFoundError(f"Missing segment file: {canonical_path}")

    summary_path = Path(str(row["output_file"]))
    if summary_path.name != canonical_path.name:
        raise ValueError(
            f"summary output_file does not match expected segment name: "
            f"{summary_path.name} != {canonical_path.name}"
        )

    epochs = np.load(canonical_path)
    if epochs.shape != EXPECTED_SEGMENT_SHAPE:
        raise ValueError(f"expected segment shape {EXPECTED_SEGMENT_SHAPE}, got {epochs.shape}")
    return epochs


def compute_coherence_matrix(epochs: np.ndarray, sfreq: float, fmin: float, fmax: float) -> np.ndarray:
    """
    Compute average magnitude-squared coherence between all channel pairs.

    The CSD/Welch calls are vectorized across epochs and channel pairs.
    """
    _, _, n_samples = epochs.shape
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

    numerator = np.abs(pxy) ** 2
    denominator = pxx[:, :, None, :] * pxx[:, None, :, :]
    coherence = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0,
    )

    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band_mask):
        raise ValueError(f"no coherence frequencies found for band {fmin}-{fmax} Hz")

    matrix = coherence[..., band_mask].mean(axis=(0, -1))
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.clip(matrix, 0.0, 1.0)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def bandpass_epochs(epochs: np.ndarray, sfreq: float, fmin: float, fmax: float) -> np.ndarray:
    """Bandpass-filter all epochs once for a frequency band."""
    nyquist = sfreq / 2.0
    low = max(fmin / nyquist, 0.001)
    high = min(fmax / nyquist, 0.999)
    sos = signal.butter(4, [low, high], btype="band", output="sos")
    return signal.sosfiltfilt(sos, epochs, axis=-1)


def compute_wpli_from_analytic(analytic: np.ndarray) -> np.ndarray:
    """Compute weighted phase lag index from band-limited analytic signals."""
    _, n_channels, _ = analytic.shape
    cross = analytic[:, :, None, :] * np.conj(analytic[:, None, :, :])
    imag_cross = np.imag(cross)

    numerator = np.abs(np.mean(imag_cross, axis=(0, 3)))
    denominator = np.mean(np.abs(imag_cross), axis=(0, 3))
    matrix = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > EPS,
    )

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.clip(matrix, 0.0, 1.0)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def compute_aec_from_analytic(analytic: np.ndarray) -> np.ndarray:
    """Compute amplitude envelope correlation from band-limited analytic signals."""
    envelopes = np.abs(analytic)
    epoch_corrs = np.array([np.corrcoef(envelopes[epoch]) for epoch in range(envelopes.shape[0])])

    matrix = np.nanmean(epoch_corrs, axis=0)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.clip(matrix, -1.0, 1.0)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def matrix_stats(matrix: np.ndarray) -> tuple[float, float]:
    """Return mean and standard deviation over off-diagonal connections."""
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    return float(np.mean(upper)), float(np.std(upper))

def validate_matrix(matrix: np.ndarray) -> None:
    """Validate one connectivity matrix before recording it as usable."""
    if matrix.shape != (22, 22):
        raise ValueError(f"expected matrix shape (22, 22), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix contains non-finite values")
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        raise ValueError("matrix is not symmetric")

def save_matrix(path: Path, matrix: np.ndarray) -> None:
    """Save one connectivity matrix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)


def summary_record(
    row: pd.Series,
    band: str,
    method: str,
    output_path: Path,
    status: str,
    mean_conn=np.nan,
    std_conn=np.nan,
    error_message: str = "",
) -> dict:
    """Build one connectivity summary row."""
    return {
        "subject_id": row["subject_id"],
        "seizure_id": row["seizure_id"],
        "window": row["window"],
        "band": band,
        "method": method,
        "mean_conn": mean_conn,
        "std_conn": std_conn,
        "saved_as": output_path.name,
        "status": status,
        "error_message": error_message,
    }


def failed_records_for_segment(row: pd.Series, error_message: str) -> list[dict]:
    """Build failed summary rows for all band/method tasks under one segment."""
    records = []
    for band in FREQ_BANDS:
        for method in METHODS:
            output_path = matrix_path(
                str(row["subject_id"]),
                str(row["seizure_id"]),
                str(row["window"]),
                band,
                method,
            )
            records.append(
                summary_record(
                    row=row,
                    band=band,
                    method=method,
                    output_path=output_path,
                    status="failed",
                    error_message=error_message,
                )
            )
    return records


def existing_matrix_record(row: pd.Series, band: str, method: str, output_path: Path) -> dict:
    """Build a summary row for an existing matrix skipped by overwrite policy."""
    try:
        matrix = np.load(output_path)
        validate_matrix(matrix)
        mean_conn, std_conn = matrix_stats(matrix)
        status = "skipped_existing"
        error_message = "output exists; pass --overwrite to regenerate"
    except Exception as exc:
        mean_conn = std_conn = np.nan
        status = "failed_existing_read"
        error_message = f"could not inspect existing matrix: {exc}"

    return summary_record(
        row=row,
        band=band,
        method=method,
        output_path=output_path,
        status=status,
        mean_conn=mean_conn,
        std_conn=std_conn,
        error_message=error_message,
    )


def compute_segment_connectivity(row: pd.Series, overwrite: bool) -> list[dict]:
    """Compute all band/method matrices for one saved segment window."""
    records = []
    subject_id = str(row["subject_id"])
    seizure_id = str(row["seizure_id"])
    window = str(row["window"])

    try:
        epochs = load_segment(row)
        print(f"\n{subject_id} {seizure_id} {window}: loaded {epochs.shape}")
    except Exception as exc:
        message = str(exc)
        print(f"\n{subject_id} {seizure_id} {window}: [fail] {message}")
        return failed_records_for_segment(row, message)

    analytic_by_band: dict[str, np.ndarray] = {}

    for band, (fmin, fmax) in FREQ_BANDS.items():
        for method in METHODS:
            output_path = matrix_path(subject_id, seizure_id, window, band, method)

            if output_path.exists() and not overwrite:
                print(f"  [skip] {band:6s} {method:9s} {output_path.name}")
                records.append(existing_matrix_record(row, band, method, output_path))
                continue

            try:
                if method == "coherence":
                    matrix = compute_coherence_matrix(epochs, SFREQ, fmin, fmax)
                else:
                    if band not in analytic_by_band:
                        filtered = bandpass_epochs(epochs, SFREQ, fmin, fmax)
                        analytic_by_band[band] = hilbert(filtered, axis=-1)
                    analytic = analytic_by_band[band]

                    if method == "wpli":
                        matrix = compute_wpli_from_analytic(analytic)
                    elif method == "aec":
                        matrix = compute_aec_from_analytic(analytic)
                    else:
                        raise ValueError(f"Unknown connectivity method: {method}")

                validate_matrix(matrix)
                save_matrix(output_path, matrix)
                mean_conn, std_conn = matrix_stats(matrix)
                print(
                    f"  [ok]   {band:6s} {method:9s} "
                    f"mean={mean_conn:.4f} std={std_conn:.4f}"
                )
                records.append(
                    summary_record(
                        row=row,
                        band=band,
                        method=method,
                        output_path=output_path,
                        status="computed",
                        mean_conn=mean_conn,
                        std_conn=std_conn,
                    )
                )
            except Exception as exc:
                print(f"  [fail] {band:6s} {method:9s} {exc}")
                records.append(
                    summary_record(
                        row=row,
                        band=band,
                        method=method,
                        output_path=output_path,
                        status="failed",
                        error_message=str(exc),
                    )
                )

    return records


def save_summary(records: list[dict]) -> pd.DataFrame:
    """Write the connectivity summary CSV."""
    CONNECTIVITY_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(records, columns=SUMMARY_COLUMNS)
    summary = summary.sort_values(
        ["subject_id", "seizure_id", "window", "band", "method"]
    ).reset_index(drop=True)
    summary.to_csv(CONNECTIVITY_SUMMARY, index=False)
    return summary


def print_run_summary(summary: pd.DataFrame, expected: int) -> None:
    """Print the required run summary."""
    computed = int((summary["status"] == "computed").sum())
    skipped = int(summary["status"].astype(str).str.startswith("skipped").sum())
    failed = int(summary["status"].astype(str).str.startswith("failed").sum())

    print("\n" + "=" * 60)
    print("Connectivity complete")
    print("=" * 60)
    print(f"Expected matrices: {expected}")
    print(f"Computed:          {computed}")
    print(f"Skipped:           {skipped}")
    print(f"Failed:            {failed}")
    print(f"Summary saved:     {CONNECTIVITY_SUMMARY}")


def main() -> None:
    """Run multi-subject connectivity from saved segmented windows."""
    args = parse_args()
    segment_summary = load_segments_summary()
    subject = args.subject if args.subject else None
    segments = select_saved_segments(segment_summary, subject)

    if segments.empty:
        if subject:
            print(f"No saved segment windows found for {subject} in {SEGMENTATION_SUMMARY}.")
        else:
            print(f"No saved segment windows found in {SEGMENTATION_SUMMARY}.")
        return

    expected = expected_task_count(len(segments))

    print("=" * 60)
    print("CHB-MIT Multi-Subject Connectivity")
    print("=" * 60)
    print(f"Segmentation summary : {SEGMENTATION_SUMMARY}")
    print(f"Subjects             : {', '.join(sorted(segments['subject_id'].unique()))}")
    print(f"Saved windows        : {len(segments)}")
    print(f"Bands                : {len(FREQ_BANDS)}")
    print(f"Methods              : {len(METHODS)}")
    print(f"Expected matrices    : {expected}")
    print(f"Overwrite            : {'YES' if args.overwrite else 'NO'}")
    print(f"Output directory     : {CONNECTIVITY_MATRIX_DIR}")

    records = []
    for _, row in segments.iterrows():
        records.extend(compute_segment_connectivity(row, overwrite=args.overwrite))

    summary = save_summary(records)
    print_run_summary(summary, expected)


if __name__ == "__main__":
    main()
