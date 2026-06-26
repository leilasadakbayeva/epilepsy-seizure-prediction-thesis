"""
02_segmentation.py
------------------
Create multi-subject seizure-aligned EEG segments from preprocessed FIF files.

The script reads results/dataset_screening/seizure_eligibility.csv, keeps only
usable seizures for processing, and extracts four 3-minute windows per seizure:

    Baseline : same-recording interictal data at least 600 seconds from seizures
    T0       : 9-6 minutes before seizure onset
    T1       : 6-3 minutes before seizure onset
    T2       : 3-0 minutes before seizure onset

Each 3-minute window is split into non-overlapping 2-second epochs and saved as
an array with shape:

    90 x 22 x 512

Run examples:
    python src/02_segmentation.py --subject chb02
    python src/02_segmentation.py --all
    python src/02_segmentation.py --subject chb02 --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd


# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"
SEGMENTS_DIR = ROOT / "data" / "segments"
SCREENING_PATH = ROOT / "results" / "dataset_screening" / "seizure_eligibility.csv"
SEGMENTATION_DIR = ROOT / "results" / "segmentation"
SUMMARY_PATH = SEGMENTATION_DIR / "segments_summary.csv"


# Segmentation parameters
WINDOW_SEC = 180
EPOCH_SEC = 2
BASELINE_BUFFER_SEC = 600
EXPECTED_SFREQ = 256
EXPECTED_N_EPOCHS = 90
EXPECTED_N_CHANNELS = 22
EXPECTED_N_SAMPLES = 512
EXPECTED_SHAPE = (EXPECTED_N_EPOCHS, EXPECTED_N_CHANNELS, EXPECTED_N_SAMPLES)

WINDOW_OFFSETS = {
    "Baseline": None,
    "T0": (-9 * 60, -6 * 60),
    "T1": (-6 * 60, -3 * 60),
    "T2": (-3 * 60, 0),
}

SUMMARY_COLUMNS = [
    "subject_id",
    "seizure_id",
    "recording_file",
    "seizure_start_sec",
    "seizure_end_sec",
    "window",
    "window_start_sec",
    "window_end_sec",
    "baseline_source_recording_file",
    "n_epochs",
    "n_channels",
    "n_samples",
    "output_file",
    "status",
    "error_message",
]

REQUIRED_COLUMNS = {
    "subject_id",
    "seizure_id",
    "recording_file",
    "seizure_start_sec",
    "seizure_end_sec",
    "usable_for_analysis",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create multi-subject CHB-MIT seizure segments from FIF files."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--subject",
        help="Process one subject, for example: --subject chb02",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all subjects with usable seizures in the screening table.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate .npy segment files even when outputs already exist.",
    )
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    """Convert common CSV boolean representations to a boolean mask."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def load_eligibility_table(path: Path = SCREENING_PATH) -> pd.DataFrame:
    """Load the seizure eligibility table and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Missing screening table: {path}")

    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    df = df.copy()
    df["usable_for_analysis"] = as_bool(df["usable_for_analysis"])
    df["subject_id"] = df["subject_id"].astype(str)
    df["seizure_id"] = df["seizure_id"].astype(str)
    df["recording_file"] = df["recording_file"].astype(str)
    df["seizure_start_sec"] = pd.to_numeric(df["seizure_start_sec"])
    df["seizure_end_sec"] = pd.to_numeric(df["seizure_end_sec"])
    return df


def select_seizures_to_process(
    eligibility_df: pd.DataFrame, subject: str | None
) -> pd.DataFrame:
    """Select usable seizure rows, optionally for one subject."""
    selected = eligibility_df[eligibility_df["usable_for_analysis"]].copy()
    if subject is not None:
        selected = selected[selected["subject_id"] == subject].copy()

    return selected.sort_values(["subject_id", "seizure_id"]).reset_index(drop=True)


def preprocessed_path_for_recording(subject_id: str, recording_file: str) -> Path:
    """Return the preprocessed FIF path for one subject recording."""
    recording_stem = Path(recording_file).stem
    return (
        DATA_PROC
        / subject_id
        / f"{recording_stem}_preprocessed_raw.fif"
    )


def output_path_for_segment(subject_id: str, seizure_id: str, window: str) -> Path:
    """Return the .npy output path for one seizure window."""
    return SEGMENTS_DIR / subject_id / f"{seizure_id}_{window}.npy"


def load_preprocessed_raw(subject_id: str, recording_file: str) -> mne.io.BaseRaw:
    """Load a preprocessed FIF file without reading raw EDF or filtering again."""
    fif_path = preprocessed_path_for_recording(subject_id, recording_file)
    if not fif_path.exists():
        raise FileNotFoundError(f"Missing preprocessed FIF: {fif_path}")
    return mne.io.read_raw_fif(fif_path, preload=False, verbose=False)


def recording_duration_sec(raw: mne.io.BaseRaw) -> float:
    """Return the exclusive-end duration of a Raw object in seconds."""
    return float(raw.n_times) / float(raw.info["sfreq"])


def extract_window(raw: mne.io.BaseRaw, start_sec: float, end_sec: float) -> np.ndarray:
    """Extract a (n_channels, n_samples) data window from a Raw object."""
    sfreq = float(raw.info["sfreq"])
    expected_samples = int(round((end_sec - start_sec) * sfreq))
    duration_sec = recording_duration_sec(raw)

    if start_sec < 0:
        raise ValueError(f"window starts before recording: {start_sec:.3f}s")
    if end_sec > duration_sec:
        raise ValueError(
            f"window ends after recording: {end_sec:.3f}s > {duration_sec:.3f}s"
        )

    start_sample = int(round(start_sec * sfreq))
    stop_sample = start_sample + expected_samples
    data = raw.get_data(start=start_sample, stop=stop_sample)

    if data.shape[1] != expected_samples:
        raise ValueError(
            f"expected {expected_samples} samples, extracted {data.shape[1]}"
        )
    return data


def split_into_epochs(window_data: np.ndarray, sfreq: float) -> np.ndarray:
    """Split one 3-minute window into non-overlapping 2-second epochs."""
    epoch_samples = int(round(EPOCH_SEC * sfreq))
    n_channels, n_samples = window_data.shape

    if n_samples % epoch_samples != 0:
        raise ValueError(
            f"{n_samples} samples cannot be split into {epoch_samples}-sample epochs"
        )

    n_epochs = n_samples // epoch_samples
    epochs = window_data.reshape(n_channels, n_epochs, epoch_samples).transpose(1, 0, 2)
    return epochs


def seizure_intervals_for_recording(
    eligibility_df: pd.DataFrame, subject_id: str, recording_file: str
) -> list[tuple[float, float]]:
    """
    Return all known seizure intervals for a subject recording.

    Baseline exclusion uses all rows in the screening table, not only usable rows,
    because unusable seizures are still ictal periods.
    """
    rows = eligibility_df[
        (eligibility_df["subject_id"] == subject_id)
        & (eligibility_df["recording_file"] == recording_file)
    ]
    return [
        (float(row["seizure_start_sec"]), float(row["seizure_end_sec"]))
        for _, row in rows.iterrows()
    ]


def find_baseline_window(
    raw: mne.io.BaseRaw,
    seizure_intervals: list[tuple[float, float]],
    window_sec: int = WINDOW_SEC,
    buffer_sec: int = BASELINE_BUFFER_SEC,
) -> tuple[float, float]:
    """Find the earliest same-recording baseline window outside seizure buffers."""
    duration_sec = recording_duration_sec(raw)
    if duration_sec < window_sec:
        raise ValueError(
            f"recording is too short for baseline: {duration_sec:.3f}s < {window_sec}s"
        )

    forbidden = []
    for seizure_start, seizure_end in seizure_intervals:
        forbidden.append(
            (
                max(0.0, seizure_start - buffer_sec),
                min(duration_sec, seizure_end + buffer_sec),
            )
        )
    forbidden = sorted(forbidden)

    merged = []
    for start, end in forbidden:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    candidate_start = 0.0
    for forbidden_start, forbidden_end in merged:
        candidate_end = candidate_start + window_sec
        if candidate_end <= forbidden_start:
            return candidate_start, candidate_end
        candidate_start = max(candidate_start, forbidden_end)

    candidate_end = candidate_start + window_sec
    if candidate_end <= duration_sec:
        return candidate_start, candidate_end

    raise ValueError(
        f"no {window_sec}s baseline at least {buffer_sec}s from seizures"
    )


def window_bounds(
    window: str,
    seizure_start_sec: float,
    raw: mne.io.BaseRaw,
    seizure_intervals: list[tuple[float, float]],
) -> tuple[float, float]:
    """Return start/end seconds for one named window."""
    if window == "Baseline":
        return find_baseline_window(raw, seizure_intervals)

    start_offset, end_offset = WINDOW_OFFSETS[window]
    return seizure_start_sec + start_offset, seizure_start_sec + end_offset


def validate_epochs(epochs: np.ndarray) -> None:
    """Ensure output arrays match the required segmentation shape."""
    if epochs.shape != EXPECTED_SHAPE:
        raise ValueError(f"expected shape {EXPECTED_SHAPE}, got {epochs.shape}")


def empty_summary_value():
    """Return a CSV-friendly missing value."""
    return np.nan


def summary_record(
    row: pd.Series,
    window: str,
    output_file: Path,
    status: str,
    window_start_sec=empty_summary_value(),
    window_end_sec=empty_summary_value(),
    baseline_source_recording_file: str = "",
    n_epochs=empty_summary_value(),
    n_channels=empty_summary_value(),
    n_samples=empty_summary_value(),
    error_message: str = "",
) -> dict:
    """Build one segmentation summary row."""
    return {
        "subject_id": row["subject_id"],
        "seizure_id": row["seizure_id"],
        "recording_file": row["recording_file"],
        "seizure_start_sec": float(row["seizure_start_sec"]),
        "seizure_end_sec": float(row["seizure_end_sec"]),
        "window": window,
        "window_start_sec": window_start_sec,
        "window_end_sec": window_end_sec,
        "baseline_source_recording_file": baseline_source_recording_file,
        "n_epochs": n_epochs,
        "n_channels": n_channels,
        "n_samples": n_samples,
        "output_file": str(output_file),
        "status": status,
        "error_message": error_message,
    }


def existing_segment_record(row: pd.Series, window: str, output_file: Path) -> dict:
    """Build a summary row for an existing output that is not overwritten."""
    try:
        existing = np.load(output_file, mmap_mode="r")
        n_epochs, n_channels, n_samples = existing.shape
        error_message = "output exists; pass --overwrite to regenerate"
        status = "skipped_existing"
        if existing.shape != EXPECTED_SHAPE:
            status = "failed_existing_shape"
            error_message = (
                f"existing output has shape {existing.shape}; "
                f"expected {EXPECTED_SHAPE}; pass --overwrite to regenerate"
            )
    except Exception as exc:
        n_epochs = n_channels = n_samples = empty_summary_value()
        status = "failed_existing_read"
        error_message = f"could not inspect existing output: {exc}"

    return summary_record(
        row=row,
        window=window,
        output_file=output_file,
        status=status,
        n_epochs=n_epochs,
        n_channels=n_channels,
        n_samples=n_samples,
        error_message=error_message,
    )


def failed_window_records(row: pd.Series, message: str) -> list[dict]:
    """Build failed summary rows for every window of a seizure."""
    return [
        summary_record(
            row=row,
            window=window,
            output_file=output_path_for_segment(
                str(row["subject_id"]), str(row["seizure_id"]), window
            ),
            status="failed",
            error_message=message,
        )
        for window in WINDOW_OFFSETS
    ]


def process_window(
    row: pd.Series,
    window: str,
    raw: mne.io.BaseRaw,
    all_seizures: pd.DataFrame,
    overwrite: bool,
) -> dict:
    """Extract, epoch, validate, and save one seizure window."""
    subject_id = str(row["subject_id"])
    seizure_id = str(row["seizure_id"])
    recording_file = str(row["recording_file"])
    output_file = output_path_for_segment(subject_id, seizure_id, window)

    if output_file.exists() and not overwrite:
        print(f"  [skip] {window}: existing output {output_file.name}")
        return existing_segment_record(row, window, output_file)

    try:
        seizure_intervals = seizure_intervals_for_recording(
            all_seizures, subject_id, recording_file
        )
        start_sec, end_sec = window_bounds(
            window,
            float(row["seizure_start_sec"]),
            raw,
            seizure_intervals,
        )
        window_data = extract_window(raw, start_sec, end_sec)
        epochs = split_into_epochs(window_data, float(raw.info["sfreq"]))
        validate_epochs(epochs)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_file, epochs)
        print(f"  [ok] {window}: {start_sec:.1f}-{end_sec:.1f}s -> {epochs.shape}")

        return summary_record(
            row=row,
            window=window,
            window_start_sec=start_sec,
            window_end_sec=end_sec,
            baseline_source_recording_file=recording_file if window == "Baseline" else "",
            n_epochs=epochs.shape[0],
            n_channels=epochs.shape[1],
            n_samples=epochs.shape[2],
            output_file=output_file,
            status="saved",
        )
    except Exception as exc:
        print(f"  [fail] {window}: {exc}")
        return summary_record(
            row=row,
            window=window,
            output_file=output_file,
            status="failed",
            error_message=str(exc),
        )


def process_seizures(
    seizures: pd.DataFrame, all_seizures: pd.DataFrame, overwrite: bool
) -> list[dict]:
    """Process selected usable seizures and return summary records."""
    records = []

    for idx, row in seizures.iterrows():
        subject_id = str(row["subject_id"])
        seizure_id = str(row["seizure_id"])
        recording_file = str(row["recording_file"])

        print(
            f"\n[{idx + 1:03d}/{len(seizures):03d}] "
            f"{subject_id} {seizure_id} {recording_file}"
        )

        try:
            raw = load_preprocessed_raw(subject_id, recording_file)
            print(
                f"  FIF: sfreq={float(raw.info['sfreq']):.1f}, "
                f"channels={len(raw.ch_names)}, duration={recording_duration_sec(raw):.1f}s"
            )
        except Exception as exc:
            message = str(exc)
            print(f"  [fail] could not load FIF: {message}")
            records.extend(failed_window_records(row, message))
            continue

        for window in WINDOW_OFFSETS:
            records.append(
                process_window(
                    row=row,
                    window=window,
                    raw=raw,
                    all_seizures=all_seizures,
                    overwrite=overwrite,
                )
            )

        if hasattr(raw, "close"):
            raw.close()

    return records


def save_summary(records: list[dict]) -> pd.DataFrame:
    """Write the segmentation summary CSV."""
    SEGMENTATION_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(records, columns=SUMMARY_COLUMNS)
    summary = summary.sort_values(["subject_id", "seizure_id", "window"]).reset_index(
        drop=True
    )
    summary.to_csv(SUMMARY_PATH, index=False)
    return summary


def main() -> None:
    """Run multi-subject segmentation from preprocessed FIF recordings."""
    args = parse_args()
    eligibility = load_eligibility_table()
    subject = args.subject if args.subject else None
    seizures = select_seizures_to_process(eligibility, subject)

    if seizures.empty:
        if subject:
            print(f"No usable seizures found for {subject} in {SCREENING_PATH}.")
        else:
            print(f"No usable seizures found in {SCREENING_PATH}.")
        return

    print("=" * 60)
    print("CHB-MIT Multi-Subject Segmentation")
    print("=" * 60)
    print(f"Screening table : {SCREENING_PATH}")
    print(f"Subjects        : {', '.join(sorted(seizures['subject_id'].unique()))}")
    print(f"Usable seizures : {len(seizures)}")
    print(f"Overwrite       : {'YES' if args.overwrite else 'NO'}")
    print(f"Segments dir    : {SEGMENTS_DIR}")
    print(f"Expected shape  : {EXPECTED_SHAPE}")

    records = process_seizures(seizures, eligibility, overwrite=args.overwrite)
    summary = save_summary(records)

    saved = int((summary["status"] == "saved").sum())
    skipped = int(summary["status"].astype(str).str.startswith("skipped").sum())
    failed = int(summary["status"].astype(str).str.startswith("failed").sum())

    print("\n" + "=" * 60)
    print("Segmentation complete")
    print("=" * 60)
    print(f"Saved this run   : {saved}")
    print(f"Skipped existing : {skipped}")
    print(f"Failed windows   : {failed}")
    print(f"Summary saved    : {SUMMARY_PATH}")
    print(f"Summary rows     : {len(summary)}")


if __name__ == "__main__":
    main()
