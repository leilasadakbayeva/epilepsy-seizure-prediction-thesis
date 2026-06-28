"""
01_preprocessing.py
-------------------
Preprocess CHB-MIT EDF recordings selected by dataset screening.

This script reads results/dataset_screening/seizure_eligibility.csv, selects
recordings linked to usable seizures, enforces the common 22-channel bipolar
montage, filters the EEG, and saves reusable MNE FIF files under:

    data/processed/<subject_id>/<recording_stem>_preprocessed_raw.fif

Run examples:
    python src/01_preprocessing.py --subject chb02
    python src/01_preprocessing.py --all
    python src/01_preprocessing.py --subject chb01 --overwrite
"""

import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd


# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
SCREENING_PATH = ROOT / "results" / "dataset_screening" / "seizure_eligibility.csv"
PREPROCESSING_DIR = ROOT / "results" / "preprocessing"
PREPROCESSING_LOG = PREPROCESSING_DIR / "preprocessing_log.csv"


# Preprocessing parameters
EXPECTED_SFREQ = 256.0
SFREQ_TOLERANCE = 1e-6
L_FREQ = 0.5
H_FREQ = 47.0


# Required 22-channel bipolar montage used by the thesis pipeline.
REQUIRED_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
    "P7-T7",
    "T7-FT9",
    "FT9-FT10",
    "FT10-T8",
]

CHANNELS_TO_DROP = {"T8-P8-1"}
NORMALIZED_CHANNELS_TO_DROP = {
    str(channel).strip().upper() for channel in CHANNELS_TO_DROP
}

LOG_COLUMNS = [
    "subject_id",
    "recording_file",
    "output_file",
    "sampling_frequency",
    "n_channels",
    "status",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess screened CHB-MIT EDF recordings."
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
        help="Regenerate FIF files even when outputs already exist.",
    )
    return parser.parse_args()


def load_eligibility_table(path: Path = SCREENING_PATH) -> pd.DataFrame:
    """Load and validate the dataset-screening seizure eligibility table."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run src/00_dataset_screening.py first."
        )

    df = pd.read_csv(path)
    required = {"subject_id", "recording_file", "usable_for_analysis"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {', '.join(missing)}"
        )

    df["usable_for_analysis"] = df["usable_for_analysis"].apply(parse_bool)
    return df


def parse_bool(value) -> bool:
    """Parse booleans stored as bools, strings, or numeric flags."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def select_recordings_to_process(
    eligibility_df: pd.DataFrame, subject: str | None
) -> pd.DataFrame:
    """
    Select unique subject/recording pairs linked to usable seizures.

    The screening row remains the source of truth for subject_id and
    recording_file, while duplicate seizure rows for the same recording are
    collapsed.
    """
    selected = eligibility_df[eligibility_df["usable_for_analysis"]].copy()
    if subject is not None:
        selected = selected[selected["subject_id"] == subject].copy()

    recordings = (
        selected[["subject_id", "recording_file"]]
        .drop_duplicates()
        .sort_values(["subject_id", "recording_file"])
        .reset_index(drop=True)
    )
    return recordings


def normalize_channel_name(channel_name: str) -> str:
    """
    Normalize MNE channel names to the canonical montage spelling.

    MNE renames duplicated channels by appending running numbers. The first
    duplicate can appear as T8-P8-0 and should be treated as canonical T8-P8.
    The second duplicate T8-P8-1 is intentionally dropped before renaming.
    """
    normalized = str(channel_name).strip().upper()
    if normalized.endswith("-0"):
        return normalized[:-2]
    return normalized


def normalize_for_drop(channel_name: str) -> str:
    """Normalize only for matching explicit duplicate channels to drop."""
    return str(channel_name).strip().upper()


def prepare_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Drop duplicate channels, canonicalize names, and enforce channel order.

    Raises:
        ValueError if any required channel is missing after normalization.
    """
    drop_channels = [
        channel
        for channel in raw.ch_names
        if normalize_for_drop(channel) in NORMALIZED_CHANNELS_TO_DROP
    ]
    if drop_channels:
        raw.drop_channels(drop_channels)

    rename_map = {}
    for channel in raw.ch_names:
        canonical = normalize_channel_name(channel)
        if canonical != channel:
            rename_map[channel] = canonical
    if rename_map:
        raw.rename_channels(rename_map)

    available = set(raw.ch_names)
    missing = [channel for channel in REQUIRED_CHANNELS if channel not in available]
    if missing:
        raise ValueError(
            "Missing required channels after normalization: "
            + ", ".join(missing)
        )

    raw.pick_channels(REQUIRED_CHANNELS, ordered=True)
    return raw


def output_path_for_recording(subject_id: str, recording_file: str) -> Path:
    """Return the FIF output path for one subject recording."""
    recording_stem = Path(recording_file).stem
    return (
        DATA_PROC
        / subject_id
        / f"{recording_stem}_preprocessed_raw.fif"
    )


def preprocess_recording(edf_path: Path) -> mne.io.BaseRaw:
    """Load one EDF, enforce channels, verify sfreq, and apply filters."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Some CHB-MIT EDF headers contain acquisition dates that are outside
    # the FIFF date range accepted by MNE when saving. The thesis uses
    # seizure annotations in seconds from file start, so the absolute
    # calendar date is not needed.
    raw.set_meas_date(None)

    raw = prepare_channels(raw)

    sfreq = float(raw.info["sfreq"])
    if abs(sfreq - EXPECTED_SFREQ) > SFREQ_TOLERANCE:
        raise ValueError(
            f"Sampling frequency mismatch: expected {EXPECTED_SFREQ} Hz, got {sfreq} Hz"
        )

    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, method="fir", verbose=False)
    return raw


def save_preprocessed_raw(raw: mne.io.BaseRaw, output_path: Path, overwrite: bool) -> None:
    """Save one preprocessed Raw object as FIF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(output_path, overwrite=overwrite, verbose=False)


def load_existing_log() -> pd.DataFrame:
    """Load an existing preprocessing log, or return an empty log frame."""
    if not PREPROCESSING_LOG.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    try:
        log_df = pd.read_csv(PREPROCESSING_LOG)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=LOG_COLUMNS)

    for column in LOG_COLUMNS:
        if column not in log_df.columns:
            log_df[column] = np.nan
    return log_df[LOG_COLUMNS]


def save_log(records: list[dict]) -> pd.DataFrame:
    """Create or update the preprocessing log."""
    PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_existing_log()
    new = pd.DataFrame(records, columns=LOG_COLUMNS)
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["subject_id", "recording_file"], keep="last"
    )
    combined = combined.sort_values(["subject_id", "recording_file"]).reset_index(
        drop=True
    )
    combined.to_csv(PREPROCESSING_LOG, index=False)
    return combined


def log_record(
    subject_id: str,
    recording_file: str,
    output_file: Path,
    sampling_frequency,
    n_channels,
    status: str,
    error_message: str = "",
) -> dict:
    """Build one preprocessing log record."""
    return {
        "subject_id": subject_id,
        "recording_file": recording_file,
        "output_file": str(output_file),
        "sampling_frequency": sampling_frequency,
        "n_channels": n_channels,
        "status": status,
        "error_message": error_message,
    }


def process_recordings(recordings: pd.DataFrame, overwrite: bool) -> list[dict]:
    """Preprocess all selected subject recordings."""
    records = []

    for idx, row in recordings.iterrows():
        subject_id = str(row["subject_id"])
        recording_file = str(row["recording_file"])
        edf_path = DATA_RAW / subject_id / recording_file
        output_path = output_path_for_recording(subject_id, recording_file)

        print(
            f"\n[{idx + 1:03d}/{len(recordings):03d}] "
            f"{subject_id} {recording_file}"
        )

        if output_path.exists() and not overwrite:
            print(f"  [skip] existing output: {output_path.name}")
            records.append(
                log_record(
                    subject_id,
                    recording_file,
                    output_path,
                    sampling_frequency=np.nan,
                    n_channels=np.nan,
                    status="skipped_existing",
                    error_message="output exists; pass --overwrite to regenerate",
                )
            )
            continue

        if not edf_path.exists():
            message = f"Missing EDF file: {edf_path}"
            print(f"  [skip] {message}")
            records.append(
                log_record(
                    subject_id,
                    recording_file,
                    output_path,
                    sampling_frequency=np.nan,
                    n_channels=np.nan,
                    status="skipped_missing_edf",
                    error_message=message,
                )
            )
            continue

        try:
            raw = preprocess_recording(edf_path)
            save_preprocessed_raw(raw, output_path, overwrite=overwrite)
            sfreq = float(raw.info["sfreq"])
            n_channels = len(raw.ch_names)
            print(f"  [ok] saved {output_path}")
            print(f"       sfreq={sfreq}, channels={n_channels}")
            records.append(
                log_record(
                    subject_id,
                    recording_file,
                    output_path,
                    sampling_frequency=sfreq,
                    n_channels=n_channels,
                    status="processed",
                )
            )
        except Exception as exc:
            print(f"  [error] {exc}")
            records.append(
                log_record(
                    subject_id,
                    recording_file,
                    output_path,
                    sampling_frequency=np.nan,
                    n_channels=np.nan,
                    status="failed",
                    error_message=str(exc),
                )
            )

    return records


def main() -> None:
    """Run multi-subject preprocessing from the screening eligibility table."""
    args = parse_args()
    eligibility = load_eligibility_table()
    subject = args.subject if args.subject else None
    recordings = select_recordings_to_process(eligibility, subject)

    if recordings.empty:
        if subject:
            print(f"No usable recordings found for {subject} in {SCREENING_PATH}.")
        else:
            print(f"No usable recordings found in {SCREENING_PATH}.")
        return

    print("=" * 60)
    print("CHB-MIT Multi-Subject Preprocessing")
    print("=" * 60)
    print(f"Screening table : {SCREENING_PATH}")
    print(f"Subjects        : {', '.join(sorted(recordings['subject_id'].unique()))}")
    print(f"Recordings      : {len(recordings)}")
    print(f"Overwrite       : {'YES' if args.overwrite else 'NO'}")
    print(f"Bandpass        : {L_FREQ}-{H_FREQ} Hz")
    print(f"Required chans  : {len(REQUIRED_CHANNELS)}")

    records = process_recordings(recordings, overwrite=args.overwrite)
    log_df = save_log(records)

    processed = sum(record["status"] == "processed" for record in records)
    skipped = sum(record["status"].startswith("skipped") for record in records)
    failed = sum(record["status"] == "failed" for record in records)

    print("\n" + "=" * 60)
    print("Preprocessing complete")
    print("=" * 60)
    print(f"Processed this run : {processed}")
    print(f"Skipped this run   : {skipped}")
    print(f"Failed this run    : {failed}")
    print(f"Log saved          : {PREPROCESSING_LOG}")
    print(f"Total log rows     : {len(log_df)}")


if __name__ == "__main__":
    main()