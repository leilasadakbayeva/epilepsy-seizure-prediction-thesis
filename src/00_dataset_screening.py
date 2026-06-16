"""
00_dataset_screening.py
-----------------------
Screen CHB-MIT subjects and annotated seizures before running the full
connectivity and graph-theory pipeline.

The script reads seizure annotations from the existing project annotation
format in data/annotations/ and, when needed, from CHB-MIT summary text files
under data/raw/<subject>/. For each annotated seizure, it checks whether the
recording can support the thesis segmentation design:

    Baseline : 3-minute interictal window away from seizures
    T0       : 9-6 minutes before seizure onset
    T1       : 6-3 minutes before seizure onset
    T2       : 3-0 minutes before seizure onset

Raw EDF files are inspected only for metadata. No EEG data are loaded into
memory.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
ANNOT_DIR = ROOT / "data" / "annotations"
RESULTS_DIR = ROOT / "results" / "dataset_screening"


# Segmentation requirements
WINDOW_SEC = 180
PREICTAL_SEC = 540
BASELINE_BUFFER_SEC = 600
EXPECTED_SFREQ = 256.0
SFREQ_TOLERANCE = 1e-6
CHANNELS_TO_DROP = {"T8-P8-1"}
NORMALIZED_CHANNELS_TO_DROP = {
    str(channel).strip().upper() for channel in CHANNELS_TO_DROP
}
MNE_MODULE = None
MNE_IMPORT_ATTEMPTED = False

# Common 22-channel bipolar montage used after dropping the duplicated T8-P8.
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


SEIZURE_COLUMNS = [
    "subject_id",
    "recording_file",
    "seizure_id",
    "seizure_start_sec",
    "seizure_end_sec",
    "available_preictal_sec",
    "has_required_preictal",
    "sampling_frequency",
    "n_channels_raw",
    "n_channels_used",
    "channel_names_raw",
    "channel_names_used",
    "missing_required_channels",
    "extra_channels",
    "channel_status",
    "montage_status",
    "usable_for_analysis",
    "exclusion_reason",
]

SUMMARY_COLUMNS = [
    "subject_id",
    "total_seizures",
    "usable_seizures",
    "excluded_seizures",
    "usable_for_study",
    "notes",
]

FLOW_COLUMNS = [
    "count_type",
    "exclusion_reason",
    "count",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Screen CHB-MIT seizure eligibility for the thesis pipeline."
    )
    parser.add_argument(
        "subjects",
        nargs="*",
        help=(
            "Optional subject IDs to screen, for example: chb01 chb05. "
            "If omitted, all annotated subjects are screened."
        ),
    )
    parser.add_argument(
        "--baseline-scope",
        choices=["same_recording", "same_subject"],
        default="same_subject",
        help=(
            "Baseline eligibility scope. same_subject allows any valid 3-minute "
            "baseline window from the same subject; same_recording requires the "
            "baseline window in the seizure recording. Default: same_subject."
        ),
    )
    return parser.parse_args()


def discover_subjects() -> list[str]:
    """Discover subjects from annotation CSVs and raw CHB-MIT summary files."""
    subjects = set()

    if ANNOT_DIR.exists():
        for path in ANNOT_DIR.glob("*_seizures.csv"):
            subjects.add(path.name.replace("_seizures.csv", ""))

    if DATA_RAW.exists():
        for path in DATA_RAW.glob("*/*-summary.txt"):
            subjects.add(path.parent.name)

    if not subjects:
        raise FileNotFoundError(
            "No seizure annotations found. Expected data/annotations/*_seizures.csv "
            "or data/raw/<subject>/<subject>-summary.txt."
        )

    return sorted(subjects)


def parse_project_seizure_csv(subject_id: str, path: Path) -> pd.DataFrame:
    """Read the existing project annotation CSV format."""
    df = pd.read_csv(path)

    column_map = {
        "file": "recording_file",
        "recording_file": "recording_file",
        "start_sec": "seizure_start_sec",
        "seizure_start_sec": "seizure_start_sec",
        "end_sec": "seizure_end_sec",
        "seizure_end_sec": "seizure_end_sec",
    }

    rename = {source: target for source, target in column_map.items() if source in df}
    df = df.rename(columns=rename)

    required = {"recording_file", "seizure_start_sec", "seizure_end_sec"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")

    df = df[["recording_file", "seizure_start_sec", "seizure_end_sec"]].copy()
    df["subject_id"] = subject_id
    df["seizure_start_sec"] = pd.to_numeric(
        df["seizure_start_sec"], errors="coerce"
    )
    df["seizure_end_sec"] = pd.to_numeric(df["seizure_end_sec"], errors="coerce")
    df = df.dropna(subset=["recording_file", "seizure_start_sec", "seizure_end_sec"])
    df = df.sort_values(["recording_file", "seizure_start_sec"]).reset_index(drop=True)
    df["seizure_id"] = [
        f"{subject_id}_sz{idx + 1:02d}" for idx in range(len(df))
    ]

    return df[
        ["subject_id", "recording_file", "seizure_id", "seizure_start_sec", "seizure_end_sec"]
    ]


def parse_chbmit_summary(subject_id: str, path: Path) -> pd.DataFrame:
    """Parse seizure events from a CHB-MIT summary text file."""
    records = []
    current_file = None

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("File Name:"):
            current_file = line.split(":", maxsplit=1)[1].strip()

        elif "Number of Seizures in File:" in line:
            n_seizures = int(re.search(r"\d+", line).group(0))
            j = i + 1
            seizures_seen = 0
            while seizures_seen < n_seizures and j < len(lines) - 1:
                start_line = lines[j].strip()
                end_line = lines[j + 1].strip()
                if "Seizure" in start_line and "Start Time:" in start_line:
                    start = int(re.findall(r"\d+", start_line)[-1])
                    end = int(re.findall(r"\d+", end_line)[-1])
                    records.append(
                        {
                            "subject_id": subject_id,
                            "recording_file": current_file,
                            "seizure_start_sec": start,
                            "seizure_end_sec": end,
                        }
                    )
                    seizures_seen += 1
                    j += 2
                else:
                    j += 1
        i += 1

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "recording_file",
                "seizure_id",
                "seizure_start_sec",
                "seizure_end_sec",
            ]
        )

    df = df.sort_values(["recording_file", "seizure_start_sec"]).reset_index(drop=True)
    df["seizure_id"] = [
        f"{subject_id}_sz{idx + 1:02d}" for idx in range(len(df))
    ]
    return df[
        ["subject_id", "recording_file", "seizure_id", "seizure_start_sec", "seizure_end_sec"]
    ]


def load_subject_annotations(subject_id: str) -> pd.DataFrame:
    """Load annotations for one subject from CSV first, summary text second."""
    csv_path = ANNOT_DIR / f"{subject_id}_seizures.csv"
    if csv_path.exists():
        return parse_project_seizure_csv(subject_id, csv_path)

    summary_path = DATA_RAW / subject_id / f"{subject_id}-summary.txt"
    if summary_path.exists():
        return parse_chbmit_summary(subject_id, summary_path)

    raise FileNotFoundError(
        f"No annotations found for {subject_id}: expected {csv_path} or {summary_path}"
    )


def recording_path(subject_id: str, recording_file: str) -> Path:
    """Return the expected EDF path for a subject recording."""
    return DATA_RAW / subject_id / recording_file


def load_mne():
    """Import MNE lazily because it is slow and only needed for EDF metadata."""
    global MNE_MODULE, MNE_IMPORT_ATTEMPTED

    if not MNE_IMPORT_ATTEMPTED:
        MNE_IMPORT_ATTEMPTED = True
        try:
            import mne

            MNE_MODULE = mne
        except ImportError:
            MNE_MODULE = None

    return MNE_MODULE


def normalize_channel_name(channel_name: str) -> str:
    """Normalize channel names for robust montage comparisons."""
    normalized = str(channel_name).strip().upper()
    if normalized.endswith("-0"):
        return normalized[:-2]
    return normalized


def join_values(values: list[str]) -> str:
    """Store list-like metadata as a stable semicolon-separated string."""
    return ";".join(str(value) for value in values)


def montage_diagnostics(channel_names_used: list[str]) -> dict:
    """Compare used channels with the required common montage."""
    required = [normalize_channel_name(channel) for channel in REQUIRED_CHANNELS]
    used = [normalize_channel_name(channel) for channel in channel_names_used]
    required_set = set(required)
    used_set = set(used)

    missing = [channel for channel in required if channel not in used_set]
    extra = [channel for channel in used if channel not in required_set]

    return {
        "missing_required_channels": missing,
        "extra_channels": extra,
        "montage_status": "montage_ok" if not missing else "missing_required_channels",
    }


def inspect_recording_metadata(subject_id: str, recording_file: str) -> dict:
    """Inspect EDF metadata needed for eligibility screening."""
    path = recording_path(subject_id, recording_file)

    metadata = {
        "recording_exists": path.exists(),
        "recording_duration_sec": np.nan,
        "sampling_frequency": np.nan,
        "n_channels_raw": np.nan,
        "n_channels_used": np.nan,
        "channel_names_raw": "",
        "channel_names_used": "",
        "missing_required_channels": "",
        "extra_channels": "",
        "channel_status": "recording_missing",
        "montage_status": "recording_missing",
        "metadata_error": "",
    }

    if not path.exists():
        metadata["metadata_error"] = f"recording file not found: {path}"
        return metadata

    mne_module = load_mne()
    if mne_module is None:
        metadata["channel_status"] = "mne_not_installed"
        metadata["montage_status"] = "mne_not_installed"
        metadata["metadata_error"] = "mne is required to inspect EDF metadata"
        return metadata

    try:
        raw = mne_module.io.read_raw_edf(path, preload=False, verbose=False)
        sampling_frequency = float(raw.info["sfreq"])
        channel_names = list(raw.ch_names)
        used_channels_raw = [
            channel
            for channel in channel_names
            if normalize_channel_name(channel) not in NORMALIZED_CHANNELS_TO_DROP
        ]
        used_channels = [normalize_channel_name(channel) for channel in used_channels_raw]
        duration_sec = float(raw.n_times / sampling_frequency)
        montage = montage_diagnostics(used_channels)

        metadata.update(
            {
                "recording_duration_sec": duration_sec,
                "sampling_frequency": sampling_frequency,
                "n_channels_raw": len(channel_names),
                "n_channels_used": len(used_channels),
                "channel_names_raw": join_values(channel_names),
                "channel_names_used": join_values(used_channels),
                "missing_required_channels": join_values(
                    montage["missing_required_channels"]
                ),
                "extra_channels": join_values(montage["extra_channels"]),
                "channel_status": channel_status(sampling_frequency),
                "montage_status": montage["montage_status"],
                "metadata_error": "",
            }
        )
    except Exception as exc:
        metadata["channel_status"] = "recording_unreadable"
        metadata["montage_status"] = "recording_unreadable"
        metadata["metadata_error"] = str(exc)

    return metadata


def channel_status(sampling_frequency: float) -> str:
    """Return a compact status string for sampling compatibility."""
    if abs(sampling_frequency - EXPECTED_SFREQ) <= SFREQ_TOLERANCE:
        return "sfreq_ok"
    return "sfreq_mismatch"


def has_compatible_recording(metadata: dict) -> bool:
    """Return True when sampling frequency and required montage are compatible."""
    return (
        metadata["channel_status"] == "sfreq_ok"
        and metadata["montage_status"] == "montage_ok"
    )


def has_valid_baseline(
    recording_seizures: pd.DataFrame,
    recording_duration_sec: float,
) -> bool:
    """Check whether a 3-minute baseline can be extracted from this recording."""
    if not np.isfinite(recording_duration_sec) or recording_duration_sec < WINDOW_SEC:
        return False

    forbidden = []
    for _, row in recording_seizures.iterrows():
        forbidden.append(
            (
                max(0.0, float(row["seizure_start_sec"]) - BASELINE_BUFFER_SEC),
                min(
                    recording_duration_sec,
                    float(row["seizure_end_sec"]) + BASELINE_BUFFER_SEC,
                ),
            )
        )

    candidates = [
        (0.0, float(WINDOW_SEC)),
        (recording_duration_sec - WINDOW_SEC, recording_duration_sec),
        (
            (recording_duration_sec / 2.0) - (WINDOW_SEC / 2.0),
            (recording_duration_sec / 2.0) + (WINDOW_SEC / 2.0),
        ),
    ]

    for start, end in candidates:
        if start < 0 or end > recording_duration_sec:
            continue
        overlaps_seizure_buffer = any(start < f_end and end > f_start for f_start, f_end in forbidden)
        if not overlaps_seizure_buffer:
            return True

    return False


def subject_recording_files(subject_id: str, annotations: pd.DataFrame) -> list[str]:
    """Return annotated and locally available EDF recording files for a subject."""
    files = set(str(name) for name in annotations["recording_file"].dropna())
    raw_subject_dir = DATA_RAW / subject_id
    if raw_subject_dir.exists():
        for path in raw_subject_dir.glob("*.edf"):
            files.add(path.name)
    return sorted(files)


def has_subject_baseline(
    recording_files: list[str],
    annotations: pd.DataFrame,
    metadata_cache: dict,
) -> bool:
    """Return True if any subject recording has a valid baseline window."""
    for recording_file in recording_files:
        metadata = metadata_cache.get(recording_file)
        if metadata is None:
            continue
        if not has_compatible_recording(metadata):
            continue
        recording_seizures = annotations[
            annotations["recording_file"] == recording_file
        ].copy()
        if has_valid_baseline(
            recording_seizures,
            float(metadata["recording_duration_sec"]),
        ):
            return True
    return False


def exclusion_reasons(
    has_required_preictal: bool,
    baseline_available: bool,
    metadata: dict,
) -> list[str]:
    """Build explicit exclusion reasons for a seizure."""
    reasons = []

    if not has_required_preictal:
        reasons.append(f"less_than_{PREICTAL_SEC}_sec_preictal_available")

    if not baseline_available:
        reasons.append("baseline_window_unavailable")

    if not metadata["recording_exists"]:
        reasons.append("recording_file_missing")
    elif metadata["channel_status"] == "mne_not_installed":
        reasons.append("mne_not_installed_for_metadata_check")
    elif metadata["channel_status"] == "recording_unreadable":
        reasons.append("recording_unreadable")
    elif not has_compatible_recording(metadata):
        if metadata["channel_status"] != "sfreq_ok":
            reasons.append(metadata["channel_status"])
        if metadata["montage_status"] != "montage_ok":
            reasons.append(metadata["montage_status"])

    if metadata.get("metadata_error"):
        reasons.append(f"metadata_error={metadata['metadata_error']}")

    return reasons


def screen_subject(subject_id: str, baseline_scope: str) -> tuple[list[dict], dict]:
    """Screen all annotated seizures for one subject."""
    annotations = load_subject_annotations(subject_id)
    seizure_rows = []

    metadata_cache = {}
    recording_files = subject_recording_files(subject_id, annotations)
    for recording_file in recording_files:
        metadata_cache[recording_file] = inspect_recording_metadata(
            subject_id, recording_file
        )

    grouped_by_recording = {
        recording_file: group.copy()
        for recording_file, group in annotations.groupby("recording_file")
    }
    subject_baseline_available = has_subject_baseline(
        recording_files, annotations, metadata_cache
    )

    for _, row in annotations.iterrows():
        recording_file = str(row["recording_file"])
        metadata = metadata_cache[recording_file]
        available_preictal_sec = float(row["seizure_start_sec"])
        has_required_preictal = available_preictal_sec >= PREICTAL_SEC
        if baseline_scope == "same_subject":
            baseline_available = subject_baseline_available
        else:
            baseline_available = has_valid_baseline(
                grouped_by_recording[recording_file],
                float(metadata["recording_duration_sec"]),
            )
        reasons = exclusion_reasons(
            has_required_preictal=has_required_preictal,
            baseline_available=baseline_available,
            metadata=metadata,
        )
        usable_for_analysis = len(reasons) == 0

        seizure_rows.append(
            {
                "subject_id": subject_id,
                "recording_file": recording_file,
                "seizure_id": row["seizure_id"],
                "seizure_start_sec": float(row["seizure_start_sec"]),
                "seizure_end_sec": float(row["seizure_end_sec"]),
                "available_preictal_sec": available_preictal_sec,
                "has_required_preictal": bool(has_required_preictal),
                "sampling_frequency": metadata["sampling_frequency"],
                "n_channels_raw": metadata["n_channels_raw"],
                "n_channels_used": metadata["n_channels_used"],
                "channel_names_raw": metadata["channel_names_raw"],
                "channel_names_used": metadata["channel_names_used"],
                "missing_required_channels": metadata["missing_required_channels"],
                "extra_channels": metadata["extra_channels"],
                "channel_status": metadata["channel_status"],
                "montage_status": metadata["montage_status"],
                "usable_for_analysis": bool(usable_for_analysis),
                "exclusion_reason": ";".join(reasons) if reasons else "none",
            }
        )

    total_seizures = len(seizure_rows)
    usable_seizures = sum(row["usable_for_analysis"] for row in seizure_rows)
    excluded_seizures = total_seizures - usable_seizures
    usable_for_study = usable_seizures > 0
    notes = (
        "ok"
        if usable_for_study
        else "no usable seizures after preictal, baseline, channel, and sfreq checks"
    )

    subject_summary = {
        "subject_id": subject_id,
        "total_seizures": int(total_seizures),
        "usable_seizures": int(usable_seizures),
        "excluded_seizures": int(excluded_seizures),
        "usable_for_study": bool(usable_for_study),
        "notes": notes,
    }

    return seizure_rows, subject_summary


def build_dataset_flow_counts(
    seizure_rows: list[dict], subject_summaries: list[dict]
) -> pd.DataFrame:
    """Build dataset-level flow counts for screening documentation."""
    seizure_df = pd.DataFrame(seizure_rows, columns=SEIZURE_COLUMNS)
    summary_df = pd.DataFrame(subject_summaries, columns=SUMMARY_COLUMNS)
    rows = []

    def add_count(count_type: str, count: int, exclusion_reason: str = "not_applicable"):
        rows.append(
            {
                "count_type": count_type,
                "exclusion_reason": exclusion_reason,
                "count": int(count),
            }
        )

    add_count("total_screened_subjects", len(summary_df))
    add_count("subjects_with_annotations", int((summary_df["total_seizures"] > 0).sum()))

    if seizure_df.empty:
        add_count("subjects_with_readable_metadata", 0)
        add_count("subjects_with_compatible_channel_sampling_metadata", 0)
        add_count("subjects_with_at_least_one_usable_seizure", 0)
        add_count("total_annotated_seizures", 0)
        add_count("usable_seizures", 0)
        add_count("excluded_seizures", 0)
        return pd.DataFrame(rows, columns=FLOW_COLUMNS)

    readable_subjects = seizure_df[
        seizure_df["sampling_frequency"].notna()
        & ~seizure_df["channel_status"].isin(
            ["recording_missing", "mne_not_installed", "recording_unreadable"]
        )
    ]["subject_id"].nunique()
    compatible_subjects = seizure_df[
        (seizure_df["channel_status"] == "sfreq_ok")
        & (seizure_df["montage_status"] == "montage_ok")
    ]["subject_id"].nunique()

    add_count("subjects_with_readable_metadata", readable_subjects)
    add_count("subjects_with_compatible_channel_sampling_metadata", compatible_subjects)
    add_count(
        "subjects_with_at_least_one_usable_seizure",
        int((summary_df["usable_seizures"] > 0).sum()),
    )
    add_count("total_annotated_seizures", len(seizure_df))
    add_count("usable_seizures", int(seizure_df["usable_for_analysis"].sum()))
    add_count("excluded_seizures", int((~seizure_df["usable_for_analysis"]).sum()))

    excluded = seizure_df[~seizure_df["usable_for_analysis"]]
    reason_counts = {}
    for reason_text in excluded["exclusion_reason"].fillna("unknown"):
        for reason in str(reason_text).split(";"):
            reason = reason.strip() or "unknown"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    for reason, count in sorted(reason_counts.items()):
        add_count("excluded_seizures_by_exclusion_reason", count, reason)

    return pd.DataFrame(rows, columns=FLOW_COLUMNS)


def save_outputs(seizure_rows: list[dict], subject_summaries: list[dict]) -> None:
    """Write dataset-screening CSV outputs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    seizure_df = pd.DataFrame(seizure_rows, columns=SEIZURE_COLUMNS)
    summary_df = pd.DataFrame(subject_summaries, columns=SUMMARY_COLUMNS)
    flow_df = build_dataset_flow_counts(seizure_rows, subject_summaries)

    seizure_path = RESULTS_DIR / "seizure_eligibility.csv"
    summary_path = RESULTS_DIR / "subject_eligibility_summary.csv"
    flow_path = RESULTS_DIR / "dataset_flow_counts.csv"
    seizure_df.to_csv(seizure_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    flow_df.to_csv(flow_path, index=False)

    print("\nOutputs saved:")
    print(f"  Seizure eligibility : {seizure_path}")
    print(f"  Subject summary     : {summary_path}")
    print(f"  Dataset flow counts : {flow_path}")


def main() -> None:
    """Run dataset screening."""
    args = parse_args()
    subjects = args.subjects if args.subjects else discover_subjects()

    print("=" * 60)
    print("CHB-MIT Dataset Screening")
    print("=" * 60)
    print(f"Repository root : {ROOT}")
    print(f"Subjects        : {', '.join(subjects)}")
    print(f"Preictal needed : {PREICTAL_SEC} seconds")
    print(f"Baseline window : {WINDOW_SEC} seconds")
    print(f"Baseline scope  : {args.baseline_scope}")
    print(f"Expected sfreq  : {EXPECTED_SFREQ} Hz")
    print(f"Required montage: {len(REQUIRED_CHANNELS)} channels")

    all_seizure_rows = []
    subject_summaries = []

    for subject_id in subjects:
        print("\n" + "-" * 60)
        print(f"Screening {subject_id}")
        try:
            seizure_rows, subject_summary = screen_subject(
                subject_id, baseline_scope=args.baseline_scope
            )
            all_seizure_rows.extend(seizure_rows)
            subject_summaries.append(subject_summary)
            print(
                f"  seizures={subject_summary['total_seizures']}, "
                f"usable={subject_summary['usable_seizures']}, "
                f"excluded={subject_summary['excluded_seizures']}"
            )
        except Exception as exc:
            print(f"  [warning] Could not screen {subject_id}: {exc}")
            subject_summaries.append(
                {
                    "subject_id": subject_id,
                    "total_seizures": 0,
                    "usable_seizures": 0,
                    "excluded_seizures": 0,
                    "usable_for_study": False,
                    "notes": f"screening_failed={exc}",
                }
            )

    save_outputs(all_seizure_rows, subject_summaries)

    print("\n" + "=" * 60)
    print("Dataset screening complete")
    print("=" * 60)
    print(f"Subjects screened : {len(subject_summaries)}")
    print(f"Seizures screened : {len(all_seizure_rows)}")
    print(f"Usable seizures   : {sum(row['usable_for_analysis'] for row in all_seizure_rows)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
