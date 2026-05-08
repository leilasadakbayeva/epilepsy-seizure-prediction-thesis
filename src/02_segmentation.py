"""
02_segmentation.py
------------------
Extracts four time windows relative to each seizure onset:

    Baseline : interictal segment, far from any seizure
    T0       : 9–6 minutes before seizure onset
    T1       : 6–3 minutes before seizure onset
    T2       : 3–0 minutes before seizure onset

For each seizure, all four segments are saved as numpy arrays
in data/processed/chb01/

Run after 01_preprocessing.py
"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
ANNOT_DIR = ROOT / "data" / "annotations"

# ── Config ─────────────────────────────────────────────────────────────────
SUBJECT       = "chb01"
L_FREQ        = 0.5
H_FREQ        = 47.0
NOTCH_FREQ    = 60.0
WINDOW_SEC    = 180        # 3 minutes per window
PRE_ICTAL_SEC = 540        # 9 minutes total pre-ictal period
EPOCH_SEC     = 2          # each window split into 2s epochs
BASELINE_BUFFER = 600      # baseline must be ≥10 min from any seizure

# ── Channel handling ───────────────────────────────────────────────────────
# CHB-MIT has a duplicate T8-P8 channel — we keep the first one
CHANNELS_TO_DROP = ['T8-P8-1']   # MNE renames duplicate to T8-P8-1


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load and preprocess one EDF file
# ══════════════════════════════════════════════════════════════════════════

def load_and_preprocess(edf_path: Path) -> mne.io.Raw:
    """Load EDF, drop duplicate channels, apply filters."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Drop duplicate channel
    existing_drops = [ch for ch in CHANNELS_TO_DROP
                      if ch in raw.ch_names]
    if existing_drops:
        raw.drop_channels(existing_drops)

    # Filters
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ,
               method='fir', verbose=False)
    raw.notch_filter(freqs=NOTCH_FREQ, verbose=False)

    return raw


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Extract one time window from a Raw object
# ══════════════════════════════════════════════════════════════════════════

def extract_window(raw: mne.io.Raw,
                   start_sec: float,
                   end_sec: float) -> np.ndarray:
    """
    Extract a segment from a Raw object between start_sec and end_sec.

    Returns array of shape (n_channels, n_samples)
    Returns None if the window is outside the file duration.
    """
    file_duration = raw.times[-1]

    if start_sec < 0 or end_sec > file_duration:
        return None

    start_idx = int(start_sec * raw.info['sfreq'])
    end_idx   = int(end_sec   * raw.info['sfreq'])

    data = raw.get_data()[:, start_idx:end_idx]
    return data


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Split a window into 2-second epochs
# ══════════════════════════════════════════════════════════════════════════

def split_into_epochs(window: np.ndarray,
                      sfreq: float,
                      epoch_sec: float = EPOCH_SEC) -> np.ndarray:
    """
    Split a (n_channels, n_samples) window into non-overlapping epochs.

    Returns array of shape (n_epochs, n_channels, epoch_samples)
    """
    epoch_samples = int(epoch_sec * sfreq)
    n_channels, n_samples = window.shape
    n_epochs = n_samples // epoch_samples

    # Trim any leftover samples that don't fill a complete epoch
    trimmed = window[:, :n_epochs * epoch_samples]

    # Reshape → (n_epochs, n_channels, epoch_samples)
    epochs = trimmed.reshape(n_channels,
                             n_epochs,
                             epoch_samples).transpose(1, 0, 2)
    return epochs


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Find a valid baseline segment
# ══════════════════════════════════════════════════════════════════════════

def find_baseline(raw: mne.io.Raw,
                  all_seizures_in_file: pd.DataFrame,
                  window_sec: int = WINDOW_SEC,
                  buffer_sec: int = BASELINE_BUFFER) -> np.ndarray:
    """
    Find a clean interictal baseline segment in this file.

    Rules:
    - Must be at least buffer_sec (10 min) away from any seizure
    - Must be window_sec (3 min) long
    - We try the beginning of the file first, then the end

    Returns (n_channels, n_samples) array or None if no valid segment found.
    """
    file_duration = raw.times[-1]
    sfreq         = raw.info['sfreq']

    # Collect all "forbidden zones" — seizure ± buffer
    forbidden = []
    for _, row in all_seizures_in_file.iterrows():
        forbidden.append((
            max(0, row['start_sec'] - buffer_sec),
            min(file_duration, row['end_sec'] + buffer_sec)
        ))

    def is_valid(start, end):
        """Check if [start, end] overlaps with any forbidden zone."""
        for (f_start, f_end) in forbidden:
            if start < f_end and end > f_start:
                return False
        return True

    # Try beginning of file
    candidate_start = 0
    candidate_end   = window_sec
    if candidate_end <= file_duration and is_valid(candidate_start,
                                                    candidate_end):
        return extract_window(raw, candidate_start, candidate_end)

    # Try end of file
    candidate_start = file_duration - window_sec
    candidate_end   = file_duration
    if candidate_start >= 0 and is_valid(candidate_start, candidate_end):
        return extract_window(raw, candidate_start, candidate_end)

    # Try middle of file
    candidate_start = (file_duration / 2) - (window_sec / 2)
    candidate_end   = candidate_start + window_sec
    if is_valid(candidate_start, candidate_end):
        return extract_window(raw, candidate_start, candidate_end)

    # No valid baseline found in this file
    return None


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Process all seizures for a subject
# ══════════════════════════════════════════════════════════════════════════

def process_subject(subject: str) -> pd.DataFrame:
    """
    For each seizure, extract Baseline/T0/T1/T2 windows,
    split into epochs, and save to disk.

    Returns a summary DataFrame of what was successfully extracted.
    """
    subject_raw_dir  = DATA_RAW  / subject
    subject_proc_dir = DATA_PROC / subject
    subject_proc_dir.mkdir(parents=True, exist_ok=True)

    # Load seizure annotations
    annot_path = ANNOT_DIR / f"{subject}_seizures.csv"
    seizure_df = pd.read_csv(annot_path)

    print(f"\nProcessing {subject} — {len(seizure_df)} seizures")
    print("=" * 60)

    summary_records = []

    for idx, row in seizure_df.iterrows():
        edf_name    = row['file']
        onset_sec   = row['start_sec']
        edf_path    = subject_raw_dir / edf_name
        seizure_id  = f"{subject}_sz{idx+1:02d}"

        print(f"\n[Seizure {idx+1}] {edf_name} — onset at {onset_sec}s")

        # Load and preprocess the EDF
        raw   = load_and_preprocess(edf_path)
        sfreq = raw.info['sfreq']

        # ── Define window boundaries ──────────────────────────────────────
        windows = {
            "T2": (onset_sec - WINDOW_SEC,
                   onset_sec),
            "T1": (onset_sec - 2 * WINDOW_SEC,
                   onset_sec - WINDOW_SEC),
            "T0": (onset_sec - 3 * WINDOW_SEC,
                   onset_sec - 2 * WINDOW_SEC),
        }

        # ── Check all pre-ictal windows fit inside the file ───────────────
        file_duration   = raw.times[-1]
        t0_start        = onset_sec - PRE_ICTAL_SEC

        if t0_start < 0:
            print(f"  [skip] Not enough pre-ictal data "
                  f"(need {PRE_ICTAL_SEC}s before onset, "
                  f"only {onset_sec}s available)")
            continue

        # ── Extract pre-ictal windows ─────────────────────────────────────
        extracted = {}
        valid     = True

        for label, (start, end) in windows.items():
            data = extract_window(raw, start, end)
            if data is None:
                print(f"  [skip] {label} window out of bounds")
                valid = False
                break
            extracted[label] = data
            print(f"  [ok] {label}: {start:.0f}s → {end:.0f}s  "
                  f"shape: {data.shape}")

        if not valid:
            continue

        # ── Extract baseline ───────────────────────────────────────────────
        # Use only seizures in this same file for the forbidden zones
        file_seizures = seizure_df[seizure_df['file'] == edf_name]
        baseline      = find_baseline(raw, file_seizures)

        if baseline is None:
            print(f"  [skip] No valid baseline found in {edf_name}")
            continue

        extracted['Baseline'] = baseline
        print(f"  [ok] Baseline: shape {baseline.shape}")

        # ── Split into 2-second epochs ─────────────────────────────────────
        print(f"  Splitting into {EPOCH_SEC}s epochs ...")
        for label, data in extracted.items():
            epochs     = split_into_epochs(data, sfreq)
            save_path  = subject_proc_dir / f"{seizure_id}_{label}.npy"
            np.save(save_path, epochs)
            print(f"  [saved] {save_path.name}  "
                  f"→ {epochs.shape[0]} epochs × "
                  f"{epochs.shape[1]} channels × "
                  f"{epochs.shape[2]} samples")

            summary_records.append({
                "subject"   : subject,
                "seizure_id": seizure_id,
                "file"      : edf_name,
                "onset_sec" : onset_sec,
                "window"    : label,
                "n_epochs"  : epochs.shape[0],
                "n_channels": epochs.shape[1],
                "n_samples" : epochs.shape[2],
                "saved_as"  : save_path.name
            })

    # ── Save summary ───────────────────────────────────────────────────────
    summary_df   = pd.DataFrame(summary_records)
    summary_path = ANNOT_DIR / f"{subject}_segments_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print(f"Segmentation complete!")
    print(f"Saved {len(summary_records)} segment files")
    print(f"Summary → {summary_path}")
    print(f"\n{summary_df.to_string(index=False)}")

    return summary_df


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print(f"  CHB-MIT Segmentation Pipeline — Subject: {SUBJECT}")
    print("=" * 60)

    summary = process_subject(SUBJECT)

    print("\n── Final summary ──")
    print(f"Total windows extracted : {len(summary)}")
    print(f"Unique seizures         : {summary['seizure_id'].nunique()}")
    print(f"Windows per seizure     : Baseline, T0, T1, T2")
    print(f"\nNext step: run 03_connectivity.py")