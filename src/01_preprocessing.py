"""
01_preprocessing.py
-------------------
Downloads CHB-MIT chb01 data from PhysioNet, parses seizure annotations,
loads EEG recordings with MNE, and applies preprocessing pipeline.

This is the foundation script — run this first before anything else.
"""

import os
import re
import urllib.request
import mne
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent          # D:\epilepsy-thesis
DATA_RAW    = ROOT / "data" / "raw"
DATA_PROC   = ROOT / "data" / "processed"
ANNOT_DIR   = ROOT / "data" / "annotations"

for folder in [DATA_RAW, DATA_PROC, ANNOT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
SUBJECT         = "chb01"
PHYSIONET_BASE  = "https://physionet.org/files/chbmit/1.0.0"
SFREQ           = 256       # Hz
L_FREQ          = 0.5       # bandpass low  (Hz)
H_FREQ          = 47.0      # bandpass high (Hz)
NOTCH_FREQ      = 60.0      # US power line (Hz)
EPOCH_LENGTH    = 2.0       # seconds

# Standard 23 channels in CHB-MIT (10-20 system)
STANDARD_CHANNELS = [
    'FP1-F7', 'F7-T7',  'T7-P7',  'P7-O1',
    'FP1-F3', 'F3-C3',  'C3-P3',  'P3-O1',
    'FP2-F4', 'F4-C4',  'C4-P4',  'P4-O2',
    'FP2-F8', 'F8-T8',  'T8-P8',  'P8-O2',
    'FZ-CZ',  'CZ-PZ',
    'P7-T7',  'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8'
]


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Download summary file and EDF files
# ══════════════════════════════════════════════════════════════════════════

def download_file(url: str, dest_path: Path) -> None:
    """Download a single file from PhysioNet if not already present."""
    if dest_path.exists():
        print(f"  [skip] already exists: {dest_path.name}")
        return
    print(f"  [download] {dest_path.name} ...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"  [ok] {dest_path.name}")
    except Exception as e:
        print(f"  [error] Could not download {dest_path.name}: {e}")

def download_subject(subject: str) -> None:
    """Download summary file + only EDF files that contain seizures."""
    subject_dir = DATA_RAW / subject
    subject_dir.mkdir(exist_ok=True)
    base_url    = f"{PHYSIONET_BASE}/{subject}"

    # Download summary file first
    summary_url  = f"{base_url}/{subject}-summary.txt"
    summary_path = subject_dir / f"{subject}-summary.txt"
    download_file(summary_url, summary_path)

    # Parse summary to find only seizure-containing files
    seizure_files = parse_seizure_filenames(summary_path)
    print(f"\nFound {len(seizure_files)} EDF files containing seizures")
    print("Files to download:")
    for f in seizure_files:
        print(f"  → {f}")

    # Download only those files
    for edf_name in seizure_files:
        edf_url  = f"{base_url}/{edf_name}"
        edf_path = subject_dir / edf_name
        download_file(edf_url, edf_path)

def parse_seizure_filenames(summary_path: Path) -> list:
    """
    Extract only EDF filenames that contain at least one seizure.
    Much smaller download than grabbing all 42 files.
    """
    seizure_files = []
    current_file  = None

    with open(summary_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()

        if line.startswith("File Name:"):
            current_file = line.split(":")[-1].strip()

        elif "Number of Seizures in File:" in line:
            n_seizures = int(line.split(":")[-1].strip())
            if n_seizures > 0 and current_file:
                seizure_files.append(current_file)

    return seizure_files

def parse_edf_filenames(summary_path: Path) -> list:
    """Extract EDF filenames mentioned in the summary file."""
    edf_files = []
    with open(summary_path, 'r') as f:
        for line in f:
            if line.strip().startswith("File Name:"):
                fname = line.strip().split(":")[-1].strip()
                edf_files.append(fname)
    return edf_files


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Parse seizure annotations
# ══════════════════════════════════════════════════════════════════════════

def parse_annotations(subject: str) -> pd.DataFrame:
    """
    Parse the summary .txt file and extract all seizure events.

    Returns a DataFrame with columns:
        file        — EDF filename
        start_sec   — seizure start in seconds from file start
        end_sec     — seizure end in seconds from file start
        duration    — seizure duration in seconds
    """
    summary_path = DATA_RAW / subject / f"{subject}-summary.txt"
    records      = []
    current_file = None

    with open(summary_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Track which file we're in
        if line.startswith("File Name:"):
            current_file = line.split(":")[-1].strip()

        # Find number of seizures in this file
        elif "Number of Seizures in File:" in line:
            n_seizures = int(line.split(":")[-1].strip())

            # Read the next n_seizures pairs of start/end lines
            j = i + 1
            seizure_count = 0
            while seizure_count < n_seizures and j < len(lines):
                l = lines[j].strip()
                if "Seizure" in l and "Start Time:" in l:
                    start = int(re.search(r'(\d+)', l).group(1))
                    end_line = lines[j+1].strip()
                    end   = int(re.search(r'(\d+)', end_line).group(1))
                    records.append({
                        "file"      : current_file,
                        "start_sec" : start,
                        "end_sec"   : end,
                        "duration"  : end - start
                    })
                    seizure_count += 1
                    j += 2
                else:
                    j += 1
        i += 1

    df = pd.DataFrame(records)

    # Save to annotations folder
    out_path = ANNOT_DIR / f"{subject}_seizures.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSeizure annotations saved → {out_path}")
    print(df.to_string(index=False))
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Load and preprocess one EDF file
# ══════════════════════════════════════════════════════════════════════════

def preprocess_edf(edf_path: Path) -> mne.io.Raw:
    """
    Load one EDF file and apply preprocessing pipeline:
        1. Load raw EEG
        2. Select EEG channels only
        3. Bandpass filter  0.5 – 47 Hz
        4. Notch filter at 60 Hz
    Returns the cleaned MNE Raw object.
    """
    print(f"\n[load] {edf_path.name}")

    # Load — preload=True loads all data into memory
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    print(f"  Channels ({len(raw.ch_names)}): {raw.ch_names}")
    print(f"  Duration: {raw.times[-1]:.1f} s  "
          f"({raw.times[-1]/60:.1f} min)")
    print(f"  Sampling rate: {raw.info['sfreq']} Hz")

    # Drop non-EEG channels (ECG, VNS, misc)
    eeg_channels = [ch for ch in raw.ch_names
                    if not any(x in ch.upper()
                               for x in ['ECG', 'VNS', 'EKG', '-'])]
    # Keep only channels that exist in this file
    available = [ch for ch in raw.ch_names if ch in raw.ch_names]
    raw.pick_channels(available)

    # Bandpass filter
    print(f"  Applying bandpass filter {L_FREQ}–{H_FREQ} Hz ...")
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ,
               method='fir', verbose=False)

    # Notch filter
    print(f"  Applying notch filter at {NOTCH_FREQ} Hz ...")
    raw.notch_filter(freqs=NOTCH_FREQ, verbose=False)

    print(f"  [ok] preprocessing complete")
    return raw


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Quick inspection & sanity check
# ══════════════════════════════════════════════════════════════════════════

def inspect_raw(raw: mne.io.Raw, seizure_df: pd.DataFrame,
                edf_name: str) -> None:
    """Print a summary and show basic signal statistics."""

    # Which seizures are in this file?
    file_seizures = seizure_df[seizure_df['file'] == edf_name]

    print(f"\n── Inspection: {edf_name} ──")
    print(f"  Channels     : {len(raw.ch_names)}")
    print(f"  Duration     : {raw.times[-1]:.1f} s")
    print(f"  Seizures     : {len(file_seizures)}")

    if not file_seizures.empty:
        for _, row in file_seizures.iterrows():
            print(f"    → seizure at {row.start_sec}s "
                  f"– {row.end_sec}s "
                  f"(duration: {row.duration}s)")

    # Signal amplitude statistics
    data = raw.get_data()   # shape: (n_channels, n_samples)
    print(f"\n  Signal statistics (µV):")
    print(f"    Mean amplitude : {np.mean(np.abs(data)) * 1e6:.2f}")
    print(f"    Max amplitude  : {np.max(np.abs(data))  * 1e6:.2f}")
    print(f"    Std            : {np.std(data)           * 1e6:.2f}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print(f"  CHB-MIT Preprocessing Pipeline — Subject: {SUBJECT}")
    print("=" * 60)

    # 1. Download data
    print("\n[STEP 1] Downloading data from PhysioNet ...")
    download_subject(SUBJECT)

    # 2. Parse seizure annotations
    print("\n[STEP 2] Parsing seizure annotations ...")
    seizure_df = parse_annotations(SUBJECT)

    # 3. Preprocess first EDF file as a test
    #    chb01_03.edf is a good first file — it contains a seizure
    test_edf = DATA_RAW / SUBJECT / "chb01_03.edf"
    print("\n[STEP 3] Preprocessing test file (chb01_03.edf) ...")
    raw = preprocess_edf(test_edf)

    # 4. Inspect
    print("\n[STEP 4] Inspecting ...")
    inspect_raw(raw, seizure_df, "chb01_03.edf")

    print("\n" + "=" * 60)
    print("  Pipeline test complete!")
    print("  Next step: run 02_segmentation.py")
    print("=" * 60)