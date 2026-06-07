# Epilepsy Seizure Prediction Thesis

This repository contains the code developed for a master's thesis on epileptic seizure prediction using scalp EEG functional connectivity and graph-theoretic analysis.

The project investigates whether pre-ictal changes in EEG brain-network topology can be detected from the CHB-MIT scalp EEG dataset. The pipeline transforms raw EEG recordings into segmented pre-seizure windows, computes functional connectivity matrices, extracts graph metrics, performs exploratory statistical analysis, and generates visualizations of the main findings.

## Thesis Topic

**Graph-Theoretic Analysis of Scalp EEG Functional Connectivity for Epileptic Seizure Prediction**

The main research question is whether graph-based features derived from scalp EEG functional connectivity networks can distinguish baseline brain activity from pre-ictal periods before epileptic seizures.

## Repository Scope

This repository contains the reproducible analysis code.

It does **not** store the full CHB-MIT EEG dataset, processed `.npy` arrays, connectivity matrices, or large generated result files. These artifacts are stored separately because they are too large for regular Git tracking.

The companion thesis-writing repository contains the LaTeX source of the written thesis.

## Pipeline Overview

The current pipeline is organized as a sequence of Python scripts:

```text
RAW EEG (.edf)
    |
    v
01_preprocessing.py
    |
    v
02_segmentation.py
    |
    v
03_connectivity.py
    |
    v
04_graph_metrics.py
    |
    v
05_statistics.py
    |
    v
06_visualization.py
```

### 01 Preprocessing

Parses CHB-MIT seizure annotations, loads EEG recordings, applies filtering, handles duplicated channels, and prepares the recordings for segmentation.

### 02 Segmentation

Extracts seizure-centered windows:

* Baseline
* T0: 9–6 minutes before seizure onset
* T1: 6–3 minutes before seizure onset
* T2: 3–0 minutes before seizure onset

Each 3-minute window is divided into 2-second epochs.

### 03 Connectivity

Computes functional connectivity matrices for each seizure, window, frequency band, and connectivity method.

Connectivity methods:

* coherence
* weighted Phase Lag Index, wPLI
* Amplitude Envelope Correlation, AEC

Frequency bands:

* delta
* theta
* alpha1
* alpha2
* beta1
* beta2

The script is designed to resume from existing matrix files and mirror outputs to Google Drive when running in Colab.

### 04 Graph Metrics

Converts connectivity matrices into weighted graphs and extracts graph-theoretic features.

Computed graph features include:

* mean connectivity
* clustering coefficient
* global efficiency
* characteristic path length
* small-worldness
* modularity
* mean betweenness centrality
* assortativity

Graphs are constructed using proportional thresholding, with 20% density used as the primary analysis setting.

### 05 Statistics

Runs exploratory repeated-measures statistics across the four temporal windows.

The repeated unit is `seizure_id`, not individual graph rows, channels, or epochs.

Statistical tests include:

* repeated-measures ANOVA
* Baseline-vs-preictal post-hoc paired tests
* Friedman non-parametric tests
* Benjamini-Hochberg FDR correction

### 06 Visualization

Generates figures and summary tables for inspecting the graph-metric and statistical results.

Current visualizations include:

* main finding line plot
* main finding mean ± SEM plot
* top ANOVA results
* top post-hoc results
* exploratory small-worldness plots

## Current Status

The pipeline has been validated on subject `chb01` from the CHB-MIT dataset.

Current completed stages for `chb01`:

```text
01_preprocessing.py       complete
02_segmentation.py        complete
03_connectivity.py        complete
04_graph_metrics.py       complete
05_statistics.py          complete
06_visualization.py       complete
```

For `chb01`, six usable seizures were retained. One seizure was excluded because there was not enough pre-ictal data before seizure onset.

The connectivity stage produced:

```text
6 seizures × 4 windows × 6 bands × 3 methods = 432 matrices
```

The graph-metric stage produced:

```text
432 graph-metric rows
```

## Preliminary Finding

The exploratory single-subject analysis on `chb01` identified one graph feature that survived FDR correction in repeated-measures ANOVA:

```text
Feature: mean betweenness centrality
Connectivity method: wPLI
Frequency band: delta
```

This feature decreased consistently from baseline to all pre-ictal windows across the six usable seizures.

However, this finding should be interpreted cautiously:

* the analysis is currently single-subject;
* post-hoc tests did not survive FDR correction;
* Friedman tests did not survive FDR correction;
* additional subjects are required before drawing general conclusions.

The current result is therefore considered a preliminary within-subject finding, not a validated seizure prediction biomarker.

## Repository Structure

```text
epilepsy-seizure-prediction-thesis/
├── data/
│   └── annotations/
├── src/
│   ├── 01_preprocessing.py
│   ├── 02_segmentation.py
│   ├── 03_connectivity.py
│   ├── 04_graph_metrics.py
│   ├── 05_statistics.py
│   └── 06_visualization.py
├── requirements.txt
├── .gitignore
└── README.md
```

Large generated folders such as the following are not intended to be tracked in Git:

```text
data/raw/
data/processed/
results/connectivity_matrices/
results/graph_metrics/
results/statistics/
results/figures/
```

## Installation

Clone the repository:

```bash
git clone https://github.com/leilasadakbayeva/epilepsy-seizure-prediction-thesis.git
cd epilepsy-seizure-prediction-thesis
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

The scripts are designed to be run from the repository root.

```bash
python src/01_preprocessing.py
python src/02_segmentation.py
python src/03_connectivity.py
python src/04_graph_metrics.py
python src/05_statistics.py
python src/06_visualization.py
```

Most scripts support subject-level processing and are intended to scale beyond `chb01`.

Example:

```bash
python src/04_graph_metrics.py chb01
python src/05_statistics.py chb01
python src/06_visualization.py chb01
```

## Google Colab Workflow

The project was primarily executed in Google Colab because the connectivity computation can be time-consuming.

The recommended Colab workflow is:

1. Mount Google Drive.
2. Clone or pull the latest GitHub version.
3. Restore processed arrays and previous results from Google Drive.
4. Run the next pipeline stage.
5. Save generated outputs back to Google Drive.

Code is stored in GitHub. Data artifacts and generated results are stored separately in Google Drive.

## Reproducibility Notes

The pipeline is designed to be reproducible and resumable.

Important reproducibility choices include:

* fixed temporal windows relative to seizure onset;
* fixed EEG frequency bands;
* subject-independent filename conventions;
* proportional graph thresholding;
* saved software versions in result tables;
* no random train/test splitting across windows from the same seizure;
* explicit correction for multiple statistical comparisons.

A numerical issue was identified and corrected in the coherence computation. The original implementation used a fixed epsilon in the denominator, which distorted coherence values for EEG signals stored in volts. The corrected implementation uses safe division based on the true spectral denominator.

## Data Availability

The CHB-MIT scalp EEG dataset is publicly available through PhysioNet.

Raw EEG files are not included in this repository. Users who want to reproduce the full pipeline must download the required CHB-MIT recordings separately and place them in the expected local or Google Drive data directory.

## Thesis Writing Repository

The LaTeX thesis source is maintained separately in a dedicated writing repository:

```text
epilepsy-thesis-writing
```

This separation keeps the analysis code and thesis document independent.

## Planned Next Steps

Planned development includes:

* exploratory prediction using graph-metric features;
* leave-one-seizure-out cross-validation;
* validation on additional CHB-MIT subjects;
* graph-threshold sensitivity analysis;
* refinement of the statistical interpretation;
* integration of final results into the written thesis.

## Author

Leila Sadakbayeva
Master's thesis project, University of Bologna

## Disclaimer

This repository is part of an academic thesis project. The current results are exploratory and should not be interpreted as a clinically validated seizure prediction system.
