# Epilepsy Seizure Prediction Thesis

This repository contains the code, selected results, figures, and LaTeX thesis source for a master's thesis project on EEG-based epileptic seizure analysis using the CHB-MIT dataset.

The project builds a full analysis pipeline for screening EEG recordings, preprocessing data, segmenting seizure-centered windows, computing functional connectivity, extracting graph metrics, performing statistical analysis, and writing the final thesis report.

## Repository Contents

```text
epilepsy-seizure-prediction-thesis/
├── data/
│   └── annotations/
├── results/
│   ├── dataset_screening/
│   ├── figures/
│   ├── graph_metrics/
│   └── statistics/
├── src/
│   ├── 01_preprocessing.py
│   ├── 02_segmentation.py
│   ├── 03_connectivity.py
│   ├── 04_graph_metrics.py
│   ├── 05_statistics.py
│   └── 06_visualization.py
├── thesis/
│   ├── appendices/
│   ├── bibliography/
│   ├── chapters/
│   ├── figures/
│   ├── ai_bo_thesis.sty
│   ├── main.tex
│   └── main.pdf
├── requirements.txt
├── README.md
└── .gitignore
```

## What This Repository Contains

This repository includes:

* Python source code for the analysis pipeline;
* dataset screening outputs;
* selected graph metric outputs;
* selected statistical result tables;
* selected result figures;
* LaTeX thesis source files;
* compiled thesis PDF.

## What This Repository Does Not Contain

This repository does **not** contain large data files or generated intermediate artifacts, including:

* the original CHB-MIT raw EEG dataset;
* full preprocessed EEG files;
* full segmented `.npy` EEG window arrays;
* full connectivity matrix folders;
* large intermediate outputs produced during pipeline execution.

These files are excluded from Git tracking because they are large and should be stored locally.

## Pipeline Overview

The main pipeline stages are:

```text
Dataset screening
        ↓
EEG preprocessing
        ↓
Temporal segmentation
        ↓
Functional connectivity computation
        ↓
Graph metric extraction
        ↓
Statistical analysis
        ↓
Visualization and thesis writing
```

Scripts are located in the `src/` folder.

## Running the Code

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline scripts from the repository root:

```bash
python src/01_preprocessing.py
python src/02_segmentation.py
python src/03_connectivity.py
python src/04_graph_metrics.py
python src/05_statistics.py
python src/06_visualization.py
```

Some stages require the CHB-MIT dataset to be downloaded separately and placed in the expected local data directory.

## Thesis Report

The thesis source is stored in:

```text
thesis/
```

Main LaTeX file:

```text
thesis/main.tex
```

Compiled thesis PDF:

```text
thesis/main.pdf
```

To compile the thesis locally:

```bash
cd thesis
latexmk -xelatex -interaction=nonstopmode -file-line-error main.tex
```

To clean LaTeX build files:

```bash
latexmk -C main.tex
```

## Notes

The repository is intended for academic research and thesis reproducibility. It is not a clinically validated seizure prediction system.

Raw EEG data must be obtained separately from the CHB-MIT dataset source.