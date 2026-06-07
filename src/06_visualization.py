"""
06_visualization.py
-------------------
Create reproducible figures and summary tables for graph-metric results.

This script is an exploratory visualization stage before prediction. It is
subject-general: subjects are discovered from results/graph_metrics/ unless
specific subject IDs are passed on the command line.

Example:
    python src/06_visualization.py chb01
    python src/06_visualization.py
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Paths
ROOT = Path(__file__).resolve().parent.parent
GRAPH_METRICS_DIR = ROOT / "results" / "graph_metrics"
STATISTICS_DIR = ROOT / "results" / "statistics"
FIGURES_DIR = ROOT / "results" / "figures"


# Config
WINDOW_ORDER = ["Baseline", "T0", "T1", "T2"]
MAIN_METRIC = "mean_betweenness_centrality"
MAIN_METHOD = "wpli"
MAIN_BAND = "delta"
TOP_N_RESULTS = 20
DPI = 300


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create exploratory graph-metric and statistics figures."
    )
    parser.add_argument(
        "subjects",
        nargs="*",
        help=(
            "Optional subject IDs to process, for example: chb01. "
            "If omitted, all subjects found in results/graph_metrics/ are processed."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving them.",
    )
    return parser.parse_args()


def discover_subjects() -> list[str]:
    """Discover subjects with graph-metrics CSV files."""
    if not GRAPH_METRICS_DIR.exists():
        raise FileNotFoundError(
            f"Missing {GRAPH_METRICS_DIR}. Run 04_graph_metrics.py first."
        )

    subjects = []
    for path in sorted(GRAPH_METRICS_DIR.glob("*_graph_metrics.csv")):
        subjects.append(path.name.replace("_graph_metrics.csv", ""))

    if not subjects:
        raise FileNotFoundError(
            f"No *_graph_metrics.csv files found in {GRAPH_METRICS_DIR}."
        )

    return subjects


def subject_paths(subject: str) -> dict[str, Path]:
    """Return expected input and output paths for one subject."""
    return {
        "graph": GRAPH_METRICS_DIR / f"{subject}_graph_metrics.csv",
        "anova": STATISTICS_DIR / f"{subject}_rm_anova_window_effects.csv",
        "posthoc": STATISTICS_DIR / f"{subject}_posthoc_baseline_vs_preictal.csv",
        "descriptive": STATISTICS_DIR / f"{subject}_descriptive_summary.csv",
        "friedman": STATISTICS_DIR / f"{subject}_friedman_window_effects.csv",
        "figures": FIGURES_DIR / subject,
    }


def load_csv(path: Path, label: str) -> pd.DataFrame | None:
    """Load a CSV file or print a clear warning if it is missing/unreadable."""
    if not path.exists():
        print(f"[warning] Missing {label}: {path}")
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[warning] Could not read {label}: {path} ({exc})")
        return None


def ensure_columns(df: pd.DataFrame, columns: list[str], label: str) -> bool:
    """Return True when all required columns are present."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        print(f"[warning] {label} is missing columns: {', '.join(missing)}")
        return False
    return True


def ordered_window_index(values: pd.Series) -> pd.Categorical:
    """Return a categorical window index using the thesis window order."""
    return pd.Categorical(values, categories=WINDOW_ORDER, ordered=True)


def save_figure(fig: plt.Figure, path: Path, show: bool) -> None:
    """Save a matplotlib figure with thesis-friendly settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def safe_neg_log10(p_values: pd.Series) -> pd.Series:
    """Convert p-values to -log10(p), handling zeros and invalid values."""
    numeric = pd.to_numeric(p_values, errors="coerce")
    numeric = numeric.where(numeric > 0)
    numeric = numeric.clip(lower=np.finfo(float).tiny)
    return -np.log10(numeric)


def sem(values: pd.Series) -> float:
    """Return standard error of the mean."""
    clean = values.dropna()
    if len(clean) == 0:
        return np.nan
    return float(clean.std(ddof=1) / math.sqrt(len(clean))) if len(clean) > 1 else 0.0


def main_metric_values(graph_df: pd.DataFrame, subject: str) -> pd.DataFrame:
    """Return main finding values for delta wPLI betweenness."""
    required = ["subject", "seizure_id", "window", "method", "band", MAIN_METRIC]
    if not ensure_columns(graph_df, required, "graph metrics CSV"):
        return pd.DataFrame()

    subset = graph_df[
        (graph_df["method"] == MAIN_METHOD)
        & (graph_df["band"] == MAIN_BAND)
        & (graph_df["window"].isin(WINDOW_ORDER))
    ].copy()

    if subset.empty:
        print(
            "[warning] No rows found for main finding: "
            f"{MAIN_METRIC}, {MAIN_METHOD}, {MAIN_BAND}"
        )
        return subset

    subset["window"] = ordered_window_index(subset["window"])
    subset = subset.sort_values(["seizure_id", "window"]).reset_index(drop=True)
    subset = subset[
        ["subject", "seizure_id", "window", "method", "band", MAIN_METRIC]
    ].copy()
    subset["figure_subject"] = subject
    return subset


def plot_main_lines(values_df: pd.DataFrame, output_dir: Path, show: bool) -> Path | None:
    """Plot one thin line per seizure and a thick black mean line."""
    if values_df.empty:
        return None

    pivot = values_df.pivot_table(
        index="seizure_id", columns="window", values=MAIN_METRIC, aggfunc="mean"
    ).reindex(columns=WINDOW_ORDER)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(WINDOW_ORDER))

    for seizure_id, row in pivot.iterrows():
        ax.plot(x, row.to_numpy(dtype=float), color="0.55", linewidth=1.0, alpha=0.75)

    mean_values = pivot.mean(axis=0, skipna=True).to_numpy(dtype=float)
    ax.plot(
        x,
        mean_values,
        color="black",
        linewidth=3.0,
        marker="o",
        markersize=6,
        label="Mean",
    )

    ax.set_title("Delta wPLI Betweenness Across Windows")
    ax.set_xlabel("Window")
    ax.set_ylabel("Mean betweenness centrality")
    ax.set_xticks(x)
    ax.set_xticklabels(WINDOW_ORDER)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    path = output_dir / "main_delta_wpli_betweenness_lines.png"
    save_figure(fig, path, show)
    return path


def plot_main_mean_sem(
    values_df: pd.DataFrame, output_dir: Path, show: bool
) -> Path | None:
    """Plot window means with SEM error bars."""
    if values_df.empty:
        return None

    summary = (
        values_df.groupby("window", observed=False)[MAIN_METRIC]
        .agg(mean="mean", sem=sem, n="count")
        .reindex(WINDOW_ORDER)
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(WINDOW_ORDER))
    ax.errorbar(
        x,
        summary["mean"].to_numpy(dtype=float),
        yerr=summary["sem"].to_numpy(dtype=float),
        color="black",
        marker="o",
        linewidth=2.5,
        capsize=5,
    )

    ax.set_title("Delta wPLI Betweenness: Mean +/- SEM")
    ax.set_xlabel("Window")
    ax.set_ylabel("Mean betweenness centrality")
    ax.set_xticks(x)
    ax.set_xticklabels(WINDOW_ORDER)
    ax.grid(axis="y", alpha=0.25)

    path = output_dir / "main_delta_wpli_betweenness_mean_sem.png"
    save_figure(fig, path, show)
    return path


def top_result_label(row: pd.Series, include_contrast: bool = False) -> str:
    """Build a compact label for top statistical results."""
    pieces = [
        str(row.get("metric", "")),
        str(row.get("method", "")),
        str(row.get("band", "")),
    ]
    if include_contrast:
        pieces.append(str(row.get("contrast", "")))
    return " | ".join(piece for piece in pieces if piece)


def prepare_top_results(
    stats_df: pd.DataFrame, include_contrast: bool = False
) -> pd.DataFrame:
    """Sort statistical results by raw p-value and keep the top rows."""
    required = ["metric", "method", "band", "p_uncorrected"]
    if include_contrast:
        required.append("contrast")
    if not ensure_columns(stats_df, required, "statistics CSV"):
        return pd.DataFrame()

    table = stats_df.copy()
    table["p_uncorrected"] = pd.to_numeric(table["p_uncorrected"], errors="coerce")
    table = table.dropna(subset=["p_uncorrected"])
    table = table.sort_values("p_uncorrected", ascending=True).head(TOP_N_RESULTS)
    table = table.reset_index(drop=True)

    if table.empty:
        return table

    table["neg_log10_p_uncorrected"] = safe_neg_log10(table["p_uncorrected"])
    table["plot_label"] = table.apply(
        lambda row: top_result_label(row, include_contrast=include_contrast), axis=1
    )
    if "p_fdr_bh" in table.columns:
        table["survives_fdr_0_05"] = pd.to_numeric(
            table["p_fdr_bh"], errors="coerce"
        ) < 0.05
    else:
        table["survives_fdr_0_05"] = False

    return table


def plot_top_results(
    table: pd.DataFrame,
    title: str,
    xlabel: str,
    output_path: Path,
    show: bool,
) -> Path | None:
    """Create a horizontal top-results -log10(p) bar plot."""
    if table.empty:
        print(f"[warning] No plottable rows for {output_path.name}")
        return None

    plot_df = table.iloc[::-1].copy()
    colors = np.where(plot_df["survives_fdr_0_05"], "black", "0.65")

    fig_height = max(5.0, 0.34 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["neg_log10_p_uncorrected"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["plot_label"], fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)

    if plot_df["survives_fdr_0_05"].any():
        for idx, survives in enumerate(plot_df["survives_fdr_0_05"]):
            if survives:
                ax.text(
                    plot_df["neg_log10_p_uncorrected"].iloc[idx] + 0.03,
                    idx,
                    "FDR < 0.05",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    save_figure(fig, output_path, show)
    return output_path


def plot_small_worldness(
    graph_df: pd.DataFrame, output_dir: Path, show: bool
) -> Path | None:
    """Plot coherence small-worldness for theta and alpha2 bands."""
    required = ["seizure_id", "window", "method", "band", "small_worldness"]
    if not ensure_columns(graph_df, required, "graph metrics CSV"):
        return None

    subset = graph_df[
        (graph_df["method"] == "coherence")
        & (graph_df["band"].isin(["theta", "alpha2"]))
        & (graph_df["window"].isin(WINDOW_ORDER))
    ].copy()
    subset["small_worldness"] = pd.to_numeric(
        subset["small_worldness"], errors="coerce"
    )
    subset = subset.dropna(subset=["small_worldness"])

    if subset.empty:
        print("[warning] No coherence theta/alpha2 small_worldness rows to plot.")
        return None

    subset["window"] = ordered_window_index(subset["window"])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    x = np.arange(len(WINDOW_ORDER))

    for ax, band in zip(axes, ["theta", "alpha2"]):
        band_df = subset[subset["band"] == band]
        pivot = band_df.pivot_table(
            index="seizure_id",
            columns="window",
            values="small_worldness",
            aggfunc="mean",
        ).reindex(columns=WINDOW_ORDER)

        for _, row in pivot.iterrows():
            ax.plot(
                x,
                row.to_numpy(dtype=float),
                color="0.60",
                linewidth=1.0,
                alpha=0.75,
            )

        mean_values = pivot.mean(axis=0, skipna=True).to_numpy(dtype=float)
        ax.plot(
            x,
            mean_values,
            color="black",
            linewidth=2.5,
            marker="o",
            label="Mean",
        )
        ax.set_title(f"Coherence {band}")
        ax.set_xlabel("Window")
        ax.set_xticks(x)
        ax.set_xticklabels(WINDOW_ORDER)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Small-worldness")
    axes[1].legend(frameon=False)
    fig.suptitle("Small-Worldness Across Windows")

    path = output_dir / "vecchio_style_small_worldness_theta_alpha2.png"
    save_figure(fig, path, show)
    return path


def write_table(df: pd.DataFrame, path: Path) -> Path:
    """Write a summary table CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def process_subject(subject: str, show: bool) -> dict:
    """Create all figures and summary tables for one subject."""
    paths = subject_paths(subject)
    output_dir = paths["figures"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"Visualization pipeline - Subject: {subject}")
    print("=" * 60)

    graph_df = load_csv(paths["graph"], "graph metrics CSV")
    anova_df = load_csv(paths["anova"], "RM ANOVA CSV")
    posthoc_df = load_csv(paths["posthoc"], "post-hoc CSV")

    found_inputs = {
        "graph": graph_df is not None,
        "anova": anova_df is not None,
        "posthoc": posthoc_df is not None,
        "descriptive": paths["descriptive"].exists(),
        "friedman": paths["friedman"].exists(),
    }

    figures_written = []
    tables_written = []

    if graph_df is not None:
        values_df = main_metric_values(graph_df, subject)
        values_path = output_dir / "main_delta_wpli_betweenness_values.csv"
        tables_written.append(write_table(values_df, values_path))

        line_path = plot_main_lines(values_df, output_dir, show)
        if line_path is not None:
            figures_written.append(line_path)

        sem_path = plot_main_mean_sem(values_df, output_dir, show)
        if sem_path is not None:
            figures_written.append(sem_path)

        small_world_path = plot_small_worldness(graph_df, output_dir, show)
        if small_world_path is not None:
            figures_written.append(small_world_path)
    else:
        print("[warning] Skipping graph-metric figures because graph CSV is missing.")

    if anova_df is not None:
        top_anova = prepare_top_results(anova_df, include_contrast=False)
        anova_table_path = output_dir / "top_anova_results.csv"
        tables_written.append(write_table(top_anova, anova_table_path))
        anova_plot_path = plot_top_results(
            top_anova,
            "Top RM ANOVA Window Effects",
            "-log10(raw p-value)",
            output_dir / "top_anova_results.png",
            show,
        )
        if anova_plot_path is not None:
            figures_written.append(anova_plot_path)
    else:
        print("[warning] Skipping top ANOVA plot/table because ANOVA CSV is missing.")

    if posthoc_df is not None:
        top_posthoc = prepare_top_results(posthoc_df, include_contrast=True)
        posthoc_table_path = output_dir / "top_posthoc_results.csv"
        tables_written.append(write_table(top_posthoc, posthoc_table_path))
        posthoc_plot_path = plot_top_results(
            top_posthoc,
            "Top Baseline vs Preictal Post-Hoc Results",
            "-log10(raw p-value)",
            output_dir / "top_posthoc_results.png",
            show,
        )
        if posthoc_plot_path is not None:
            figures_written.append(posthoc_plot_path)
    else:
        print("[warning] Skipping top post-hoc plot/table because post-hoc CSV is missing.")

    print("\n-- Validation summary --")
    print(f"Subject processed : {subject}")
    print("Input files found :")
    for label, found in found_inputs.items():
        print(f"  {label:12s}: {'yes' if found else 'no'}")
    print(f"Figures written   : {len(figures_written)}")
    for path in figures_written:
        print(f"  - {path}")
    print(f"Tables written    : {len(tables_written)}")
    for path in tables_written:
        print(f"  - {path}")
    print(f"Output folder     : {output_dir}")

    return {
        "subject": subject,
        "input_files_found": found_inputs,
        "figures_written": figures_written,
        "tables_written": tables_written,
        "output_dir": output_dir,
    }


def main() -> None:
    """Run visualization for requested or discovered subjects."""
    args = parse_args()
    subjects = args.subjects if args.subjects else discover_subjects()

    print("=" * 60)
    print("Visualization Pipeline")
    print("=" * 60)
    print(f"Repository root : {ROOT}")
    print(f"Subjects        : {', '.join(subjects)}")
    print(f"Show figures    : {'yes' if args.show else 'no'}")

    summaries = []
    for subject in subjects:
        summaries.append(process_subject(subject, show=args.show))

    print("\n" + "=" * 60)
    print("Final validation")
    print("=" * 60)
    for summary in summaries:
        print(
            f"{summary['subject']}: "
            f"figures={len(summary['figures_written'])}, "
            f"tables={len(summary['tables_written'])}, "
            f"output={summary['output_dir']}"
        )

    print("\nVisualization complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
