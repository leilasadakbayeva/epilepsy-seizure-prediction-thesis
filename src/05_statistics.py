"""
05_statistics.py
----------------
Exploratory repeated-measures statistics for graph metrics.

This script treats seizure_id as the repeated-measures unit. It does not treat
graph-metric rows, frequency bands, connectivity methods, epochs, or channels
as independent biological samples.

For each metric x connectivity method x frequency band, the script tests
whether graph metrics change across four windows:
    Baseline, T0, T1, T2

Outputs are saved under results/statistics/.
"""

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import scipy
    from scipy import stats
except ImportError:
    scipy = None
    stats = None

try:
    import pingouin as pg
except ImportError:
    pg = None


# Paths
ROOT = Path(__file__).resolve().parent.parent
GRAPH_METRICS_DIR = ROOT / "results" / "graph_metrics"
STATISTICS_DIR = ROOT / "results" / "statistics"


# Config
WINDOW_ORDER = ["Baseline", "T0", "T1", "T2"]
PREICTAL_WINDOWS = ["T0", "T1", "T2"]
MIN_COMPLETE_REPEATS = 4

GRAPH_METRICS = [
    "mean_connectivity",
    "clustering_coefficient",
    "global_efficiency",
    "characteristic_path_length",
    "small_worldness",
    "modularity",
    "mean_betweenness_centrality",
    "assortativity",
]

SOFTWARE_COLUMNS = [
    "python_version",
    "numpy_version",
    "pandas_version",
    "scipy_version",
    "pingouin_version",
]


def software_versions() -> dict:
    """Return software versions used by this analysis."""
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__ if scipy is not None else "not_installed",
        "pingouin_version": pg.__version__ if pg is not None else "not_installed",
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run exploratory repeated-measures statistics for graph metrics."
    )
    parser.add_argument(
        "subjects",
        nargs="*",
        help=(
            "Optional subject IDs to process, for example: chb01 chb05. "
            "If omitted, all subjects found in results/graph_metrics/ are processed."
        ),
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


def input_path_for_subject(subject: str) -> Path:
    """Return graph-metrics input path for a subject."""
    return GRAPH_METRICS_DIR / f"{subject}_graph_metrics.csv"


def output_paths_for_subject(subject: str) -> dict[str, Path]:
    """Return all statistics output paths for a subject."""
    return {
        "anova": STATISTICS_DIR / f"{subject}_rm_anova_window_effects.csv",
        "posthoc": STATISTICS_DIR / f"{subject}_posthoc_baseline_vs_preictal.csv",
        "descriptive": STATISTICS_DIR / f"{subject}_descriptive_summary.csv",
        "friedman": STATISTICS_DIR / f"{subject}_friedman_window_effects.csv",
    }


def load_graph_metrics(subject: str) -> pd.DataFrame:
    """Load one subject graph-metrics CSV."""
    path = input_path_for_subject(subject)
    if not path.exists():
        raise FileNotFoundError(f"Missing graph-metrics file: {path}")

    df = pd.read_csv(path)
    required = {"subject", "seizure_id", "window", "band", "method"}
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        raise ValueError(
            f"{path} is missing required columns: {', '.join(missing_required)}"
        )

    return df


def available_metrics(df: pd.DataFrame) -> list[str]:
    """Return configured graph metrics present in the input file."""
    present = [metric for metric in GRAPH_METRICS if metric in df.columns]
    missing = [metric for metric in GRAPH_METRICS if metric not in df.columns]

    if missing:
        print(f"[warning] Missing graph metric columns: {', '.join(missing)}")
    if not present:
        raise ValueError("No configured graph metric columns are present.")

    return present


def make_long_dataframe(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Convert graph metrics from wide metric columns to a long table."""
    id_columns = ["subject", "seizure_id", "window", "band", "method"]
    long_df = df.melt(
        id_vars=id_columns,
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df.loc[~np.isfinite(long_df["value"]), "value"] = np.nan
    return long_df


def descriptive_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive summaries by metric, method, band, and window."""
    grouped = long_df.groupby(
        ["subject", "metric", "method", "band", "window"], dropna=False
    )["value"]

    summary = grouped.agg(
        n="count",
        mean="mean",
        std="std",
        median="median",
        min="min",
        max="max",
    ).reset_index()

    return add_software_columns(summary)


def complete_window_matrix(combo_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Return a seizure x window matrix with complete finite repeated measures.

    Duplicate rows for the same seizure_id x window are averaged as a defensive
    guard against accidental duplicate graph-metric rows.
    """
    pivot = combo_df.pivot_table(
        index="seizure_id",
        columns="window",
        values="value",
        aggfunc="mean",
    )
    duplicate_counts = combo_df.groupby(["seizure_id", "window"]).size()
    if (duplicate_counts > 1).any():
        metric = combo_df["metric"].iloc[0] if "metric" in combo_df else "unknown_metric"
        method = combo_df["method"].iloc[0] if "method" in combo_df else "unknown_method"
        band = combo_df["band"].iloc[0] if "band" in combo_df else "unknown_band"
        print(
            "[warning] duplicate seizure_id x window rows detected for "
            f"{metric}/{method}/{band} - averaging"
        )

    available_columns = [window for window in WINDOW_ORDER if window in pivot.columns]
    pivot = pivot.reindex(columns=WINDOW_ORDER)
    n_before_complete_filter = len(pivot)

    if len(available_columns) < len(WINDOW_ORDER):
        return pivot.iloc[0:0], n_before_complete_filter

    pivot = pivot.replace([np.inf, -np.inf], np.nan)
    complete = pivot.dropna(subset=WINDOW_ORDER)
    return complete, n_before_complete_filter


def manual_rm_anova(values: np.ndarray) -> dict:
    """
    Compute one-way repeated-measures ANOVA for a balanced n x k matrix.

    The p-value is returned only when scipy is available. Pingouin is preferred
    when installed because it can also report sphericity/Greenhouse-Geisser
    information.
    """
    n_subjects, n_windows = values.shape
    grand_mean = float(np.mean(values))
    subject_means = np.mean(values, axis=1, keepdims=True)
    window_means = np.mean(values, axis=0, keepdims=True)

    ss_total = float(np.sum((values - grand_mean) ** 2))
    ss_subjects = float(n_windows * np.sum((subject_means - grand_mean) ** 2))
    ss_window = float(n_subjects * np.sum((window_means - grand_mean) ** 2))
    ss_error = float(ss_total - ss_subjects - ss_window)

    df_window = n_windows - 1
    df_error = (n_subjects - 1) * (n_windows - 1)
    ms_window = ss_window / df_window if df_window > 0 else np.nan
    ms_error = ss_error / df_error if df_error > 0 else np.nan

    f_value = ms_window / ms_error if ms_error and ms_error > 0 else np.nan
    p_value = (
        float(stats.f.sf(f_value, df_window, df_error))
        if stats is not None and np.isfinite(f_value)
        else np.nan
    )
    partial_eta_sq = (
        ss_window / (ss_window + ss_error)
        if (ss_window + ss_error) > 0
        else np.nan
    )

    return {
        "anova_source": "manual",
        "F": float(f_value) if np.isfinite(f_value) else np.nan,
        "ddof1": float(df_window),
        "ddof2": float(df_error),
        "p_uncorrected": p_value,
        "partial_eta_sq": float(partial_eta_sq)
        if np.isfinite(partial_eta_sq)
        else np.nan,
        "p_gg_corrected": np.nan,
        "epsilon_gg": np.nan,
    }


def pingouin_rm_anova(complete: pd.DataFrame) -> dict:
    """Run Pingouin repeated-measures ANOVA and normalize output columns."""
    long_complete = (
        complete.reset_index()
        .melt(
            id_vars="seizure_id",
            value_vars=WINDOW_ORDER,
            var_name="window",
            value_name="value",
        )
        .dropna(subset=["value"])
    )

    result = pg.rm_anova(
        data=long_complete,
        dv="value",
        within="window",
        subject="seizure_id",
        detailed=True,
        correction=True,
        effsize="np2",
    )

    row = result[result["Source"] == "window"]
    if row.empty:
        row = result.iloc[[0]]
    row = row.iloc[0]
    error_row = result[result["Source"] == "Error"]
    ddof2 = np.nan
    if "ddof2" in row:
        ddof2 = row.get("ddof2", np.nan)
    elif not error_row.empty:
        error = error_row.iloc[0]
        if "ddof2" in error:
            ddof2 = error.get("ddof2", np.nan)
        elif "DF" in error:
            ddof2 = error.get("DF", np.nan)

    return {
        "anova_source": "pingouin",
        "F": float(row.get("F", np.nan)),
        "ddof1": float(row.get("ddof1", row.get("DF", np.nan))),
        "ddof2": float(row.get("ddof2", ddof2)),
        "p_uncorrected": float(row.get("p-unc", np.nan)),
        "partial_eta_sq": float(row.get("np2", np.nan)),
        "p_gg_corrected": float(row.get("p-GG-corr", np.nan)),
        "epsilon_gg": float(row.get("eps", np.nan)),
    }


def run_rm_anova_for_combo(
    combo_df: pd.DataFrame,
    subject: str,
    metric: str,
    method: str,
    band: str,
) -> dict:
    """Run or skip repeated-measures ANOVA for one metric x method x band."""
    complete, n_candidate_seizures = complete_window_matrix(combo_df)
    n_complete = len(complete)

    base = {
        "subject": subject,
        "metric": metric,
        "method": method,
        "band": band,
        "n_candidate_seizures": int(n_candidate_seizures),
        "n_complete_seizures": int(n_complete),
        "windows": ",".join(WINDOW_ORDER),
        "test_completed": False,
        "skip_reason": "",
        "anova_source": "",
        "F": np.nan,
        "ddof1": np.nan,
        "ddof2": np.nan,
        "p_uncorrected": np.nan,
        "partial_eta_sq": np.nan,
        "p_gg_corrected": np.nan,
        "epsilon_gg": np.nan,
    }

    if n_complete < MIN_COMPLETE_REPEATS:
        base["skip_reason"] = (
            f"fewer than {MIN_COMPLETE_REPEATS} complete seizure repeats"
        )
        return base

    try:
        if pg is not None:
            stats_row = pingouin_rm_anova(complete)
        else:
            stats_row = manual_rm_anova(complete[WINDOW_ORDER].to_numpy(dtype=float))
        base.update(stats_row)
        base["test_completed"] = bool(np.isfinite(base["F"]))
        if not base["test_completed"]:
            base["skip_reason"] = "test statistic was not finite"
    except Exception as exc:
        base["skip_reason"] = f"rm_anova failed: {exc}"

    return base


def paired_t_test(baseline: np.ndarray, comparison: np.ndarray) -> dict:
    """Run a paired t-test and paired Cohen's dz effect size."""
    diff = comparison - baseline
    n_pairs = len(diff)
    df = n_pairs - 1
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1)) if n_pairs > 1 else np.nan
    cohen_dz = mean_diff / sd_diff if sd_diff and sd_diff > 0 else np.nan

    if stats is not None and n_pairs >= 2:
        test = stats.ttest_rel(comparison, baseline, nan_policy="omit")
        t_value = float(test.statistic)
        p_value = float(test.pvalue)
    elif sd_diff and sd_diff > 0:
        t_value = float(mean_diff / (sd_diff / np.sqrt(n_pairs)))
        p_value = np.nan
    else:
        t_value = np.nan
        p_value = np.nan

    return {
        "t": t_value,
        "df": float(df),
        "p_uncorrected": p_value,
        "mean_difference": mean_diff,
        "cohen_dz": float(cohen_dz) if np.isfinite(cohen_dz) else np.nan,
    }


def run_posthoc_for_combo(
    combo_df: pd.DataFrame,
    subject: str,
    metric: str,
    method: str,
    band: str,
) -> list[dict]:
    """Run Baseline-vs-preictal paired tests for one metric x method x band."""
    complete, n_candidate_seizures = complete_window_matrix(combo_df)
    rows = []

    for comparison in PREICTAL_WINDOWS:
        base = {
            "subject": subject,
            "metric": metric,
            "method": method,
            "band": band,
            "contrast": f"Baseline_vs_{comparison}",
            "baseline_window": "Baseline",
            "comparison_window": comparison,
            "n_candidate_seizures": int(n_candidate_seizures),
            "n_complete_seizures": int(len(complete)),
            "test_completed": False,
            "skip_reason": "",
            "t": np.nan,
            "df": np.nan,
            "p_uncorrected": np.nan,
            "p_fdr_bh": np.nan,
            "mean_difference": np.nan,
            "cohen_dz": np.nan,
        }

        if len(complete) < MIN_COMPLETE_REPEATS:
            base["skip_reason"] = (
                f"fewer than {MIN_COMPLETE_REPEATS} complete seizure repeats"
            )
            rows.append(base)
            continue

        try:
            stats_row = paired_t_test(
                complete["Baseline"].to_numpy(dtype=float),
                complete[comparison].to_numpy(dtype=float),
            )
            base.update(stats_row)
            base["test_completed"] = bool(np.isfinite(base["t"]))
            if not base["test_completed"]:
                base["skip_reason"] = "paired t statistic was not finite"
        except Exception as exc:
            base["skip_reason"] = f"paired test failed: {exc}"

        rows.append(base)

    return rows


def friedman_test(values: np.ndarray) -> dict:
    """Run or manually compute the Friedman test for a complete n x k matrix."""
    n_subjects, n_windows = values.shape

    if stats is not None:
        test = stats.friedmanchisquare(*[values[:, i] for i in range(n_windows)])
        return {
            "friedman_source": "scipy",
            "chi_square": float(test.statistic),
            "df": float(n_windows - 1),
            "p_uncorrected": float(test.pvalue),
        }

    ranks = np.apply_along_axis(rank_values, 1, values)
    rank_sums = np.sum(ranks, axis=0)
    chi_square = (
        12.0 / (n_subjects * n_windows * (n_windows + 1.0))
    ) * np.sum(rank_sums**2) - 3.0 * n_subjects * (n_windows + 1.0)

    return {
        "friedman_source": "manual_no_scipy_p",
        "chi_square": float(chi_square),
        "df": float(n_windows - 1),
        "p_uncorrected": np.nan,
    }


def rank_values(values: np.ndarray) -> np.ndarray:
    """Rank one vector using average ranks for ties."""
    series = pd.Series(values)
    return series.rank(method="average").to_numpy(dtype=float)


def run_friedman_for_combo(
    combo_df: pd.DataFrame,
    subject: str,
    metric: str,
    method: str,
    band: str,
) -> dict:
    """Run or skip Friedman test for one metric x method x band."""
    complete, n_candidate_seizures = complete_window_matrix(combo_df)
    n_complete = len(complete)

    base = {
        "subject": subject,
        "metric": metric,
        "method": method,
        "band": band,
        "n_candidate_seizures": int(n_candidate_seizures),
        "n_complete_seizures": int(n_complete),
        "windows": ",".join(WINDOW_ORDER),
        "test_completed": False,
        "skip_reason": "",
        "friedman_source": "",
        "chi_square": np.nan,
        "df": np.nan,
        "p_uncorrected": np.nan,
    }

    if n_complete < MIN_COMPLETE_REPEATS:
        base["skip_reason"] = (
            f"fewer than {MIN_COMPLETE_REPEATS} complete seizure repeats"
        )
        return base

    try:
        stats_row = friedman_test(complete[WINDOW_ORDER].to_numpy(dtype=float))
        base.update(stats_row)
        base["test_completed"] = bool(np.isfinite(base["chi_square"]))
        if not base["test_completed"]:
            base["skip_reason"] = "friedman statistic was not finite"
    except Exception as exc:
        base["skip_reason"] = f"friedman test failed: {exc}"

    return base


def fdr_bh(p_values: pd.Series) -> pd.Series:
    """
    Benjamini-Hochberg FDR correction for a pandas Series of p-values.

    NaN p-values remain NaN.
    """
    corrected = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna()

    if valid.empty:
        return corrected

    order = valid.sort_values().index
    sorted_p = valid.loc[order].to_numpy(dtype=float)
    m = len(sorted_p)
    adjusted = sorted_p * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    corrected.loc[order] = adjusted
    return corrected


def add_software_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Append reproducibility metadata columns to an output dataframe."""
    versions = software_versions()
    for column in SOFTWARE_COLUMNS:
        df[column] = versions[column]
    return df


def add_completed_test_fdr(df: pd.DataFrame) -> pd.DataFrame:
    """Add FDR/BH p-values only for completed tests."""
    df["p_fdr_bh"] = np.nan
    completed = df["test_completed"] == True
    df.loc[completed, "p_fdr_bh"] = fdr_bh(df.loc[completed, "p_uncorrected"])
    return df


def iter_metric_method_band(long_df: pd.DataFrame):
    """Yield grouped metric x method x band dataframes in deterministic order."""
    group_columns = ["metric", "method", "band"]
    for (metric, method, band), combo_df in long_df.groupby(group_columns, sort=True):
        yield metric, method, band, combo_df


def run_subject_statistics(subject: str) -> dict:
    """Run all exploratory statistics for one subject."""
    input_path = input_path_for_subject(subject)
    output_paths = output_paths_for_subject(subject)

    df = load_graph_metrics(subject)
    metrics = available_metrics(df)
    long_df = make_long_dataframe(df, metrics)

    print("\n" + "=" * 60)
    print(f"Statistics pipeline - Subject: {subject}")
    print("=" * 60)
    print(f"Input CSV          : {input_path}")
    print(f"Input rows         : {len(df)}")
    print(f"Metrics analyzed   : {len(metrics)}")
    print(f"Methods            : {df['method'].nunique()}")
    print(f"Bands              : {df['band'].nunique()}")
    print(f"Repeated unit      : seizure_id")
    print(f"Window order       : {', '.join(WINDOW_ORDER)}")

    descriptive_df = descriptive_summary(long_df)

    anova_rows = []
    posthoc_rows = []
    friedman_rows = []

    for metric, method, band, combo_df in iter_metric_method_band(long_df):
        anova_rows.append(
            run_rm_anova_for_combo(combo_df, subject, metric, method, band)
        )
        posthoc_rows.extend(
            run_posthoc_for_combo(combo_df, subject, metric, method, band)
        )
        friedman_rows.append(
            run_friedman_for_combo(combo_df, subject, metric, method, band)
        )

    anova_df = pd.DataFrame(anova_rows)
    if not anova_df.empty:
        anova_df = add_completed_test_fdr(anova_df)
    anova_df = add_software_columns(anova_df)
    posthoc_df = pd.DataFrame(posthoc_rows)
    if not posthoc_df.empty:
        posthoc_df["p_fdr_bh"] = fdr_bh(posthoc_df["p_uncorrected"])
    posthoc_df = add_software_columns(posthoc_df)
    friedman_df = pd.DataFrame(friedman_rows)
    if not friedman_df.empty:
        friedman_df = add_completed_test_fdr(friedman_df)
    friedman_df = add_software_columns(friedman_df)

    STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
    descriptive_df.to_csv(output_paths["descriptive"], index=False)
    anova_df.to_csv(output_paths["anova"], index=False)
    posthoc_df.to_csv(output_paths["posthoc"], index=False)
    friedman_df.to_csv(output_paths["friedman"], index=False)

    tests_attempted = len(anova_df) + len(posthoc_df) + len(friedman_df)
    tests_completed = int(
        anova_df["test_completed"].sum()
        + posthoc_df["test_completed"].sum()
        + friedman_df["test_completed"].sum()
    )
    tests_skipped = tests_attempted - tests_completed

    print("\n-- Validation summary --")
    print(f"Input rows        : {len(df)}")
    print(f"ANOVA tests       : {len(anova_df)} attempted, {anova_df['test_completed'].sum()} completed")
    print(f"Post-hoc tests    : {len(posthoc_df)} attempted, {posthoc_df['test_completed'].sum()} completed")
    print(f"Friedman tests    : {len(friedman_df)} attempted, {friedman_df['test_completed'].sum()} completed")
    print(f"Tests attempted   : {tests_attempted}")
    print(f"Tests completed   : {tests_completed}")
    print(f"Tests skipped     : {tests_skipped}")
    for label, path in output_paths.items():
        print(f"{label:12s} output : {path}")

    return {
        "subject": subject,
        "input_rows": len(df),
        "tests_attempted": tests_attempted,
        "tests_completed": tests_completed,
        "tests_skipped": tests_skipped,
        "output_paths": output_paths,
    }


def main() -> None:
    """Run statistics for requested or discovered subjects."""
    args = parse_args()
    subjects = args.subjects if args.subjects else discover_subjects()

    versions = software_versions()
    print("=" * 60)
    print("Exploratory Statistics Pipeline")
    print("=" * 60)
    print(f"Repository root   : {ROOT}")
    print(f"Subjects          : {', '.join(subjects)}")
    print(f"Python            : {versions['python_version']}")
    print(f"NumPy             : {versions['numpy_version']}")
    print(f"pandas            : {versions['pandas_version']}")
    print(f"SciPy             : {versions['scipy_version']}")
    print(f"Pingouin          : {versions['pingouin_version']}")

    summaries = []
    for subject in subjects:
        summaries.append(run_subject_statistics(subject))

    print("\n" + "=" * 60)
    print("Final validation")
    print("=" * 60)
    print(f"Subjects processed : {len(summaries)}")
    for summary in summaries:
        print(
            f"{summary['subject']}: "
            f"input rows={summary['input_rows']}, "
            f"tests attempted={summary['tests_attempted']}, "
            f"completed={summary['tests_completed']}, "
            f"skipped={summary['tests_skipped']}"
        )
        for label, path in summary["output_paths"].items():
            print(f"  {label:12s}: {path}")

    print("\nStatistics complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
