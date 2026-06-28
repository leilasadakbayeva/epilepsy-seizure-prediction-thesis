"""
05_statistics.py
----------------
Exploratory full-dataset statistics for graph-theoretic EEG connectivity metrics.

This script is an exploratory/statistical association analysis for the thesis
pipeline; it is not a seizure prediction model. The primary analysis is a
seizure-level mixed model: seizure_id is the repeated-measures unit, seizures
are nested within subject_id, and graph-metric rows, frequency bands,
connectivity methods, channels, and epochs are not treated as independent
biological samples.

Supervisor feedback is incorporated through a subject-average sensitivity and
descriptive analysis. Each subject contributes one average trajectory per
metric/method/band/window to evaluate whether group-level effects reflect
consistent subject-level patterns. Because averaging across seizures reduces
within-subject seizure variability, this subject-average analysis is not used
as the only analysis.

Planned posthoc tests compare Baseline with T0/T1/T2. They are computed for
transparency, but should be interpreted mainly when the corresponding omnibus
window effect is significant after FDR correction. Benjamini-Hochberg FDR is
applied across each output family, not separately per subject.

Run:
    python src/05_statistics.py
"""

from __future__ import annotations

import argparse
import platform
import sys
import warnings
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

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except ImportError:
    sm = None
    smf = None


# Paths
ROOT = Path(__file__).resolve().parent.parent
GRAPH_METRICS_DIR = ROOT / "results" / "graph_metrics"
STATISTICS_DIR = ROOT / "results" / "statistics"

COMBINED_LONG_PATH = STATISTICS_DIR / "combined_graph_metrics_long.csv"
DESCRIPTIVE_PATH = STATISTICS_DIR / "descriptive_summary.csv"
SEIZURE_OMNIBUS_PATH = STATISTICS_DIR / "seizure_level_window_effects_omnibus.csv"
SEIZURE_POSTHOC_PATH = STATISTICS_DIR / "seizure_level_baseline_vs_preictal_posthoc.csv"
SEIZURE_PREICTAL_TREND_PATH = STATISTICS_DIR / "seizure_level_preictal_trend_T0_T1_T2.csv"
OMNIBUS_ALIAS_PATH = STATISTICS_DIR / "window_effects_omnibus.csv"
POSTHOC_ALIAS_PATH = STATISTICS_DIR / "baseline_vs_preictal_posthoc.csv"
PREICTAL_TREND_ALIAS_PATH = STATISTICS_DIR / "preictal_trend_T0_T1_T2.csv"
SUBJECT_AVERAGE_LONG_PATH = STATISTICS_DIR / "subject_average_graph_metrics_long.csv"
SUBJECT_AVERAGE_DESCRIPTIVE_PATH = STATISTICS_DIR / "subject_average_descriptive_summary.csv"
SUBJECT_AVERAGE_OMNIBUS_PATH = STATISTICS_DIR / "subject_average_window_effects.csv"
SUBJECT_AVERAGE_POSTHOC_PATH = STATISTICS_DIR / "subject_average_baseline_vs_preictal_posthoc.csv"


# Analysis config
WINDOW_ORDER = ["Baseline", "T0", "T1", "T2"]
PREICTAL_WINDOWS = ["T0", "T1", "T2"]
PREICTAL_TIME_MIN = {"T0": -7.5, "T1": -4.5, "T2": -1.5}
POSTHOC_CONTRASTS = [("Baseline", "T0"), ("Baseline", "T1"), ("Baseline", "T2")]

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

DEFAULT_MIN_COMPLETE_REPEATS = 4
OMNIBUS_COLUMNS = [
    "metric",
    "method",
    "band",
    "n_complete_seizures",
    "n_subjects",
    "model_type",
    "test_name",
    "statistic",
    "df1",
    "df2",
    "p_uncorrected",
    "p_fdr_bh",
    "effect_size",
    "skip_reason",
    "fallback_reason",
]
POSTHOC_COLUMNS = [
    "metric",
    "method",
    "band",
    "contrast",
    "n_pairs",
    "n_subjects",
    "mean_baseline",
    "mean_comparison",
    "mean_difference",
    "t",
    "t_df",
    "t_p_uncorrected",
    "t_p_fdr_bh",
    "cohen_dz",
    "wilcoxon_statistic",
    "wilcoxon_p_uncorrected",
    "wilcoxon_p_fdr_bh",
    "omnibus_p_fdr_bh",
    "interpret_posthoc",
    "skip_reason",
]
TREND_COLUMNS = [
    "metric",
    "method",
    "band",
    "n_complete_seizures",
    "n_subjects",
    "model_type",
    "slope_per_minute",
    "statistic",
    "p_uncorrected",
    "p_fdr_bh",
    "skip_reason",
    "fallback_reason",
]
SUBJECT_AVERAGE_COLUMNS = [
    "subject_id",
    "window",
    "band",
    "method",
    "metric",
    "mean_value",
    "std_across_seizures",
    "n_seizures",
]
SUBJECT_AVERAGE_DESCRIPTIVE_COLUMNS = [
    "metric",
    "method",
    "band",
    "window",
    "n_subjects",
    "mean",
    "std",
    "median",
    "min",
    "max",
    "sem",
    "ci95_lower",
    "ci95_upper",
]
SUBJECT_AVERAGE_OMNIBUS_COLUMNS = [
    "metric",
    "method",
    "band",
    "n_subjects_complete",
    "model_type",
    "test_name",
    "statistic",
    "df1",
    "df2",
    "p_uncorrected",
    "p_fdr_bh",
    "effect_size",
    "skip_reason",
    "fallback_reason",
]
SUBJECT_AVERAGE_POSTHOC_COLUMNS = [
    "metric",
    "method",
    "band",
    "contrast",
    "n_subjects",
    "mean_baseline",
    "mean_comparison",
    "mean_difference",
    "t",
    "t_df",
    "t_p_uncorrected",
    "t_p_fdr_bh",
    "cohen_dz",
    "wilcoxon_statistic",
    "wilcoxon_p_uncorrected",
    "wilcoxon_p_fdr_bh",
    "omnibus_p_fdr_bh",
    "interpret_posthoc",
    "skip_reason",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run full-dataset exploratory statistics for graph metrics."
    )
    parser.add_argument(
        "--min-complete-repeats",
        type=int,
        default=DEFAULT_MIN_COMPLETE_REPEATS,
        help=f"Minimum complete seizure repeats required per test. Default: {DEFAULT_MIN_COMPLETE_REPEATS}",
    )
    parser.add_argument(
        "--no-mixed-models",
        action="store_true",
        help="Skip statsmodels mixed models and use repeated-measures fallbacks.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha used for descriptive confidence intervals. Default: 0.05",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing statistics output files.",
    )
    return parser.parse_args()


def discover_graph_metric_files() -> list[Path]:
    """Find all subject graph-metric CSV files."""
    if not GRAPH_METRICS_DIR.exists():
        raise FileNotFoundError(f"Missing graph metrics directory: {GRAPH_METRICS_DIR}")

    files = sorted(GRAPH_METRICS_DIR.glob("*_graph_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No *_graph_metrics.csv files found in {GRAPH_METRICS_DIR}")
    return files


def load_graph_metrics(files: list[Path]) -> pd.DataFrame:
    """Read and combine all graph-metric CSV files."""
    required = {"subject", "seizure_id", "window", "band", "method", "matrix_file"}
    frames = []

    for path in files:
        df = pd.read_csv(path)
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"subject": "subject_id"})
    combined["subject_id"] = combined["subject_id"].astype(str)
    combined["seizure_id"] = combined["seizure_id"].astype(str)
    combined["window"] = combined["window"].astype(str)
    combined["band"] = combined["band"].astype(str)
    combined["method"] = combined["method"].astype(str)
    return combined


def available_metric_columns(df: pd.DataFrame) -> list[str]:
    """Return graph metrics present in the combined input."""
    metrics = [metric for metric in GRAPH_METRICS if metric in df.columns]
    missing = [metric for metric in GRAPH_METRICS if metric not in df.columns]
    if missing:
        print(f"[warning] missing graph metric columns: {', '.join(missing)}")
    if not metrics:
        raise ValueError("No configured graph metric columns are present.")
    return metrics


def validate_wide_input(df: pd.DataFrame) -> None:
    """Run defensive checks on the combined matrix-derived graph rows."""
    invalid_windows = sorted(set(df["window"]) - set(WINDOW_ORDER))
    if invalid_windows:
        raise ValueError(f"Unexpected window labels: {', '.join(invalid_windows)}")

    key = ["subject_id", "seizure_id", "window", "band", "method"]
    duplicates = df.duplicated(subset=key)
    if duplicates.any():
        examples = df.loc[duplicates, key].head(5).to_dict("records")
        raise ValueError(f"Duplicate matrix-derived graph rows detected: {examples}")


def make_long_dataframe(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Convert graph metrics to long format."""
    id_columns = ["subject_id", "seizure_id", "window", "band", "method"]
    long_df = df.melt(
        id_vars=id_columns,
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    long_df["window"] = pd.Categorical(long_df["window"], categories=WINDOW_ORDER, ordered=True)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df.loc[~np.isfinite(long_df["value"]), "value"] = np.nan
    long_df = long_df.sort_values(
        ["subject_id", "seizure_id", "window", "band", "method", "metric"]
    ).reset_index(drop=True)
    return long_df


def validate_long_dataframe(long_df: pd.DataFrame, input_rows: int, n_metrics: int) -> None:
    """Run defensive checks on the melted long dataframe."""
    expected_rows = input_rows * n_metrics
    if len(long_df) != expected_rows:
        raise ValueError(f"Long row count mismatch: expected {expected_rows}, got {len(long_df)}")

    key = ["subject_id", "seizure_id", "window", "band", "method", "metric"]
    duplicates = long_df.duplicated(subset=key)
    if duplicates.any():
        examples = long_df.loc[duplicates, key].head(5).to_dict("records")
        raise ValueError(f"Duplicate long metric rows detected: {examples}")

    counts = long_df.groupby(["subject_id", "seizure_id", "window", "band", "method"]).size()
    bad_counts = counts[counts != n_metrics]
    if not bad_counts.empty:
        raise ValueError("At least one matrix-derived row does not map to one value per metric.")


def fdr_bh(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR correction, preserving NaN values."""
    corrected = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna()
    if valid.empty:
        return corrected

    order = valid.sort_values().index
    sorted_p = valid.loc[order].to_numpy(dtype=float)
    m = len(sorted_p)
    adjusted = sorted_p * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    corrected.loc[order] = np.clip(adjusted, 0.0, 1.0)
    return corrected


def descriptive_summary(long_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Compute summaries by metric, method, band, and window."""
    grouped = long_df.groupby(["metric", "method", "band", "window"], observed=False)["value"]
    summary = grouped.agg(
        n="count",
        mean="mean",
        std="std",
        median="median",
        min="min",
        max="max",
    ).reset_index()
    summary["sem"] = summary["std"] / np.sqrt(summary["n"])

    ci_multiplier = np.nan
    if stats is not None:
        valid = summary["n"] > 1
        summary["ci95_lower"] = np.nan
        summary["ci95_upper"] = np.nan
        multipliers = stats.t.ppf(1 - alpha / 2, summary.loc[valid, "n"] - 1)
        summary.loc[valid, "ci95_lower"] = summary.loc[valid, "mean"] - multipliers * summary.loc[valid, "sem"]
        summary.loc[valid, "ci95_upper"] = summary.loc[valid, "mean"] + multipliers * summary.loc[valid, "sem"]
    else:
        if alpha == 0.05:
            ci_multiplier = 1.96
        summary["ci95_lower"] = summary["mean"] - ci_multiplier * summary["sem"]
        summary["ci95_upper"] = summary["mean"] + ci_multiplier * summary["sem"]
        summary.loc[summary["n"] <= 1, ["ci95_lower", "ci95_upper"]] = np.nan

    return summary


def subject_average_long(long_df: pd.DataFrame) -> pd.DataFrame:
    """Average graph metrics across seizures within subject/window/band/method/metric."""
    grouped = long_df.groupby(
        ["subject_id", "window", "band", "method", "metric"],
        observed=True,
    )["value"]
    averaged = grouped.agg(
        mean_value="mean",
        std_across_seizures="std",
        n_seizures="count",
    ).reset_index()
    averaged = averaged.sort_values(
        ["subject_id", "window", "band", "method", "metric"]
    ).reset_index(drop=True)
    return averaged[SUBJECT_AVERAGE_COLUMNS]


def subject_average_descriptive_summary(
    subject_avg: pd.DataFrame, alpha: float
) -> pd.DataFrame:
    """Summarize subject-average trajectories by metric/method/band/window."""
    grouped = subject_avg.groupby(
        ["metric", "method", "band", "window"], observed=False
    )["mean_value"]
    summary = grouped.agg(
        n_subjects="count",
        mean="mean",
        std="std",
        median="median",
        min="min",
        max="max",
    ).reset_index()
    summary["sem"] = summary["std"] / np.sqrt(summary["n_subjects"])
    summary["ci95_lower"] = np.nan
    summary["ci95_upper"] = np.nan

    valid = summary["n_subjects"] > 1
    if stats is not None:
        multipliers = stats.t.ppf(1 - alpha / 2, summary.loc[valid, "n_subjects"] - 1)
    else:
        multipliers = 1.96 if alpha == 0.05 else np.nan
    summary.loc[valid, "ci95_lower"] = (
        summary.loc[valid, "mean"] - multipliers * summary.loc[valid, "sem"]
    )
    summary.loc[valid, "ci95_upper"] = (
        summary.loc[valid, "mean"] + multipliers * summary.loc[valid, "sem"]
    )
    return summary[SUBJECT_AVERAGE_DESCRIPTIVE_COLUMNS]


def combo_base(metric: str, method: str, band: str) -> dict:
    """Base identifiers for metric x method x band rows."""
    return {"metric": metric, "method": method, "band": band}


def complete_matrix(combo_df: pd.DataFrame, windows: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return complete seizure x window values plus seizure-subject mapping."""
    subject_map = combo_df[["seizure_id", "subject_id"]].drop_duplicates("seizure_id")
    pivot = combo_df.pivot(index="seizure_id", columns="window", values="value")
    pivot = pivot.reindex(columns=windows).replace([np.inf, -np.inf], np.nan)
    complete = pivot.dropna(subset=windows)
    return complete, subject_map


def complete_subject_average_matrix(
    combo_df: pd.DataFrame, windows: list[str]
) -> pd.DataFrame:
    """Return complete subject x window mean values for subject-average analyses."""
    pivot = combo_df.pivot(index="subject_id", columns="window", values="mean_value")
    pivot = pivot.reindex(columns=windows).replace([np.inf, -np.inf], np.nan)
    return pivot.dropna(subset=windows)


def complete_subject_average_long(
    combo_df: pd.DataFrame, windows: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return complete subject-average long data and complete pivot table."""
    complete = complete_subject_average_matrix(combo_df, windows)
    if complete.empty:
        return combo_df.iloc[0:0].copy(), complete

    complete_ids = set(complete.index)
    complete_long = combo_df[
        combo_df["subject_id"].isin(complete_ids) & combo_df["window"].isin(windows)
    ].copy()
    complete_long["window"] = pd.Categorical(
        complete_long["window"].astype(str), categories=windows, ordered=True
    )
    complete_long["value"] = complete_long["mean_value"]
    return complete_long, complete


def complete_long_with_subjects(
    combo_df: pd.DataFrame, windows: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return complete long data and its complete pivot table."""
    complete, subject_map = complete_matrix(combo_df, windows)
    if complete.empty:
        return combo_df.iloc[0:0].copy(), complete

    complete_ids = set(complete.index)
    complete_long = combo_df[
        combo_df["seizure_id"].isin(complete_ids) & combo_df["window"].isin(windows)
    ].copy()
    complete_long["window"] = pd.Categorical(
        complete_long["window"].astype(str), categories=windows, ordered=True
    )
    complete_long = complete_long.merge(subject_map, on="seizure_id", suffixes=("", "_map"))
    if "subject_id_map" in complete_long.columns:
        complete_long["subject_id"] = complete_long["subject_id"].fillna(complete_long["subject_id_map"])
        complete_long = complete_long.drop(columns=["subject_id_map"])
    return complete_long, complete


def n_subjects_for_complete(combo_df: pd.DataFrame, complete: pd.DataFrame) -> int:
    """Count subjects contributing complete seizures."""
    if complete.empty:
        return 0
    subject_map = combo_df[["seizure_id", "subject_id"]].drop_duplicates("seizure_id")
    return int(subject_map[subject_map["seizure_id"].isin(complete.index)]["subject_id"].nunique())


def manual_rm_anova(values: np.ndarray) -> dict:
    """Manual one-way repeated-measures ANOVA for balanced n x k data."""
    n, k = values.shape
    grand_mean = float(np.mean(values))
    subject_means = np.mean(values, axis=1, keepdims=True)
    window_means = np.mean(values, axis=0, keepdims=True)

    ss_total = float(np.sum((values - grand_mean) ** 2))
    ss_subject = float(k * np.sum((subject_means - grand_mean) ** 2))
    ss_window = float(n * np.sum((window_means - grand_mean) ** 2))
    ss_error = max(float(ss_total - ss_subject - ss_window), 0.0)

    df1 = k - 1
    df2 = (n - 1) * (k - 1)
    ms_window = ss_window / df1 if df1 > 0 else np.nan
    ms_error = ss_error / df2 if df2 > 0 else np.nan
    f_value = ms_window / ms_error if np.isfinite(ms_error) and ms_error > 0 else np.nan
    p_value = float(stats.f.sf(f_value, df1, df2)) if stats is not None and np.isfinite(f_value) else np.nan
    effect_size = ss_window / (ss_window + ss_error) if (ss_window + ss_error) > 0 else np.nan

    return {
        "model_type": "manual_repeated_measures_anova",
        "test_name": "one_way_rm_anova_window",
        "statistic": float(f_value) if np.isfinite(f_value) else np.nan,
        "df1": float(df1),
        "df2": float(df2),
        "p_uncorrected": p_value,
        "effect_size": float(effect_size) if np.isfinite(effect_size) else np.nan,
    }


def pingouin_rm_anova(
    complete: pd.DataFrame, windows: list[str], subject_column: str
) -> dict:
    """Run Pingouin repeated-measures ANOVA."""
    long = complete.reset_index().melt(
        id_vars=subject_column,
        value_vars=windows,
        var_name="window",
        value_name="value",
    )
    result = pg.rm_anova(
        data=long,
        dv="value",
        within="window",
        subject=subject_column,
        detailed=True,
        correction=True,
        effsize="np2",
    )
    row = result[result["Source"] == "window"]
    if row.empty:
        row = result.iloc[[0]]
    row = row.iloc[0]
    return {
        "model_type": "pingouin_repeated_measures_anova",
        "test_name": "one_way_rm_anova_window",
        "statistic": float(row.get("F", np.nan)),
        "df1": float(row.get("ddof1", row.get("DF", np.nan))),
        "df2": float(row.get("ddof2", np.nan)),
        "p_uncorrected": float(row.get("p-unc", np.nan)),
        "effect_size": float(row.get("np2", np.nan)),
    }


def mixed_window_effect(
    complete_long: pd.DataFrame,
    windows: list[str],
    groups: pd.Series,
    vc_formula: dict[str, str] | None,
    model_type: str,
) -> dict:
    """Run mixed model omnibus window effect via Wald test."""
    if sm is None or smf is None:
        raise RuntimeError("statsmodels is not installed")

    data = complete_long.copy()
    reference = windows[0]
    data["window"] = pd.Categorical(data["window"], categories=windows, ordered=True)
    formula = f'value ~ C(window, Treatment(reference="{reference}"))'

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(
            formula,
            data=data,
            groups=groups,
            re_formula="1",
            vc_formula=vc_formula,
        )
        result = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)

    window_params = [
        idx for idx, name in enumerate(result.params.index)
        if name.startswith(f'C(window, Treatment(reference="{reference}"))')
    ]
    if not window_params:
        raise RuntimeError("mixed model did not estimate window parameters")

    constraints = np.zeros((len(window_params), len(result.params)))
    for row_idx, param_idx in enumerate(window_params):
        constraints[row_idx, param_idx] = 1.0

    wald = result.wald_test(constraints, scalar=True)
    statistic = float(wald.statistic)
    p_value = float(wald.pvalue)
    return {
        "model_type": model_type,
        "test_name": "wald_chi_square_window_terms",
        "statistic": statistic,
        "df1": float(len(window_params)),
        "df2": np.nan,
        "p_uncorrected": p_value,
        "effect_size": np.nan,
    }


def mixed_seizure_window_effect(complete_long: pd.DataFrame) -> dict:
    """Run primary seizure-level mixed model for window effects."""
    return mixed_window_effect(
        complete_long=complete_long,
        windows=WINDOW_ORDER,
        groups=complete_long["subject_id"],
        vc_formula={"seizure": "0 + C(seizure_id)"},
        model_type="linear_mixed_effects_subject_random_intercept_seizure_vc",
    )


def mixed_subject_average_window_effect(complete_long: pd.DataFrame) -> dict:
    """Run subject-average mixed model for window effects."""
    return mixed_window_effect(
        complete_long=complete_long,
        windows=WINDOW_ORDER,
        groups=complete_long["subject_id"],
        vc_formula=None,
        model_type="linear_mixed_effects_subject_random_intercept",
    )


def run_omnibus_for_combo(
    combo_df: pd.DataFrame,
    metric: str,
    method: str,
    band: str,
    min_complete_repeats: int,
    use_mixed_models: bool,
) -> dict:
    """Run or skip one Baseline-inclusive omnibus window test."""
    complete_long, complete = complete_long_with_subjects(combo_df, WINDOW_ORDER)
    n_complete = len(complete)
    n_subjects = n_subjects_for_complete(combo_df, complete)
    base = {
        **combo_base(metric, method, band),
        "n_complete_seizures": int(n_complete),
        "n_subjects": int(n_subjects),
        "model_type": "",
        "test_name": "",
        "statistic": np.nan,
        "df1": np.nan,
        "df2": np.nan,
        "p_uncorrected": np.nan,
        "p_fdr_bh": np.nan,
        "effect_size": np.nan,
        "skip_reason": "",
        "fallback_reason": "",
    }

    if n_complete < min_complete_repeats:
        base["skip_reason"] = f"fewer than {min_complete_repeats} complete four-window seizures"
        return base

    if use_mixed_models:
        try:
            base.update(mixed_seizure_window_effect(complete_long))
            return base
        except Exception as exc:
            base["fallback_reason"] = f"mixed model failed: {exc}"

    try:
        if pg is not None:
            base.update(
                pingouin_rm_anova(
                    complete,
                    windows=WINDOW_ORDER,
                    subject_column="seizure_id",
                )
            )
        else:
            base.update(manual_rm_anova(complete[WINDOW_ORDER].to_numpy(dtype=float)))
        if not np.isfinite(base["statistic"]):
            base["skip_reason"] = "fallback failed: omnibus statistic was not finite"
    except Exception as exc:
        base["skip_reason"] = f"fallback failed: {exc}"

    return base


def run_subject_average_omnibus_for_combo(
    combo_df: pd.DataFrame,
    metric: str,
    method: str,
    band: str,
    min_complete_repeats: int,
    use_mixed_models: bool,
) -> dict:
    """Run or skip one subject-average omnibus window test."""
    complete_long, complete = complete_subject_average_long(combo_df, WINDOW_ORDER)
    n_complete = len(complete)
    base = {
        **combo_base(metric, method, band),
        "n_subjects_complete": int(n_complete),
        "model_type": "",
        "test_name": "",
        "statistic": np.nan,
        "df1": np.nan,
        "df2": np.nan,
        "p_uncorrected": np.nan,
        "p_fdr_bh": np.nan,
        "effect_size": np.nan,
        "skip_reason": "",
        "fallback_reason": "",
    }

    if n_complete < min_complete_repeats:
        base["skip_reason"] = f"fewer than {min_complete_repeats} complete subjects"
        return base

    if use_mixed_models:
        try:
            base.update(mixed_subject_average_window_effect(complete_long))
            return base
        except Exception as exc:
            base["fallback_reason"] = f"mixed model failed: {exc}"

    try:
        if pg is not None:
            base.update(
                pingouin_rm_anova(
                    complete,
                    windows=WINDOW_ORDER,
                    subject_column="subject_id",
                )
            )
        else:
            base.update(manual_rm_anova(complete[WINDOW_ORDER].to_numpy(dtype=float)))
        if not np.isfinite(base["statistic"]):
            base["skip_reason"] = "fallback failed: omnibus statistic was not finite"
    except Exception as exc:
        base["skip_reason"] = f"fallback failed: {exc}"

    return base


def paired_stats(baseline: np.ndarray, comparison: np.ndarray) -> dict:
    """Run paired t-test, Wilcoxon sensitivity test, and Cohen's dz."""
    diff = comparison - baseline
    n = len(diff)
    mean_diff = float(np.mean(diff)) if n else np.nan
    sd_diff = float(np.std(diff, ddof=1)) if n > 1 else np.nan
    cohen_dz = mean_diff / sd_diff if np.isfinite(sd_diff) and sd_diff > 0 else np.nan

    out = {
        "mean_baseline": float(np.mean(baseline)) if n else np.nan,
        "mean_comparison": float(np.mean(comparison)) if n else np.nan,
        "mean_difference": mean_diff,
        "t": np.nan,
        "t_df": float(n - 1) if n else np.nan,
        "t_p_uncorrected": np.nan,
        "cohen_dz": float(cohen_dz) if np.isfinite(cohen_dz) else np.nan,
        "wilcoxon_statistic": np.nan,
        "wilcoxon_p_uncorrected": np.nan,
    }

    if stats is not None and n >= 2:
        t_res = stats.ttest_rel(comparison, baseline, nan_policy="omit")
        out["t"] = float(t_res.statistic)
        out["t_p_uncorrected"] = float(t_res.pvalue)
        try:
            w_res = stats.wilcoxon(comparison, baseline, zero_method="wilcox", alternative="two-sided")
            out["wilcoxon_statistic"] = float(w_res.statistic)
            out["wilcoxon_p_uncorrected"] = float(w_res.pvalue)
        except Exception:
            pass
    return out


def run_posthoc_for_combo(
    combo_df: pd.DataFrame,
    metric: str,
    method: str,
    band: str,
    min_complete_repeats: int,
) -> list[dict]:
    """Run planned Baseline-vs-preictal paired comparisons."""
    rows = []
    for baseline_window, comparison_window in POSTHOC_CONTRASTS:
        windows = [baseline_window, comparison_window]
        complete, _ = complete_matrix(combo_df, windows)
        n_pairs = len(complete)
        n_subjects = n_subjects_for_complete(combo_df, complete)
        row = {
            **combo_base(metric, method, band),
            "contrast": f"{baseline_window}_vs_{comparison_window}",
            "n_pairs": int(n_pairs),
            "n_subjects": int(n_subjects),
            "mean_baseline": np.nan,
            "mean_comparison": np.nan,
            "mean_difference": np.nan,
            "t": np.nan,
            "t_df": np.nan,
            "t_p_uncorrected": np.nan,
            "t_p_fdr_bh": np.nan,
            "cohen_dz": np.nan,
            "wilcoxon_statistic": np.nan,
            "wilcoxon_p_uncorrected": np.nan,
            "wilcoxon_p_fdr_bh": np.nan,
            "omnibus_p_fdr_bh": np.nan,
            "interpret_posthoc": False,
            "skip_reason": "",
        }

        if n_pairs < min_complete_repeats:
            row["skip_reason"] = f"fewer than {min_complete_repeats} paired seizures"
            rows.append(row)
            continue

        try:
            stats_row = paired_stats(
                complete[baseline_window].to_numpy(dtype=float),
                complete[comparison_window].to_numpy(dtype=float),
            )
            row.update(stats_row)
            if not np.isfinite(row["mean_difference"]):
                row["skip_reason"] = "paired mean difference was not finite"
        except Exception as exc:
            row["skip_reason"] = f"paired comparison failed: {exc}"

        rows.append(row)
    return rows


def run_subject_average_posthoc_for_combo(
    combo_df: pd.DataFrame,
    metric: str,
    method: str,
    band: str,
    min_complete_repeats: int,
) -> list[dict]:
    """Run planned paired contrasts on subject-average values."""
    rows = []
    for baseline_window, comparison_window in POSTHOC_CONTRASTS:
        windows = [baseline_window, comparison_window]
        complete = complete_subject_average_matrix(combo_df, windows)
        n_subjects = len(complete)
        row = {
            **combo_base(metric, method, band),
            "contrast": f"{baseline_window}_vs_{comparison_window}",
            "n_subjects": int(n_subjects),
            "mean_baseline": np.nan,
            "mean_comparison": np.nan,
            "mean_difference": np.nan,
            "t": np.nan,
            "t_df": np.nan,
            "t_p_uncorrected": np.nan,
            "t_p_fdr_bh": np.nan,
            "cohen_dz": np.nan,
            "wilcoxon_statistic": np.nan,
            "wilcoxon_p_uncorrected": np.nan,
            "wilcoxon_p_fdr_bh": np.nan,
            "omnibus_p_fdr_bh": np.nan,
            "interpret_posthoc": False,
            "skip_reason": "",
        }

        if n_subjects < min_complete_repeats:
            row["skip_reason"] = f"fewer than {min_complete_repeats} paired subjects"
            rows.append(row)
            continue

        try:
            stats_row = paired_stats(
                complete[baseline_window].to_numpy(dtype=float),
                complete[comparison_window].to_numpy(dtype=float),
            )
            row.update(stats_row)
            if not np.isfinite(row["mean_difference"]):
                row["skip_reason"] = "paired mean difference was not finite"
        except Exception as exc:
            row["skip_reason"] = f"paired comparison failed: {exc}"

        rows.append(row)
    return rows


def mixed_preictal_trend(complete_long: pd.DataFrame) -> dict:
    """Run mixed model preictal slope test."""
    if smf is None:
        raise RuntimeError("statsmodels is not installed")

    data = complete_long.copy()
    data["time_to_seizure"] = data["window"].map(PREICTAL_TIME_MIN).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(
            "value ~ time_to_seizure",
            data=data,
            groups=data["subject_id"],
            re_formula="1",
            vc_formula={"seizure": "0 + C(seizure_id)"},
        )
        result = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)

    slope = float(result.params.get("time_to_seizure", np.nan))
    statistic = float(result.tvalues.get("time_to_seizure", np.nan))
    p_value = float(result.pvalues.get("time_to_seizure", np.nan))
    return {
        "model_type": "linear_mixed_effects_subject_random_intercept_seizure_vc",
        "slope_per_minute": slope,
        "statistic": statistic,
        "p_uncorrected": p_value,
    }


def fallback_preictal_trend(complete: pd.DataFrame) -> dict:
    """Fallback repeated-measures linear trend using per-seizure slopes."""
    times = np.array([PREICTAL_TIME_MIN[window] for window in PREICTAL_WINDOWS], dtype=float)
    slopes = []
    for _, row in complete[PREICTAL_WINDOWS].iterrows():
        fit = np.polyfit(times, row.to_numpy(dtype=float), 1)
        slopes.append(fit[0])
    slopes = np.asarray(slopes, dtype=float)
    slope_mean = float(np.mean(slopes)) if len(slopes) else np.nan

    statistic = np.nan
    p_value = np.nan
    if stats is not None and len(slopes) >= 2:
        res = stats.ttest_1samp(slopes, popmean=0.0, nan_policy="omit")
        statistic = float(res.statistic)
        p_value = float(res.pvalue)

    return {
        "model_type": "fallback_per_seizure_slope_one_sample_t",
        "slope_per_minute": slope_mean,
        "statistic": statistic,
        "p_uncorrected": p_value,
    }


def run_preictal_trend_for_combo(
    combo_df: pd.DataFrame,
    metric: str,
    method: str,
    band: str,
    min_complete_repeats: int,
    use_mixed_models: bool,
) -> dict:
    """Run or skip one T0/T1/T2 preictal trend test."""
    complete_long, complete = complete_long_with_subjects(combo_df, PREICTAL_WINDOWS)
    n_complete = len(complete)
    n_subjects = n_subjects_for_complete(combo_df, complete)
    row = {
        **combo_base(metric, method, band),
        "n_complete_seizures": int(n_complete),
        "n_subjects": int(n_subjects),
        "model_type": "",
        "slope_per_minute": np.nan,
        "statistic": np.nan,
        "p_uncorrected": np.nan,
        "p_fdr_bh": np.nan,
        "skip_reason": "",
        "fallback_reason": "",
    }

    if n_complete < min_complete_repeats:
        row["skip_reason"] = f"fewer than {min_complete_repeats} complete preictal seizures"
        return row

    if use_mixed_models:
        try:
            row.update(mixed_preictal_trend(complete_long))
            return row
        except Exception as exc:
            row["fallback_reason"] = f"mixed model failed: {exc}"

    try:
        row.update(fallback_preictal_trend(complete))
        if not np.isfinite(row["slope_per_minute"]):
            row["skip_reason"] = "fallback failed: trend slope was not finite"
    except Exception as exc:
        row["skip_reason"] = f"fallback failed: {exc}"
    return row


def iter_metric_method_band(long_df: pd.DataFrame):
    """Yield metric x method x band groups in deterministic order."""
    for (metric, method, band), combo_df in long_df.groupby(["metric", "method", "band"], sort=True, observed=True):
        yield metric, method, band, combo_df


def add_fdr_columns(
    omnibus: pd.DataFrame,
    posthoc: pd.DataFrame,
    trend: pd.DataFrame,
    subject_omnibus: pd.DataFrame,
    subject_posthoc: pd.DataFrame,
) -> None:
    """Apply BH-FDR separately within required output families."""
    omnibus["p_fdr_bh"] = fdr_bh(omnibus["p_uncorrected"])
    posthoc["t_p_fdr_bh"] = fdr_bh(posthoc["t_p_uncorrected"])
    posthoc["wilcoxon_p_fdr_bh"] = fdr_bh(posthoc["wilcoxon_p_uncorrected"])
    trend["p_fdr_bh"] = fdr_bh(trend["p_uncorrected"])
    subject_omnibus["p_fdr_bh"] = fdr_bh(subject_omnibus["p_uncorrected"])
    subject_posthoc["t_p_fdr_bh"] = fdr_bh(subject_posthoc["t_p_uncorrected"])
    subject_posthoc["wilcoxon_p_fdr_bh"] = fdr_bh(
        subject_posthoc["wilcoxon_p_uncorrected"]
    )


def attach_omnibus_interpretation(
    posthoc: pd.DataFrame, omnibus: pd.DataFrame, alpha: float
) -> pd.DataFrame:
    """Attach omnibus FDR p-values and interpretation flags to posthoc rows."""
    lookup = omnibus[["metric", "method", "band", "p_fdr_bh"]].rename(
        columns={"p_fdr_bh": "omnibus_p_fdr_bh"}
    )
    posthoc = posthoc.drop(columns=["omnibus_p_fdr_bh"], errors="ignore").merge(
        lookup,
        on=["metric", "method", "band"],
        how="left",
    )
    posthoc["interpret_posthoc"] = posthoc["omnibus_p_fdr_bh"].lt(alpha).fillna(False)
    return posthoc


def output_paths() -> list[Path]:
    """All files written by this script."""
    return [
        COMBINED_LONG_PATH,
        DESCRIPTIVE_PATH,
        SEIZURE_OMNIBUS_PATH,
        SEIZURE_POSTHOC_PATH,
        SEIZURE_PREICTAL_TREND_PATH,
        OMNIBUS_ALIAS_PATH,
        POSTHOC_ALIAS_PATH,
        PREICTAL_TREND_ALIAS_PATH,
        SUBJECT_AVERAGE_LONG_PATH,
        SUBJECT_AVERAGE_DESCRIPTIVE_PATH,
        SUBJECT_AVERAGE_OMNIBUS_PATH,
        SUBJECT_AVERAGE_POSTHOC_PATH,
    ]


def ensure_outputs_can_be_written(overwrite: bool) -> None:
    """Protect existing outputs unless --overwrite is passed."""
    existing = [path for path in output_paths() if path.exists()]
    if existing and not overwrite:
        names = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(f"Statistics outputs already exist; pass --overwrite to replace:\n  {names}")


def write_outputs(
    long_df: pd.DataFrame,
    descriptive: pd.DataFrame,
    omnibus: pd.DataFrame,
    posthoc: pd.DataFrame,
    trend: pd.DataFrame,
    subject_avg: pd.DataFrame,
    subject_avg_descriptive: pd.DataFrame,
    subject_omnibus: pd.DataFrame,
    subject_posthoc: pd.DataFrame,
    overwrite: bool,
) -> None:
    """Write all statistics outputs."""
    ensure_outputs_can_be_written(overwrite)
    STATISTICS_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(COMBINED_LONG_PATH, index=False)
    descriptive.to_csv(DESCRIPTIVE_PATH, index=False)
    omnibus.to_csv(SEIZURE_OMNIBUS_PATH, index=False)
    posthoc.to_csv(SEIZURE_POSTHOC_PATH, index=False)
    trend.to_csv(SEIZURE_PREICTAL_TREND_PATH, index=False)
    omnibus.to_csv(OMNIBUS_ALIAS_PATH, index=False)
    posthoc.to_csv(POSTHOC_ALIAS_PATH, index=False)
    trend.to_csv(PREICTAL_TREND_ALIAS_PATH, index=False)
    subject_avg.to_csv(SUBJECT_AVERAGE_LONG_PATH, index=False)
    subject_avg_descriptive.to_csv(SUBJECT_AVERAGE_DESCRIPTIVE_PATH, index=False)
    subject_omnibus.to_csv(SUBJECT_AVERAGE_OMNIBUS_PATH, index=False)
    subject_posthoc.to_csv(SUBJECT_AVERAGE_POSTHOC_PATH, index=False)


def count_complete_seizures(long_df: pd.DataFrame, windows: list[str]) -> int:
    """Count unique seizures with all requested windows present in the long table."""
    availability = long_df[["subject_id", "seizure_id", "window"]].drop_duplicates()
    counts = availability[availability["window"].isin(windows)].groupby(["subject_id", "seizure_id"])["window"].nunique()
    return int((counts == len(windows)).sum())


def count_complete_subjects(subject_avg: pd.DataFrame, windows: list[str]) -> int:
    """Count subjects with all requested subject-average windows present."""
    availability = subject_avg[["subject_id", "window"]].drop_duplicates()
    counts = (
        availability[availability["window"].isin(windows)]
        .groupby("subject_id")["window"]
        .nunique()
    )
    return int((counts == len(windows)).sum())


def run_statistics(args: argparse.Namespace) -> dict:
    """Run the full-dataset statistics workflow."""
    files = discover_graph_metric_files()
    wide_df = load_graph_metrics(files)
    metrics = available_metric_columns(wide_df)

    validate_wide_input(wide_df)
    long_df = make_long_dataframe(wide_df, metrics)
    validate_long_dataframe(long_df, len(wide_df), len(metrics))

    descriptive = descriptive_summary(long_df, alpha=args.alpha)
    subject_avg = subject_average_long(long_df)
    subject_avg_descriptive = subject_average_descriptive_summary(
        subject_avg, alpha=args.alpha
    )

    omnibus_rows = []
    posthoc_rows = []
    trend_rows = []
    subject_omnibus_rows = []
    subject_posthoc_rows = []
    use_mixed_models = (not args.no_mixed_models) and smf is not None

    for metric, method, band, combo_df in iter_metric_method_band(long_df):
        omnibus_rows.append(
            run_omnibus_for_combo(
                combo_df,
                metric,
                method,
                band,
                args.min_complete_repeats,
                use_mixed_models,
            )
        )
        posthoc_rows.extend(
            run_posthoc_for_combo(
                combo_df,
                metric,
                method,
                band,
                args.min_complete_repeats,
            )
        )
        trend_rows.append(
            run_preictal_trend_for_combo(
                combo_df,
                metric,
                method,
                band,
                args.min_complete_repeats,
                use_mixed_models,
            )
        )

    for metric, method, band, combo_df in iter_metric_method_band(subject_avg):
        subject_omnibus_rows.append(
            run_subject_average_omnibus_for_combo(
                combo_df,
                metric,
                method,
                band,
                args.min_complete_repeats,
                use_mixed_models,
            )
        )
        subject_posthoc_rows.extend(
            run_subject_average_posthoc_for_combo(
                combo_df,
                metric,
                method,
                band,
                args.min_complete_repeats,
            )
        )

    omnibus = pd.DataFrame(omnibus_rows, columns=OMNIBUS_COLUMNS)
    posthoc = pd.DataFrame(posthoc_rows, columns=POSTHOC_COLUMNS)
    trend = pd.DataFrame(trend_rows, columns=TREND_COLUMNS)
    subject_omnibus = pd.DataFrame(
        subject_omnibus_rows, columns=SUBJECT_AVERAGE_OMNIBUS_COLUMNS
    )
    subject_posthoc = pd.DataFrame(
        subject_posthoc_rows, columns=SUBJECT_AVERAGE_POSTHOC_COLUMNS
    )
    add_fdr_columns(omnibus, posthoc, trend, subject_omnibus, subject_posthoc)
    posthoc = attach_omnibus_interpretation(posthoc, omnibus, args.alpha)
    subject_posthoc = attach_omnibus_interpretation(
        subject_posthoc, subject_omnibus, args.alpha
    )
    posthoc = posthoc[POSTHOC_COLUMNS]
    subject_posthoc = subject_posthoc[SUBJECT_AVERAGE_POSTHOC_COLUMNS]

    write_outputs(
        long_df,
        descriptive,
        omnibus,
        posthoc,
        trend,
        subject_avg,
        subject_avg_descriptive,
        subject_omnibus,
        subject_posthoc,
        args.overwrite,
    )

    return {
        "wide_df": wide_df,
        "long_df": long_df,
        "metrics": metrics,
        "omnibus": omnibus,
        "posthoc": posthoc,
        "trend": trend,
        "subject_avg": subject_avg,
        "subject_avg_descriptive": subject_avg_descriptive,
        "subject_omnibus": subject_omnibus,
        "subject_posthoc": subject_posthoc,
        "files": files,
    }


def print_validation(results: dict) -> None:
    """Print required validation output."""
    wide_df = results["wide_df"]
    long_df = results["long_df"]
    metrics = results["metrics"]
    omnibus = results["omnibus"]
    posthoc = results["posthoc"]
    trend = results["trend"]
    subject_avg = results["subject_avg"]
    subject_omnibus = results["subject_omnibus"]
    subject_posthoc = results["subject_posthoc"]

    subjects = wide_df["subject_id"].nunique()
    seizures = wide_df["seizure_id"].nunique()
    complete_four = count_complete_seizures(long_df, WINDOW_ORDER)
    complete_preictal = count_complete_seizures(long_df, PREICTAL_WINDOWS)
    combinations = long_df[["metric", "method", "band"]].drop_duplicates().shape[0]
    completed_omnibus = int(omnibus["p_uncorrected"].notna().sum())
    completed_posthoc = int(posthoc["t_p_uncorrected"].notna().sum())
    completed_trend = int(trend["p_uncorrected"].notna().sum())
    subject_avg_complete_four = count_complete_subjects(subject_avg, WINDOW_ORDER)
    completed_subject_omnibus = int(subject_omnibus["p_uncorrected"].notna().sum())
    completed_subject_posthoc = int(subject_posthoc["t_p_uncorrected"].notna().sum())

    print("\n" + "=" * 60)
    print("Validation summary")
    print("=" * 60)
    print(f"Total input graph rows              : {len(wide_df)}")
    print(f"Graph metric columns analyzed       : {len(metrics)}")
    print(f"Total long rows                     : {len(long_df)}")
    print(f"Number of subjects                  : {subjects}")
    print(f"Number of seizures                  : {seizures}")
    print(f"Complete four-window seizures       : {complete_four}")
    print(f"Preictal-only complete seizures     : {complete_preictal}")
    print(f"Metric x method x band combinations : {combinations}")
    print(f"Seizure-level omnibus tests completed        : {completed_omnibus}")
    print(f"Seizure-level posthoc contrasts completed    : {completed_posthoc}")
    print(f"Seizure-level preictal trend tests completed : {completed_trend}")
    print(f"Subject-average rows                         : {len(subject_avg)}")
    print(f"Subject-average complete four-window subjects: {subject_avg_complete_four}")
    print(f"Subject-average omnibus tests completed      : {completed_subject_omnibus}")
    print(f"Subject-average posthoc contrasts completed  : {completed_subject_posthoc}")
    print("\nOutput files:")
    for path in output_paths():
        print(f"  {path}")

    print("\nSoftware:")
    print(f"  Python      : {platform.python_version()}")
    print(f"  NumPy       : {np.__version__}")
    print(f"  pandas      : {pd.__version__}")
    print(f"  SciPy       : {scipy.__version__ if scipy is not None else 'not_installed'}")
    print(f"  Pingouin    : {pg.__version__ if pg is not None else 'not_installed'}")
    print(f"  statsmodels : {sm.__version__ if sm is not None else 'not_installed'}")


def main() -> None:
    """Run full-dataset exploratory statistics."""
    args = parse_args()
    print("=" * 60)
    print("CHB-MIT Full-Dataset Graph Statistics")
    print("=" * 60)
    print(f"Graph metrics directory : {GRAPH_METRICS_DIR}")
    print(f"Statistics directory    : {STATISTICS_DIR}")
    print(f"Mixed models            : {'OFF' if args.no_mixed_models else 'ON if available'}")
    print(f"Min complete repeats    : {args.min_complete_repeats}")
    print(f"Alpha                   : {args.alpha}")
    print(f"Overwrite outputs       : {'YES' if args.overwrite else 'NO'}")

    results = run_statistics(args)
    print_validation(results)
    print("\nStatistics complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
