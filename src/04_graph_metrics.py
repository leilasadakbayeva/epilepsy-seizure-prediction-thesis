"""
04_graph_metrics.py
-------------------
Computes graph-level features from functional connectivity matrices.

The script is designed for the full CHB-MIT thesis pipeline:
- discovers available subjects under results/connectivity_matrices/
- processes one or more subjects passed on the command line
- processes all discovered subjects if no subject is provided
- writes one graph-metrics CSV per subject under results/graph_metrics/

Graph construction is deterministic and reproducible. Matrices are treated as
undirected weighted connectivity networks, self-loops are removed, and topology
metrics are computed after proportional thresholding. Negative connectivity
values are preserved for mean_connectivity but set to zero for topology.
"""

import argparse
import math
import platform
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parent.parent
CONNECTIVITY_DIR = ROOT / "results" / "connectivity_matrices"
GRAPH_METRICS_DIR = ROOT / "results" / "graph_metrics"

# Config
DEFAULT_THRESHOLD_DENSITY = 0.20
DEFAULT_N_RANDOM_GRAPHS = 20
DEFAULT_RANDOM_SEED = 42
EPS = 1e-12

REQUIRED_OUTPUT_COLUMNS = [
    "subject",
    "seizure_id",
    "window",
    "band",
    "method",
    "matrix_file",
    "mean_connectivity",
    "density",
    "clustering_coefficient",
    "global_efficiency",
    "characteristic_path_length",
    "small_worldness",
    "modularity",
    "mean_betweenness_centrality",
    "assortativity",
    "threshold_density",
    "threshold_weight",
    "negative_edge_fraction",
    "n_zero_edges_after_cleaning",
    "n_nodes",
    "n_possible_edges",
    "n_edges",
    "is_connected",
    "n_components",
    "largest_component_size",
    "n_random_graphs",
    "random_seed",
    "python_version",
    "numpy_version",
    "pandas_version",
    "networkx_version",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute graph-level metrics from connectivity matrices."
    )
    parser.add_argument(
        "subjects",
        nargs="*",
        help=(
            "Optional subject IDs to process, for example: chb01 chb05. "
            "If omitted, all subjects found in results/connectivity_matrices/ "
            "are processed."
        ),
    )
    parser.add_argument(
        "--density",
        type=float,
        default=DEFAULT_THRESHOLD_DENSITY,
        help=f"Proportional threshold density. Default: {DEFAULT_THRESHOLD_DENSITY}",
    )
    parser.add_argument(
        "--n-random-graphs",
        type=int,
        default=DEFAULT_N_RANDOM_GRAPHS,
        help=(
            "Number of degree-preserving random graphs for small-worldness. "
            f"Default: {DEFAULT_N_RANDOM_GRAPHS}"
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for reproducible random graphs. Default: {DEFAULT_RANDOM_SEED}",
    )
    return parser.parse_args()


def discover_subjects() -> list[str]:
    """Return subject directories available under results/connectivity_matrices/."""
    if not CONNECTIVITY_DIR.exists():
        raise FileNotFoundError(
            f"Missing {CONNECTIVITY_DIR}. Run 03_connectivity.py first."
        )

    subjects = sorted(
        path.name for path in CONNECTIVITY_DIR.iterdir() if path.is_dir()
    )
    if not subjects:
        raise FileNotFoundError(
            f"No subject folders found in {CONNECTIVITY_DIR}."
        )
    return subjects


def parse_matrix_filename(path: Path) -> dict:
    """
    Parse metadata from a connectivity matrix filename.

    Expected pattern:
        <subject>_sz##_Window_band_method.npy

    Example:
        chb01_sz01_Baseline_alpha1_aec.npy
    """
    parts = path.stem.split("_")
    if len(parts) < 5:
        raise ValueError(
            "Expected filename pattern "
            "<subject>_sz##_Window_band_method.npy"
        )

    method = parts[-1]
    band = parts[-2]
    window = parts[-3]
    seizure_id = "_".join(parts[:-3])

    if "_sz" in seizure_id:
        subject = seizure_id.split("_sz", maxsplit=1)[0]
    else:
        subject = parts[0]

    return {
        "subject": subject,
        "seizure_id": seizure_id,
        "window": window,
        "band": band,
        "method": method,
        "matrix_file": path.name,
    }


def matrix_off_diagonal(matrix: np.ndarray) -> np.ndarray:
    """Return upper-triangle off-diagonal matrix values."""
    return matrix[np.triu_indices_from(matrix, k=1)]


def clean_matrix_for_topology(matrix: np.ndarray) -> tuple[np.ndarray, float, int]:
    """
    Symmetrize a matrix, remove self-loops, and remove invalid topology weights.

    Returns:
        cleaned_matrix:
            Symmetric non-negative matrix with zero diagonal.
        negative_edge_fraction:
            Fraction of off-diagonal upper-triangle values that were negative
            before clipping.
        n_zero_edges_after_cleaning:
            Number of upper-triangle off-diagonal values equal to zero after
            invalid values and negatives have been set to zero.
    """
    clean = np.array(matrix, dtype=float, copy=True)
    clean = np.nan_to_num(clean, nan=0.0, posinf=0.0, neginf=0.0)
    clean = (clean + clean.T) / 2.0
    np.fill_diagonal(clean, 0.0)

    off_diag = matrix_off_diagonal(clean)
    n_possible = len(off_diag)
    negative_edge_fraction = (
        float(np.sum(off_diag < 0.0) / n_possible) if n_possible else np.nan
    )

    clean[clean < 0.0] = 0.0
    np.fill_diagonal(clean, 0.0)
    cleaned_off_diag = matrix_off_diagonal(clean)
    n_zero_edges_after_cleaning = int(np.sum(cleaned_off_diag <= 0.0))

    return clean, negative_edge_fraction, n_zero_edges_after_cleaning


def proportional_threshold(
    matrix: np.ndarray, threshold_density: float
) -> tuple[np.ndarray, float, int]:
    """
    Keep the strongest positive edges up to a target proportional density.

    If there are fewer positive edges than requested, all positive edges are
    kept and the realized graph density is lower than the requested density.
    Ties are resolved deterministically by node index.
    """
    if not 0.0 < threshold_density <= 1.0:
        raise ValueError("threshold_density must be in the interval (0, 1].")

    n_nodes = matrix.shape[0]
    rows, cols = np.triu_indices(n_nodes, k=1)
    weights = matrix[rows, cols]
    n_possible_edges = len(weights)

    target_edges = int(math.ceil(threshold_density * n_possible_edges))
    positive_mask = weights > 0.0
    positive_rows = rows[positive_mask]
    positive_cols = cols[positive_mask]
    positive_weights = weights[positive_mask]

    if len(positive_weights) == 0:
        return np.zeros_like(matrix), np.nan, 0

    keep_count = min(target_edges, len(positive_weights))
    order = np.lexsort((positive_cols, positive_rows, -positive_weights))
    keep_idx = order[:keep_count]

    thresholded = np.zeros_like(matrix)
    kept_rows = positive_rows[keep_idx]
    kept_cols = positive_cols[keep_idx]
    kept_weights = positive_weights[keep_idx]
    thresholded[kept_rows, kept_cols] = kept_weights
    thresholded[kept_cols, kept_rows] = kept_weights

    threshold_weight = float(np.min(kept_weights))
    return thresholded, threshold_weight, keep_count


def build_graph(matrix: np.ndarray) -> nx.Graph:
    """
    Build an undirected graph with separate strength and distance attributes.

    Edge attributes:
        weight:
            Retained connectivity strength.
        distance:
            Inverse connectivity strength, used for weighted shortest paths.
    """
    graph = nx.Graph()
    n_nodes = matrix.shape[0]
    graph.add_nodes_from(range(n_nodes))

    rows, cols = np.triu_indices(n_nodes, k=1)
    for i, j, weight in zip(rows, cols, matrix[rows, cols]):
        if weight > 0.0:
            graph.add_edge(
                int(i),
                int(j),
                weight=float(weight),
                distance=float(1.0 / max(weight, EPS)),
            )
    return graph


def largest_connected_component(graph: nx.Graph) -> nx.Graph:
    """Return the largest connected component as a copied subgraph."""
    if graph.number_of_nodes() == 0:
        return graph.copy()
    components = list(nx.connected_components(graph))
    if not components:
        return graph.copy()
    largest_nodes = max(components, key=len)
    return graph.subgraph(largest_nodes).copy()


def weighted_global_efficiency(graph: nx.Graph) -> float:
    """
    Compute weighted global efficiency using inverse-strength distances.

    Unreachable node pairs contribute zero. This intentionally avoids
    networkx.global_efficiency(), which ignores edge weights.
    """
    n_nodes = graph.number_of_nodes()
    if n_nodes < 2:
        return np.nan

    path_lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight="distance"))
    efficiency_sum = 0.0

    for source in graph.nodes:
        for target in graph.nodes:
            if source == target:
                continue
            distance = path_lengths.get(source, {}).get(target)
            if distance is not None and np.isfinite(distance) and distance > 0.0:
                efficiency_sum += 1.0 / distance

    return float(efficiency_sum / (n_nodes * (n_nodes - 1)))


def characteristic_path_length(graph: nx.Graph) -> float:
    """Compute weighted path length on the largest connected component."""
    if graph.number_of_edges() == 0:
        return np.nan

    lcc = largest_connected_component(graph)
    if lcc.number_of_nodes() < 2 or lcc.number_of_edges() == 0:
        return np.nan

    try:
        return float(nx.average_shortest_path_length(lcc, weight="distance"))
    except (nx.NetworkXError, ZeroDivisionError, ValueError):
        return np.nan


def graph_connectivity_diagnostics(graph: nx.Graph) -> tuple[bool, int, int]:
    """Return connectedness, component count, and largest component size."""
    if graph.number_of_nodes() == 0:
        return False, 0, 0

    components = list(nx.connected_components(graph))
    n_components = len(components)
    largest_component_size = max((len(c) for c in components), default=0)
    is_connected = n_components == 1
    return bool(is_connected), int(n_components), int(largest_component_size)


def compute_modularity(graph: nx.Graph, random_seed: int) -> float:
    """Compute weighted modularity using Louvain communities when available."""
    if graph.number_of_edges() == 0 or graph.number_of_nodes() < 2:
        return np.nan

    try:
        if hasattr(nx.community, "louvain_communities"):
            communities = nx.community.louvain_communities(
                graph, weight="weight", seed=random_seed
            )
        else:
            communities = nx.community.greedy_modularity_communities(
                graph, weight="weight"
            )

        if not communities:
            return np.nan
        return float(nx.community.modularity(graph, communities, weight="weight"))
    except (nx.NetworkXError, ZeroDivisionError, ValueError):
        return np.nan


def compute_assortativity(graph: nx.Graph) -> float:
    """Compute degree assortativity and return NaN if undefined."""
    if graph.number_of_edges() == 0 or graph.number_of_nodes() < 2:
        return np.nan

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            value = nx.degree_assortativity_coefficient(graph)
        return float(value) if np.isfinite(value) else np.nan
    except (nx.NetworkXError, ZeroDivisionError, ValueError):
        return np.nan


def assign_shuffled_weights(
    random_graph: nx.Graph, weights: np.ndarray, rng: np.random.Generator
) -> nx.Graph:
    """Assign the observed weight distribution to a randomized topology."""
    graph = random_graph.copy()
    shuffled_weights = rng.permutation(weights)
    edges = sorted(graph.edges())

    for edge, weight in zip(edges, shuffled_weights):
        graph.edges[edge]["weight"] = float(weight)
        graph.edges[edge]["distance"] = float(1.0 / max(weight, EPS))

    return graph


def compute_small_worldness(
    graph: nx.Graph, n_random_graphs: int, random_seed: int
) -> float:
    """
    Compute weighted small-worldness against randomized comparison graphs.

    Random graphs preserve the observed degree sequence through double-edge
    swaps when possible. The observed edge-weight distribution is then shuffled
    onto each randomized topology. If valid random comparisons cannot be
    generated, NaN is returned.
    """
    if (
        graph.number_of_nodes() < 4
        or graph.number_of_edges() < 2
        or n_random_graphs <= 0
    ):
        return np.nan

    c_obs = nx.average_clustering(graph, weight="weight")
    l_obs = characteristic_path_length(graph)
    if not np.isfinite(c_obs) or not np.isfinite(l_obs) or c_obs <= 0.0 or l_obs <= 0.0:
        return np.nan

    observed_weights = np.array(
        [data["weight"] for _, _, data in sorted(graph.edges(data=True))],
        dtype=float,
    )

    rng = np.random.default_rng(random_seed)
    random_clusterings = []
    random_path_lengths = []

    for _ in range(n_random_graphs):
        randomized = nx.Graph()
        randomized.add_nodes_from(graph.nodes())
        randomized.add_edges_from(graph.edges())

        try:
            n_swaps = max(1, 10 * graph.number_of_edges())
            max_tries = max(100, 100 * graph.number_of_edges())
            swap_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            nx.double_edge_swap(
                randomized,
                nswap=n_swaps,
                max_tries=max_tries,
                seed=swap_seed,
            )
        except (nx.NetworkXError, nx.NetworkXAlgorithmError):
            continue

        randomized = assign_shuffled_weights(randomized, observed_weights, rng)
        c_rand = nx.average_clustering(randomized, weight="weight")
        l_rand = characteristic_path_length(randomized)

        if (
            np.isfinite(c_rand)
            and np.isfinite(l_rand)
            and c_rand > 0.0
            and l_rand > 0.0
        ):
            random_clusterings.append(c_rand)
            random_path_lengths.append(l_rand)

    if not random_clusterings or not random_path_lengths:
        return np.nan

    c_random_mean = float(np.mean(random_clusterings))
    l_random_mean = float(np.mean(random_path_lengths))

    if c_random_mean <= 0.0 or l_random_mean <= 0.0:
        return np.nan

    return float((c_obs / c_random_mean) / (l_obs / l_random_mean))


def load_existing_output(
    output_path: Path, threshold_density: float, n_random_graphs: int, random_seed: int
) -> tuple[list[dict], set[str]]:
    """
    Load matching previous output rows so the script can resume safely.

    Rows are reused only when the key reproducibility parameters match the
    current run.
    """
    if not output_path.exists():
        return [], set()

    try:
        existing = pd.read_csv(output_path)
    except (OSError, pd.errors.ParserError):
        print(f"[warning] Could not read existing output: {output_path}")
        return [], set()

    missing_columns = [
        column for column in REQUIRED_OUTPUT_COLUMNS if column not in existing.columns
    ]
    if missing_columns:
        print(
            "[warning] Existing output is missing required columns; "
            "recomputing all rows."
        )
        return [], set()

    matching = existing[
        np.isclose(existing["threshold_density"], threshold_density)
        & (existing["n_random_graphs"] == n_random_graphs)
        & (existing["random_seed"] == random_seed)
    ].copy()

    records = matching.to_dict("records")
    completed_files = set(matching["matrix_file"].astype(str))
    return records, completed_files


def save_records(records: list[dict], output_path: Path) -> pd.DataFrame:
    """Save graph metric records in deterministic order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records, columns=REQUIRED_OUTPUT_COLUMNS)
    if not df.empty:
        df = (
            df.drop_duplicates(subset=["matrix_file"], keep="last")
            .sort_values(["seizure_id", "window", "band", "method", "matrix_file"])
            .reset_index(drop=True)
        )
    df.to_csv(output_path, index=False)
    return df


def compute_metrics_for_matrix(
    matrix_path: Path,
    threshold_density: float,
    n_random_graphs: int,
    random_seed: int,
) -> dict:
    """Load one matrix and compute graph-level metrics."""
    metadata = parse_matrix_filename(matrix_path)
    matrix = np.load(matrix_path)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix must be square; got shape {matrix.shape}")

    raw = np.array(matrix, dtype=float, copy=True)
    raw = (raw + raw.T) / 2.0
    np.fill_diagonal(raw, 0.0)
    raw_off_diag = matrix_off_diagonal(raw)
    mean_connectivity = float(np.nanmean(raw_off_diag)) if len(raw_off_diag) else np.nan

    clean_matrix, negative_fraction, n_zero_edges = clean_matrix_for_topology(raw)
    thresholded, threshold_weight, n_edges = proportional_threshold(
        clean_matrix, threshold_density
    )

    graph = build_graph(thresholded)
    n_nodes = graph.number_of_nodes()
    n_possible_edges = n_nodes * (n_nodes - 1) // 2
    realized_density = (
        float(n_edges / n_possible_edges) if n_possible_edges > 0 else np.nan
    )
    is_connected, n_components, largest_component_size = graph_connectivity_diagnostics(
        graph
    )

    if graph.number_of_edges() > 0:
        clustering = float(nx.average_clustering(graph, weight="weight"))
        global_efficiency = weighted_global_efficiency(graph)
        path_length = characteristic_path_length(graph)
        modularity = compute_modularity(graph, random_seed)
        betweenness = nx.betweenness_centrality(
            graph, normalized=True, weight="distance"
        )
        mean_betweenness = float(np.mean(list(betweenness.values())))
        assortativity = compute_assortativity(graph)
        small_worldness = compute_small_worldness(
            graph, n_random_graphs, random_seed
        )
    else:
        clustering = np.nan
        global_efficiency = 0.0 if n_nodes >= 2 else np.nan
        path_length = np.nan
        modularity = np.nan
        mean_betweenness = 0.0 if n_nodes > 0 else np.nan
        assortativity = np.nan
        small_worldness = np.nan

    return {
        **metadata,
        "mean_connectivity": mean_connectivity,
        "density": realized_density,
        "clustering_coefficient": clustering,
        "global_efficiency": global_efficiency,
        "characteristic_path_length": path_length,
        "small_worldness": small_worldness,
        "modularity": modularity,
        "mean_betweenness_centrality": mean_betweenness,
        "assortativity": assortativity,
        "threshold_density": float(threshold_density),
        "threshold_weight": threshold_weight,
        "negative_edge_fraction": negative_fraction,
        "n_zero_edges_after_cleaning": int(n_zero_edges),
        "n_nodes": int(n_nodes),
        "n_possible_edges": int(n_possible_edges),
        "n_edges": int(n_edges),
        "is_connected": bool(is_connected),
        "n_components": int(n_components),
        "largest_component_size": int(largest_component_size),
        "n_random_graphs": int(n_random_graphs),
        "random_seed": int(random_seed),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "networkx_version": nx.__version__,
    }


def process_subject(
    subject: str,
    threshold_density: float,
    n_random_graphs: int,
    random_seed: int,
) -> dict:
    """Process all connectivity matrices for one subject."""
    subject_dir = CONNECTIVITY_DIR / subject
    output_path = GRAPH_METRICS_DIR / f"{subject}_graph_metrics.csv"

    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject folder not found: {subject_dir}")

    matrix_paths = sorted(subject_dir.glob("*.npy"))
    records, completed_files = load_existing_output(
        output_path, threshold_density, n_random_graphs, random_seed
    )

    print("\n" + "=" * 60)
    print(f"Graph metrics pipeline - Subject: {subject}")
    print("=" * 60)
    print(f"Input folder       : {subject_dir}")
    print(f"Output CSV         : {output_path}")
    print(f"Matrices found     : {len(matrix_paths)}")
    print(f"Rows already done  : {len(completed_files)}")
    print(f"Threshold density  : {threshold_density}")
    print(f"Random graphs      : {n_random_graphs}")
    print(f"Random seed        : {random_seed}")

    skipped_files = []

    for idx, matrix_path in enumerate(matrix_paths, start=1):
        if matrix_path.name in completed_files:
            print(f"  [{idx:03d}/{len(matrix_paths)}] loaded   {matrix_path.name}")
            continue

        try:
            row = compute_metrics_for_matrix(
                matrix_path,
                threshold_density=threshold_density,
                n_random_graphs=n_random_graphs,
                random_seed=random_seed,
            )
            records.append(row)
            save_records(records, output_path)
            print(
                f"  [{idx:03d}/{len(matrix_paths)}] computed {matrix_path.name} "
                f"density={row['density']:.4f} "
                f"eff={row['global_efficiency']:.4f}"
            )
        except Exception as exc:
            skipped_files.append({"matrix_file": matrix_path.name, "error": str(exc)})
            print(f"  [{idx:03d}/{len(matrix_paths)}] skipped  {matrix_path.name}: {exc}")

    final_df = save_records(records, output_path)

    print("\n-- Validation summary --")
    print(f"Matrices found          : {len(matrix_paths)}")
    print(f"Rows written            : {len(final_df)}")
    print(f"Corrupted/skipped files : {len(skipped_files)}")
    print(f"Output path             : {output_path}")

    if skipped_files:
        print("\nSkipped file details:")
        for item in skipped_files:
            print(f"  - {item['matrix_file']}: {item['error']}")

    if len(final_df) != len(matrix_paths):
        print(
            "[warning] Row count does not match discovered matrix count. "
            "Check skipped files and previous output parameters."
        )

    return {
        "subject": subject,
        "matrices_found": len(matrix_paths),
        "rows_written": len(final_df),
        "skipped_files": skipped_files,
        "output_path": output_path,
    }


def main() -> None:
    """Run graph metrics for requested or discovered subjects."""
    args = parse_args()

    subjects = args.subjects if args.subjects else discover_subjects()

    print("=" * 60)
    print("Graph Metrics Pipeline")
    print("=" * 60)
    print(f"Repository root     : {ROOT}")
    print(f"Subjects            : {', '.join(subjects)}")
    print(f"Threshold density   : {args.density}")
    print(f"Random graphs       : {args.n_random_graphs}")
    print(f"Random seed         : {args.random_seed}")
    print(f"Python              : {platform.python_version()}")
    print(f"NumPy               : {np.__version__}")
    print(f"pandas              : {pd.__version__}")
    print(f"NetworkX            : {nx.__version__}")

    summaries = []
    for subject in subjects:
        summaries.append(
            process_subject(
                subject=subject,
                threshold_density=args.density,
                n_random_graphs=args.n_random_graphs,
                random_seed=args.random_seed,
            )
        )

    print("\n" + "=" * 60)
    print("Final validation")
    print("=" * 60)
    for summary in summaries:
        print(
            f"{summary['subject']}: "
            f"matrices found={summary['matrices_found']}, "
            f"rows written={summary['rows_written']}, "
            f"corrupted/skipped={len(summary['skipped_files'])}, "
            f"output={summary['output_path']}"
        )

    print("\nGraph metrics complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
