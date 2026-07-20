"""Calculate all-weight graph features from Koopman singular vectors."""

import gc
import json
import os
import pickle
import time
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count

import igraph as ig
import numpy as np
import pandas as pd
from tqdm import tqdm


warnings.filterwarnings("ignore")


class ImprovedGraphMetrics:
    """Reconstruct weighted graphs and calculate four node-level metrics."""

    def individual_zscore_normalization_koopman(self, eigenvector_data):
        """Apply one global z-score normalization within a subject run."""
        all_data = np.asarray(eigenvector_data, dtype=np.float32)
        flattened_data = all_data.ravel()

        individual_mean = np.mean(flattened_data)
        individual_std = np.std(flattened_data)
        if individual_std == 0:
            individual_std = 1.0

        normalized_data = (all_data - individual_mean) / individual_std
        norm_stats = {
            "mean": float(individual_mean),
            "std": float(individual_std),
            "original_range": [
                float(np.min(flattened_data)),
                float(np.max(flattened_data)),
            ],
            "normalized_range": [
                float(np.min(normalized_data)),
                float(np.max(normalized_data)),
            ],
        }
        return normalized_data, norm_stats

    def reconstruct_connectivity_matrices(
        self, eigenvectors, n_channels, n_eigenvectors=10
    ):
        """Reconstruct symmetric connectivity matrices from edge vectors."""
        n_koopman_windows = eigenvectors.shape[0]
        upper_rows, upper_cols = np.triu_indices(n_channels, k=1)
        connectivity_matrices = np.zeros(
            (
                n_koopman_windows,
                n_channels,
                n_channels,
                n_eigenvectors,
            ),
            dtype=np.float32,
        )

        for window in range(n_koopman_windows):
            for eig_idx in range(n_eigenvectors):
                matrix = np.zeros((n_channels, n_channels), dtype=np.float32)
                matrix[upper_rows, upper_cols] = eigenvectors[
                    window, :, eig_idx
                ]
                matrix += matrix.T
                connectivity_matrices[window, :, :, eig_idx] = matrix

            if window % 100 == 0:
                gc.collect()

        return connectivity_matrices

    def create_weighted_graph(self, matrix, n_channels):
        """Create an undirected graph using every nonzero absolute weight."""
        abs_matrix = np.abs(matrix)
        edges = []
        similarity_weights = []
        distance_weights = []

        for source in range(n_channels):
            for target in range(source + 1, n_channels):
                similarity = float(abs_matrix[source, target])
                if similarity > 0.0:
                    edges.append((source, target))
                    similarity_weights.append(similarity)
                    distance_weights.append(1.0 / (similarity + 1e-8))

        graph = ig.Graph(n=n_channels, directed=False)
        if edges:
            graph.add_edges(edges)
            graph.es["similarity_weight"] = similarity_weights
            graph.es["distance_weight"] = distance_weights

        return graph

    def compute_selected_metrics(self, matrix, n_channels):
        """Calculate node strength, degree spread, centrality, and efficiency."""
        try:
            graph = self.create_weighted_graph(matrix, n_channels)
            abs_matrix = np.abs(matrix)

            if graph.ecount() > 0:
                node_strength = graph.strength(weights="similarity_weight")
            else:
                node_strength = [0.0] * n_channels

            metrics = {
                "node_strength": node_strength,
                "degree_std": np.std(abs_matrix, axis=0).tolist(),
            }

            if graph.ecount() > 0:
                try:
                    metrics["eigenvector_centrality"] = (
                        graph.eigenvector_centrality(
                            directed=False,
                            weights="similarity_weight",
                            scale=False,
                        )
                    )
                except Exception:
                    metrics["eigenvector_centrality"] = [0.0] * n_channels
            else:
                metrics["eigenvector_centrality"] = [0.0] * n_channels

            if graph.ecount() > 0:
                try:
                    distances = graph.distances(weights="distance_weight")
                    global_efficiency = []
                    for source in range(n_channels):
                        finite_distances = [
                            distance
                            for distance in distances[source]
                            if distance != float("inf") and distance > 0
                        ]
                        if finite_distances:
                            global_efficiency.append(
                                np.mean(
                                    [
                                        1.0 / distance
                                        for distance in finite_distances
                                    ]
                                )
                            )
                        else:
                            global_efficiency.append(0.0)
                    metrics["global_efficiency"] = global_efficiency
                except Exception:
                    metrics["global_efficiency"] = [0.0] * n_channels
            else:
                metrics["global_efficiency"] = [0.0] * n_channels

            return metrics, True
        except Exception:
            default_metrics = {
                "node_strength": [0.0] * n_channels,
                "degree_std": [0.0] * n_channels,
                "eigenvector_centrality": [0.0] * n_channels,
                "global_efficiency": [0.0] * n_channels,
            }
            return default_metrics, False


def process_single_window_improved(args):
    """Calculate all graph metrics for one Koopman window."""
    window_idx, matrices_window, n_channels, n_eigenvectors = args
    calculator = ImprovedGraphMetrics()
    window_results = {
        "window_idx": window_idx,
        "graph_metrics": {
            "node_strength": np.zeros((n_channels, n_eigenvectors)),
            "degree_std": np.zeros((n_channels, n_eigenvectors)),
            "eigenvector_centrality": np.zeros(
                (n_channels, n_eigenvectors)
            ),
            "global_efficiency": np.zeros((n_channels, n_eigenvectors)),
        },
        "success_count": 0,
    }

    for eig_idx in range(n_eigenvectors):
        matrix = matrices_window[:, :, eig_idx]
        metrics, success = calculator.compute_selected_metrics(
            matrix, n_channels
        )
        if success:
            for metric_name, values in metrics.items():
                window_results["graph_metrics"][metric_name][
                    :, eig_idx
                ] = values
            window_results["success_count"] += 1

    return window_results


def parallel_compute_improved_metrics(
    connectivity_matrices,
    n_channels,
    n_eigenvectors,
    n_cores=6,
):
    """Calculate all-weight graph metrics in parallel over time windows."""
    n_cores = max(1, min(n_cores, cpu_count()))
    n_windows = connectivity_matrices.shape[0]
    start_time = time.time()

    print(
        "    Calculating four all-weight graph metrics "
        f"with {n_cores} processes"
    )
    tasks = [
        (
            window,
            connectivity_matrices[window],
            n_channels,
            n_eigenvectors,
        )
        for window in range(n_windows)
    ]

    window_results = []
    with Pool(processes=n_cores) as pool:
        with tqdm(
            total=n_windows,
            desc="Graph metrics",
            leave=False,
        ) as progress:
            for result in pool.imap(process_single_window_improved, tasks):
                window_results.append(result)
                progress.update(1)

    graph_metrics = {
        "node_strength": np.zeros(
            (n_windows, n_channels, n_eigenvectors)
        ),
        "degree_std": np.zeros(
            (n_windows, n_channels, n_eigenvectors)
        ),
        "eigenvector_centrality": np.zeros(
            (n_windows, n_channels, n_eigenvectors)
        ),
        "global_efficiency": np.zeros(
            (n_windows, n_channels, n_eigenvectors)
        ),
    }

    total_success_count = 0
    for result in window_results:
        window_idx = result["window_idx"]
        for metric_name in graph_metrics:
            graph_metrics[metric_name][window_idx] = result[
                "graph_metrics"
            ][metric_name]
        total_success_count += result["success_count"]

    computation_time = time.time() - start_time
    total_computations = n_windows * n_eigenvectors
    graph_metrics_data = {
        "processing_mode": "all_weights",
        "n_eigenvectors": n_eigenvectors,
        "n_windows": n_windows,
        "n_channels": n_channels,
        "graph_metrics": graph_metrics,
        "processing_stats": {
            "mode": "all_weights",
            "matrices_modified": False,
            "total_connections_per_matrix": int(
                n_channels * (n_channels - 1) / 2
            ),
        },
        "quality_control": {
            "success_rate": total_success_count / total_computations,
            "total_successful_computations": total_success_count,
            "total_attempted_computations": total_computations,
        },
        "computation_metadata": {
            "timestamp": datetime.now().isoformat(),
            "computation_time_seconds": computation_time,
            "n_cores_used": n_cores,
            "software_version": "3.0.0_all_weights",
            "metrics_computed": [
                "node_strength",
                "degree_std",
                "eigenvector_centrality",
                "global_efficiency",
            ],
        },
    }

    success_rate = total_success_count / total_computations
    print(
        f"    Completed in {computation_time:.1f}s; "
        f"success rate: {success_rate:.1%}"
    )
    return graph_metrics_data


def save_improved_results(
    graph_metrics_data, subject_id, run_index, save_root_path
):
    """Save graph metrics for one subject run."""
    run_path = os.path.join(
        save_root_path,
        f"subject_{subject_id}",
        f"run_{run_index}",
    )
    os.makedirs(run_path, exist_ok=True)
    filepath = os.path.join(run_path, "improved_features_all_weights.pkl")

    graph_metrics_data["subject_id"] = subject_id
    graph_metrics_data["run_index"] = run_index
    with open(filepath, "wb") as file:
        pickle.dump(graph_metrics_data, file, protocol=4)
    return filepath


def process_subject_run_improved(
    normalized_eigenvectors,
    subject_id,
    run_index,
    n_channels,
    save_path,
    n_eigenvectors=10,
    n_cores=6,
):
    """Process and save one subject run in all-weight mode."""
    print(f"  Run {run_index}: processing all weights")
    run_start_time = time.time()
    calculator = ImprovedGraphMetrics()

    connectivity_matrices = calculator.reconstruct_connectivity_matrices(
        normalized_eigenvectors, n_channels, n_eigenvectors
    )
    print(f"    Connectivity matrices: {connectivity_matrices.shape}")

    graph_metrics_data = parallel_compute_improved_metrics(
        connectivity_matrices,
        n_channels,
        n_eigenvectors,
        n_cores=n_cores,
    )
    saved_file = save_improved_results(
        graph_metrics_data, subject_id, run_index, save_path
    )
    run_time = time.time() - run_start_time
    success_rate = graph_metrics_data["quality_control"]["success_rate"]

    del connectivity_matrices
    gc.collect()

    print(
        f"  Run {run_index}: completed in {run_time:.1f}s; "
        f"success rate: {success_rate:.1%}"
    )
    return {
        "run_index": run_index,
        "success_rate": success_rate,
        "computation_time": run_time,
        "file_saved": os.path.basename(saved_file),
        "processing_mode": "all_weights",
    }


def compute_all_subjects_improved(
    subject_eigenvector,
    subject_index,
    channel_list,
    save_path,
    n_eigenvectors=10,
    n_cores=6,
):

    os.makedirs(save_path, exist_ok=True)
    processing_log = {
        "start_time": datetime.now().isoformat(),
        "processing_mode": "all_weights",
        "metrics_computed": [
            "node_strength",
            "degree_std",
            "eigenvector_centrality",
            "global_efficiency",
        ],
        "n_eigenvectors": n_eigenvectors,
        "subjects": [],
    }

    total_subjects = len(subject_eigenvector)
    for subject_idx in range(total_subjects):
        subject_id = subject_index[subject_idx]
        n_channels = int(channel_list[subject_idx])
        subject_data = subject_eigenvector[subject_idx]
        n_runs = len(subject_data)

        print(
            f"\nSubject {subject_id} ({subject_idx + 1}/{total_subjects}); "
            f"channels: {n_channels}; runs: {n_runs}"
        )
        subject_start_time = time.time()
        completed_runs = 0
        run_stats = []
        calculator = ImprovedGraphMetrics()

        for run_idx in range(n_runs):
            try:
                normalized_data, _ = (
                    calculator.individual_zscore_normalization_koopman(
                        subject_data[run_idx]
                    )
                )
                run_result = process_subject_run_improved(
                    normalized_data,
                    subject_id,
                    run_idx,
                    n_channels,
                    save_path,
                    n_eigenvectors=n_eigenvectors,
                    n_cores=n_cores,
                )
                run_stats.append(run_result)
                completed_runs += 1
                del normalized_data
                gc.collect()
            except Exception as error:
                print(f"  Run {run_idx}: failed - {error}")
                run_stats.append(
                    {
                        "run_index": run_idx,
                        "error": str(error),
                        "success_rate": 0.0,
                        "processing_mode": "all_weights",
                    }
                )

        subject_time = time.time() - subject_start_time
        average_success_rate = np.mean(
            [stat.get("success_rate", 0.0) for stat in run_stats]
        )
        processing_log["subjects"].append(
            {
                "subject_id": subject_id,
                "subject_index": subject_idx,
                "n_channels": n_channels,
                "total_runs": n_runs,
                "completed_runs": completed_runs,
                "average_success_rate": average_success_rate,
                "processing_time_seconds": subject_time,
                "run_details": run_stats,
            }
        )
        print(
            f"Subject {subject_id}: {completed_runs}/{n_runs} runs completed "
            f"in {subject_time:.1f}s"
        )
        gc.collect()

    start_timestamp = datetime.fromisoformat(
        processing_log["start_time"]
    ).timestamp()
    total_time = time.time() - start_timestamp
    processing_log["total_processing_time"] = total_time
    processing_log["completion_time"] = datetime.now().isoformat()

    log_file = os.path.join(
        save_path, "processing_all_weights.json"
    )
    with open(log_file, "w", encoding="utf-8") as file:
        json.dump(processing_log, file, indent=2)

    total_runs = sum(
        subject["completed_runs"] for subject in processing_log["subjects"]
    )
    average_run_time = total_time / total_runs if total_runs else 0.0
    print(
        f"\nCompleted {total_runs} runs from {total_subjects} subjects in "
        f"{total_time:.1f}s; average per run: {average_run_time:.1f}s"
    )
    return processing_log


def load_pkl(file_name):
    """Load a pickle file from the project result directory."""
    folder_path = (
        "//10.20.37.22/dataset0/codes/"
        "graph_koopman_epilepsy/result"
    ) #example path, please change it to your own path
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "rb") as file:
        return pickle.load(file)


if __name__ == "__main__":
    participant_file = r"D:/project/graph_koopman/participants.xlsx"
    participants = pd.read_excel(participant_file)
    participants = participants[participants["adopt"] == 1]

    subject_index = participants["participant_id"].values.tolist()
    channel_list = participants["all_nodes"].values.tolist()
    subject_eigenvector = load_pkl(
        "high_gamma_80-200_subject_corr_svd_"
        "eigenvector_top10_win200.pkl"
    )#example path, please change it to your own path


    save_path = (
        "//10.20.37.22/dataset0/codes/"
        "graph_koopman_epilepsy/result/"
        "graph_allweight_4metrics_results"
    ) #example path, please change it to your own path

    compute_all_subjects_improved(
        subject_eigenvector=subject_eigenvector,
        subject_index=subject_index,
        channel_list=channel_list,
        save_path=save_path,
        n_eigenvectors=10,
        n_cores=6,
    )
