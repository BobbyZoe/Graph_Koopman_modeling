from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.sparse.linalg import eigsh
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans

from .data import make_windows, normalize_scores


def dynamic_envelope_corr_layers(
    X: np.ndarray,
    sfreq: float,
    window_s: float = 0.5,
    step_s: float = 0.25,
    threshold_quantile: float = 0.8,
    absolute: bool = True,
) -> List[np.ndarray]:
    """Build dynamic high-frequency synchronization layers from 80--200 Hz data."""
    X = np.asarray(X, dtype=float)
    env = np.abs(hilbert(X, axis=1))
    layers: List[np.ndarray] = []
    for s, e in make_windows(env, sfreq, window_s, step_s):
        W = env[:, s:e]
        C = np.corrcoef(W)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        if absolute:
            C = np.abs(C)
        np.fill_diagonal(C, 0.0)
        vals = C[C > 0]
        if vals.size:
            thr = np.quantile(vals, threshold_quantile)
            C = np.where(C >= thr, C, 0.0)
        layers.append(C.astype(float))
    return layers


def build_supra_adjacency(layers: List[np.ndarray], coupling: float = 1.0) -> sparse.csr_matrix:
    T = len(layers)
    if T == 0:
        raise ValueError("No layers provided")
    N = layers[0].shape[0]
    blocks = [[None for _ in range(T)] for _ in range(T)]
    I = sparse.eye(N, format="csr") * coupling
    for t in range(T):
        blocks[t][t] = sparse.csr_matrix(layers[t])
        if t < T - 1:
            blocks[t][t + 1] = I
            blocks[t + 1][t] = I
    for i in range(T):
        for j in range(T):
            if blocks[i][j] is None:
                blocks[i][j] = sparse.csr_matrix((N, N))
    return sparse.bmat(blocks, format="csr")


def multilayer_evc(
    layers: List[np.ndarray],
    coupling: float = 1.0,
    max_eigs: int | None = None,
) -> np.ndarray:
    """Compute mlEVC: weighted sum of top eigenvectors of the supra-graph.

    Returns matrix [n_channels, n_layers].
    """
    T = len(layers)
    N = layers[0].shape[0]
    A = build_supra_adjacency(layers, coupling=coupling)
    requested = T if max_eigs is None else min(T, max_eigs)
    k = int(min(max(2, requested), A.shape[0] - 2))
    try:
        vals, vecs = eigsh(A, k=k, which="LA", maxiter=5000)
    except Exception:
        dense = A.toarray()
        vals, vecs = np.linalg.eigh(dense)
        vals, vecs = vals[-k:], vecs[:, -k:]
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    ml = np.sum(np.abs(vecs * vals.reshape(1, -1)), axis=1)
    return ml.reshape(T, N).T


def quantize_mlevc(ml: np.ndarray, percentile: float = 20.0) -> np.ndarray:
    lo = np.percentile(ml, percentile / 2.0)
    hi = np.percentile(ml, 100.0 - percentile / 2.0)
    Q = np.zeros_like(ml)
    Q[ml <= lo] = -1.0
    Q[ml >= hi] = 1.0
    return Q


def mlevc_svd_features(ml_list: List[np.ndarray], n_components: int = 4, quantile_percentile: float = 20.0) -> np.ndarray:
    """Return the paper's left singular vectors after concatenating quantized mlEVC."""
    Q = np.concatenate([quantize_mlevc(ml, quantile_percentile) for ml in ml_list], axis=1)
    n_components = min(n_components, min(Q.shape))
    if n_components < 1:
        return Q.mean(axis=1, keepdims=True)
    U, _, _ = np.linalg.svd(Q, full_matrices=False)
    return U[:, :n_components]


def cluster_target_nodes(features: np.ndarray) -> np.ndarray:
    """Two-cluster hierarchical clustering. Returns binary target labels for smaller, more distinct cluster."""
    Z = np.asarray(features, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    Z = (Z - Z.mean(axis=0, keepdims=True)) / (Z.std(axis=0, keepdims=True) + 1e-12)
    if Z.shape[0] <= 2:
        return np.ones(Z.shape[0], dtype=int)
    L = linkage(pdist(Z, metric="euclidean"), method="centroid")
    labels = fcluster(L, t=2, criterion="maxclust")
    counts = {lab: np.sum(labels == lab) for lab in np.unique(labels)}
    small = min(counts, key=counts.get)
    return (labels == small).astype(int)


def dynamic_lagged_coherence_layers(
    X: np.ndarray,
    sfreq: float,
    band: Tuple[float, float],
    window_s: float = 2.5,
    step_s: float = 0.5,
    seizure_onset_s: float = 50.0,
) -> List[np.ndarray]:
    """Paper-aligned lagged-coherence layers for one high-frequency band.

    Connectivity is standardized edge-wise against windows centered before
    seizure onset and mapped to ``[0, 1)`` with ``1-exp(-max(z, 0))``. The
    latter is the practical exponential mapping used here because the paper
    cites the transform but does not state its explicit formula.
    """
    X = np.asarray(X, dtype=float)
    low, high = map(float, band)
    nyquist = sfreq / 2.0
    if not (0.0 < low < high < nyquist):
        raise ValueError(f"Invalid band {band} for sfreq={sfreq}")
    sos = butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, X, axis=1)
    analytic = hilbert(filtered, axis=1)
    windows = make_windows(analytic, sfreq, window_s, step_s)
    if not windows:
        raise ValueError("No windows available for multilayer connectivity")
    raw_layers: List[np.ndarray] = []
    centers = []
    for start, stop in windows:
        Z = analytic[:, start:stop]
        cross = Z @ Z.conj().T / max(1, Z.shape[1])
        power = np.maximum(np.real(np.diag(cross)), 1e-12)
        coherency = cross / np.sqrt(power[:, None] * power[None, :])
        denominator = np.sqrt(np.maximum(1.0 - np.real(coherency) ** 2, 1e-12))
        lagged = np.abs(np.imag(coherency)) / denominator
        lagged = np.nan_to_num(lagged, nan=0.0, posinf=0.0, neginf=0.0)
        lagged = np.clip((lagged + lagged.T) / 2.0, 0.0, None)
        np.fill_diagonal(lagged, 0.0)
        raw_layers.append(lagged)
        centers.append((start + stop) / (2.0 * sfreq))

    stack = np.stack(raw_layers, axis=0)
    preictal = np.asarray(centers) < seizure_onset_s
    if preictal.sum() < 2:
        n_baseline = max(2, int(np.ceil(0.2 * len(stack))))
        preictal = np.zeros(len(stack), dtype=bool)
        preictal[: min(n_baseline, len(stack))] = True
    baseline = stack[preictal]
    mean = baseline.mean(axis=0, keepdims=True)
    std = baseline.std(axis=0, keepdims=True) + 1e-12
    zscore = (stack - mean) / std
    mapped = 1.0 - np.exp(-np.clip(zscore, 0.0, 50.0))
    for layer in mapped:
        np.fill_diagonal(layer, 0.0)
    return [layer for layer in mapped]


def _target_cluster_and_performance(features: np.ndarray) -> Tuple[np.ndarray, float]:
    Z = np.asarray(features, dtype=float)
    Z = (Z - Z.mean(axis=0, keepdims=True)) / (Z.std(axis=0, keepdims=True) + 1e-12)
    n = Z.shape[0]
    if n < 3:
        return np.ones(n, dtype=int), 0.0
    linkage_matrix = linkage(pdist(Z, metric="euclidean"), method="centroid")
    labels = fcluster(linkage_matrix, t=2, criterion="maxclust")
    unique, counts = np.unique(labels, return_counts=True)
    target_label = unique[np.argmin(counts)]
    if counts.min() < max(1, int(np.ceil(0.05 * n))) and n >= 4:
        labels = fcluster(linkage_matrix, t=3, criterion="maxclust")
        unique, counts = np.unique(labels, return_counts=True)
        order = np.argsort(counts)
        target_label = unique[order[min(1, len(order) - 1)]]
    target = labels == target_label
    other = ~target
    if target.sum() < 2 or other.sum() == 0:
        return target.astype(int), 0.0
    target_points = Z[target]
    target_centroid = target_points.mean(axis=0)
    other_centroid = Z[other].mean(axis=0)
    separation = float(np.sum((target_centroid - other_centroid) ** 2))
    pairwise = pdist(target_points, metric="euclidean")
    mean_pairwise = float(pairwise.mean()) if pairwise.size else 0.0
    radius = float(np.linalg.norm(target_points - target_centroid, axis=1).max())
    compactness = mean_pairwise * radius
    performance = separation / (compactness + 1e-12) if compactness > 0 else 0.0
    return target.astype(int), float(performance)


def weighted_consensus_from_mlevc(
    mlevc_by_coupling: Dict[float, List[np.ndarray]],
    quantile_percentiles: Tuple[float, ...] = tuple(np.arange(10.0, 81.0, 10.0)),
) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted consensus EZ score from paper-defined SVD feature combinations."""
    feature_combinations = ((0, 1), (0, 2), (1, 2), (0, 1, 2), (0, 1, 2, 3))
    consensus_labels = []
    consensus_weights = []
    for _, ml_list in mlevc_by_coupling.items():
        for percentile in quantile_percentiles:
            U = mlevc_svd_features(ml_list, n_components=4, quantile_percentile=percentile)
            labels_for_case = []
            weights_for_case = []
            for indices in feature_combinations:
                valid = [idx for idx in indices if idx < U.shape[1]]
                if len(valid) < 2:
                    continue
                labels, performance = _target_cluster_and_performance(U[:, valid])
                if performance > 0 and np.isfinite(performance):
                    labels_for_case.append(labels)
                    weights_for_case.append(performance)
            if not weights_for_case:
                continue
            weights = np.asarray(weights_for_case, dtype=float)
            probability = np.average(np.stack(labels_for_case), axis=0, weights=weights)
            threshold = (len(weights) - 1) / len(weights) if len(weights) > 1 else 0.5
            consensus_labels.append((probability > threshold).astype(float))
            consensus_weights.append(float(weights.sum()))
    if not consensus_weights:
        n = next(iter(mlevc_by_coupling.values()))[0].shape[0]
        return np.zeros(n), np.zeros(n, dtype=int)
    score = np.average(
        np.stack(consensus_labels), axis=0, weights=np.asarray(consensus_weights)
    )
    score = normalize_scores(score)
    if np.unique(score).size < 3:
        target = (score >= np.max(score)).astype(int)
    else:
        labels = KMeans(n_clusters=3, random_state=0, n_init=20).fit_predict(score[:, None])
        means = {label: score[labels == label].mean() for label in np.unique(labels)}
        target = (labels == max(means, key=means.get)).astype(int)
    return score, target


def multilayer_consensus_features_for_subject(
    runs: List[np.ndarray],
    sfreq: float,
    bands: Tuple[Tuple[float, float], ...] = ((80.0, 140.0), (140.0, 200.0)),
    window_s: float = 2.5,
    step_s: float = 0.5,
    seizure_onset_s: float = 50.0,
    couplings: Tuple[float, ...] = tuple(range(1, 11)) + (15.0,),
    quantile_percentiles: Tuple[float, ...] = tuple(np.arange(10.0, 81.0, 10.0)),
) -> Dict[str, np.ndarray]:
    """Extract only the paper-native continuous consensus score and target label."""
    if not runs:
        raise ValueError("At least one run is required")
    n_channels = np.asarray(runs[0]).shape[0]
    if any(np.asarray(run).shape[0] != n_channels for run in runs):
        raise ValueError("All runs for a subject must have identical channels")
    layer_sets = []
    for run in runs:
        for band in bands:
            layer_sets.append(
                dynamic_lagged_coherence_layers(
                    run,
                    sfreq,
                    band=band,
                    window_s=window_s,
                    step_s=step_s,
                    seizure_onset_s=seizure_onset_s,
                )
            )
    by_coupling: Dict[float, List[np.ndarray]] = {}
    for coupling in couplings:
        by_coupling[float(coupling)] = [
            multilayer_evc(layers, coupling=float(coupling), max_eigs=None)
            for layers in layer_sets
        ]
    score, target = weighted_consensus_from_mlevc(
        by_coupling, quantile_percentiles=quantile_percentiles
    )
    return {"mlevc_consensus_score": score, "mlevc_target": target.astype(float)}


def multilayer_features_for_run(
    X: np.ndarray,
    sfreq: float,
    window_s: float = 0.5,
    step_s: float = 0.25,
    threshold_quantile: float = 0.8,
    coupling: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Return only the paper-native mlEVC matrix for one run.

    Subject-level EZ identification should use
    :func:`multilayer_consensus_features_for_subject` instead.
    """
    layers = dynamic_envelope_corr_layers(
        X, sfreq, window_s=window_s, step_s=step_s, threshold_quantile=threshold_quantile
    )
    ml = multilayer_evc(layers, coupling=coupling)
    return {"mlevc": ml}
