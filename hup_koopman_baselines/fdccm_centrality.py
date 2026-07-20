from __future__ import annotations

from typing import Dict, Optional

import networkx as nx
import numpy as np
from scipy.signal import hilbert, resample_poly
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def _embed(x: np.ndarray, E: int = 3, tau: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(x) - (E - 1) * tau
    if n <= E + 2:
        raise ValueError("time series too short for embedding")
    return np.column_stack([x[i * tau : i * tau + n] for i in range(E)])


def ccm_predict_score(source: np.ndarray, target: np.ndarray, E: int = 3, tau: int = 1, k: Optional[int] = None) -> float:
    """CCM-like score: reconstruct target from the delay manifold of source.

    This is a practical implementation for directional EC benchmarking. For influence source->target,
    a high score means target can be reconstructed from source's manifold in the selected frequency band.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    n0 = min(len(source), len(target))
    source = source[:n0]
    target = target[:n0]
    M = _embed(source, E=E, tau=tau)
    offset = (E - 1) * tau
    y = target[offset : offset + M.shape[0]]
    if k is None:
        k = E + 1
    k = min(k, M.shape[0] - 1)
    if k < 2 or np.nanstd(y) < 1e-12:
        return 0.0
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(M)
    dist, ind = nn.kneighbors(M, return_distance=True)
    dist = dist[:, 1:]
    ind = ind[:, 1:]
    d0 = dist[:, [0]] + 1e-12
    w = np.exp(-dist / d0)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
    pred = np.sum(w * y[ind], axis=1)
    if np.nanstd(pred) < 1e-12:
        return 0.0
    r = np.corrcoef(pred, y)[0, 1]
    if not np.isfinite(r):
        return 0.0
    return float(max(0.0, r))


def fdccm_connectivity(
    X: np.ndarray,
    sfreq: float,
    use_envelope: bool = True,
    downsample_to: float = 50.0,
    max_points: int = 1500,
    E: int = 3,
    tau: int = 1,
    show_progress: bool = False,
) -> np.ndarray:
    """Compute a directed FD-CCM-like connectivity matrix on 80--200 Hz data.

    Since the input is already band-passed, the 'frequency-domain' component is represented
    by the high-frequency analytic envelope by default. Set use_envelope=False to use the raw
    band-passed waveform.
    """
    X = np.asarray(X, dtype=float)
    if use_envelope:
        Z = np.abs(hilbert(X, axis=1))
    else:
        Z = X.copy()
    if downsample_to and downsample_to < sfreq:
        # Use integer approximation for stable resampling.
        q = int(round(sfreq / downsample_to))
        q = max(q, 1)
        Z = resample_poly(Z, up=1, down=q, axis=1)
    if Z.shape[1] > max_points:
        idx = np.linspace(0, Z.shape[1] - 1, max_points).astype(int)
        Z = Z[:, idx]
    Z = (Z - Z.mean(axis=1, keepdims=True)) / (Z.std(axis=1, keepdims=True) + 1e-12)

    n_ch = Z.shape[0]
    A = np.zeros((n_ch, n_ch), dtype=float)
    pairs = [(i, j) for i in range(n_ch) for j in range(n_ch) if i != j]
    iterator = pairs
    if show_progress:
        iterator = tqdm(pairs, desc="FD-CCM pairs", leave=False)
    for i, j in iterator:
        A[i, j] = ccm_predict_score(Z[i], Z[j], E=E, tau=tau)
    np.fill_diagonal(A, 0.0)
    return A


def _mapping_to_array(values: Dict[int, float], n: int) -> np.ndarray:
    return np.asarray([values.get(i, 0.0) for i in range(n)], dtype=float)


def _weighted_local_efficiency(G: nx.DiGraph) -> np.ndarray:
    """Directed, weighted local efficiency using reciprocal shortest paths.

    For each node, efficiency is evaluated on the subgraph induced by the
    union of its predecessors and successors. Edge ``distance`` must be the
    reciprocal of connectivity strength.
    """
    result = np.zeros(G.number_of_nodes(), dtype=float)
    for node in G.nodes:
        neighbors = set(G.predecessors(node)) | set(G.successors(node))
        m = len(neighbors)
        if m < 2:
            continue
        subgraph = G.subgraph(neighbors)
        inverse_distance_sum = 0.0
        for source, lengths in nx.all_pairs_dijkstra_path_length(subgraph, weight="distance"):
            for target, distance in lengths.items():
                if source != target and distance > 0:
                    inverse_distance_sum += 1.0 / distance
        result[node] = inverse_distance_sum / (m * (m - 1))
    return result


def centrality_features_from_adjacency(A: np.ndarray, threshold_quantile: float = 0.9) -> Dict[str, np.ndarray]:
    """Return the ten directed-graph centralities listed in Balaji & Parhi (2024).

    ``threshold_quantile=0.9`` retains the strongest 10% of non-zero directed
    edges, matching the FD-CCM sparsity highlighted in the paper.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    thr = np.quantile(A[A > 0], threshold_quantile) if np.any(A > 0) else np.inf
    B = np.where(A >= thr, A, 0.0)
    G = nx.from_numpy_array(B, create_using=nx.DiGraph)
    for u, v, d in G.edges(data=True):
        # NetworkX shortest path centralities interpret larger weight as larger distance.
        d["distance"] = 1.0 / (float(d.get("weight", 0.0)) + 1e-12)

    # Degree/strength must be evaluated on the sparsified graph used by all
    # other centralities, rather than on the original dense matrix.
    out: Dict[str, np.ndarray] = {
        "fdccm_indegree": B.sum(axis=0),
        "fdccm_outdegree": B.sum(axis=1),
    }
    try:
        out["fdccm_in_closeness"] = _mapping_to_array(
            nx.closeness_centrality(G, distance="distance"), n
        )
        out["fdccm_out_closeness"] = _mapping_to_array(
            nx.closeness_centrality(G.reverse(copy=False), distance="distance"), n
        )
    except Exception:
        out["fdccm_in_closeness"] = np.zeros(n)
        out["fdccm_out_closeness"] = np.zeros(n)
    try:
        out["fdccm_local_clustering"] = _mapping_to_array(nx.clustering(G, weight="weight"), n)
    except Exception:
        out["fdccm_local_clustering"] = np.zeros(n)
    try:
        out["fdccm_local_efficiency"] = _weighted_local_efficiency(G)
    except Exception:
        out["fdccm_local_efficiency"] = np.zeros(n)
    try:
        out["fdccm_betweenness"] = _mapping_to_array(
            nx.betweenness_centrality(G, weight="distance"), n
        )
    except Exception:
        out["fdccm_betweenness"] = np.zeros(n)
    try:
        out["fdccm_pagerank"] = _mapping_to_array(nx.pagerank(G, weight="weight"), n)
    except Exception:
        out["fdccm_pagerank"] = np.zeros(n)
    try:
        hubs, authorities = nx.hits(G, max_iter=1000, normalized=True)
        out["fdccm_hub"] = _mapping_to_array(hubs, n)
        out["fdccm_authority"] = _mapping_to_array(authorities, n)
    except Exception:
        out["fdccm_hub"] = np.zeros(n)
        out["fdccm_authority"] = np.zeros(n)
    for k in list(out.keys()):
        out[k] = np.nan_to_num(out[k], nan=0.0, posinf=0.0, neginf=0.0)
    return out


def fdccm_centrality_features(
    X: np.ndarray,
    sfreq: float,
    threshold_quantile: float = 0.9,
    **kwargs,
) -> Dict[str, np.ndarray]:
    A = fdccm_connectivity(X, sfreq, **kwargs)
    return centrality_features_from_adjacency(A, threshold_quantile=threshold_quantile)
