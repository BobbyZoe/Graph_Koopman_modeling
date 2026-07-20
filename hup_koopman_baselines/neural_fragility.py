from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

from .data import make_windows, normalize_scores


def estimate_A_ridge(Xw: np.ndarray, alpha: float = 1e-4, stabilize: bool = True, radius: float = 0.99) -> np.ndarray:
    """Fit x(t+1)=A x(t) by ridge regression for one window [channels, samples].

    The ridge matrix is symmetric positive definite, so a direct linear solve is
    both more appropriate and more reliable than an SVD-based pseudoinverse.
    A small adaptive diagonal jitter is used only if the LAPACK solve fails.
    """
    Xw = np.asarray(Xw, dtype=np.float64)
    if Xw.ndim != 2 or Xw.shape[1] < 2:
        raise ValueError(f"Xw must have shape [channels, >=2 samples], got {Xw.shape}")
    if not np.isfinite(Xw).all():
        raise ValueError("Xw contains NaN or Inf")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    X0 = Xw[:, :-1]
    X1 = Xw[:, 1:]
    C = X0 @ X0.T
    cross = X1 @ X0.T
    C = 0.5 * (C + C.T)
    eye = np.eye(C.shape[0], dtype=C.dtype)
    ridge = max(float(alpha), np.finfo(C.dtype).eps)
    system = C + ridge * eye
    try:
        A = np.linalg.solve(system, cross.T).T
    except np.linalg.LinAlgError:
        scale = max(float(np.max(np.abs(np.diag(C)))), 1.0)
        for multiplier in (1e-10, 1e-8, 1e-6, 1e-4):
            try:
                A = np.linalg.solve(system + multiplier * scale * eye, cross.T).T
                break
            except np.linalg.LinAlgError:
                continue
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(system)
            cutoff = np.finfo(C.dtype).eps * max(system.shape) * max(
                float(np.max(np.abs(eigenvalues))), 1.0
            )
            inverse_eigenvalues = np.where(eigenvalues > cutoff, 1.0 / eigenvalues, 0.0)
            inverse = (eigenvectors * inverse_eigenvalues) @ eigenvectors.T
            A = cross @ inverse
    if stabilize:
        eig = np.linalg.eigvals(A)
        rho = np.max(np.abs(eig)) if eig.size else 0.0
        if rho > radius:
            A = A * (radius / (rho + 1e-12))
    return A


def structured_perturbation_norms_from_A(
    A: np.ndarray,
    target_radius: float = 1.0,
    n_angles: int = 64,
) -> np.ndarray:
    """Minimum real column-perturbation norm over a grid on the unit circle.

    For a perturbation restricted to column ``i``, the matrix determinant lemma
    gives the boundary constraint

    ``e_i.T @ inv(lambda * I - A) @ delta_i = 1``.

    The minimum-norm real ``delta_i`` is solved for every channel and every
    candidate ``lambda`` on the stability boundary. This follows the structured
    perturbation definition in Li et al. (2021), with angular discretization used
    for practical computation.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"A must be square, got {A.shape}")
    if n_angles < 3:
        raise ValueError("n_angles must be at least 3")
    identity = np.eye(n)
    best = np.full(n, np.inf, dtype=float)
    # A is real, so conjugate points give identical real perturbation norms.
    angles = np.linspace(0.0, np.pi, n_angles, endpoint=True)
    for theta in angles:
        lam = target_radius * np.exp(1j * theta)
        try:
            resolvent = np.linalg.solve(lam * identity - A, identity)
        except np.linalg.LinAlgError:
            resolvent = np.linalg.pinv(lam * identity - A)
        real = resolvent.real
        imag = resolvent.imag
        aa = np.einsum("ij,ij->i", real, real)
        bb = np.einsum("ij,ij->i", imag, imag)
        ab = np.einsum("ij,ij->i", real, imag)
        determinant = aa * bb - ab * ab
        energy = np.full(n, np.inf, dtype=float)
        real_only = bb <= 1e-12
        energy[real_only] = 1.0 / np.sqrt(np.maximum(aa[real_only], 1e-24))
        full_rank = (~real_only) & (determinant > 1e-18)
        energy[full_rank] = np.sqrt(bb[full_rank] / determinant[full_rank])
        best = np.minimum(best, energy)
    return np.nan_to_num(best, nan=np.inf, posinf=np.inf, neginf=np.inf)


def fragility_scores_from_A(
    A: np.ndarray,
    target_radius: float = 1.0,
    n_angles: int = 64,
) -> np.ndarray:
    """Return high-is-fragile scores from minimum structured perturbation norms."""
    energy = structured_perturbation_norms_from_A(
        A, target_radius=target_radius, n_angles=n_angles
    )
    finite = np.isfinite(energy) & (energy >= 0)
    score = np.zeros_like(energy)
    score[finite] = 1.0 / (energy[finite] + 1e-12)
    return normalize_scores(score)


def fragility_heatmap(
    X: np.ndarray,
    sfreq: float,
    window_ms: float = 250.0,
    step_ms: float = 125.0,
    alpha: float = 1e-4,
    n_angles: int = 64,
    show_progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must have shape [channels, samples], got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or Inf")
    centered = X - X.mean(axis=1, keepdims=True)
    # Scaling before the RMS calculation prevents overflow while remaining
    # algebraically equivalent to channel-wise standardization.
    max_abs = np.max(np.abs(centered), axis=1, keepdims=True)
    safe_max_abs = np.where(max_abs > 0.0, max_abs, 1.0)
    scaled = centered / safe_max_abs
    rms = np.sqrt(np.mean(scaled * scaled, axis=1, keepdims=True))
    X = scaled / np.where(rms > 1e-12, rms, 1.0)
    wins = make_windows(X, sfreq, window_ms / 1000.0, step_ms / 1000.0)
    H = np.zeros((X.shape[0], len(wins)), dtype=float)
    iterator = enumerate(wins)
    if show_progress:
        iterator = tqdm(iterator, total=len(wins), desc="Fragility", leave=False)
    times = []
    for k, (s, e) in iterator:
        A = estimate_A_ridge(X[:, s:e], alpha=alpha)
        H[:, k] = fragility_scores_from_A(A, n_angles=n_angles)
        times.append((s + e) / 2.0 / sfreq)
    return H, np.asarray(times)


def neural_fragility_features(
    X: np.ndarray,
    sfreq: float,
    window_ms: float = 250.0,
    step_ms: float = 125.0,
    alpha: float = 1e-4,
    n_angles: int = 64,
    show_progress: bool = False,
) -> Dict[str, np.ndarray]:
    """Return only the paper-native channel-by-time fragility heatmap."""
    H, _ = fragility_heatmap(
        X,
        sfreq,
        window_ms=window_ms,
        step_ms=step_ms,
        alpha=alpha,
        n_angles=n_angles,
        show_progress=show_progress,
    )
    return {"neural_fragility": H}
