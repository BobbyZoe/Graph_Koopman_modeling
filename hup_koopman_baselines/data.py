from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class RunData:
    subject: str
    run: str
    X: np.ndarray  # channels x samples
    y: np.ndarray  # channels, binary SOZ labels
    sfreq: float
    ch_names: List[str]
    path: str = ""
    meta: Optional[Dict[str, Any]] = None

    def validate(self) -> "RunData":
        self.X = np.asarray(self.X, dtype=np.float64)
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2D [channels, samples], got {self.X.shape}")
        self.y = np.asarray(self.y).astype(int).reshape(-1)
        if self.y.shape[0] != self.X.shape[0]:
            raise ValueError(f"y length {len(self.y)} does not match channels {self.X.shape[0]}")
        if not self.ch_names:
            self.ch_names = [f"ch{i:03d}" for i in range(self.X.shape[0])]
        return self


def infer_subject_run(path: Path) -> Tuple[str, str]:
    name = path.stem
    sub_match = re.search(r"(HUP\d+|sub-[A-Za-z0-9]+|pt\d+|patient\d+)", name, flags=re.I)
    run_match = re.search(r"(run[-_]?\d+|sz[-_]?\d+|seizure[-_]?\d+)", name, flags=re.I)
    subject = sub_match.group(1) if sub_match else name.split("_")[0]
    run = run_match.group(1).replace("_", "-") if run_match else "run-unknown"
    return subject, run


def _read_any(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            return obj
        raise ValueError(f"Pickle must contain a dict, got {type(obj)} from {path}")
    if path.suffix.lower() == ".npz":
        obj = np.load(path, allow_pickle=True)
        return {k: obj[k].item() if obj[k].shape == () else obj[k] for k in obj.files}
    if path.suffix.lower() == ".npy":
        return {"data": np.load(path, allow_pickle=True)}
    raise ValueError(f"Unsupported file type: {path}")


def _first_key(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return default


def load_run(path: str | Path, default_sfreq: Optional[float] = None) -> RunData:
    path = Path(path)
    d = _read_any(path)

    X = _first_key(d, ["data", "X", "x", "signals", "ieeg", "eeg"])
    if X is None:
        raise KeyError(f"No data key found in {path}; expected one of data/X/signals/ieeg/eeg")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim > 2:
        X = np.squeeze(X)
    if X.shape[0] > X.shape[1]:
        # Most iEEG arrays are channels x samples. If samples x channels, transpose.
        # This heuristic assumes n_samples >> n_channels.
        X = X.T

    sfreq = _first_key(d, ["sfreq", "fs", "sampling_rate", "sample_rate"], default_sfreq)
    if sfreq is None:
        raise KeyError(f"No sfreq found in {path}; pass --sfreq or include sfreq in file")
    sfreq = float(sfreq)

    y = _first_key(d, ["y", "labels", "soz_labels", "soz_mask"])
    if y is None:
        soz_index = _first_key(d, ["soz_index", "soz_indices", "soz_ch_idx", "SOZ_index", "soz"])
        y = np.zeros(X.shape[0], dtype=int)
        if soz_index is not None:
            idx = np.asarray(soz_index).astype(int).reshape(-1)
            idx = idx[(idx >= 0) & (idx < X.shape[0])]
            y[idx] = 1
    y = np.asarray(y).astype(int).reshape(-1)

    ch_names = _first_key(d, ["ch_names", "channels", "channel_names"], None)
    if ch_names is None:
        ch_names = [f"ch{i:03d}" for i in range(X.shape[0])]
    else:
        ch_names = [str(c) for c in list(ch_names)]

    inferred_subject, inferred_run = infer_subject_run(path)
    subject = str(_first_key(d, ["subject", "sub", "patient"], inferred_subject))
    run = str(_first_key(d, ["run", "seizure", "session"], inferred_run))

    return RunData(subject=subject, run=run, X=X, y=y, sfreq=sfreq, ch_names=ch_names, path=str(path), meta=d).validate()


def load_dataset(data_dir: str | Path, pattern: str = "**/*.pkl", default_sfreq: Optional[float] = None) -> List[RunData]:
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob(pattern))
    runs = []
    for p in paths:
        try:
            runs.append(load_run(p, default_sfreq=default_sfreq))
        except Exception as e:
            print(f"[WARN] skipped {p}: {e}")
    if not runs:
        raise RuntimeError(f"No valid runs found in {data_dir} with pattern {pattern}")
    return runs


def robust_zscore(x: np.ndarray, axis: Optional[int] = None, eps: float = 1e-12) -> np.ndarray:
    med = np.nanmedian(x, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(x - med), axis=axis, keepdims=True)
    return (x - med) / (1.4826 * mad + eps)


def make_windows(X: np.ndarray, sfreq: float, window_s: float, step_s: float) -> List[Tuple[int, int]]:
    n = X.shape[-1]
    w = max(1, int(round(window_s * sfreq)))
    s = max(1, int(round(step_s * sfreq)))
    return [(i, i + w) for i in range(0, max(1, n - w + 1), s) if i + w <= n]


def normalize_scores(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < eps:
        return np.zeros_like(v, dtype=float)
    return (v - lo) / (hi - lo + eps)


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, score))


def score_metrics(y: np.ndarray, score: np.ndarray, threshold: Optional[float] = None) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    if threshold is None:
        # Predict the same number of positive channels as in y for fair ranking-based comparison.
        k = int(max(1, y.sum()))
        idx = np.argsort(score)[::-1][:k]
        pred = np.zeros_like(y)
        pred[idx] = 1
    else:
        pred = (score >= threshold).astype(int)
    out: Dict[str, float] = {
        "roc_auc": safe_auc(y, score),
        "auprc": float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else np.nan,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else np.nan,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    for k in [1, 3, 5, int(max(1, y.sum()))]:
        kk = min(k, len(y))
        pred_idx = np.argsort(score)[::-1][:kk]
        out[f"hit_at_{kk}"] = float(y[pred_idx].max() > 0)
        out[f"recall_at_{kk}"] = float(y[pred_idx].sum() / max(1, y.sum()))
    return out


def features_to_frame(run: RunData, features: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = {
        "subject": run.subject,
        "run": run.run,
        "channel": np.arange(run.X.shape[0]),
        "ch_name": run.ch_names,
        "y": run.y.astype(int),
    }
    for k, v in features.items():
        v = np.asarray(v)
        if v.ndim == 1:
            rows[k] = v
        else:
            for j in range(v.shape[1]):
                rows[f"{k}_{j}"] = v[:, j]
    return pd.DataFrame(rows)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
