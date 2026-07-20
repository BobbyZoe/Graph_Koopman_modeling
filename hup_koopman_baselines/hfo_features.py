from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import hilbert
from tqdm import tqdm

from .data import normalize_scores


@dataclass
class HFOEvent:
    start: int
    stop: int
    peak_amp: float
    mean_amp: float
    power: float
    duration_s: float


def _merge_segments(segments: List[Tuple[int, int]], gap: int) -> List[Tuple[int, int]]:
    if not segments:
        return []
    merged = [segments[0]]
    for s, e in segments[1:]:
        ps, pe = merged[-1]
        if s - pe <= gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def detect_hfo_events_channel(
    x: np.ndarray,
    sfreq: float,
    threshold_std: float = 5.0,
    min_duration_ms: float = 20.0,
    max_duration_ms: float = 200.0,
    merge_ms: float = 10.0,
    min_cycles_freq: float = 200.0,
) -> List[HFOEvent]:
    """Detect HFO-like events in a single already 80--200 Hz filtered channel.

    The detector uses the Hilbert envelope and a robust median/MAD threshold.
    For strict ripple definitions, increase min_duration_ms to 50 ms.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.nanmedian(x)
    env = np.abs(hilbert(x))
    med = np.nanmedian(env)
    mad = np.nanmedian(np.abs(env - med))
    thr = med + threshold_std * 1.4826 * (mad + 1e-12)

    above = env > thr
    idx = np.flatnonzero(above)
    if idx.size == 0:
        return []

    splits = np.where(np.diff(idx) > 1)[0] + 1
    segments = [(int(s[0]), int(s[-1]) + 1) for s in np.split(idx, splits)]
    segments = _merge_segments(segments, int(round(merge_ms / 1000 * sfreq)))

    min_len = max(int(round(min_duration_ms / 1000 * sfreq)), int(round(4 / min_cycles_freq * sfreq)))
    max_len = int(round(max_duration_ms / 1000 * sfreq))
    events: List[HFOEvent] = []
    for s, e in segments:
        if e - s < min_len or e - s > max_len:
            continue
        seg_x = x[s:e]
        seg_env = env[s:e]
        events.append(
            HFOEvent(
                start=s,
                stop=e,
                peak_amp=float(np.nanmax(seg_env)),
                mean_amp=float(np.nanmean(seg_env)),
                power=float(np.nanmean(seg_x ** 2)),
                duration_s=float((e - s) / sfreq),
            )
        )
    return events


def hfo_rate_features_for_run(
    X: np.ndarray,
    sfreq: float,
    threshold_std: float = 5.0,
    min_duration_ms: float = 20.0,
    max_duration_ms: float = 200.0,
    show_progress: bool = False,
) -> Dict[str, np.ndarray]:
    """Return only HFO event rate per minute for every channel."""
    X = np.asarray(X, dtype=float)
    n_ch, n_samples = X.shape
    duration_min = n_samples / sfreq / 60.0
    rate = np.zeros(n_ch, dtype=float)
    iterator = range(n_ch)
    if show_progress:
        iterator = tqdm(iterator, desc="HFO rate", leave=False)
    for ch in iterator:
        events = detect_hfo_events_channel(
            X[ch],
            sfreq,
            threshold_std=threshold_std,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
        )
        rate[ch] = len(events) / max(duration_min, 1e-12)
    return {"hfo_rate_per_min": rate}


def hfo_features_for_run(
    X: np.ndarray,
    sfreq: float,
    threshold_std: float = 5.0,
    min_duration_ms: float = 20.0,
    max_duration_ms: float = 200.0,
    show_progress: bool = False,
) -> Dict[str, np.ndarray]:
    """Return channel-level HFO rate/amplitude/power features.

    Output keys are 1D arrays of length n_channels.
    """
    X = np.asarray(X, dtype=float)
    n_ch, n_samples = X.shape
    duration_min = n_samples / sfreq / 60.0
    out = {
        "hfo_rate_per_min": np.zeros(n_ch),
        "hfo_peak_amp_mean": np.zeros(n_ch),
        "hfo_peak_amp_median": np.zeros(n_ch),
        "hfo_event_power_mean": np.zeros(n_ch),
        "hfo_event_duration_mean": np.zeros(n_ch),
        "band_power_80_200": np.nanmean(X ** 2, axis=1),
        "hfo_count": np.zeros(n_ch),
    }
    iterator = range(n_ch)
    if show_progress:
        iterator = tqdm(iterator, desc="HFO features", leave=False)
    for ch in iterator:
        events = detect_hfo_events_channel(
            X[ch],
            sfreq,
            threshold_std=threshold_std,
            min_duration_ms=min_duration_ms,
            max_duration_ms=max_duration_ms,
        )
        out["hfo_count"][ch] = len(events)
        out["hfo_rate_per_min"][ch] = len(events) / max(duration_min, 1e-12)
        if events:
            peaks = np.asarray([e.peak_amp for e in events])
            powers = np.asarray([e.power for e in events])
            durs = np.asarray([e.duration_s for e in events])
            out["hfo_peak_amp_mean"][ch] = float(np.mean(peaks))
            out["hfo_peak_amp_median"][ch] = float(np.median(peaks))
            out["hfo_event_power_mean"][ch] = float(np.mean(powers))
            out["hfo_event_duration_mean"][ch] = float(np.mean(durs))
    # normalized versions are useful for unsupervised ranking
    for k in list(out.keys()):
        if k != "hfo_count":
            out[f"{k}_norm"] = normalize_scores(out[k])
    return out
