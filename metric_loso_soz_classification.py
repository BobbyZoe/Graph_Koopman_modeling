"""
LOSO SOZ localization using Koopman graph metric features.

Each channel is one sample. For every LOSO fold, all runs from the held-out
subject are tested and all runs from the remaining subjects are used for
training. Subject-level predictions are the average channel probabilities
across that subject's runs.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler


DEFAULT_BASE_PATH = Path(
    r"\\10.20.37.22\dataset0\weiting\graph_koopman\final_version\koopman_data\graph_all_weight_4metrics_results"
)
DEFAULT_PREPROCESSED_PATH = Path(
    r"\\10.20.37.22\dataset0\weiting\graph_koopman\preprocessed_HUPdata_500resample_80_200_100s"
)
DEFAULT_OUTPUT_DIR = Path(
    r"\\10.20.37.22\dataset0\weiting\graph_koopman\final_version\koopman_data\metric_loso_results"
)
# examle path, please change to your own path 

METRIC_NAMES = [
    "node_strength",
    "degree_std",
    "eigenvector_centrality",
    "global_efficiency",
]


@dataclass
class RunData:
    subject: str
    run_index: int
    features: np.ndarray
    labels: np.ndarray
    soz_channels: list[int]


def normalize_data(data: np.ndarray) -> np.ndarray:
    """Min-max normalize each graph metric over channels, windows, and modes."""
    n_channels, n_windows, n_eig, n_metrics = data.shape
    normalized = np.array(data, dtype=np.float32, copy=True)

    for metric_idx in range(n_metrics):
        metric_data = normalized[:, :, :, metric_idx]
        flat_data = metric_data.reshape(-1)
        valid_mask = np.isfinite(flat_data)
        if not np.any(valid_mask):
            normalized[:, :, :, metric_idx] = 0.0
            continue

        valid_data = flat_data[valid_mask]
        data_min = float(np.min(valid_data))
        data_max = float(np.max(valid_data))
        data_range = data_max - data_min
        if data_range > 0:
            scaled = (flat_data - data_min) / data_range
            scaled = np.clip(scaled, 0.0, 1.0)
            normalized[:, :, :, metric_idx] = scaled.reshape(n_channels, n_windows, n_eig)
        else:
            normalized[:, :, :, metric_idx] = 0.5

    return normalized


def process_data(raw_data: np.ndarray) -> np.ndarray:
    """
    Convert raw graph metric data to (channels, windows, eig, metrics).

    Current files are stored as (windows, channels, eig, metrics), while the
    notebook loader returns the transposed normalized array.
    """
    if not isinstance(raw_data, np.ndarray) or raw_data.ndim != 4:
        raise ValueError(f"Expected 4D ndarray, got {type(raw_data)} with shape {getattr(raw_data, 'shape', None)}")

    if raw_data.shape[0] == 240 and raw_data.shape[2:] == (10, 4):
        data = np.transpose(raw_data, (1, 0, 2, 3))
    elif raw_data.shape[1] == 240 and raw_data.shape[2:] == (10, 4):
        data = raw_data
    else:
        raise ValueError(f"Unexpected graph metric shape: {raw_data.shape}")

    return normalize_data(data)


def aggregate_time_bins(data: np.ndarray, time_bins: int, top_eigs: int) -> np.ndarray:
    """Average windows into time bins and flatten to channel-level features."""
    if time_bins < 1:
        raise ValueError("time_bins must be >= 1")
    if top_eigs < 1:
        raise ValueError("top_eigs must be >= 1")

    n_channels, n_windows, n_eig, n_metrics = data.shape
    if top_eigs > n_eig:
        raise ValueError(f"top_eigs={top_eigs} exceeds available eig dimension {n_eig}")

    data = data[:, :, :top_eigs, :]
    bin_indices = np.array_split(np.arange(n_windows), time_bins)
    aggregated = np.zeros((n_channels, time_bins, top_eigs, n_metrics), dtype=np.float32)

    for bin_idx, indices in enumerate(bin_indices):
        aggregated[:, bin_idx, :, :] = np.nanmean(data[:, indices, :, :], axis=1)

    features = aggregated.reshape(n_channels, -1)
    return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)


def parse_run_index(path: Path) -> int | None:
    match = re.search(r"run(\d+)", path.stem)
    return int(match.group(1)) if match else None


def discover_subjects(base_path: Path, preprocessed_path: Path, requested_subjects: list[str] | None) -> list[str]:
    if requested_subjects:
        return sorted({subject.replace("subject_", "") for subject in requested_subjects})

    subjects = []
    for subject_dir in sorted(base_path.glob("subject_HUP*")):
        subject = subject_dir.name.replace("subject_", "")
        if (preprocessed_path / subject / "soz_info.pkl").exists():
            subjects.append(subject)
    return subjects


def load_soz_channels(subject: str, preprocessed_path: Path) -> list[int]:
    soz_path = preprocessed_path / subject / "soz_info.pkl"
    with soz_path.open("rb") as f:
        soz_info = pickle.load(f)
    return [int(idx) for idx in soz_info.get("soz_channel_indices", [])]


def load_subject_runs(
    subject: str,
    base_path: Path,
    preprocessed_path: Path,
    time_bins: int,
    top_eigs: int,
) -> list[RunData]:
    subject_dir = base_path / f"subject_{subject}"
    run_files = sorted(subject_dir.glob("improved_features_all_weights_run*.pkl"), key=lambda p: parse_run_index(p) or -1)
    if not run_files:
        return []

    soz_channels = load_soz_channels(subject, preprocessed_path)
    runs = []
    for run_file in run_files:
        run_index = parse_run_index(run_file)
        if run_index is None:
            continue

        with run_file.open("rb") as f:
            raw_data = pickle.load(f)
        normalized_data = process_data(raw_data)
        features = aggregate_time_bins(normalized_data, time_bins, top_eigs)

        labels = np.zeros(features.shape[0], dtype=np.int64)
        valid_soz = [idx for idx in soz_channels if 0 <= idx < features.shape[0]]
        labels[valid_soz] = 1

        runs.append(
            RunData(
                subject=subject,
                run_index=run_index,
                features=features,
                labels=labels,
                soz_channels=valid_soz,
            )
        )

    return runs


def balance_training_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    method: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "none":
        return x_train, y_train

    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return x_train, y_train

    if method == "undersample":
        n = min(len(pos_idx), len(neg_idx))
        selected_pos = rng.choice(pos_idx, size=n, replace=False)
        selected_neg = rng.choice(neg_idx, size=n, replace=False)
    elif method == "oversample":
        n = max(len(pos_idx), len(neg_idx))
        selected_pos = rng.choice(pos_idx, size=n, replace=len(pos_idx) < n)
        selected_neg = rng.choice(neg_idx, size=n, replace=len(neg_idx) < n)
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    selected = np.concatenate([selected_pos, selected_neg])
    rng.shuffle(selected)
    return x_train[selected], y_train[selected]


def positive_class_probability(model: RandomForestClassifier, x_test: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(x_test)
    if 1 in model.classes_:
        return proba[:, int(np.where(model.classes_ == 1)[0][0])]
    return np.zeros(x_test.shape[0], dtype=np.float32)


def calculate_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (y_score >= threshold).astype(np.int64)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()

    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    sen = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5

    return {
        "AUC": float(auc),
        "ACC": float(acc),
        "SEN": float(sen),
        "SPEC": float(spec),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def feature_names(time_bins: int, top_eigs: int) -> list[dict[str, int | str]]:
    names = []
    for bin_idx in range(time_bins):
        for eig_idx in range(top_eigs):
            for metric_idx, metric_name in enumerate(METRIC_NAMES):
                names.append(
                    {
                        "feature_index": len(names),
                        "time_bin": bin_idx,
                        "eig_index": eig_idx,
                        "metric_index": metric_idx,
                        "metric_name": metric_name,
                    }
                )
    return names


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_excel(path: Path, sheets: dict[str, list[dict]]) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("pandas is not installed; skipped Excel output.")
        return

    try:
        with pd.ExcelWriter(path) as writer:
            for sheet_name, rows in sheets.items():
                pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name[:31], index=False)
    except ImportError as exc:
        print(f"Excel writer dependency is missing ({exc}); CSV files were still saved.")
    except ModuleNotFoundError as exc:
        print(f"Excel writer dependency is missing ({exc}); CSV files were still saved.")


def summarize_subject_metrics(subject_rows: list[dict]) -> list[dict]:
    summary = []
    for metric in ["AUC", "ACC", "SEN", "SPEC"]:
        values = np.array([float(row[metric]) for row in subject_rows], dtype=float)
        summary.append(
            {
                "Metric": metric,
                "Mean": float(np.mean(values)) if len(values) else np.nan,
                "Std": float(np.std(values)) if len(values) else np.nan,
                "Min": float(np.min(values)) if len(values) else np.nan,
                "Max": float(np.max(values)) if len(values) else np.nan,
                "N_Subjects": int(len(values)),
            }
        )
    return summary


def run_loso(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    subjects = discover_subjects(args.base_path, args.preprocessed_path, args.subjects)
    if args.max_subjects:
        subjects = subjects[: args.max_subjects]

    all_data: dict[str, list[RunData]] = {}
    skipped = []
    for subject in subjects:
        try:
            runs = load_subject_runs(
                subject,
                args.base_path,
                args.preprocessed_path,
                args.time_bins,
                args.top_eigs,
            )
            if runs and np.any(runs[0].labels == 1):
                all_data[subject] = runs
            else:
                skipped.append({"subject": subject, "reason": "no runs or no valid SOZ labels"})
        except Exception as exc:
            skipped.append({"subject": subject, "reason": str(exc)})

    valid_subjects = sorted(all_data)
    if len(valid_subjects) < 2:
        raise RuntimeError(f"Need at least 2 valid subjects for LOSO, got {len(valid_subjects)}")

    subject_rows = []
    run_rows = []
    prediction_rows = []
    importance_rows = []

    for fold_idx, test_subject in enumerate(valid_subjects, start=1):
        train_subjects = [subject for subject in valid_subjects if subject != test_subject]
        x_train = np.concatenate([run.features for subject in train_subjects for run in all_data[subject]], axis=0)
        y_train = np.concatenate([run.labels for subject in train_subjects for run in all_data[subject]], axis=0)

        if len(np.unique(y_train)) < 2:
            skipped.append({"subject": test_subject, "reason": "training data has only one class"})
            continue

        x_balanced, y_balanced = balance_training_data(x_train, y_train, args.sampling, rng)
        scaler = StandardScaler()
        x_balanced = scaler.fit_transform(x_balanced)

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )
        model.fit(x_balanced, y_balanced)

        run_scores = []
        y_true_subject = None
        for run in all_data[test_subject]:
            x_test = scaler.transform(run.features)
            y_score = positive_class_probability(model, x_test)
            run_scores.append(y_score)

            metrics = calculate_metrics(run.labels, y_score, args.threshold)
            run_rows.append(
                {
                    "fold": fold_idx,
                    "test_subject": test_subject,
                    "run_index": run.run_index,
                    "n_channels": int(len(run.labels)),
                    "n_soz": int(np.sum(run.labels)),
                    **metrics,
                }
            )

            if y_true_subject is None:
                y_true_subject = run.labels
            elif len(y_true_subject) != len(run.labels) or not np.array_equal(y_true_subject, run.labels):
                skipped.append(
                    {
                        "subject": test_subject,
                        "reason": f"run {run.run_index} labels/channel count differ; subject average may be invalid",
                    }
                )

        if not run_scores or y_true_subject is None:
            continue

        y_score_subject = np.mean(run_scores, axis=0)
        subject_metrics = calculate_metrics(y_true_subject, y_score_subject, args.threshold)
        subject_rows.append(
            {
                "fold": fold_idx,
                "test_subject": test_subject,
                "n_train_subjects": len(train_subjects),
                "n_runs": len(run_scores),
                "n_channels": int(len(y_true_subject)),
                "n_soz": int(np.sum(y_true_subject)),
                "train_samples_raw": int(len(y_train)),
                "train_soz_raw": int(np.sum(y_train)),
                "train_samples_after_sampling": int(len(y_balanced)),
                "train_soz_after_sampling": int(np.sum(y_balanced)),
                **subject_metrics,
            }
        )

        y_pred_subject = (y_score_subject >= args.threshold).astype(np.int64)
        for channel_idx, (label, pred, score) in enumerate(zip(y_true_subject, y_pred_subject, y_score_subject)):
            prediction_rows.append(
                {
                    "subject": test_subject,
                    "channel_index": channel_idx,
                    "y_true": int(label),
                    "y_pred": int(pred),
                    "y_score": float(score),
                }
            )

        for feature_info, importance in zip(feature_names(args.time_bins, args.top_eigs), model.feature_importances_):
            importance_rows.append(
                {
                    "fold": fold_idx,
                    "test_subject": test_subject,
                    **feature_info,
                    "importance": float(importance),
                }
            )

        print(
            f"[{fold_idx:02d}/{len(valid_subjects)}] {test_subject}: "
            f"AUC={subject_metrics['AUC']:.4f}, ACC={subject_metrics['ACC']:.4f}, "
            f"SEN={subject_metrics['SEN']:.4f}, SPEC={subject_metrics['SPEC']:.4f}"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = summarize_subject_metrics(subject_rows)
    write_csv(output_dir / "subject_metrics.csv", subject_rows)
    write_csv(output_dir / "run_metrics.csv", run_rows)
    write_csv(output_dir / "channel_predictions.csv", prediction_rows)
    write_csv(output_dir / "summary_metrics.csv", summary_rows)
    write_csv(output_dir / "skipped_subjects.csv", skipped)

    if importance_rows:
        write_csv(output_dir / "feature_importance_by_fold.csv", importance_rows)
        aggregate_importance = []
        for feature_info in feature_names(args.time_bins, args.top_eigs):
            idx = feature_info["feature_index"]
            vals = [row["importance"] for row in importance_rows if row["feature_index"] == idx]
            aggregate_importance.append(
                {
                    **feature_info,
                    "mean_importance": float(np.mean(vals)),
                    "std_importance": float(np.std(vals)),
                }
            )
        aggregate_importance.sort(key=lambda row: row["mean_importance"], reverse=True)
        write_csv(output_dir / "feature_importance_mean.csv", aggregate_importance)

    if not args.no_excel:
        write_excel(
            output_dir / "metric_loso_soz_results.xlsx",
            {
                "summary": summary_rows,
                "subject_metrics": subject_rows,
                "run_metrics": run_rows,
                "channel_predictions": prediction_rows,
                "skipped": skipped,
            },
        )

    print("\nSummary:")
    for row in summary_rows:
        print(f"{row['Metric']}: {row['Mean']:.4f} +/- {row['Std']:.4f} (n={row['N_Subjects']})")
    print(f"\nSaved results to: {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LOSO SOZ classification on graph metric features.")
    parser.add_argument("--base-path", type=Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--preprocessed-path", type=Path, default=DEFAULT_PREPROCESSED_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--time-bins", type=int, default=6)
    parser.add_argument("--top-eigs", type=int, default=10, help="Use the first N values from the eig dimension.")
    parser.add_argument("--sampling", choices=["undersample", "oversample", "none"], default="undersample")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subject IDs, e.g. HUP070 HUP074")
    parser.add_argument("--max-subjects", type=int, default=None, help="Debug option: only use first N subjects.")
    parser.add_argument("--no-excel", action="store_true", help="Only write CSV files.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_loso(args)


if __name__ == "__main__":
    main()
