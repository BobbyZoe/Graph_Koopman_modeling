from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from .data import features_to_frame, load_dataset, save_json
from .evaluation import loso_supervised_eval
from .hfo_features import hfo_rate_features_for_run
from .fdccm_centrality import fdccm_centrality_features
from .multilayer_network import multilayer_features_for_run
from .neural_fragility import neural_fragility_features


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pattern", default="**/*.pkl")
    ap.add_argument("--sfreq", type=float, default=None)
    ap.add_argument("--methods", nargs="+", default=["hfo", "fdccm", "mlevc", "fragility"])
    ap.add_argument("--max_points", type=int, default=1500, help="FD-CCM max resampled points per run")
    ap.add_argument("--show_progress", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_dataset(args.data_dir, pattern=args.pattern, default_sfreq=args.sfreq)

    method_to_frames: Dict[str, List[pd.DataFrame]] = {m: [] for m in args.methods}
    for run in tqdm(runs, desc="Runs"):
        if "hfo" in args.methods:
            feats = hfo_rate_features_for_run(run.X, run.sfreq, show_progress=args.show_progress)
            df = features_to_frame(run, feats)
            method_to_frames["hfo"].append(df)
            p = out_dir / "features" / "hfo" / f"{run.subject}_{run.run}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, index=False)

        if "fdccm" in args.methods:
            feats = fdccm_centrality_features(run.X, run.sfreq, max_points=args.max_points, show_progress=args.show_progress)
            df = features_to_frame(run, feats)
            method_to_frames["fdccm"].append(df)
            p = out_dir / "features" / "fdccm" / f"{run.subject}_{run.run}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, index=False)

        if "mlevc" in args.methods:
            feats = multilayer_features_for_run(run.X, run.sfreq)
            df = features_to_frame(run, feats)
            method_to_frames["mlevc"].append(df)
            p = out_dir / "features" / "mlevc" / f"{run.subject}_{run.run}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, index=False)

        if "fragility" in args.methods:
            feats = neural_fragility_features(run.X, run.sfreq, show_progress=args.show_progress)
            df = features_to_frame(run, feats)
            method_to_frames["fragility"].append(df)
            p = out_dir / "features" / "fragility" / f"{run.subject}_{run.run}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, index=False)

    summary = {}
    for method, frames in method_to_frames.items():
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(out_dir / f"{method}_all_features.csv", index=False)
        feature_cols = [c for c in df.columns if c not in {"subject", "run", "channel", "ch_name", "y"}]
        pred, met = loso_supervised_eval(df, feature_cols, model="rf")
        pred.to_csv(out_dir / f"{method}_predictions.csv", index=False)
        met.to_csv(out_dir / f"{method}_loso_metrics.csv", index=False)
        summary[method] = met.mean(numeric_only=True).to_dict()
    save_json(summary, out_dir / "summary_metrics.json")
    print(pd.DataFrame(summary).T)


if __name__ == "__main__":
    main()
