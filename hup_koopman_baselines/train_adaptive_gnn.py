from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .adaptive_gnn import train_one_subject_loso
from .data import load_dataset, score_metrics, save_json


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pattern", default="**/*.pkl")
    ap.add_argument("--sfreq", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--hidden_dim", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_dataset(args.data_dir, pattern=args.pattern, default_sfreq=args.sfreq)
    subjects = sorted(set(r.subject for r in runs))

    pred_rows = []
    metrics_rows = []
    for fold, sub in enumerate(tqdm(subjects, desc="GNN LOSO")):
        scores, model = train_one_subject_loso(
            runs,
            test_subject=sub,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            device=args.device,
            seed=fold,
        )
        for r in runs:
            if r.subject != sub:
                continue
            s = scores[id(r)]
            for ch, sc in enumerate(s):
                pred_rows.append(
                    {
                        "subject": r.subject,
                        "run": r.run,
                        "channel": ch,
                        "ch_name": r.ch_names[ch],
                        "y": int(r.y[ch]),
                        "score": float(sc),
                    }
                )
        torch.save(model.state_dict(), out_dir / f"adaptive_gnn_fold_{sub}.pt")

    pred = pd.DataFrame(pred_rows)
    pred.to_csv(out_dir / "adaptive_gnn_predictions.csv", index=False)
    for sub, g in pred.groupby("subject"):
        m = score_metrics(g["y"].values, g["score"].values)
        m.update({"subject": sub, "n_channels": int(len(g)), "n_soz": int(g["y"].sum())})
        metrics_rows.append(m)
    met = pd.DataFrame(metrics_rows)
    met.to_csv(out_dir / "adaptive_gnn_loso_metrics.csv", index=False)
    save_json({"adaptive_gnn": met.mean(numeric_only=True).to_dict()}, out_dir / "adaptive_gnn_summary.json")
    print(met.mean(numeric_only=True))


if __name__ == "__main__":
    main()
