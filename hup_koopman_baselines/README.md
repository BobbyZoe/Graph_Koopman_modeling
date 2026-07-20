# HUP SOZ baseline reproduction code

This package implements reproducible baselines for channel-level SOZ localization on preprocessed HUP iEEG/ECoG/SEEG data filtered in 80--200 Hz:

1. HFO rate
2. Static FC/EC using FD-CCM-like directed connectivity + graph centrality
3. Dynamic FC using multilayer network + mlEVC + SVD/clustering
4. Neural fragility from sliding-window linear dynamics
5. Adaptive graph learning GNN implemented in pure PyTorch

The code is designed as a practical reproduction/benchmark scaffold. It is not a byte-for-byte copy of any paper's private code. For papers with unavailable official code, the implementation follows the algorithmic description and exposes parameters so you can tune them to your manuscript.

## Expected input format

Put each preprocessed run in a `.pkl` or `.npz` file. Each file should contain a dictionary with:

```python
{
    "data": np.ndarray,          # shape: [n_channels, n_samples], already filtered 80--200 Hz
    "sfreq": 500,                # sampling frequency
    "ch_names": [...],           # optional
    "soz_index": [0, 3, 10],     # SOZ channel indices, or use "y" below
    "y": np.ndarray,             # optional binary labels shape [n_channels]
    "subject": "HUP123",         # optional; inferred from file name if absent
    "run": "run-01"              # optional; inferred from file name if absent
}
```

If your current pickle files have different key names, edit `hup_baselines/data.py::load_run`.

## Install

```bash
pip install numpy scipy pandas scikit-learn networkx tqdm torch
```

## Run feature baselines

```bash
python -m hup_baselines.run_all \
  --data_dir /path/to/preprocessed_80_200 \
  --out_dir ./results \
  --methods hfo fdccm mlevc fragility \
  --pattern "**/*.pkl" \
  --sfreq 500
```

This saves per-run feature CSVs and prediction/evaluation files.

## Run adaptive graph learning GNN

```bash
python -m hup_baselines.train_adaptive_gnn \
  --data_dir /path/to/preprocessed_80_200 \
  --out_dir ./results_gnn \
  --pattern "**/*.pkl" \
  --epochs 10 \
  --sfreq 500
```

## Notes

- HFO detector assumes the signal is already filtered in 80--200 Hz. It uses the Hilbert envelope and robust thresholding.
- FD-CCM can be slow for many channels. Start with short segments or `--max_points 1000`.
- The paper-aligned multilayer comparison splits 80--200 Hz into 80--140 and 140--200 Hz, uses 2.5 s lagged-coherence layers with 80% overlap, and returns only the subject-level mlEVC weighted-consensus score. The exponential pre-ictal mapping is a documented practical implementation because the paper does not state its explicit formula.
- Neural fragility uses a 250 ms/125 ms sliding first-order model and minimum real column perturbations evaluated on a discretized unit circle. Its native output is the channel-by-time fragility heatmap; no mean/max/percentile composite indicators are added.
- Adaptive ST GNN uses one-second raw 80--200 Hz segments, an IEEG-TCN attribute encoder, learned adjacency, one GCN layer, a 20-segment node LSTM, label-smoothed ictal/interictal weak supervision, and summed node-readout attention as the only localization score. The notebook replaces the paper's patient-specific seizure split with cross-subject LOSO for the common comparison protocol.
- All supervised feature baselines use Leave-One-Subject-Out evaluation in `evaluation.py`.
