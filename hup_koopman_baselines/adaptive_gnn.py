from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import RunData, normalize_scores


class GatedCausalBlock(nn.Module):
    """Residual gated dilated-convolution block used by IEEG-TCN."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.filter_conv = nn.utils.weight_norm(
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=self.padding)
        )
        self.gate_conv = nn.utils.weight_norm(
            nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=self.padding)
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        filtered = self.filter_conv(x)
        gated = self.gate_conv(x)
        if self.padding:
            filtered = filtered[..., :-self.padding]
            gated = gated[..., :-self.padding]
        update = torch.tanh(filtered) * torch.sigmoid(gated)
        return F.relu(x + self.norm(update))


class IEEGTCN(nn.Module):
    """Learn a paper-defined attribute vector from each one-second channel segment."""

    def __init__(self, attribute_dim: int = 10):
        super().__init__()
        self.input_conv = nn.Conv1d(1, 12, kernel_size=5, padding=2)
        self.input_norm = nn.BatchNorm1d(12)
        self.blocks = nn.ModuleList(
            [GatedCausalBlock(12, kernel_size=3, dilation=2**level) for level in range(3)]
        )
        self.output_conv = nn.Conv1d(12, 24, kernel_size=1)
        self.output_norm = nn.BatchNorm1d(24)
        self.temporal_attention = nn.Parameter(torch.empty(24))
        self.attribute_projection = nn.Linear(24, attribute_dim)
        nn.init.normal_(self.temporal_attention, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: batch_of_channels x samples
        h = F.relu(self.input_norm(self.input_conv(x[:, None, :])))
        for block in self.blocks:
            h = block(h)
        h = F.relu(self.output_norm(self.output_conv(h)))
        logits = torch.einsum("bft,f->bt", torch.tanh(h), self.temporal_attention)
        attention = torch.softmax(logits, dim=-1)
        pooled = torch.einsum("bft,bt->bf", h, attention)
        return self.attribute_projection(pooled), attention


class AdaptiveSTGNN(nn.Module):
    """IEEG-TCN + adaptive graph + one-layer GCN + node LSTM + attention readout.

    Input has shape ``[batch, sequence, channels, samples]``. The output is an
    ictal/interictal graph-level logit. Node attention, rather than a supervised
    node logit, is the paper-defined channel-localization signal.
    """

    def __init__(
        self,
        attribute_dim: int = 10,
        graph_dim: int = 8,
        gcn_dim: int = 8,
        lstm_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attribute_dim = attribute_dim
        self.tcn = IEEGTCN(attribute_dim=attribute_dim)
        self.graph_embedding = nn.Linear(attribute_dim, graph_dim, bias=False)
        self.gcn = nn.Linear(attribute_dim, gcn_dim, bias=False)
        self.lstm = nn.LSTMCell(attribute_dim + gcn_dim, lstm_dim)
        self.node_attention = nn.Parameter(torch.empty(lstm_dim))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(lstm_dim, 2))
        nn.init.normal_(self.node_attention, std=0.02)

    @staticmethod
    def _normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
        channels = adjacency.shape[-1]
        identity = torch.eye(channels, device=adjacency.device, dtype=adjacency.dtype)[None]
        adjacency = adjacency + identity
        degree = adjacency.sum(dim=-1).clamp_min(1e-8)
        inv_sqrt = degree.rsqrt()
        return inv_sqrt.unsqueeze(-1) * adjacency * inv_sqrt.unsqueeze(-2)

    def encode_segments(self, segments: torch.Tensor) -> torch.Tensor:
        """Encode one-second segments once per run.

        ``segments`` has shape ``[segments, channels, samples]`` and the output
        has shape ``[segments, channels, attribute_dim]``.
        """
        n_segments, channels, samples = segments.shape
        flat = segments.reshape(n_segments * channels, samples)
        attributes, _ = self.tcn(flat)
        return attributes.reshape(n_segments, channels, self.attribute_dim)

    def forward_from_attributes(
        self,
        attributes: torch.Tensor,
        return_attention: bool = True,
        return_adjacency: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch, steps, channels, _ = attributes.shape
        hidden = torch.zeros(batch * channels, self.lstm.hidden_size, device=attributes.device)
        cell = torch.zeros_like(hidden)
        node_attention_sequence = [] if return_attention else None
        adjacency_sequence = [] if return_adjacency else None
        graph_logits = []

        for step in range(steps):
            node_features = attributes[:, step]
            embedding = self.graph_embedding(node_features)
            similarity = torch.matmul(embedding, embedding.transpose(-1, -2))
            adjacency = F.relu(torch.tanh(similarity))
            normalized = self._normalize_adjacency(adjacency)
            spatial = F.relu(torch.matmul(normalized, self.gcn(node_features)))
            recurrent_input = torch.cat([node_features, spatial], dim=-1).reshape(batch * channels, -1)
            hidden, cell = self.lstm(recurrent_input, (hidden, cell))
            node_hidden = hidden.reshape(batch, channels, -1)
            attention_logits = torch.einsum("bch,h->bc", torch.tanh(node_hidden), self.node_attention)
            node_attention = torch.softmax(attention_logits, dim=-1)
            graph_embedding = torch.einsum("bc,bch->bh", node_attention, node_hidden)
            graph_logits.append(self.classifier(graph_embedding))
            if node_attention_sequence is not None:
                node_attention_sequence.append(node_attention)
            if adjacency_sequence is not None:
                adjacency_sequence.append(adjacency)

        return (
            graph_logits[-1],
            torch.stack(node_attention_sequence, dim=1) if node_attention_sequence is not None else None,
            torch.stack(adjacency_sequence, dim=1) if adjacency_sequence is not None else None,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = True,
        return_adjacency: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        batch, steps, channels, samples = x.shape
        flat = x.reshape(batch * steps * channels, samples)
        attributes, _ = self.tcn(flat)
        attributes = attributes.reshape(batch, steps, channels, self.attribute_dim)
        return self.forward_from_attributes(
            attributes,
            return_attention=return_attention,
            return_adjacency=return_adjacency,
        )


def segment_run(
    X: np.ndarray,
    sfreq: float,
    segment_s: float = 1.0,
) -> np.ndarray:
    """Zero-mean/unit-variance channel normalization followed by 1 s segments."""
    X = np.asarray(X, dtype=np.float32)
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-6)
    samples = int(round(segment_s * sfreq))
    n_segments = X.shape[1] // samples
    if n_segments < 2:
        raise ValueError("Run is too short for segmentation")
    trimmed = X[:, : n_segments * samples]
    return trimmed.reshape(X.shape[0], n_segments, samples).transpose(1, 0, 2).copy()


def sequence_targets(
    n_segments: int,
    sequence_length: int = 20,
    onset_segment: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return prediction indices and weak ictal/interictal labels."""
    targets = np.arange(sequence_length, n_segments, dtype=int)
    labels = (targets >= onset_segment).astype(np.int64)
    return targets, labels


def make_sequence_batch(
    segments: np.ndarray,
    target_indices: np.ndarray,
    sequence_length: int = 20,
) -> np.ndarray:
    return np.stack(
        [segments[target - sequence_length : target] for target in target_indices], axis=0
    ).astype(np.float32, copy=False)


def make_sequence_batch_tensor(
    segments: torch.Tensor,
    target_indices: torch.Tensor,
    sequence_length: int = 20,
) -> torch.Tensor:
    """Build a sequence batch from an on-device segment tensor.

    ``segments`` has shape ``[segments, channels, samples]`` and target index
    ``t`` maps to ``segments[t - sequence_length:t]``.
    """
    target_indices = target_indices.to(device=segments.device, dtype=torch.long, non_blocking=True)
    offsets = torch.arange(sequence_length, device=segments.device, dtype=torch.long)
    indices = target_indices[:, None] - sequence_length + offsets[None, :]
    selected = segments.index_select(0, indices.reshape(-1))
    return selected.reshape(target_indices.numel(), sequence_length, *segments.shape[1:])


def localization_score_from_attention(attention: np.ndarray) -> np.ndarray:
    """Equations 13--14: sum node attention over time, then min-max normalize."""
    attention = np.asarray(attention, dtype=float)
    if attention.ndim == 3:  # sequences x time x channels
        importance = attention.sum(axis=(0, 1))
    elif attention.ndim == 2:  # time x channels
        importance = attention.sum(axis=0)
    else:
        raise ValueError(f"Unexpected attention shape {attention.shape}")
    return normalize_scores(importance)


def label_smoothed_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smoothing: float = 0.1,
) -> torch.Tensor:
    return F.cross_entropy(logits, labels, label_smoothing=smoothing)


# Backward-compatible alias for callers that imported the previous class name.
AdaptiveGraphLSTM = AdaptiveSTGNN


def train_one_subject_loso(
    runs: list[RunData],
    test_subject: str,
    epochs: int = 10,
    lr: float = 1e-4,
    hidden_dim: int = 32,
    device: str = "cpu",
    window_s: float = 1.0,
    step_s: float = 0.5,
    seed: int = 0,
) -> Tuple[Dict[int, np.ndarray], AdaptiveSTGNN]:
    """Compatibility entry point using paper-style weak seizure-state supervision.

    ``step_s`` is retained for API compatibility; the paper uses contiguous
    one-second segments and sequences of 20 segments.
    """
    del step_s
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_runs = [run for run in runs if run.subject != test_subject]
    test_runs = [run for run in runs if run.subject == test_subject]
    if not train_runs or not test_runs:
        raise ValueError("Need at least one training and one test run")
    model = AdaptiveSTGNN(lstm_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.25)
    amp_enabled = str(device).startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    for _ in range(epochs):
        model.train()
        for run_index in rng.permutation(len(train_runs)):
            segments = segment_run(train_runs[run_index].X, train_runs[run_index].sfreq, window_s)
            targets, labels = sequence_targets(len(segments), sequence_length=20, onset_segment=50)
            negative = np.flatnonzero(labels == 0)
            positive = np.flatnonzero(labels == 1)
            n_each = min(len(negative), len(positive))
            selected = np.concatenate(
                [rng.choice(negative, n_each, replace=False), rng.choice(positive, n_each, replace=False)]
            )
            rng.shuffle(selected)
            targets, labels = targets[selected], labels[selected]
            segments_tensor = torch.from_numpy(segments).to(device)
            targets_tensor = torch.as_tensor(targets, dtype=torch.long, device=segments_tensor.device)
            labels_tensor = torch.as_tensor(labels, dtype=torch.long, device=segments_tensor.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                attributes = model.encode_segments(segments_tensor)
                attribute_batch = make_sequence_batch_tensor(attributes, targets_tensor, 20)
                logits, _, _ = model.forward_from_attributes(
                    attribute_batch, return_attention=False, return_adjacency=False
                )
                loss = label_smoothed_cross_entropy(logits, labels_tensor, smoothing=0.1)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

    model.eval()
    scores: Dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for run in test_runs:
            segments = segment_run(run.X, run.sfreq, window_s)
            targets, _ = sequence_targets(len(segments), sequence_length=20, onset_segment=50)
            segments_tensor = torch.from_numpy(segments).to(device)
            targets_tensor = torch.as_tensor(targets, dtype=torch.long, device=segments_tensor.device)
            attention_batches = []
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                attributes = model.encode_segments(segments_tensor)
            for start in range(0, len(targets), 64):
                batch_targets = targets_tensor[start : start + 64]
                attribute_batch = make_sequence_batch_tensor(attributes, batch_targets, 20)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    _, attention, _ = model.forward_from_attributes(
                        attribute_batch, return_attention=True, return_adjacency=False
                    )
                attention_batches.append(attention.cpu().numpy())
            scores[id(run)] = localization_score_from_attention(
                np.concatenate(attention_batches, axis=0)
            )
    return scores, model
