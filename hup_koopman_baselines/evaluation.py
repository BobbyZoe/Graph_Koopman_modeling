from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .data import score_metrics


def balanced_train_indices(y: np.ndarray, random_state: int = 0) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    y = np.asarray(y).astype(int)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return np.arange(len(y))
    n = min(len(neg), len(pos))
    neg_sel = rng.choice(neg, size=n, replace=False)
    return np.concatenate([pos, neg_sel])


def loso_supervised_eval(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    model: str = "rf",
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-subject-out channel-level classification."""
    df = df.copy().reset_index(drop=True)
    groups = df["subject"].astype(str).values
    y = df["y"].astype(int).values
    X = df[list(feature_cols)].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(float)
    logo = LeaveOneGroupOut()
    scores = np.full(len(df), np.nan, dtype=float)
    fold_rows = []
    for fold, (tr, te) in enumerate(logo.split(X, y, groups)):
        tr_bal = tr[balanced_train_indices(y[tr], random_state=random_state + fold)]
        if model == "logreg":
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
            )
        else:
            clf = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                random_state=random_state + fold,
                n_jobs=-1,
            )
        clf.fit(X[tr_bal], y[tr_bal])
        if hasattr(clf, "predict_proba"):
            s = clf.predict_proba(X[te])[:, 1]
        else:
            s = clf.decision_function(X[te])
        scores[te] = s
        sub = str(groups[te][0])
        m = score_metrics(y[te], s)
        m.update({"fold": fold, "subject": sub, "n_channels": int(len(te)), "n_soz": int(y[te].sum())})
        fold_rows.append(m)
    pred_df = df[["subject", "run", "channel", "ch_name", "y"]].copy()
    pred_df["score"] = scores
    metrics_df = pd.DataFrame(fold_rows)
    return pred_df, metrics_df


def unsupervised_eval(df: pd.DataFrame, score_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pred_df = df[["subject", "run", "channel", "ch_name", "y", score_col]].copy()
    pred_df = pred_df.rename(columns={score_col: "score"})
    rows = []
    for subject, g in pred_df.groupby("subject"):
        m = score_metrics(g["y"].values, g["score"].values)
        m.update({"subject": subject, "n_channels": int(len(g)), "n_soz": int(g["y"].sum())})
        rows.append(m)
    return pred_df, pd.DataFrame(rows)
