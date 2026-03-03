"""
utility_functions.py
--------------------
Utility functions for Assignment 1: Sepsis Onset Prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, recall_score, precision_score
)


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_all_partitions(base_path: str = "raw_data") -> pd.DataFrame:
    """
    Load and concatenate the four SepsisExp partitions (A–D).

    Returns a single DataFrame with all patients and timesteps.
    Prints a summary of each partition.
    """
    partitions = ["A", "B", "C", "D"]
    dfs = []
    for part in partitions:
        path = f"{base_path}/sepsisexp_timeseries_partition-{part}.tsv"
        df_part = pd.read_csv(path, sep="\t")
        dfs.append(df_part)
        print(f"  Partition {part}: {df_part.shape[0]:,} rows, "
              f"{df_part['id'].nunique()} patients")
    df = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total: {df.shape[0]:,} rows, {df['id'].nunique()} patients")
    return df


def get_onset_map(df: pd.DataFrame) -> dict:
    """
    Compute first timestep where severity > 0 for each sepsis patient.
    Returns dict: {patient_id: onset_time_hours}
    """
    return (
        df[(df["sepsis"] == 1) & (df["severity"] > 0)]
        .groupby("id")["timestep"]
        .min()
        .to_dict()
    )


# ── Sliding Window Dataset ────────────────────────────────────────────────────

def build_windows(
    df: pd.DataFrame,
    feature_cols: list,
    window_size: int = 12,
    horizon_hours: float = 4.0,
    stride: int = 2
):
    """
    Build a sliding-window dataset for sepsis onset prediction.

    Each window of `window_size` timesteps is labelled:
      1  if: patient has sepsis AND onset falls within the next `horizon_hours`
             after the window's last timestep (and before onset).
      0  otherwise.

    Windows that already contain onset data (severity > 0) are excluded
    to prevent data leakage.

    Parameters
    ----------
    window_size   : int   — number of 30-min timesteps per window (default 12 = 6h)
    horizon_hours : float — prediction horizon in hours (2, 4, or 6)
    stride        : int   — step between windows (default 2 = 1h)

    Returns
    -------
    X   : np.ndarray  shape (n_windows, window_size, n_features)
    y   : np.ndarray  shape (n_windows,)  int labels
    pids: np.ndarray  shape (n_windows,)  patient IDs
    """
    X_list, y_list, pid_list = [], [], []
    sepsis_ids = set(df[df["sepsis"] == 1]["id"].unique())
    onset_map  = get_onset_map(df)

    for pid, group in df.groupby("id"):
        group    = group.sort_values("timestep").reset_index(drop=True)
        features = group[feature_cols].values.astype(np.float32)
        times    = group["timestep"].values
        onset_t  = onset_map.get(pid, None)

        for start in range(0, len(group) - window_size + 1, stride):
            end           = start + window_size
            win_end_time  = times[end - 1]

            # Skip if the window already overlaps with sepsis onset
            if onset_t is not None and win_end_time >= onset_t:
                continue

            if pid in sepsis_ids and onset_t is not None:
                label = int(0 < (onset_t - win_end_time) <= horizon_hours)
            else:
                label = 0

            X_list.append(features[start:end])
            y_list.append(label)
            pid_list.append(pid)

    return (
        np.array(X_list,   dtype=np.float32),
        np.array(y_list,   dtype=np.int64),
        np.array(pid_list)
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model_nn(model, X, y, device, batch_size=256, threshold=0.5):
    """
    Evaluate a PyTorch model, returning probabilities and a metrics dict.

    Returns
    -------
    probs   : np.ndarray — predicted probabilities
    labels  : np.ndarray — true labels
    metrics : dict       — auc_roc, auc_pr, f1, recall, precision
    """
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()
    ds     = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            all_probs.extend(model(Xb.to(device)).cpu().numpy())
            all_labels.extend(yb.numpy())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= threshold).astype(int)

    metrics = {
        "auc_roc":   roc_auc_score(labels, probs) if labels.sum() > 0 else 0,
        "auc_pr":    average_precision_score(labels, probs) if labels.sum() > 0 else 0,
        "f1":        f1_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
    }
    return probs, labels, metrics


def plot_confusion(y_true, y_pred, title="Confusion Matrix", ax=None):
    """Plot a labelled confusion matrix."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Sepsis", "Sepsis"],
        yticklabels=["No Sepsis", "Sepsis"],
        ax=ax
    )
    ax.set_title(title)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")


def plot_pr_curve(y_true, y_prob, label="", ax=None, color="steelblue"):
    """Plot a precision-recall curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob) if y_true.sum() > 0 else 0
    ax.plot(rec, prec, color=color, label=f"{label} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    return ap


def summarise_cv_results(all_results: dict) -> pd.DataFrame:
    """
    Build a summary DataFrame from cross-validation results.

    Parameters
    ----------
    all_results : dict  {horizon_int: [fold_metrics_dict, ...]}

    Returns
    -------
    summary_df : pd.DataFrame  with mean ± std per horizon
    """
    rows = []
    for horizon, folds in all_results.items():
        for m in folds:
            rows.append({
                "Horizon": f"{horizon}h",
                "Fold":    m["fold"],
                "AUC-ROC": m["auc_roc"],
                "AUC-PR":  m["auc_pr"],
                "Recall":  m["recall"],
                "Precision": m["precision"],
                "F1":      m["f1"],
            })
    df = pd.DataFrame(rows)
    mean_std = df.groupby("Horizon")[
        ["AUC-ROC", "AUC-PR", "Recall", "Precision", "F1"]
    ].agg(lambda x: f"{x.mean():.3f} ± {x.std():.3f}")
    return mean_std
