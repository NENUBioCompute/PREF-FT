#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train PREF-FT with an internal validation split and optional external cohorts.

Class 0 is unaffected and class 1 is affected. Only the training split fits the
imputer/scaler. Default label smoothing 0.05 follows the archived run settings;
set --label-smoothing 0 for hard-label cross entropy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import copy
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if __package__:
    from .pref_ft_model import PREF_FT
else:
    from pref_ft_model import PREF_FT


# ============================================================
# Configuration
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class Config:
    train_features: str = str(PROJECT_ROOT / "data/GSE243529_intersection_CpGs_after_com.csv.gz")
    train_metadata: str = str(PROJECT_ROOT / "data/GSE243529_pheno.csv")
    val_features: Optional[str] = None
    val_metadata: Optional[str] = None
    external_tests_json: Optional[str] = str(PROJECT_ROOT / "data/external_tests.json")

    outdir: str = str(PROJECT_ROOT / "runs/pref_ft")
    matrix_orientation: str = "cpg_rows"
    sample_col: str = "SampleID"
    label_col: str = "disease.state"
    class_order: str = "unaffected, affected"

    seed: int = 42
    val_size: float = 0.15
    stratify_val: bool = True

    external_feature_policy: str = "strict"  # strict or fill_train_median
    min_feature_overlap_ratio: float = 0.98

    d_model: int = 128
    dropout: float = 0.2
    n_transformer_layers: int = 3
    num_heads: int = 4
    tabm_num_members: int = 4
    use_linear_residual: bool = True
    gate_dropout: float = 0.05
    linear_dropout: float = 0.1
    use_positive_ovr: bool = True
    use_prototype_residual: bool = True

    epochs: int = 180
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-3
    num_workers: int = 0
    patience: int = 30
    amp: bool = False
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.05
    use_class_weight: bool = False

    save_logits: bool = True


@dataclass
class DatasetSpec:
    name: str
    features: str
    metadata: str
    matrix_orientation: str = "cpg_rows"
    sample_col: str = "SampleID"
    label_col: str = "disease.state"
    label_map: Optional[Dict[str, str]] = None


@dataclass
class TaskDefinition:
    class_names: List[str]
    label_to_id: Dict[str, int]
    id_to_label: Dict[int, str]
    num_classes: int
    task_type: str = "binary"


# ============================================================
# Basic utilities
# ============================================================

def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def to_jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v) if pd.notna(v) else None
    if isinstance(v, np.ndarray):
        return v.tolist()
    if torch.is_tensor(v):
        return v.detach().cpu().tolist()
    try:
        if np.isscalar(v) and pd.isna(v):
            return None
    except Exception:
        pass
    return v


def metrics_to_jsonable(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {k: to_jsonable(v) for k, v in metrics.items()}


def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    encodings = ["utf-8", "utf-8-sig", "gb18030", "gbk", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception as e:
            last_err = e
    raise ValueError(f"Failed to read file: {path}. Last error: {last_err}")


def ensure_sample_id_column(df: pd.DataFrame, sample_col: str) -> pd.DataFrame:
    df = df.copy()
    if sample_col in df.columns:
        df[sample_col] = df[sample_col].astype(str)
        return df
    if df.index.name == sample_col:
        df = df.reset_index()
        df[sample_col] = df[sample_col].astype(str)
        return df
    if "index" in df.columns:
        df = df.rename(columns={"index": sample_col})
        df[sample_col] = df[sample_col].astype(str)
        return df
    first_col = df.columns[0]
    df = df.rename(columns={first_col: sample_col})
    df[sample_col] = df[sample_col].astype(str)
    return df


def parse_class_order_to_task(class_order: str) -> TaskDefinition:
    class_names = [x.strip() for x in class_order.split(",") if x.strip()]
    if len(class_names) != 2:
        raise ValueError(f"This cleaned project only supports binary tasks, got class_order={class_names}")
    if len(set(class_names)) != 2:
        raise ValueError(f"class_order contains duplicate labels: {class_names}")
    return TaskDefinition(
        class_names=class_names,
        label_to_id={name: i for i, name in enumerate(class_names)},
        id_to_label={i: name for i, name in enumerate(class_names)},
        num_classes=2,
    )


def inverse_transform_labels(y: np.ndarray, task: TaskDefinition) -> List[str]:
    return [task.id_to_label[int(i)] for i in y]


def safe_metric_name(name: str) -> str:
    return str(name).replace(" ", "_").replace("/", "_").replace("-", "_").replace("'", "")


# ============================================================
# Dataset loading and feature alignment
# ============================================================

def canonicalize_label_series(series: pd.Series, label_map: Optional[Dict[str, str]] = None) -> pd.Series:
    out = series.astype(str).copy()
    if label_map is not None:
        out = out.map(lambda x: label_map.get(x, x))
    return out.astype(str)


def numeric_feature_block(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Convert features to numeric and collapse duplicate feature names by row-wise mean."""
    block = df.loc[:, feature_cols].copy()
    block = block.apply(pd.to_numeric, errors="coerce")
    if block.columns.duplicated().any():
        # Collapse duplicate CpG columns produced by duplicated probe IDs.
        block = block.T.groupby(level=0, sort=False).mean().T
    return block


def matrix_to_sample_feature_df(
    feat: pd.DataFrame,
    meta: pd.DataFrame,
    matrix_orientation: str,
    sample_col: str,
    dataset_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Return a sample x feature dataframe aligned to metadata order."""
    if matrix_orientation == "sample_rows":
        feat = ensure_sample_id_column(feat, sample_col)
        common = [s for s in meta[sample_col].astype(str).tolist() if s in set(feat[sample_col].astype(str))]
        if not common:
            raise ValueError(f"[{dataset_name}] No overlapping samples between features and metadata.")
        feat = feat[feat[sample_col].astype(str).isin(common)].copy()
        meta = meta[meta[sample_col].astype(str).isin(common)].copy()
        feat = feat.set_index(sample_col).loc[common].reset_index()
        meta = meta.set_index(sample_col).loc[common].reset_index()
        feature_cols = [c for c in feat.columns if c != sample_col]
        block = numeric_feature_block(feat, feature_cols)
        feat_out = pd.concat([feat[[sample_col]].reset_index(drop=True), block.reset_index(drop=True)], axis=1)
        return feat_out, meta, block.columns.tolist()

    if matrix_orientation == "cpg_rows":
        feat = feat.copy()
        cpg_col = feat.columns[0]
        feat[cpg_col] = feat[cpg_col].astype(str)
        feat = feat.set_index(cpg_col)
        feat.columns = feat.columns.astype(str)
        common = [s for s in meta[sample_col].astype(str).tolist() if s in set(feat.columns)]
        if not common:
            raise ValueError(f"[{dataset_name}] No overlapping samples between cpg_rows matrix and metadata.")

        # Keep only sample columns, convert once, and average duplicated CpG/probe IDs.
        feat = feat.loc[:, common].apply(pd.to_numeric, errors="coerce")
        if feat.index.duplicated().any():
            feat = feat.groupby(level=0, sort=False).mean()

        feat_t = feat.T.reset_index().rename(columns={"index": sample_col})
        meta = meta[meta[sample_col].astype(str).isin(common)].copy()
        feat_t = feat_t.set_index(sample_col).loc[common].reset_index()
        meta = meta.set_index(sample_col).loc[common].reset_index()
        feature_cols = [c for c in feat_t.columns if c != sample_col]
        return feat_t, meta, feature_cols

    raise ValueError("matrix_orientation must be 'sample_rows' or 'cpg_rows'.")


def load_matrix_and_metadata(spec: DatasetSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feat = read_table(spec.features)
    meta = read_table(spec.metadata)
    meta = ensure_sample_id_column(meta, spec.sample_col)
    if spec.label_col not in meta.columns:
        raise ValueError(f"[{spec.name}] label_col={spec.label_col} not found in metadata columns: {meta.columns.tolist()}")
    meta[spec.sample_col] = meta[spec.sample_col].astype(str)
    meta[spec.label_col] = canonicalize_label_series(meta[spec.label_col], spec.label_map)
    return feat, meta


def filter_by_task_labels(
    feat_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    sample_col: str,
    label_col: str,
    task: TaskDefinition,
    dataset_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    allowed = set(task.class_names)
    labels_before = meta_df[label_col].astype(str)
    keep_mask = labels_before.isin(allowed)
    feat_kept = feat_df.loc[keep_mask.values].reset_index(drop=True)
    meta_kept = meta_df.loc[keep_mask.values].reset_index(drop=True)
    info = {
        "n_samples_before_label_filter": int(len(meta_df)),
        "n_samples_after_label_filter": int(len(meta_kept)),
        "n_samples_dropped_by_label_filter": int((~keep_mask).sum()),
        "allowed_labels": list(task.class_names),
        "dropped_label_counts": labels_before.loc[~keep_mask].value_counts().to_dict(),
        "kept_label_counts_raw": labels_before.loc[keep_mask].value_counts().to_dict(),
    }
    if len(meta_kept) == 0:
        raise ValueError(f"[{dataset_name}] No samples remain after class_order filtering: {task.class_names}")
    feat_kept = feat_kept.set_index(sample_col).loc[meta_kept[sample_col].tolist()].reset_index()
    return feat_kept, meta_kept, info


def encode_labels(meta: pd.DataFrame, label_col: str, task: TaskDefinition, dataset_name: str) -> np.ndarray:
    labels = meta[label_col].astype(str).values.tolist()
    unknown = sorted(list(set(labels) - set(task.class_names)))
    if unknown:
        raise ValueError(f"[{dataset_name}] labels contain unseen classes after filtering: {unknown}")
    return np.array([task.label_to_id[x] for x in labels], dtype=np.int64)


def load_dataset_as_arrays(
    spec: DatasetSpec,
    task: TaskDefinition,
    expected_feature_order: Optional[List[str]] = None,
    external_feature_policy: str = "strict",
    min_feature_overlap_ratio: float = 0.98,
) -> Dict[str, Any]:
    feat, meta = load_matrix_and_metadata(spec)
    feat_df, meta_df, feature_cols = matrix_to_sample_feature_df(
        feat=feat,
        meta=meta,
        matrix_orientation=spec.matrix_orientation,
        sample_col=spec.sample_col,
        dataset_name=spec.name,
    )
    feat_df, meta_df, filter_info = filter_by_task_labels(
        feat_df=feat_df,
        meta_df=meta_df,
        sample_col=spec.sample_col,
        label_col=spec.label_col,
        task=task,
        dataset_name=spec.name,
    )
    y = encode_labels(meta_df, spec.label_col, task, spec.name)
    sample_ids = feat_df[spec.sample_col].astype(str).values
    numeric_df = feat_df[feature_cols].copy()

    overlap_info = {
        "n_raw_features": int(len(feature_cols)),
        "n_overlap_with_train": None,
        "overlap_ratio_against_train": None,
        "missing_train_features": None,
        "extra_external_features": None,
    }

    if expected_feature_order is None:
        return {
            "X_raw": numeric_df.values.astype(np.float32),
            "y": y,
            "sample_ids": sample_ids,
            "feature_names": feature_cols,
            "overlap_info": overlap_info,
            "filter_info": filter_info,
        }

    present = [f for f in expected_feature_order if f in numeric_df.columns]
    missing = [f for f in expected_feature_order if f not in numeric_df.columns]
    extra = [f for f in numeric_df.columns if f not in set(expected_feature_order)]
    overlap_ratio = len(present) / max(1, len(expected_feature_order))
    overlap_info.update({
        "n_overlap_with_train": int(len(present)),
        "overlap_ratio_against_train": float(overlap_ratio),
        "missing_train_features": int(len(missing)),
        "extra_external_features": int(len(extra)),
    })
    if external_feature_policy == "strict" and missing:
        raise ValueError(f"[{spec.name}] strict feature policy failed: {len(missing)} training features are missing.")
    if overlap_ratio < min_feature_overlap_ratio:
        raise ValueError(f"[{spec.name}] feature overlap ratio too low: {overlap_ratio:.4f} < {min_feature_overlap_ratio:.4f}.")
    aligned_df = numeric_df.reindex(columns=expected_feature_order)
    return {
        "X_raw": aligned_df.values.astype(np.float32),
        "y": y,
        "sample_ids": sample_ids,
        "feature_names": list(expected_feature_order),
        "overlap_info": overlap_info,
        "filter_info": filter_info,
    }


# ============================================================
# Preprocessing
# ============================================================

def fit_preprocessor_on_train(X_train_raw: np.ndarray) -> Dict[str, Any]:
    train_medians = np.nanmedian(X_train_raw, axis=0)
    train_medians = np.where(np.isnan(train_medians), 0.0, train_medians).astype(np.float32)
    X_imp = X_train_raw.copy()
    inds = np.where(np.isnan(X_imp))
    if len(inds[0]) > 0:
        X_imp[inds] = np.take(train_medians, inds[1])
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_imp).astype(np.float32)
    return {"train_medians": train_medians, "scaler": scaler, "X_train_scaled": X_scaled}


def transform_with_preprocessor(X_raw: np.ndarray, preproc: Dict[str, Any]) -> np.ndarray:
    X_imp = X_raw.copy()
    inds = np.where(np.isnan(X_imp))
    if len(inds[0]) > 0:
        X_imp[inds] = np.take(preproc["train_medians"], inds[1])
    return preproc["scaler"].transform(X_imp).astype(np.float32)


# ============================================================
# Dataset / DataLoader
# ============================================================

class MethylationDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(X: np.ndarray, y: np.ndarray, cfg: Config, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    return DataLoader(
        MethylationDataset(X, y),
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


# ============================================================
# Metrics
# ============================================================

def safe_divide(num: float, den: float) -> float:
    return float(num / den) if den != 0 else np.nan


def binary_mcc_from_counts(tp: int, fp: int, tn: int, fn: int) -> float:
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if den == 0:
        return np.nan
    return float(((tp * tn) - (fp * fn)) / np.sqrt(den))


def binary_auc(y_bin: np.ndarray, score: np.ndarray) -> float:
    try:
        if len(np.unique(y_bin)) < 2:
            return np.nan
        return float(roc_auc_score(y_bin, score))
    except Exception:
        return np.nan


def binary_auprc(y_bin: np.ndarray, score: np.ndarray) -> float:
    try:
        if len(np.unique(y_bin)) < 2:
            return np.nan
        return float(average_precision_score(y_bin, score))
    except Exception:
        return np.nan


def compute_per_class_metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray, class_names: List[str]) -> pd.DataFrame:
    rows = []
    n = int(len(y_true))
    for i, cls in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        p_bin = (pred == i).astype(int)
        tp = int(((y_bin == 1) & (p_bin == 1)).sum())
        fp = int(((y_bin == 0) & (p_bin == 1)).sum())
        tn = int(((y_bin == 0) & (p_bin == 0)).sum())
        fn = int(((y_bin == 1) & (p_bin == 0)).sum())
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        specificity = safe_divide(tn, tn + fp)
        f1 = safe_divide(2 * precision * recall, precision + recall) if not (np.isnan(precision) or np.isnan(recall)) else np.nan
        rows.append({
            "class_id": int(i),
            "class_name": cls,
            "support": int(tp + fn),
            "n_negative": int(tn + fp),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "acc": safe_divide(tp + tn, n),
            "balanced_acc": float(np.nanmean([recall, specificity])),
            "precision": precision,
            "ppv": precision,
            "recall": recall,
            "sensitivity": recall,
            "specificity": specificity,
            "f1": f1,
            "mcc": binary_mcc_from_counts(tp, fp, tn, fn),
            "auc_ovr": binary_auc(y_bin, prob[:, i]) if prob.ndim == 2 and prob.shape[1] > i else np.nan,
            "auprc_ovr": binary_auprc(y_bin, prob[:, i]) if prob.ndim == 2 and prob.shape[1] > i else np.nan,
            "npv": safe_divide(tn, tn + fn),
            "fpr": safe_divide(fp, fp + tn),
            "fnr": safe_divide(fn, fn + tp),
            "prevalence": safe_divide(tp + fn, n),
            "pred_positive_rate": safe_divide(tp + fp, n),
        })
    return pd.DataFrame(rows)


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    prob = np.asarray(prob)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    n = int(tp + tn + fp + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    npv = safe_divide(tn, tn + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall) if not (np.isnan(precision) or np.isnan(recall)) else np.nan
    y_bin = (y_true == 1).astype(int)
    pos_score = prob[:, 1] if prob.ndim == 2 and prob.shape[1] > 1 else np.full_like(y_bin, np.nan, dtype=float)
    bal = float(np.nanmean([recall, specificity]))
    per_class = compute_per_class_metrics(y_true, prob, pred, class_names)
    return {
        "acc": safe_divide(tp + tn, n),
        "bal_acc": bal,
        "balanced_acc": bal,
        "precision": precision,
        "ppv": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "npv": npv,
        "f1": f1,
        "mcc": binary_mcc_from_counts(tp, fp, tn, fn),
        "auc_binary": binary_auc(y_bin, pos_score),
        "auprc_binary": binary_auprc(y_bin, pos_score),
        "fpr": safe_divide(fp, fp + tn),
        "fnr": safe_divide(fn, fn + tp),
        "prevalence": safe_divide(tp + fn, n),
        "pred_positive_rate": safe_divide(tp + fp, n),
        "tp": float(tp), "fp": float(fp), "tn": float(tn), "fn": float(fn),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "mcc_multiclass": float(matthews_corrcoef(y_true, pred)),
        "macro_auc_ovr_from_per_class": float(np.nanmean(per_class["auc_ovr"].to_numpy(dtype=float))),
        "macro_auprc_ovr_from_per_class": float(np.nanmean(per_class["auprc_ovr"].to_numpy(dtype=float))),
    }


def compute_class_weights(y: np.ndarray, num_classes: int = 2) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return (weights / weights.mean()).astype(np.float32)


def select_validation_score(metrics: Dict[str, float]) -> Tuple[str, float]:
    for name in ["auc_binary", "f1", "bal_acc"]:
        value = metrics.get(name, np.nan)
        try:
            if pd.notna(value):
                return name, float(value)
        except Exception:
            pass
    return "none", -np.inf


# ============================================================
# Training / prediction
# ============================================================

class EarlyStopping:
    def __init__(self, patience: int = 30):
        self.patience = int(patience)
        self.best: Optional[float] = None
        self.bad_epochs = 0
        self.best_state: Optional[Dict[str, torch.Tensor]] = None

    def step(self, score: float, model: nn.Module) -> bool:
        improved = self.best is None or score > self.best
        if improved:
            self.best = float(score)
            self.bad_epochs = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_grad_norm: float = 1.0,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and is_train and device.type == "cuda"))

    losses: List[float] = []
    probs_all: List[np.ndarray] = []
    preds_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=(amp_enabled and device.type == "cuda")):
                logits, _ = model(xb)
                loss = criterion(logits, yb)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if amp_enabled and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    optimizer.step()

        prob = torch.softmax(logits.detach(), dim=1).cpu().numpy()
        pred = prob.argmax(axis=1)
        probs_all.append(prob)
        preds_all.append(pred)
        y_all.append(yb.detach().cpu().numpy())
        losses.append(float(loss.detach().cpu().item()))

    probs = np.concatenate(probs_all, axis=0)
    preds = np.concatenate(preds_all, axis=0)
    y_true = np.concatenate(y_all, axis=0)
    metrics = compute_metrics(y_true, probs, preds, ["negative", "positive"])
    metrics["loss"] = float(np.mean(losses))
    return metrics, probs, preds, y_true


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, amp_enabled: bool):
    return run_epoch(model, loader, criterion, device, amp_enabled, optimizer=None)


# ============================================================
# Output helpers
# ============================================================

def plot_history(hist_df: pd.DataFrame, outdir: str) -> None:
    for metric, ylabel, fname in [
        ("loss", "Loss", "loss_curve.png"),
        ("auc_binary", "Binary ROC-AUC", "auc_binary_curve.png"),
        ("bal_acc", "Balanced Accuracy", "bal_acc_curve.png"),
    ]:
        train_col = f"train_{metric}"
        val_col = f"val_{metric}"
        if train_col not in hist_df.columns or val_col not in hist_df.columns:
            continue
        plt.figure(figsize=(6, 4))
        plt.plot(hist_df["epoch"], hist_df[train_col], label="train")
        plt.plot(hist_df["epoch"], hist_df[val_col], label="val")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=220)
        plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str, title: str) -> None:
    plt.figure(figsize=(5.2, 4.3))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=25, ha="right")
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def save_evaluation_outputs(
    save_dir: str,
    split_name: str,
    class_names: List[str],
    metrics: Dict[str, float],
    probs: np.ndarray,
    preds: np.ndarray,
    y_true: np.ndarray,
    sample_ids: np.ndarray,
    overlap_info: Optional[Dict[str, Any]] = None,
    filter_info: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    per_class_df = compute_per_class_metrics(y_true, probs, preds, class_names)
    per_class_df.to_csv(os.path.join(save_dir, "per_class_metrics.csv"), index=False)
    detailed = metrics_to_jsonable(metrics)
    detailed["per_class"] = per_class_df.replace({np.nan: None}).to_dict(orient="records")
    save_json(detailed, os.path.join(save_dir, "metrics.json"))

    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(os.path.join(save_dir, "confusion_matrix.csv"))
    plot_confusion_matrix(cm, class_names, os.path.join(save_dir, "confusion_matrix.png"), title=split_name)

    report = classification_report(y_true, preds, labels=[0, 1], target_names=class_names, digits=4, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(os.path.join(save_dir, "classification_report.csv"))

    pred_df = pd.DataFrame({
        "sample_id": sample_ids,
        "y_true": y_true,
        "y_pred": preds,
        "y_true_name": [class_names[i] for i in y_true],
        "y_pred_name": [class_names[i] for i in preds],
        f"prob__{safe_metric_name(class_names[0])}": probs[:, 0],
        f"prob__{safe_metric_name(class_names[1])}": probs[:, 1],
    })
    pred_df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

    if overlap_info is not None:
        save_json({k: to_jsonable(v) for k, v in overlap_info.items()}, os.path.join(save_dir, "feature_overlap_info.json"))
    if filter_info is not None:
        save_json(filter_info, os.path.join(save_dir, "label_filter_info.json"))
    np.save(os.path.join(save_dir, "probs.npy"), probs)
    np.save(os.path.join(save_dir, "preds.npy"), preds)
    np.save(os.path.join(save_dir, "y_true.npy"), y_true)


def make_external_summary_row(dataset_name: str, y_ext: np.ndarray, metrics: Dict[str, float], pack: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["acc", "bal_acc", "precision", "recall", "sensitivity", "specificity", "f1", "mcc", "auc_binary", "auprc_binary", "macro_f1", "weighted_f1"]
    row = {"dataset": dataset_name, "n_samples": int(len(y_ext))}
    for k in keys:
        row[k] = to_jsonable(metrics.get(k, np.nan))
    row.update(pack.get("overlap_info", {}))
    for k, v in pack.get("filter_info", {}).items():
        row[f"filter__{k}"] = json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v
    return row


# ============================================================
# Main training routine
# ============================================================

def split_train_val(X_raw: np.ndarray, y: np.ndarray, sample_ids: np.ndarray, cfg: Config) -> Dict[str, Any]:
    if cfg.stratify_val:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=cfg.val_size, random_state=cfg.seed)
        train_idx, val_idx = next(splitter.split(X_raw, y))
    else:
        rng = np.random.default_rng(cfg.seed)
        idx = np.arange(len(y))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(y) * cfg.val_size)))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
    return {
        "X_train_raw": X_raw[train_idx],
        "y_train": y[train_idx],
        "train_ids": sample_ids[train_idx],
        "X_val_raw": X_raw[val_idx],
        "y_val": y[val_idx],
        "val_ids": sample_ids[val_idx],
    }


def load_external_test_specs(path: Optional[str], cfg: Config) -> List[DatasetSpec]:
    if path is None or str(path).strip() == "":
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = [raw]
    specs = []
    for item in raw:
        specs.append(DatasetSpec(
            name=item["name"],
            features=str((Path(path).resolve().parent / item["features"]).resolve()),
            metadata=str((Path(path).resolve().parent / item["metadata"]).resolve()),
            matrix_orientation=item.get("matrix_orientation", cfg.matrix_orientation),
            sample_col=item.get("sample_col", cfg.sample_col),
            label_col=item.get("label_col", cfg.label_col),
            label_map=item.get("label_map"),
        ))
    return specs


def build_model(cfg: Config, num_features: int) -> PREF_FT:
    return PREF_FT(
        num_features=num_features,
        num_classes=2,
        d_model=cfg.d_model,
        n_transformer_layers=cfg.n_transformer_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        tabm_num_members=cfg.tabm_num_members,
        use_linear_residual=cfg.use_linear_residual,
        gate_dropout=cfg.gate_dropout,
        linear_dropout=cfg.linear_dropout,
        use_positive_ovr=cfg.use_positive_ovr,
        use_prototype_residual=cfg.use_prototype_residual,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args_dict = vars(args).copy()
    args_dict["stratify_val"] = not args_dict.pop("no_stratify_val")
    cfg = Config(**args_dict)

    if (cfg.val_features is None) != (cfg.val_metadata is None):
        raise ValueError("Provide both --val-features and --val-metadata, or neither.")
    for value in (cfg.train_features, cfg.train_metadata):
        if not Path(value).is_file():
            raise FileNotFoundError(f"Input not found: {value}. See data/README.md or pass explicit paths.")
    if cfg.epochs < 1 or cfg.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")
    set_seed(cfg.seed)
    task = parse_class_order_to_task(cfg.class_order)
    os.makedirs(cfg.outdir, exist_ok=True)

    save_json({**asdict(cfg), "model_name": "pref_ft", "binary_positive_class": task.class_names[1], "binary_negative_class": task.class_names[0]}, os.path.join(cfg.outdir, "config.json"))
    save_json({"class_names": task.class_names, "label_to_id": task.label_to_id, "id_to_label": task.id_to_label}, os.path.join(cfg.outdir, "label_mapping.json"))

    train_spec = DatasetSpec(
        name="train_cohort",
        features=cfg.train_features,
        metadata=cfg.train_metadata,
        matrix_orientation=cfg.matrix_orientation,
        sample_col=cfg.sample_col,
        label_col=cfg.label_col,
    )
    train_pack = load_dataset_as_arrays(train_spec, task=task)
    label_counts = np.bincount(train_pack["y"], minlength=2)
    missing = [task.id_to_label[i] for i, c in enumerate(label_counts) if c == 0]
    if missing:
        raise ValueError(f"Training cohort misses required class(es): {missing}")

    if cfg.val_features is not None and cfg.val_metadata is not None:
        val_spec = DatasetSpec(
            name="explicit_val",
            features=cfg.val_features,
            metadata=cfg.val_metadata,
            matrix_orientation=cfg.matrix_orientation,
            sample_col=cfg.sample_col,
            label_col=cfg.label_col,
        )
        val_pack = load_dataset_as_arrays(
            val_spec,
            task=task,
            expected_feature_order=train_pack["feature_names"],
            external_feature_policy=cfg.external_feature_policy,
            min_feature_overlap_ratio=cfg.min_feature_overlap_ratio,
        )
        X_train_raw, y_train, train_ids = train_pack["X_raw"], train_pack["y"], train_pack["sample_ids"]
        X_val_raw, y_val, val_ids = val_pack["X_raw"], val_pack["y"], val_pack["sample_ids"]
        val_overlap_info, val_filter_info = val_pack["overlap_info"], val_pack["filter_info"]
    else:
        split = split_train_val(train_pack["X_raw"], train_pack["y"], train_pack["sample_ids"], cfg)
        X_train_raw, y_train, train_ids = split["X_train_raw"], split["y_train"], split["train_ids"]
        X_val_raw, y_val, val_ids = split["X_val_raw"], split["y_val"], split["val_ids"]
        val_overlap_info, val_filter_info = None, None
        pd.DataFrame({
            "sample_id": np.concatenate([train_ids, val_ids]),
            "split": ["train"] * len(train_ids) + ["val"] * len(val_ids),
            "label_id": np.concatenate([y_train, y_val]),
            "label_name": inverse_transform_labels(np.concatenate([y_train, y_val]), task),
        }).to_csv(os.path.join(cfg.outdir, "train_val_split_samples.csv"), index=False)

    save_json(train_pack["filter_info"], os.path.join(cfg.outdir, "train_label_filter_info.json"))
    pd.DataFrame({"feature_name": train_pack["feature_names"]}).to_csv(os.path.join(cfg.outdir, "feature_order.csv"), index=False)

    preproc = fit_preprocessor_on_train(X_train_raw)
    X_train = preproc["X_train_scaled"]
    X_val = transform_with_preprocessor(X_val_raw, preproc)
    np.save(os.path.join(cfg.outdir, "train_feature_medians.npy"), preproc["train_medians"])
    np.save(os.path.join(cfg.outdir, "scaler_mean.npy"), preproc["scaler"].mean_.astype(np.float32))
    np.save(os.path.join(cfg.outdir, "scaler_scale.npy"), preproc["scaler"].scale_.astype(np.float32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, num_features=X_train.shape[1]).to(device)
    class_weights = torch.tensor(compute_class_weights(y_train), dtype=torch.float32, device=device) if cfg.use_class_weight else None
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    early = EarlyStopping(patience=cfg.patience)

    train_loader = build_loader(X_train, y_train, cfg, shuffle=True)
    val_loader = build_loader(X_val, y_val, cfg, shuffle=False)

    history = []
    best_metric_name = "none"
    for epoch in range(1, cfg.epochs + 1):
        train_metrics, _, _, _ = run_epoch(model, train_loader, criterion, device, cfg.amp, optimizer=optimizer, max_grad_norm=cfg.max_grad_norm)
        val_metrics, _, _, _ = predict(model, val_loader, criterion, device, cfg.amp)
        scheduler.step()

        metric_name, score = select_validation_score(val_metrics)
        best_metric_name = metric_name
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row["val_selection_metric"] = metric_name
        row["val_selection_score"] = score
        history.append(row)
        print(f"Epoch {epoch:03d} | train_loss={train_metrics['loss']:.4f} | val_{metric_name}={score:.4f} | val_auc={val_metrics.get('auc_binary', np.nan):.4f}")

        if early.step(score, model):
            print(f"Early stopping at epoch {epoch}.")
            break

    if early.best_state is not None:
        model.load_state_dict(early.best_state)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(cfg.outdir, "history.csv"), index=False)
    plot_history(hist_df, cfg.outdir)
    torch.save(model.state_dict(), os.path.join(cfg.outdir, "best_model.pt"))

    train_final, train_probs, train_preds, train_true = predict(model, build_loader(X_train, y_train, cfg, shuffle=False), criterion, device, cfg.amp)
    val_final, val_probs, val_preds, val_true = predict(model, val_loader, criterion, device, cfg.amp)
    save_evaluation_outputs(os.path.join(cfg.outdir, "eval_train"), "train", task.class_names, train_final, train_probs, train_preds, train_true, train_ids, None, train_pack["filter_info"])
    save_evaluation_outputs(os.path.join(cfg.outdir, "eval_val"), "val", task.class_names, val_final, val_probs, val_preds, val_true, val_ids, val_overlap_info, val_filter_info)

    external_rows = []
    for spec in load_external_test_specs(cfg.external_tests_json, cfg):
        pack = load_dataset_as_arrays(
            spec,
            task=task,
            expected_feature_order=train_pack["feature_names"],
            external_feature_policy=cfg.external_feature_policy,
            min_feature_overlap_ratio=cfg.min_feature_overlap_ratio,
        )
        X_ext = transform_with_preprocessor(pack["X_raw"], preproc)
        ext_loader = build_loader(X_ext, pack["y"], cfg, shuffle=False)
        ext_metrics, ext_probs, ext_preds, ext_true = predict(model, ext_loader, criterion, device, cfg.amp)
        save_evaluation_outputs(os.path.join(cfg.outdir, "external_tests", spec.name), f"external_test__{spec.name}", task.class_names, ext_metrics, ext_probs, ext_preds, ext_true, pack["sample_ids"], pack["overlap_info"], pack["filter_info"])
        external_rows.append(make_external_summary_row(spec.name, pack["y"], ext_metrics, pack))

    pd.DataFrame(external_rows).to_csv(os.path.join(cfg.outdir, "external_test_summary.csv"), index=False)
    final_metric_name, final_score = select_validation_score(val_final)
    summary = {
        "task_type": task.task_type,
        "class_names": task.class_names,
        "binary_positive_class": task.class_names[1],
        "binary_negative_class": task.class_names[0],
        "model_name": "pref_ft",
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_features": int(X_train.shape[1]),
        "best_val_selection_metric": best_metric_name,
        "best_val_selection_score": to_jsonable(early.best),
        "final_val_selection_metric": final_metric_name,
        "final_val_selection_score": to_jsonable(final_score),
        "selection_rule": "binary: auc_binary -> f1 -> bal_acc",
        "train_metrics": metrics_to_jsonable(train_final),
        "val_metrics": metrics_to_jsonable(val_final),
        "n_external_tests": int(len(external_rows)),
        "external_tests": external_rows,
    }
    save_json(summary, os.path.join(cfg.outdir, "summary.json"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# ============================================================
# CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PREF-FT training for binary DNA methylation classification.")
    parser.add_argument("--train-features", type=str, default=Config.train_features)
    parser.add_argument("--train-metadata", type=str, default=Config.train_metadata)
    parser.add_argument("--val-features", type=str, default=None)
    parser.add_argument("--val-metadata", type=str, default=None)
    parser.add_argument("--external-tests-json", type=str, default=Config.external_tests_json)
    parser.add_argument("--outdir", type=str, default=Config.outdir)
    parser.add_argument("--matrix-orientation", type=str, default=Config.matrix_orientation, choices=["sample_rows", "cpg_rows"])
    parser.add_argument("--sample-col", type=str, default=Config.sample_col)
    parser.add_argument("--label-col", type=str, default=Config.label_col)
    parser.add_argument("--class-order", type=str, default=Config.class_order)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--val-size", type=float, default=Config.val_size)
    parser.add_argument("--no-stratify-val", action="store_true")
    parser.add_argument("--external-feature-policy", type=str, default=Config.external_feature_policy, choices=["fill_train_median", "strict"])
    parser.add_argument("--min-feature-overlap-ratio", type=float, default=Config.min_feature_overlap_ratio)

    parser.add_argument("--d-model", type=int, default=Config.d_model)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument("--n-transformer-layers", type=int, default=Config.n_transformer_layers)
    parser.add_argument("--num-heads", type=int, default=Config.num_heads)
    parser.add_argument("--ensemble-members", "--tabm-num-members", dest="tabm_num_members", type=int, default=Config.tabm_num_members)
    parser.add_argument("--disable-linear-residual", dest="use_linear_residual", action="store_false")
    parser.set_defaults(use_linear_residual=Config.use_linear_residual)
    parser.add_argument("--gate-dropout", type=float, default=Config.gate_dropout)
    parser.add_argument("--linear-dropout", type=float, default=Config.linear_dropout)
    parser.add_argument("--disable-positive-evidence", "--disable-positive-ovr", dest="use_positive_ovr", action="store_false")
    parser.set_defaults(use_positive_ovr=Config.use_positive_ovr)
    parser.add_argument("--disable-prototype-residual", dest="use_prototype_residual", action="store_false")
    parser.set_defaults(use_prototype_residual=Config.use_prototype_residual)

    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--patience", type=int, default=Config.patience)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=Config.max_grad_norm)
    parser.add_argument("--label-smoothing", type=float, default=Config.label_smoothing)
    parser.add_argument("--use-class-weight", action="store_true")
    parser.add_argument("--save-logits", action="store_true")
    return parser


if __name__ == "__main__":
    main()
