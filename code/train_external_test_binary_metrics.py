#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train on one dataset, validate on an internal split or an explicit validation set,
and independently test the best checkpoint on one or more external datasets.

Updated version
---------------
This file keeps the original training / validation / external-test workflow, but
uses task-aware metrics:

1) If --class-order contains 2 labels, run one binary classification task.
   - class_names[0] is treated as the negative class.
   - class_names[1] is treated as the positive class.
   - Example: --class-order "unaffected, affected"
       unaffected -> 0 -> negative
       affected   -> 1 -> positive
   - Main metrics are computed from the direct 2x2 confusion matrix:
       ACC, Balanced ACC, Sensitivity, Specificity, Precision/PPV, NPV,
       F1, MCC, ROC-AUC, PR-AUC/AUPRC.
   - ROC-AUC / AUPRC are computed using prob[:, 1], i.e. the positive-class
     probability.

2) If --class-order contains 3 labels, run one 3-class classification task.
   - The original macro / weighted / one-vs-rest metrics are retained.

3) Label filtering is applied consistently to train / val / external cohorts.
   Samples whose labels are not in class-order are discarded.

4) HFR-style models / hierarchical loss are only enabled for a 3-class HC/MCI/AD
   compatible task. For binary tasks they are disabled or rejected with a clear
   error.
"""

import os
import json
import copy
import random
import argparse
from pathlib import Path
import warnings
import pickle
import sys

# -----------------------------------------------------------------------------
# XGBoost / OpenMP static TLS compatibility
# -----------------------------------------------------------------------------
# On some Linux/conda systems, importing xgboost after PyTorch/other OpenMP-heavy
# libraries can fail with: "libgomp... cannot allocate memory in static TLS block".
# If this run requests --model-name xgboost, import xgboost as early as possible,
# before numpy/pandas/sklearn/torch are imported.
def _argv_model_name(argv):
    for i, arg in enumerate(argv):
        if arg in {"--model-name", "--model_name"} and i + 1 < len(argv):
            return argv[i + 1].strip().lower()
        if arg.startswith("--model-name="):
            return arg.split("=", 1)[1].strip().lower()
        if arg.startswith("--model_name="):
            return arg.split("=", 1)[1].strip().lower()
    return None


_EARLY_XGB_IMPORT_ERROR = None
if _argv_model_name(sys.argv) == "xgboost":
    try:
        import xgboost as _xgboost_early  # noqa: F401
    except Exception as _e:
        _EARLY_XGB_IMPORT_ERROR = _e

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from methylation_model_zoo_dg_methods_extand import build_model
except ImportError:
    from methylation_model_zoo_dg_methods import build_model


# ============================================================
# Constants
# ============================================================

HFR_MODEL_NAMES = {
    "hfr_net",
    "rank_hfr_mlp",
    "sparse_rank_hfr_mlp",
    "linear_residual_rank_hfr_mlp",
    "cross_cohort_linear_residual_rank_hfr_mlp",
    "cross_cohort_dg_linear_residual_rank_hfr_mlp",
}

HIERARCHICAL_AUX_MODEL_NAMES = HFR_MODEL_NAMES | {
    "ft_transformer",
    "ft",
    "modernnca",
    "modern_nca",
    "nca",
    "tabm",
    "tabular_mixup",
    "dann_ft",
    "dann_ft_transformer",
    "dann+ft",
    "enhanced_dann_ft_transformer",
    "dann_ft_transformer_plus",
    "stable_dann_ft_transformer",
    "enhanced_ft_transformer",
    "pref_ft",
    "binary_positive_aware_ft",
    "binary_positive_aware_ft_2",
    "ft_transformer_plus",
    "stable_ft_transformer",
    "stable_mlp",
    "stable_methyl_res_mlp",
    "stable_methylation_res_mlp",
}

MIXUP_MODEL_NAMES = {"tabm", "tabular_mixup"}
FOUNDATION_MODEL_NAMES = {"tabpfn", "tabicl"}
TABPFN_CONSERVATIVE_MAX_FEATURES = 100

SKLEARN_MODEL_NAMES = {
    "elastic_net_lr",
    "random_forest",
    "extra_trees",
    "svm_rbf",
    "xgboost",
    "lightgbm",
}

CANONICAL_HFR_CLASSES = [
    "healthy control",
    "Mild Cognitive Impairment",
    "Alzheimer's disease",
]

HFR_ROLE_ORDER = ("hc", "mci", "ad")

HFR_LABEL_ROLE_ALIASES = {
    "hc": "hc",
    "cn": "hc",
    "control": "hc",
    "controls": "hc",
    "healthy": "hc",
    "healthy control": "hc",
    "healthy controls": "hc",
    "normal": "hc",
    "normal control": "hc",
    "normal controls": "hc",
    "cognitively normal": "hc",
    "cognitively normal control": "hc",
    "cognitively normal controls": "hc",
    "unaffected": "hc",

    "mci": "mci",
    "mild cognitive impairment": "mci",
    "mild cognitive impaired": "mci",
    "mild cognitive impairment case": "mci",

    "ad": "ad",
    "alzheimers disease": "ad",
    "alzheimer disease": "ad",
    "alzheimer's disease": "ad",
    "alzheimer’s disease": "ad",
    "alzheimer dementia": "ad",
    "dementia due to alzheimer disease": "ad",
}


def normalize_label_key(x: Any) -> str:
    """Normalize label text for robust alias matching."""
    s = str(x).strip().lower()
    s = s.replace("\ufeff", "")
    s = s.replace("’", "'").replace("`", "'")
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    return s


def resolve_hfr_role(label: Any) -> Optional[str]:
    key = normalize_label_key(label)
    if key in HFR_LABEL_ROLE_ALIASES:
        return HFR_LABEL_ROLE_ALIASES[key]
    key_no_apostrophe = key.replace("'", "")
    return HFR_LABEL_ROLE_ALIASES.get(key_no_apostrophe)


def is_hfr_compatible_3way_task(task: "TaskDefinition") -> bool:
    """Return True when the 3 task labels can be resolved to HC, MCI and AD."""
    if task.num_classes != 3:
        return False
    try:
        infer_hfr_indices(task.class_names)
        return True
    except ValueError:
        return False


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    # main train cohort
    train_features: str
    train_metadata: str

    # optional explicit validation cohort
    val_features: Optional[str] = None
    val_metadata: Optional[str] = None

    # optional external tests json
    external_tests_json: Optional[str] = None

    outdir: str = "runs_external_test_framework"
    matrix_orientation: str = "sample_rows"  # sample_rows or cpg_rows
    sample_col: str = "sample_id"
    label_col: str = "label"
    class_order: str = "unaffected, affected"

    # split / reproducibility
    seed: int = 42
    val_size: float = 0.15
    stratify_val: bool = True

    # external feature alignment
    external_feature_policy: str = "strict"
    min_feature_overlap_ratio: float = 0.98

    # model
    model_name: str = "hybrid_ms_mamba"
    d_model: int = 128
    dropout: float = 0.25
    patch_scales: str = "4,16,64"
    n_mamba_layers: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2

    # advanced tabular model parameters
    n_transformer_layers: int = 3
    num_heads: int = 4
    num_prototypes_per_class: int = 4
    tabm_num_members: int = 4
    use_linear_residual: bool = True

    # DANN / domain adversarial training
    domain_col: Optional[str] = None
    use_dann_loss: bool = False
    dann_weight: float = 0.05
    grl_lambda: float = 0.1
    num_domains: int = 1

    # DG / DA methods
    dg_method: str = "erm"
    dg_weight: float = 0.1
    groupdro_eta: float = 0.01
    irm_weight: float = 0.1
    vrex_weight: float = 0.1
    coral_weight: float = 0.1
    mmd_weight: float = 0.1
    cdan_weight: float = 0.1
    cdan_hidden_dim: int = 128

    # pseudo-label self-training
    pseudo_unlabeled_json: Optional[str] = None
    pseudo_threshold: float = 0.9
    pseudo_weight: float = 0.2
    pseudo_warmup_epochs: int = 10
    pseudo_max_steps_per_epoch: int = 0

    # mixup
    mixup_alpha: float = 0.0

    # Foundation / in-context model branches
    tabpfn_max_features: int = 100
    tabpfn_device: str = "auto"
    tabicl_max_features: int = 512
    tabicl_device: str = "auto"
    tabicl_n_estimators: int = 8
    tabicl_batch_size: int = 8
    tabicl_kv_cache: bool = False

    # TabR / ExcelFormer PyTorch baselines
    tabr_k_neighbors: int = 8

    # classical sklearn / GBDT model parameters
    enet_C: float = 1.0
    enet_l1_ratio: float = 0.5
    enet_max_iter: int = 5000
    n_estimators: int = 500
    max_depth: int = 0
    min_samples_leaf: int = 1
    max_features: str = "sqrt"
    svm_C: float = 1.0
    svm_gamma: str = "scale"
    gb_learning_rate: float = 0.03
    gb_max_depth: int = 6
    gb_num_leaves: int = 31
    gb_subsample: float = 0.9
    gb_colsample_bytree: float = 0.8
    gb_min_child_samples: int = 20
    gb_reg_lambda: float = 1.0

    # optimization
    epochs: int = 150
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    patience: int = 25
    amp: bool = True
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.05

    # loss tricks
    use_class_weight: bool = False
    use_ordinal_loss: bool = False
    ordinal_weight: float = 0.3

    use_ovr_loss: bool = False
    ovr_weight: float = 0.3

    use_hierarchical_loss: bool = False
    hier_stage1_weight: float = 0.6
    hier_stage2_weight: float = 0.9
    hier_aux3_weight: float = 0.2

    use_enet_reg: bool = False
    enet_l1_weight: float = 1e-4
    enet_l2_weight: float = 1e-4

    use_linear_residual_reg: bool = False
    linear_l1_weight: float = 1e-5
    linear_l2_weight: float = 1e-4

    # logging
    save_logits: bool = True


@dataclass
class DatasetSpec:
    name: str
    features: str
    metadata: str
    matrix_orientation: str = "sample_rows"
    sample_col: str = "sample_id"
    label_col: str = "label"
    label_map: Optional[Dict[str, str]] = None
    domain_col: Optional[str] = None


@dataclass
class TaskDefinition:
    class_names: List[str]
    label_to_id: Dict[str, int]
    id_to_label: Dict[int, str]
    num_classes: int
    task_type: str


# ============================================================
# Reproducibility / basic utils
# ============================================================


def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
        except Exception:
            pass


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


def parse_patch_scales(x: str) -> Tuple[int, ...]:
    vals = tuple(int(v.strip()) for v in x.split(",") if v.strip())
    if len(vals) == 0:
        raise ValueError("patch_scales cannot be empty")
    return vals


def safe_metric_name(name: str) -> str:
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace("'", "")
    )


def compute_class_weights_from_y(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def make_ordinal_targets(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    thresholds = torch.arange(num_classes - 1, device=y.device).unsqueeze(0)
    return (y.unsqueeze(1) > thresholds).float()


def parse_class_order_to_task(class_order: str) -> TaskDefinition:
    class_names = [x.strip() for x in class_order.split(",") if x.strip()]
    if len(class_names) not in (2, 3):
        raise ValueError(
            f"class-order must contain exactly 2 or 3 labels, got {len(class_names)}: {class_names}"
        )
    if len(set(class_names)) != len(class_names):
        raise ValueError(f"class-order contains duplicate labels: {class_names}")

    label_to_id = {name: idx for idx, name in enumerate(class_names)}
    id_to_label = {idx: name for name, idx in label_to_id.items()}
    task_type = "binary" if len(class_names) == 2 else "multiclass_3way"
    return TaskDefinition(
        class_names=class_names,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        num_classes=len(class_names),
        task_type=task_type,
    )


def inverse_transform_labels(y: np.ndarray, task: TaskDefinition) -> List[str]:
    return [task.id_to_label[int(i)] for i in y]


def _to_jsonable_value(v: Any):
    """Convert numpy scalar / NaN values into JSON-friendly values."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v) if pd.notna(v) else None
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    if pd.isna(v) if np.isscalar(v) else False:
        return None
    return v


def _metrics_to_jsonable(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _to_jsonable_value(v) for k, v in metrics.items()}


# ============================================================
# Dataset loading / feature alignment
# ============================================================


def canonicalize_label_series(series: pd.Series, label_map: Optional[Dict[str, str]] = None) -> pd.Series:
    out = series.astype(str).copy()
    if label_map is not None:
        out = out.map(lambda x: label_map.get(x, x))
    return out.astype(str)


def load_matrix_and_metadata(spec: DatasetSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feat = read_table(spec.features)
    meta = read_table(spec.metadata)
    meta = ensure_sample_id_column(meta, spec.sample_col)

    if spec.label_col not in meta.columns:
        raise ValueError(
            f"[{spec.name}] label_col={spec.label_col} not found in metadata columns: {meta.columns.tolist()}"
        )
    if spec.domain_col is not None and spec.domain_col not in meta.columns:
        raise ValueError(
            f"[{spec.name}] domain_col={spec.domain_col} not found in metadata columns: {meta.columns.tolist()}"
        )

    meta[spec.sample_col] = meta[spec.sample_col].astype(str)
    meta[spec.label_col] = canonicalize_label_series(meta[spec.label_col], spec.label_map)
    return feat, meta


def matrix_to_sample_feature_df(
    feat: pd.DataFrame,
    meta: pd.DataFrame,
    matrix_orientation: str,
    sample_col: str,
    dataset_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Convert any supported feature matrix into a sample x feature dataframe with a sample column.
    Return: feat_df, aligned_meta_df, feature_names
    """
    if matrix_orientation == "sample_rows":
        feat = ensure_sample_id_column(feat, sample_col)
        feat[sample_col] = feat[sample_col].astype(str)
        common = [s for s in meta[sample_col].tolist() if s in set(feat[sample_col])]
        if len(common) == 0:
            raise ValueError(f"[{dataset_name}] No overlapping samples between features and metadata.")
        feat = feat[feat[sample_col].isin(common)].copy()
        meta = meta[meta[sample_col].isin(common)].copy()
        feat = feat.set_index(sample_col).loc[common].reset_index()
        meta = meta.set_index(sample_col).loc[common].reset_index()
        feature_cols = [c for c in feat.columns if c != sample_col]
        for c in feature_cols:
            feat[c] = pd.to_numeric(feat[c], errors="coerce")
        return feat, meta, feature_cols

    if matrix_orientation == "cpg_rows":
        feat = feat.copy()
        feat.iloc[:, 0] = feat.iloc[:, 0].astype(str)
        cpg_col = feat.columns[0]
        feat = feat.set_index(cpg_col)
        feat.columns = feat.columns.astype(str)
        common = [s for s in meta[sample_col].tolist() if s in set(feat.columns)]
        if len(common) == 0:
            raise ValueError(f"[{dataset_name}] No overlapping samples between cpg_rows matrix and metadata.")
        feat = feat.loc[:, common].T.reset_index().rename(columns={"index": sample_col})
        meta = meta[meta[sample_col].isin(common)].copy()
        feat = feat.set_index(sample_col).loc[common].reset_index()
        meta = meta.set_index(sample_col).loc[common].reset_index()
        feature_cols = [c for c in feat.columns if c != sample_col]
        for c in feature_cols:
            feat[c] = pd.to_numeric(feat[c], errors="coerce")
        return feat, meta, feature_cols

    raise ValueError("matrix_orientation must be 'sample_rows' or 'cpg_rows'")


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

    filter_info = {
        "n_samples_before_label_filter": int(len(meta_df)),
        "n_samples_after_label_filter": int(len(meta_kept)),
        "n_samples_dropped_by_label_filter": int((~keep_mask).sum()),
        "allowed_labels": list(task.class_names),
        "dropped_label_counts": labels_before.loc[~keep_mask].value_counts().to_dict(),
        "kept_label_counts_raw": labels_before.loc[keep_mask].value_counts().to_dict(),
    }

    if len(meta_kept) == 0:
        raise ValueError(
            f"[{dataset_name}] No samples remain after filtering by class-order labels {task.class_names}."
        )

    feat_kept = feat_kept.set_index(sample_col).loc[meta_kept[sample_col].tolist()].reset_index()
    return feat_kept, meta_kept, filter_info


def encode_labels(meta: pd.DataFrame, label_col: str, task: TaskDefinition, dataset_name: str) -> np.ndarray:
    labels = meta[label_col].astype(str).values.tolist()
    unknown = sorted(list(set(labels) - set(task.class_names)))
    if len(unknown) > 0:
        raise ValueError(
            f"[{dataset_name}] labels contain unseen classes after filtering {unknown}. Expected classes={task.class_names}"
        )
    return np.array([task.label_to_id[x] for x in labels], dtype=np.int64)


def load_dataset_as_arrays(
    spec: DatasetSpec,
    task: TaskDefinition,
    expected_feature_order: Optional[List[str]] = None,
    external_feature_policy: str = "strict",
    min_feature_overlap_ratio: float = 0.98,
) -> Dict[str, Any]:
    """
    Returns a dict with keys:
      X_raw, y, sample_ids, feature_names, domain_values, overlap_info, filter_info
    If expected_feature_order is given, features are aligned to that order.
    """
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
    domain_values = None
    if spec.domain_col is not None:
        domain_values = meta_df[spec.domain_col].astype(str).values

    numeric_df = feat_df[feature_cols].copy()
    X_raw = numeric_df.values.astype(np.float32)

    overlap_info = {
        "n_raw_features": int(len(feature_cols)),
        "n_overlap_with_train": None,
        "overlap_ratio_against_train": None,
        "missing_train_features": None,
        "extra_external_features": None,
    }

    if expected_feature_order is None:
        return {
            "X_raw": X_raw,
            "y": y,
            "sample_ids": sample_ids,
            "feature_names": feature_cols,
            "domain_values": domain_values,
            "overlap_info": overlap_info,
            "filter_info": filter_info,
        }

    aligned_df = numeric_df.reindex(columns=expected_feature_order)
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

    if external_feature_policy == "strict" and len(missing) > 0:
        raise ValueError(
            f"[{spec.name}] strict feature policy failed: {len(missing)} training features are missing."
        )

    if overlap_ratio < min_feature_overlap_ratio:
        raise ValueError(
            f"[{spec.name}] feature overlap ratio too low: {overlap_ratio:.4f} < {min_feature_overlap_ratio:.4f}."
        )

    X_raw = aligned_df.values.astype(np.float32)
    return {
        "X_raw": X_raw,
        "y": y,
        "sample_ids": sample_ids,
        "feature_names": list(expected_feature_order),
        "domain_values": domain_values,
        "overlap_info": overlap_info,
        "filter_info": filter_info,
    }


# ============================================================
# Preprocessing
# ============================================================


def fit_preprocessor_on_train(X_train_raw: np.ndarray) -> Dict[str, Any]:
    """Fit median imputation + standard scaler on train only."""
    train_medians = np.nanmedian(X_train_raw, axis=0)
    train_medians = np.where(np.isnan(train_medians), 0.0, train_medians).astype(np.float32)

    X_train_imp = X_train_raw.copy()
    inds = np.where(np.isnan(X_train_imp))
    if len(inds[0]) > 0:
        X_train_imp[inds] = np.take(train_medians, inds[1])

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train_imp).astype(np.float32)

    return {
        "train_medians": train_medians,
        "scaler": scaler,
        "X_train_scaled": X_train_scaled,
    }


def transform_with_preprocessor(X_raw: np.ndarray, preproc: Dict[str, Any]) -> np.ndarray:
    X_imp = X_raw.copy()
    inds = np.where(np.isnan(X_imp))
    if len(inds[0]) > 0:
        X_imp[inds] = np.take(preproc["train_medians"], inds[1])
    X_scaled = preproc["scaler"].transform(X_imp).astype(np.float32)
    return X_scaled


# ============================================================
# DataLoader
# ============================================================


class MethylationDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, domains: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.domains = None if domains is None else torch.tensor(domains, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.domains is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx], self.domains[idx]


def _worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(X, y, cfg: Config, shuffle: bool, domains: Optional[np.ndarray] = None):
    ds = MethylationDataset(X, y, domains=domains)
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=g,
        worker_init_fn=_worker_init_fn,
    )


# ============================================================
# Loss / metrics
# ============================================================


class CompositeCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        use_ordinal_loss: bool = False,
        ordinal_weight: float = 0.3,
        use_ovr_loss: bool = False,
        ovr_weight: float = 0.3,
        use_hierarchical_loss: bool = False,
        hier_stage1_weight: float = 0.6,
        hier_stage2_weight: float = 0.9,
        hier_aux3_weight: float = 0.2,
        use_enet_reg: bool = False,
        enet_l1_weight: float = 1e-4,
        enet_l2_weight: float = 1e-4,
        use_linear_residual_reg: bool = False,
        linear_l1_weight: float = 1e-5,
        linear_l2_weight: float = 1e-4,
        hierarchical_class_indices: Optional[Dict[str, int]] = None,
        use_dann_loss: bool = False,
        dann_weight: float = 0.05,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_ordinal_loss = use_ordinal_loss
        self.ordinal_weight = ordinal_weight
        self.use_ovr_loss = use_ovr_loss
        self.ovr_weight = ovr_weight
        self.use_hierarchical_loss = use_hierarchical_loss
        self.hier_stage1_weight = hier_stage1_weight
        self.hier_stage2_weight = hier_stage2_weight
        self.hier_aux3_weight = hier_aux3_weight
        self.hierarchical_class_indices = hierarchical_class_indices
        self.use_enet_reg = use_enet_reg
        self.enet_l1_weight = enet_l1_weight
        self.enet_l2_weight = enet_l2_weight
        self.use_linear_residual_reg = use_linear_residual_reg
        self.linear_l1_weight = linear_l1_weight
        self.linear_l2_weight = linear_l2_weight
        self.use_dann_loss = use_dann_loss
        self.dann_weight = dann_weight

        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.bce = nn.BCEWithLogitsLoss()
        self.domain_ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, y: torch.Tensor, aux: Optional[Dict] = None, domains: Optional[torch.Tensor] = None):
        loss_main = self.ce(logits, y)
        total_loss = loss_main
        loss_dict = {"loss_main": loss_main.detach().item()}

        if self.use_ordinal_loss and aux is not None and "ordinal_logits" in aux:
            ordinal_logits = aux["ordinal_logits"]
            ordinal_targets = make_ordinal_targets(y, self.num_classes)
            loss_ord = self.bce(ordinal_logits, ordinal_targets)
            total_loss = total_loss + self.ordinal_weight * loss_ord
            loss_dict["loss_ordinal"] = loss_ord.detach().item()
        else:
            loss_dict["loss_ordinal"] = 0.0

        if self.use_ovr_loss and aux is not None and "ovr_logits" in aux:
            ovr_logits = aux["ovr_logits"]
            ovr_targets = F.one_hot(y, num_classes=self.num_classes).float()
            loss_ovr = self.bce(ovr_logits, ovr_targets)
            total_loss = total_loss + self.ovr_weight * loss_ovr
            loss_dict["loss_ovr"] = loss_ovr.detach().item()
        else:
            loss_dict["loss_ovr"] = 0.0

        if self.use_hierarchical_loss and aux is not None:
            if self.hierarchical_class_indices is None:
                raise ValueError("hierarchical_class_indices is required when use_hierarchical_loss=True")

            idx_ad = self.hierarchical_class_indices["idx_ad"]
            idx_mci = self.hierarchical_class_indices["idx_mci"]
            idx_hc = self.hierarchical_class_indices["idx_hc"]
            zero = logits.new_tensor(0.0)

            if "hc_vs_disease_logits" in aux:
                stage1_logits = aux["hc_vs_disease_logits"].to(logits.device).view(-1)
                target_stage1 = (y != idx_hc).float()
                loss_stage1 = self.bce(stage1_logits, target_stage1)
            else:
                loss_stage1 = zero

            if "ad_vs_mci_logits" in aux:
                stage2_logits = aux["ad_vs_mci_logits"].to(logits.device).view(-1)
                mask = (y == idx_ad) | (y == idx_mci)
                if mask.any():
                    target_stage2 = (y[mask] == idx_ad).float()
                    loss_stage2 = self.bce(stage2_logits[mask], target_stage2)
                else:
                    loss_stage2 = zero
            else:
                loss_stage2 = zero

            if "aux3_logits" in aux:
                aux3_logits = aux["aux3_logits"].to(logits.device)
                loss_aux3 = self.ce(aux3_logits, y)
            else:
                loss_aux3 = zero

            total_loss = (
                total_loss
                + self.hier_stage1_weight * loss_stage1
                + self.hier_stage2_weight * loss_stage2
                + self.hier_aux3_weight * loss_aux3
            )
            loss_dict["loss_hier_stage1"] = loss_stage1.detach().item()
            loss_dict["loss_hier_stage2"] = loss_stage2.detach().item()
            loss_dict["loss_hier_aux3"] = loss_aux3.detach().item()
        else:
            loss_dict["loss_hier_stage1"] = 0.0
            loss_dict["loss_hier_stage2"] = 0.0
            loss_dict["loss_hier_aux3"] = 0.0

        if self.use_linear_residual_reg and aux is not None:
            zero = logits.new_tensor(0.0)
            loss_linear_l1 = aux["linear_l1"] if "linear_l1" in aux else zero
            loss_linear_l2 = aux["linear_l2"] if "linear_l2" in aux else zero
            total_loss = total_loss + self.linear_l1_weight * loss_linear_l1 + self.linear_l2_weight * loss_linear_l2
            loss_dict["loss_linear_l1"] = float(loss_linear_l1.detach().item())
            loss_dict["loss_linear_l2"] = float(loss_linear_l2.detach().item())
        else:
            loss_dict["loss_linear_l1"] = 0.0
            loss_dict["loss_linear_l2"] = 0.0

        if self.use_dann_loss and aux is not None and domains is not None and "domain_logits" in aux:
            domain_logits = aux["domain_logits"].to(logits.device)
            loss_domain = self.domain_ce(domain_logits, domains.to(logits.device))
            total_loss = total_loss + self.dann_weight * loss_domain
            loss_dict["loss_domain"] = loss_domain.detach().item()
        else:
            loss_dict["loss_domain"] = 0.0

        loss_dict["loss_total"] = total_loss.detach().item()
        return total_loss, loss_dict


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or NaN when denominator is zero."""
    return float(numerator / denominator) if denominator != 0 else np.nan


def _binary_mcc_from_counts(tp: int, fp: int, tn: int, fn: int) -> float:
    """Matthews correlation coefficient for a one-vs-rest binary confusion matrix."""
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        return np.nan
    return float(((tp * tn) - (fp * fn)) / np.sqrt(denom))


def _binary_auc_ovr(y_bin: np.ndarray, score: np.ndarray) -> float:
    """One-vs-rest AUC. Return NaN if the current class has only positives or only negatives."""
    try:
        if len(np.unique(y_bin)) < 2:
            return np.nan
        return float(roc_auc_score(y_bin, score))
    except Exception:
        return np.nan


def _binary_auprc(y_bin: np.ndarray, score: np.ndarray) -> float:
    """Average precision / PR-AUC. Return NaN if only one class is present."""
    try:
        if len(np.unique(y_bin)) < 2:
            return np.nan
        return float(average_precision_score(y_bin, score))
    except Exception:
        return np.nan


def compute_per_class_metrics(y_true, prob, pred, class_names: List[str]) -> pd.DataFrame:
    """
    Compute detailed one-vs-rest metrics for every class.

    This remains useful for both binary and 3-class tasks. For binary main metrics,
    compute_metrics() additionally reports direct positive-class metrics where
    class_names[1] is treated as the positive class.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    prob = np.asarray(prob)
    n = int(len(y_true))

    rows = []
    for i, cls in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        p_bin = (pred == i).astype(int)

        tp = int(((y_bin == 1) & (p_bin == 1)).sum())
        fp = int(((y_bin == 0) & (p_bin == 1)).sum())
        tn = int(((y_bin == 0) & (p_bin == 0)).sum())
        fn = int(((y_bin == 1) & (p_bin == 0)).sum())

        acc = _safe_divide(tp + tn, n)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        sensitivity = recall
        specificity = _safe_divide(tn, tn + fp)
        f1 = _safe_divide(2 * precision * recall, precision + recall) if not (np.isnan(precision) or np.isnan(recall)) else np.nan
        mcc = _binary_mcc_from_counts(tp=tp, fp=fp, tn=tn, fn=fn)
        npv = _safe_divide(tn, tn + fn)
        fpr = _safe_divide(fp, fp + tn)
        fnr = _safe_divide(fn, fn + tp)
        fdr = _safe_divide(fp, fp + tp)
        prevalence = _safe_divide(tp + fn, n)
        pred_positive_rate = _safe_divide(tp + fp, n)
        balanced_acc = np.nanmean([sensitivity, specificity])
        youden_j = sensitivity + specificity - 1.0 if not (np.isnan(sensitivity) or np.isnan(specificity)) else np.nan
        auc_i = _binary_auc_ovr(y_bin, prob[:, i]) if prob.ndim == 2 and prob.shape[1] > i else np.nan
        auprc_i = _binary_auprc(y_bin, prob[:, i]) if prob.ndim == 2 and prob.shape[1] > i else np.nan

        rows.append({
            "class_id": int(i),
            "class_name": cls,
            "support": int(tp + fn),
            "n_negative": int(tn + fp),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "acc": float(acc) if not np.isnan(acc) else np.nan,
            "balanced_acc": float(balanced_acc) if not np.isnan(balanced_acc) else np.nan,
            "precision": float(precision) if not np.isnan(precision) else np.nan,
            "ppv": float(precision) if not np.isnan(precision) else np.nan,
            "recall": float(recall) if not np.isnan(recall) else np.nan,
            "sensitivity": float(sensitivity) if not np.isnan(sensitivity) else np.nan,
            "specificity": float(specificity) if not np.isnan(specificity) else np.nan,
            "f1": float(f1) if not np.isnan(f1) else np.nan,
            "mcc": float(mcc) if not np.isnan(mcc) else np.nan,
            "auc_ovr": float(auc_i) if not np.isnan(auc_i) else np.nan,
            "auprc_ovr": float(auprc_i) if not np.isnan(auprc_i) else np.nan,
            "npv": float(npv) if not np.isnan(npv) else np.nan,
            "fpr": float(fpr) if not np.isnan(fpr) else np.nan,
            "fnr": float(fnr) if not np.isnan(fnr) else np.nan,
            "fdr": float(fdr) if not np.isnan(fdr) else np.nan,
            "prevalence": float(prevalence) if not np.isnan(prevalence) else np.nan,
            "pred_positive_rate": float(pred_positive_rate) if not np.isnan(pred_positive_rate) else np.nan,
            "youden_j": float(youden_j) if not np.isnan(youden_j) else np.nan,
        })

    return pd.DataFrame(rows)


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, pred: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    """
    Compute task-aware metrics.

    Binary task:
        class_names[0] is negative and class_names[1] is positive.
        The main binary metrics are calculated directly from the 2x2 confusion
        matrix and positive-class probability prob[:, 1].

    Multiclass task:
        Keep macro / weighted / one-vs-rest evaluation.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    prob = np.asarray(prob)
    num_classes = len(class_names)

    per_class_df = compute_per_class_metrics(y_true, prob, pred, class_names)
    support = per_class_df["support"].to_numpy(dtype=float)
    support_sum = float(np.nansum(support))

    def nanmean_col(col: str) -> float:
        vals = per_class_df[col].to_numpy(dtype=float)
        if vals.size == 0 or np.isnan(vals).all():
            return np.nan
        return float(np.nanmean(vals))

    def weighted_col(col: str) -> float:
        vals = per_class_df[col].to_numpy(dtype=float)
        valid = ~np.isnan(vals)
        if support_sum == 0 or not valid.any():
            return np.nan
        return float(np.nansum(vals[valid] * support[valid]) / np.nansum(support[valid]))

    # ========================================================
    # Binary classification: direct positive-class metrics
    # ========================================================
    if num_classes == 2:
        neg_id, pos_id = 0, 1
        cm = confusion_matrix(y_true, pred, labels=[neg_id, pos_id])
        tn, fp, fn, tp = cm.ravel()
        n = int(tp + tn + fp + fn)

        acc = _safe_divide(tp + tn, n)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        sensitivity = recall
        specificity = _safe_divide(tn, tn + fp)
        npv = _safe_divide(tn, tn + fn)
        fpr = _safe_divide(fp, fp + tn)
        fnr = _safe_divide(fn, fn + tp)
        fdr = _safe_divide(fp, fp + tp)
        prevalence = _safe_divide(tp + fn, n)
        pred_positive_rate = _safe_divide(tp + fp, n)
        balanced_acc = np.nanmean([sensitivity, specificity])
        youden_j = sensitivity + specificity - 1.0 if not (np.isnan(sensitivity) or np.isnan(specificity)) else np.nan

        if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
            f1 = np.nan
        else:
            f1 = float(2 * precision * recall / (precision + recall))

        mcc = _binary_mcc_from_counts(tp=tp, fp=fp, tn=tn, fn=fn)

        y_bin = (y_true == pos_id).astype(int)
        if prob.ndim == 2 and prob.shape[1] > pos_id:
            pos_score = prob[:, pos_id]
        else:
            pos_score = np.full_like(y_bin, np.nan, dtype=float)

        auc_binary = _binary_auc_ovr(y_bin, pos_score)
        auprc_binary = _binary_auprc(y_bin, pos_score)

        return {
            "acc": float(acc) if not np.isnan(acc) else np.nan,
            "bal_acc": float(balanced_acc) if not np.isnan(balanced_acc) else np.nan,
            "balanced_acc": float(balanced_acc) if not np.isnan(balanced_acc) else np.nan,

            "precision": float(precision) if not np.isnan(precision) else np.nan,
            "ppv": float(precision) if not np.isnan(precision) else np.nan,
            "recall": float(recall) if not np.isnan(recall) else np.nan,
            "sensitivity": float(sensitivity) if not np.isnan(sensitivity) else np.nan,
            "specificity": float(specificity) if not np.isnan(specificity) else np.nan,
            "npv": float(npv) if not np.isnan(npv) else np.nan,

            "f1": float(f1) if not np.isnan(f1) else np.nan,
            "mcc": float(mcc) if not np.isnan(mcc) else np.nan,
            "auc_binary": float(auc_binary) if not np.isnan(auc_binary) else np.nan,
            "auprc_binary": float(auprc_binary) if not np.isnan(auprc_binary) else np.nan,

            "fpr": float(fpr) if not np.isnan(fpr) else np.nan,
            "fnr": float(fnr) if not np.isnan(fnr) else np.nan,
            "fdr": float(fdr) if not np.isnan(fdr) else np.nan,
            "prevalence": float(prevalence) if not np.isnan(prevalence) else np.nan,
            "pred_positive_rate": float(pred_positive_rate) if not np.isnan(pred_positive_rate) else np.nan,
            "youden_j": float(youden_j) if not np.isnan(youden_j) else np.nan,

            "tp": float(tp),
            "fp": float(fp),
            "tn": float(tn),
            "fn": float(fn),
            "positive_class_id": float(pos_id),
            "negative_class_id": float(neg_id),

            # Compatibility keys for existing summary / plotting code
            "macro_precision": nanmean_col("precision"),
            "macro_recall": nanmean_col("recall"),
            "macro_sensitivity": nanmean_col("sensitivity"),
            "macro_specificity": nanmean_col("specificity"),
            "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
            "macro_mcc_ovr": nanmean_col("mcc"),
            "macro_auc_ovr_from_per_class": nanmean_col("auc_ovr"),
            "macro_auprc_ovr_from_per_class": nanmean_col("auprc_ovr"),

            "weighted_precision": weighted_col("precision"),
            "weighted_recall": weighted_col("recall"),
            "weighted_sensitivity": weighted_col("sensitivity"),
            "weighted_specificity": weighted_col("specificity"),
            "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
            "weighted_mcc_ovr": weighted_col("mcc"),
            "weighted_auc_ovr_from_per_class": weighted_col("auc_ovr"),
            "weighted_auprc_ovr_from_per_class": weighted_col("auprc_ovr"),

            # Binary task does not use multiclass OVR-AUC as a main metric
            "auc_ovr_macro": np.nan,
            "auc_ovr_weighted": np.nan,
            "mcc_multiclass": float(matthews_corrcoef(y_true, pred)),
        }

    # ========================================================
    # 3-class / multiclass task: original macro/weighted metrics
    # ========================================================
    out = {
        "acc": float(accuracy_score(y_true, pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, pred)),

        "macro_precision": nanmean_col("precision"),
        "macro_recall": nanmean_col("recall"),
        "macro_sensitivity": nanmean_col("sensitivity"),
        "macro_specificity": nanmean_col("specificity"),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "macro_mcc_ovr": nanmean_col("mcc"),
        "macro_auc_ovr_from_per_class": nanmean_col("auc_ovr"),
        "macro_auprc_ovr_from_per_class": nanmean_col("auprc_ovr"),

        "weighted_precision": weighted_col("precision"),
        "weighted_recall": weighted_col("recall"),
        "weighted_sensitivity": weighted_col("sensitivity"),
        "weighted_specificity": weighted_col("specificity"),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "weighted_mcc_ovr": weighted_col("mcc"),
        "weighted_auc_ovr_from_per_class": weighted_col("auc_ovr"),
        "weighted_auprc_ovr_from_per_class": weighted_col("auprc_ovr"),

        "mcc_multiclass": float(matthews_corrcoef(y_true, pred)),

        # Generic aliases for easier downstream summary
        "precision": nanmean_col("precision"),
        "recall": nanmean_col("recall"),
        "sensitivity": nanmean_col("sensitivity"),
        "specificity": nanmean_col("specificity"),
        "f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),

        # Binary-only metrics
        "auc_binary": np.nan,
        "auprc_binary": np.nan,
    }

    try:
        out["auc_ovr_macro"] = float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
    except Exception:
        out["auc_ovr_macro"] = np.nan

    try:
        out["auc_ovr_weighted"] = float(roc_auc_score(y_true, prob, multi_class="ovr", average="weighted"))
    except Exception:
        out["auc_ovr_weighted"] = np.nan

    return out


def select_validation_score(metrics: Dict[str, float], task: TaskDefinition) -> Tuple[str, float]:
    """
    Choose validation score for early stopping.

    Binary:
        Prefer AUC, then F1, then balanced accuracy.

    3-class:
        Prefer macro-F1, then macro OVR-AUC, then balanced accuracy.
    """
    if task.num_classes == 2:
        candidates = ["auc_binary", "f1", "bal_acc"]
    else:
        candidates = ["macro_f1", "auc_ovr_macro", "bal_acc"]

    for name in candidates:
        val = metrics.get(name, np.nan)
        try:
            if pd.notna(val):
                return name, float(val)
        except Exception:
            continue

    return "none", -np.inf


# ============================================================
# Training helpers
# ============================================================


class EarlyStopping:
    def __init__(self, patience=20, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.bad_epochs = 0
        self.best_state = None

    def step(self, score, model):
        improved = False
        if self.best is None:
            improved = True
        elif self.mode == "max" and score > self.best:
            improved = True
        elif self.mode == "min" and score < self.best:
            improved = True

        if improved:
            self.best = score
            self.bad_epochs = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def move_aux_to_device(aux: Dict, device: torch.device) -> Dict:
    if aux is None:
        return {}
    out = {}
    for k, v in aux.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, list):
            new_list = []
            for item in v:
                new_list.append(item.to(device) if torch.is_tensor(item) else item)
            out[k] = new_list
        else:
            out[k] = v
    return out


def infer_hfr_indices(class_names: List[str]) -> Dict[str, int]:
    """Infer HC/MCI/AD indices from class names or common aliases."""
    role_to_idx: Dict[str, int] = {}
    unmatched = []

    for i, cls in enumerate(class_names):
        role = resolve_hfr_role(cls)
        if role is None:
            unmatched.append(cls)
            continue
        if role in role_to_idx:
            prev = class_names[role_to_idx[role]]
            raise ValueError(
                f"Duplicate HFR semantic role '{role}' in class-order: {prev!r} and {cls!r}. "
                f"Current class_names={class_names}"
            )
        role_to_idx[role] = i

    missing_roles = [r for r in HFR_ROLE_ORDER if r not in role_to_idx]
    if unmatched or missing_roles:
        raise ValueError(
            "HFR-style training requires exactly one HC/CN, one MCI, and one AD class. "
            f"Current class_names={class_names}; unmatched={unmatched}; missing_roles={missing_roles}. "
            "Supported aliases include CN/HC/healthy control, MCI/Mild Cognitive Impairment, and AD/Alzheimer's disease."
        )

    return {
        "idx_ad": role_to_idx["ad"],
        "idx_mci": role_to_idx["mci"],
        "idx_hc": role_to_idx["hc"],
    }


def validate_task_and_model_compatibility(cfg: Config, task: TaskDefinition):
    is_hfr_task = is_hfr_compatible_3way_task(task)

    if cfg.model_name in HFR_MODEL_NAMES and not is_hfr_task:
        if task.num_classes != 3:
            raise ValueError(
                f"model_name='{cfg.model_name}' is designed for the 3-class HC/MCI/AD hierarchy, "
                f"but current class-order defines {task.num_classes} classes: {task.class_names}. "
                "For 2-class tasks, use a generic model such as mlp/cnn/ms_mamba/hybrid_ms_mamba/"
                "auc_boost_tabular/subspace_ensemble/ft_transformer/modernnca/tabm/dann_ft/tabpfn/tabicl/tabr/excelformer."
            )
        raise ValueError(
            f"model_name='{cfg.model_name}' requires a 3-class HC/MCI/AD hierarchy, "
            f"but current class-order labels cannot be resolved to HC/MCI/AD: {task.class_names}."
        )

    if cfg.use_hierarchical_loss and not is_hfr_task:
        warnings.warn(
            "use_hierarchical_loss=True is only valid when class-order can be resolved to the 3-class HC/MCI/AD task. "
            "It will be automatically disabled for the current run.",
            UserWarning,
        )


# ============================================================
# Domain generalization / adaptation utilities
# ============================================================

DG_METHODS = {"erm", "groupdro", "vrex", "irm", "coral", "mmd", "dann", "cdan", "self_training"}
DOMAIN_REQUIRED_DG_METHODS = {"groupdro", "vrex", "irm", "coral", "mmd", "dann", "cdan"}


def normalize_dg_method(method: Optional[str]) -> str:
    method = "erm" if method is None else str(method).strip().lower().replace("-", "_")
    aliases = {
        "none": "erm",
        "group_dro": "groupdro",
        "group_dro_loss": "groupdro",
        "v_rex": "vrex",
        "vr_ex": "vrex",
        "deep_coral": "coral",
        "dan": "mmd",
        "mmd_dan": "mmd",
        "mmd/dan": "mmd",
        "pseudo": "self_training",
        "pseudo_label": "self_training",
        "pseudo_label_self_training": "self_training",
    }
    method = aliases.get(method, method)
    if method not in DG_METHODS:
        raise ValueError(f"Unsupported dg_method={method!r}. Supported methods: {sorted(DG_METHODS)}")
    return method


def dg_requires_domain_labels(cfg: Config) -> bool:
    return normalize_dg_method(cfg.dg_method) in DOMAIN_REQUIRED_DG_METHODS or bool(cfg.use_dann_loss)


def _ce_none(criterion: CompositeCriterion, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    weight = getattr(criterion.ce, "weight", None)
    label_smoothing = float(getattr(criterion.ce, "label_smoothing", 0.0))
    return F.cross_entropy(logits, y, weight=weight, reduction="none", label_smoothing=label_smoothing)


def _domain_mean_losses(per_sample_loss: torch.Tensor, domains: torch.Tensor, num_domains: int) -> Tuple[List[torch.Tensor], List[int]]:
    losses, domain_ids = [], []
    if domains is None:
        return losses, domain_ids
    for d in torch.unique(domains.detach()).tolist():
        d_int = int(d)
        if d_int < 0 or d_int >= num_domains:
            continue
        mask = domains == d_int
        if mask.any():
            losses.append(per_sample_loss[mask].mean())
            domain_ids.append(d_int)
    return losses, domain_ids


def _get_model_features(aux: Optional[Dict], logits: torch.Tensor) -> torch.Tensor:
    if aux is None:
        return logits
    for key in ("features", "feature", "feat", "embedding", "z", "h"):
        v = aux.get(key)
        if torch.is_tensor(v):
            if v.dim() > 2:
                v = v.flatten(start_dim=1)
            return v
    return logits


def _covariance(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if x.shape[0] <= 1:
        return x.new_zeros((x.shape[1], x.shape[1]))
    x = x - x.mean(dim=0, keepdim=True)
    return (x.t() @ x) / (x.shape[0] - 1 + eps)


def deep_coral_loss(features: torch.Tensor, domains: torch.Tensor, num_domains: int) -> torch.Tensor:
    if domains is None:
        return features.new_tensor(0.0)
    unique_domains = [int(d) for d in torch.unique(domains.detach()).tolist()]
    if len(unique_domains) < 2:
        return features.new_tensor(0.0)
    means, covs = [], []
    for d in unique_domains:
        z = features[domains == d]
        if z.shape[0] == 0:
            continue
        means.append(z.mean(dim=0))
        covs.append(_covariance(z))
    if len(means) < 2:
        return features.new_tensor(0.0)
    losses = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            mean_loss = (means[i] - means[j]).pow(2).mean()
            cov_loss = (covs[i] - covs[j]).pow(2).mean()
            losses.append(mean_loss + cov_loss)
    return torch.stack(losses).mean() if losses else features.new_tensor(0.0)


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: Optional[float] = None) -> torch.Tensor:
    dist = torch.cdist(x, y).pow(2)
    if gamma is None:
        with torch.no_grad():
            vals = dist.detach().flatten()
            vals = vals[vals > 0]
            median = vals.median() if vals.numel() > 0 else torch.tensor(1.0, device=x.device)
            gamma = float(1.0 / (2.0 * median.item() + 1e-6))
    return torch.exp(-gamma * dist)


def mmd_rbf_loss(features: torch.Tensor, domains: torch.Tensor, num_domains: int) -> torch.Tensor:
    if domains is None:
        return features.new_tensor(0.0)
    unique_domains = [int(d) for d in torch.unique(domains.detach()).tolist()]
    if len(unique_domains) < 2:
        return features.new_tensor(0.0)
    losses = []
    for i, di in enumerate(unique_domains):
        xi = features[domains == di]
        if xi.shape[0] < 2:
            continue
        for dj in unique_domains[i + 1:]:
            xj = features[domains == dj]
            if xj.shape[0] < 2:
                continue
            k_xx = _rbf_kernel(xi, xi, None).mean()
            k_yy = _rbf_kernel(xj, xj, None).mean()
            k_xy = _rbf_kernel(xi, xj, None).mean()
            losses.append(k_xx + k_yy - 2.0 * k_xy)
    return torch.stack(losses).mean() if losses else features.new_tensor(0.0)


def irm_penalty(logits: torch.Tensor, y: torch.Tensor, domains: torch.Tensor, criterion: CompositeCriterion) -> torch.Tensor:
    if domains is None or len(torch.unique(domains)) < 2:
        return logits.new_tensor(0.0)
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    penalties = []
    for d in torch.unique(domains.detach()).tolist():
        mask = domains == int(d)
        if mask.sum() < 2:
            continue
        loss_d = criterion.ce(logits[mask] * scale, y[mask])
        grad = torch.autograd.grad(loss_d, [scale], create_graph=True)[0]
        penalties.append(grad.pow(2))
    return torch.stack(penalties).mean() if penalties else logits.new_tensor(0.0)


class _GradientReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradientReverseFn.apply(x, lambd)


class CDANDomainHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, num_domains: int, hidden_dim: int = 128):
        super().__init__()
        in_dim = int(feature_dim) * int(num_classes)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, features: torch.Tensor, class_prob: torch.Tensor, grl_lambda: float = 0.1) -> torch.Tensor:
        cond = torch.bmm(class_prob.unsqueeze(2), features.unsqueeze(1)).flatten(start_dim=1)
        cond = grad_reverse(cond, grl_lambda)
        return self.net(cond)


def apply_dg_loss(
    loss: torch.Tensor,
    loss_dict: Dict[str, float],
    logits: torch.Tensor,
    y: torch.Tensor,
    aux: Optional[Dict],
    domains: Optional[torch.Tensor],
    criterion: CompositeCriterion,
    cfg: Config,
    dg_state: Optional[Dict[str, Any]],
    is_train: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if cfg is None or not is_train:
        return loss, loss_dict
    method = normalize_dg_method(cfg.dg_method)
    if method in {"erm", "self_training"}:
        return loss, loss_dict
    if domains is None:
        loss_dict["loss_dg"] = 0.0
        return loss, loss_dict

    num_domains = int(max(cfg.num_domains, int(domains.max().item()) + 1 if domains.numel() else 1))
    features = _get_model_features(aux, logits)
    per_sample_ce = _ce_none(criterion, logits, y)
    domain_losses, domain_ids = _domain_mean_losses(per_sample_ce, domains, num_domains)
    dg_loss = logits.new_tensor(0.0)

    if method == "groupdro":
        if len(domain_losses) > 0:
            if dg_state is None:
                dg_state = {}
            q = dg_state.get("groupdro_q")
            if q is None or q.numel() != num_domains:
                q = torch.ones(num_domains, device=logits.device) / float(num_domains)
            q = q.to(logits.device)
            with torch.no_grad():
                for loss_d, d in zip(domain_losses, domain_ids):
                    q[d] = q[d] * torch.exp(torch.clamp(cfg.groupdro_eta * loss_d.detach(), max=20.0))
                q = q / q.sum().clamp_min(1e-12)
                dg_state["groupdro_q"] = q.detach()
            dg_loss = torch.stack([q[d] * ld for ld, d in zip(domain_losses, domain_ids)]).sum()
            ce_mean = criterion.ce(logits, y)
            loss = loss - ce_mean + dg_loss

    elif method == "vrex":
        if len(domain_losses) >= 2:
            risks = torch.stack(domain_losses)
            dg_loss = risks.var(unbiased=False)
            loss = loss + cfg.vrex_weight * dg_loss

    elif method == "irm":
        dg_loss = irm_penalty(logits, y, domains, criterion)
        loss = loss + cfg.irm_weight * dg_loss

    elif method == "coral":
        dg_loss = deep_coral_loss(features, domains, num_domains)
        loss = loss + cfg.coral_weight * dg_loss

    elif method == "mmd":
        dg_loss = mmd_rbf_loss(features, domains, num_domains)
        loss = loss + cfg.mmd_weight * dg_loss

    elif method == "dann":
        dg_loss = logits.new_tensor(float(loss_dict.get("loss_domain", 0.0)))

    elif method == "cdan":
        if dg_state is None or dg_state.get("cdan_head") is None:
            raise RuntimeError("CDAN selected but cdan_head was not initialized.")
        class_prob = torch.softmax(logits, dim=1)
        domain_logits = dg_state["cdan_head"](features, class_prob, grl_lambda=cfg.grl_lambda)
        domain_ce = F.cross_entropy(domain_logits, domains)
        dg_loss = domain_ce
        loss = loss + cfg.cdan_weight * dg_loss
        loss_dict["loss_cdan_domain_ce"] = float(domain_ce.detach().item())

    loss_dict["loss_dg"] = float(dg_loss.detach().item()) if torch.is_tensor(dg_loss) else float(dg_loss)
    loss_dict["loss_total"] = float(loss.detach().item())
    return loss, loss_dict


def build_dg_state(cfg: Config, model: nn.Module, X_probe: np.ndarray, task: TaskDefinition, device: torch.device) -> Dict[str, Any]:
    method = normalize_dg_method(cfg.dg_method)
    state: Dict[str, Any] = {"method": method}
    if method == "groupdro":
        state["groupdro_q"] = torch.ones(int(cfg.num_domains), device=device) / float(max(1, int(cfg.num_domains)))
    if method == "cdan":
        model.eval()
        with torch.no_grad():
            xb = torch.tensor(X_probe[: min(4, len(X_probe))], dtype=torch.float32, device=device)
            logits, aux = model(xb)
            aux = move_aux_to_device(aux, device)
            feat = _get_model_features(aux, logits)
            feat_dim = int(feat.flatten(start_dim=1).shape[1])
        state["cdan_head"] = CDANDomainHead(
            feature_dim=feat_dim,
            num_classes=task.num_classes,
            num_domains=int(cfg.num_domains),
            hidden_dim=int(cfg.cdan_hidden_dim),
        ).to(device)
    return state


def dg_parameters(dg_state: Optional[Dict[str, Any]]) -> List[nn.Parameter]:
    if not dg_state:
        return []
    params: List[nn.Parameter] = []
    if dg_state.get("cdan_head") is not None:
        params += list(dg_state["cdan_head"].parameters())
    return params


def sample_mixup(x: torch.Tensor, y: torch.Tensor, domains: Optional[torch.Tensor], alpha: float):
    if alpha <= 0.0 or x.shape[0] < 2:
        return x, y, y, domains, domains, 1.0
    beta = torch.distributions.Beta(alpha, alpha)
    lam = float(beta.sample().item())
    perm = torch.randperm(x.shape[0], device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[perm]
    y_a, y_b = y, y[perm]
    if domains is None:
        d_a = d_b = None
    else:
        d_a, d_b = domains, domains[perm]
    return x_mix, y_a, y_b, d_a, d_b, lam


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    amp_enabled,
    is_train,
    criterion,
    class_names,
    max_grad_norm: float = 1.0,
    mixup_alpha: float = 0.0,
    cfg: Optional[Config] = None,
    dg_state: Optional[Dict[str, Any]] = None,
):
    model.train(is_train)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and is_train and device.type == "cuda"))

    losses = []
    loss_main_list = []
    loss_ord_list = []
    probs_all, preds_all, y_all = [], [], []
    scale_weights_all = []

    for batch in loader:
        if len(batch) == 3:
            xb, yb, db = batch
            db = db.to(device, non_blocking=True)
        else:
            xb, yb = batch
            db = None
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        use_mixup = is_train and mixup_alpha > 0.0 and (cfg is None or normalize_dg_method(cfg.dg_method) in {"erm", "self_training"})
        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=(amp_enabled and device.type == "cuda")):
                if use_mixup:
                    x_in, y_a, y_b, d_a, d_b, lam = sample_mixup(xb, yb, db, mixup_alpha)
                    logits, aux = model(x_in)
                    logits = logits.to(device)
                    aux = move_aux_to_device(aux, device)
                    loss_a, loss_dict_a = criterion(logits, y_a, aux, domains=d_a)
                    loss_b, loss_dict_b = criterion(logits, y_b, aux, domains=d_b)
                    loss = lam * loss_a + (1.0 - lam) * loss_b
                    loss_dict = {
                        k: lam * float(loss_dict_a.get(k, 0.0)) + (1.0 - lam) * float(loss_dict_b.get(k, 0.0))
                        for k in set(loss_dict_a) | set(loss_dict_b)
                    }
                    loss_dict["loss_total"] = float(loss.detach().item())
                else:
                    logits, aux = model(xb)
                    logits = logits.to(device)
                    aux = move_aux_to_device(aux, device)
                    loss, loss_dict = criterion(logits, yb, aux, domains=db)

                loss, loss_dict = apply_dg_loss(
                    loss=loss,
                    loss_dict=loss_dict,
                    logits=logits,
                    y=yb,
                    aux=aux,
                    domains=db,
                    criterion=criterion,
                    cfg=cfg,
                    dg_state=dg_state,
                    is_train=is_train,
                )

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

        if use_mixup:
            with torch.no_grad():
                logits_eval, aux_eval = model(xb)
                logits_for_metrics = logits_eval
                aux_for_logs = aux_eval
        else:
            logits_for_metrics = logits
            aux_for_logs = aux

        prob = torch.softmax(logits_for_metrics, dim=1).detach().cpu().numpy()
        pred = prob.argmax(axis=1)
        probs_all.append(prob)
        preds_all.append(pred)
        y_all.append(yb.detach().cpu().numpy())
        losses.append(float(loss_dict.get("loss_total", 0.0)))
        loss_main_list.append(float(loss_dict.get("loss_main", 0.0)))
        loss_ord_list.append(float(loss_dict.get("loss_ordinal", 0.0)))

        if aux_for_logs is not None and "scale_weights" in aux_for_logs and torch.is_tensor(aux_for_logs["scale_weights"]):
            scale_weights_all.append(aux_for_logs["scale_weights"].detach().cpu().numpy())

    probs_all = np.concatenate(probs_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    metrics = compute_metrics(y_all, probs_all, preds_all, class_names=class_names)
    metrics["loss"] = float(np.mean(losses))
    metrics["loss_main"] = float(np.mean(loss_main_list))
    metrics["loss_ordinal"] = float(np.mean(loss_ord_list))

    aux_logs = {}
    if len(scale_weights_all) > 0:
        aux_logs["scale_weights"] = np.concatenate(scale_weights_all, axis=0)
    return metrics, probs_all, preds_all, y_all, aux_logs


@torch.no_grad()
def predict(model, loader, device, amp_enabled, criterion, class_names):
    return run_epoch(
        model=model,
        loader=loader,
        optimizer=None,
        device=device,
        amp_enabled=amp_enabled,
        is_train=False,
        criterion=criterion,
        class_names=class_names,
        max_grad_norm=0.0,
        mixup_alpha=0.0,
    )


# ============================================================
# Visualization
# ============================================================


def plot_history(hist_df: pd.DataFrame, outdir: str):
    for metric_name, ylabel, fname in [
        ("loss", "Loss", "loss_curve.png"),
        ("macro_f1", "Macro-F1", "macro_f1_curve.png"),
        ("bal_acc", "Balanced Accuracy", "bal_acc_curve.png"),
        ("selection_score", "Validation selection score", "selection_score_curve.png"),
    ]:
        if f"train_{metric_name}" not in hist_df.columns and metric_name != "selection_score":
            continue
        plt.figure(figsize=(6, 4))
        if metric_name == "selection_score":
            if "val_selection_score" not in hist_df.columns:
                plt.close()
                continue
            plt.plot(hist_df["epoch"], hist_df["val_selection_score"], label="val_selection_score")
        else:
            if f"train_{metric_name}" in hist_df.columns:
                plt.plot(hist_df["epoch"], hist_df[f"train_{metric_name}"], label="train")
            if f"val_{metric_name}" in hist_df.columns:
                plt.plot(hist_df["epoch"], hist_df[f"val_{metric_name}"], label="val")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=220)
        plt.close()


def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


# ============================================================
# Evaluation output
# ============================================================


def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


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
):
    os.makedirs(save_dir, exist_ok=True)

    per_class_df = compute_per_class_metrics(y_true, probs, preds, class_names)
    per_class_df.to_csv(os.path.join(save_dir, "per_class_metrics.csv"), index=False)

    detailed_metrics = dict(metrics)
    detailed_metrics["per_class"] = per_class_df.replace({np.nan: None}).to_dict(orient="records")
    save_json(detailed_metrics, os.path.join(save_dir, "metrics.json"))
    save_json({"per_class": detailed_metrics["per_class"]}, os.path.join(save_dir, "per_class_metrics.json"))

    cm = confusion_matrix(y_true, preds, labels=list(range(len(class_names))))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        os.path.join(save_dir, "confusion_matrix.csv")
    )
    plot_confusion_matrix(cm, class_names, os.path.join(save_dir, "confusion_matrix.png"), title=split_name)

    report = classification_report(
        y_true,
        preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(os.path.join(save_dir, "classification_report.csv"))

    pred_df = pd.DataFrame({
        "sample_id": sample_ids,
        "y_true": y_true,
        "y_pred": preds,
        "y_true_name": [class_names[i] for i in y_true],
        "y_pred_name": [class_names[i] for i in preds],
    })
    for i, cls in enumerate(class_names):
        pred_df[f"prob__{safe_metric_name(cls)}"] = probs[:, i]
    pred_df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

    if overlap_info is not None:
        save_json(overlap_info, os.path.join(save_dir, "feature_overlap_info.json"))
    if filter_info is not None:
        save_json(filter_info, os.path.join(save_dir, "label_filter_info.json"))

    np.save(os.path.join(save_dir, "probs.npy"), probs)
    np.save(os.path.join(save_dir, "preds.npy"), preds)
    np.save(os.path.join(save_dir, "y_true.npy"), y_true)


def make_external_summary_row(spec_name: str, y_ext: np.ndarray, ext_metrics: Dict[str, Any], pack: Dict[str, Any]) -> Dict[str, Any]:
    """Create a task-aware external-test summary row."""
    def get_float(key: str):
        v = ext_metrics.get(key, np.nan)
        return float(v) if pd.notna(v) else np.nan

    row = {
        "dataset": spec_name,
        "n_samples": int(len(y_ext)),
        "acc": get_float("acc"),
        "bal_acc": get_float("bal_acc"),

        # Binary positive-class metrics; for 3-class they are macro aliases or NaN where appropriate.
        "precision": get_float("precision"),
        "recall": get_float("recall"),
        "sensitivity": get_float("sensitivity"),
        "specificity": get_float("specificity"),
        "f1": get_float("f1"),
        "mcc": get_float("mcc"),
        "auc_binary": get_float("auc_binary"),
        "auprc_binary": get_float("auprc_binary"),

        # Compatibility with multiclass tables
        "macro_f1": get_float("macro_f1"),
        "weighted_f1": get_float("weighted_f1"),
        "auc_ovr_macro": get_float("auc_ovr_macro"),
        "auc_ovr_weighted": get_float("auc_ovr_weighted"),
    }
    row.update(pack["overlap_info"])
    row.update({
        f"filter__{k}": v if not isinstance(v, dict) else json.dumps(v, ensure_ascii=False)
        for k, v in pack["filter_info"].items()
    })
    return row


# ============================================================
# Pseudo-label self-training utilities
# ============================================================


def load_unlabeled_matrix_and_metadata(spec: DatasetSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feat = read_table(spec.features)
    meta = read_table(spec.metadata)
    meta = ensure_sample_id_column(meta, spec.sample_col)
    if spec.domain_col is not None and spec.domain_col not in meta.columns:
        raise ValueError(f"[{spec.name}] domain_col={spec.domain_col} not found in metadata columns: {meta.columns.tolist()}")
    meta[spec.sample_col] = meta[spec.sample_col].astype(str)
    return feat, meta


def load_unlabeled_dataset_as_arrays(
    spec: DatasetSpec,
    expected_feature_order: List[str],
    external_feature_policy: str = "strict",
    min_feature_overlap_ratio: float = 0.98,
) -> Dict[str, Any]:
    feat, meta = load_unlabeled_matrix_and_metadata(spec)
    feat_df, meta_df, feature_cols = matrix_to_sample_feature_df(
        feat=feat,
        meta=meta,
        matrix_orientation=spec.matrix_orientation,
        sample_col=spec.sample_col,
        dataset_name=spec.name,
    )
    sample_ids = feat_df[spec.sample_col].astype(str).values
    domain_values = None
    if spec.domain_col is not None:
        domain_values = meta_df[spec.domain_col].astype(str).values
    numeric_df = feat_df[feature_cols].copy()
    aligned_df = numeric_df.reindex(columns=expected_feature_order)
    present = [f for f in expected_feature_order if f in numeric_df.columns]
    missing = [f for f in expected_feature_order if f not in numeric_df.columns]
    extra = [f for f in numeric_df.columns if f not in set(expected_feature_order)]
    overlap_ratio = len(present) / max(1, len(expected_feature_order))
    overlap_info = {
        "n_raw_features": int(len(feature_cols)),
        "n_overlap_with_train": int(len(present)),
        "overlap_ratio_against_train": float(overlap_ratio),
        "missing_train_features": int(len(missing)),
        "extra_external_features": int(len(extra)),
    }
    if external_feature_policy == "strict" and len(missing) > 0:
        raise ValueError(f"[{spec.name}] strict feature policy failed: {len(missing)} training features are missing.")
    if overlap_ratio < min_feature_overlap_ratio:
        raise ValueError(f"[{spec.name}] feature overlap ratio too low: {overlap_ratio:.4f} < {min_feature_overlap_ratio:.4f}.")
    return {
        "X_raw": aligned_df.values.astype(np.float32),
        "sample_ids": sample_ids,
        "domain_values": domain_values,
        "feature_names": list(expected_feature_order),
        "overlap_info": overlap_info,
    }


def load_pseudo_unlabeled_specs(path: Optional[str], cfg: Config) -> List[DatasetSpec]:
    if path is None:
        return []
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if isinstance(items, dict):
        items = [items]
    specs = []
    for item in items:
        specs.append(DatasetSpec(
            name=item.get("name", os.path.splitext(os.path.basename(item["features"]))[0]),
            features=str((Path(path).resolve().parent / item["features"]).resolve()),
            metadata=str((Path(path).resolve().parent / item["metadata"]).resolve()),
            matrix_orientation=item.get("matrix_orientation", cfg.matrix_orientation),
            sample_col=item.get("sample_col", cfg.sample_col),
            label_col=item.get("label_col", cfg.label_col),
            label_map=item.get("label_map", None),
            domain_col=item.get("domain_col", cfg.domain_col),
        ))
    return specs


def build_pseudo_loader(cfg: Config, train_pack: Dict[str, Any], preproc: Dict[str, Any]) -> Optional[DataLoader]:
    if normalize_dg_method(cfg.dg_method) != "self_training" or not cfg.pseudo_unlabeled_json:
        return None
    xs, ids = [], []
    for spec in load_pseudo_unlabeled_specs(cfg.pseudo_unlabeled_json, cfg):
        pack = load_unlabeled_dataset_as_arrays(
            spec,
            expected_feature_order=train_pack["feature_names"],
            external_feature_policy=cfg.external_feature_policy,
            min_feature_overlap_ratio=cfg.min_feature_overlap_ratio,
        )
        xs.append(transform_with_preprocessor(pack["X_raw"], preproc))
        ids.extend(pack["sample_ids"].tolist())
    if not xs:
        return None
    X_pseudo = np.concatenate(xs, axis=0)
    y_dummy = np.zeros(X_pseudo.shape[0], dtype=np.int64)
    pd.DataFrame({"sample_id": ids}).to_csv(os.path.join(cfg.outdir, "pseudo_unlabeled_samples.csv"), index=False)
    return build_loader(X_pseudo, y_dummy, cfg, shuffle=True, domains=None)


def run_pseudo_label_epoch(
    model: nn.Module,
    loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_enabled: bool,
    cfg: Config,
    max_grad_norm: float,
) -> Dict[str, float]:
    if loader is None:
        return {"pseudo_loss": 0.0, "pseudo_kept": 0, "pseudo_total": 0, "pseudo_keep_rate": 0.0}
    model.train(True)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and device.type == "cuda"))
    losses, kept, total, steps = [], 0, 0, 0
    for batch in loader:
        xb = batch[0].to(device, non_blocking=True)
        total += int(xb.shape[0])
        with torch.set_grad_enabled(True):
            with torch.cuda.amp.autocast(enabled=(amp_enabled and device.type == "cuda")):
                logits, aux = model(xb)
                prob = torch.softmax(logits, dim=1)
                conf, pseudo_y = prob.max(dim=1)
                mask = conf >= float(cfg.pseudo_threshold)
                if not mask.any():
                    continue
                loss = F.cross_entropy(logits[mask], pseudo_y[mask]) * float(cfg.pseudo_weight)
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
        kept += int(mask.sum().item())
        losses.append(float(loss.detach().item()))
        steps += 1
        if cfg.pseudo_max_steps_per_epoch > 0 and steps >= cfg.pseudo_max_steps_per_epoch:
            break
    return {
        "pseudo_loss": float(np.mean(losses)) if losses else 0.0,
        "pseudo_kept": int(kept),
        "pseudo_total": int(total),
        "pseudo_keep_rate": float(kept / max(1, total)),
    }


# ============================================================
# Main training routine helpers
# ============================================================


def build_model_and_criterion(cfg: Config, task: TaskDefinition, num_features: int, y_train: np.ndarray, device):
    class_names = task.class_names
    num_classes = task.num_classes

    hierarchical_class_indices = None
    is_hfr_task = is_hfr_compatible_3way_task(task)

    if cfg.model_name in HIERARCHICAL_AUX_MODEL_NAMES and is_hfr_task:
        hierarchical_class_indices = infer_hfr_indices(class_names)
    elif cfg.model_name in HFR_MODEL_NAMES:
        hierarchical_class_indices = infer_hfr_indices(class_names)

    class_weights_tensor = None
    if cfg.use_class_weight:
        class_weights_np = compute_class_weights_from_y(y_train, num_classes)
        class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    else:
        class_weights_np = None

    resolved_use_hierarchical_loss = cfg.use_hierarchical_loss and is_hfr_task

    model = build_model(
        model_name=cfg.model_name,
        num_features=num_features,
        num_classes=num_classes,
        d_model=cfg.d_model,
        dropout=cfg.dropout,
        patch_scales=parse_patch_scales(cfg.patch_scales),
        n_mamba_layers=cfg.n_mamba_layers,
        d_state=cfg.d_state,
        d_conv=cfg.d_conv,
        expand=cfg.expand,
        hierarchical_class_indices=hierarchical_class_indices,
        num_domains=cfg.num_domains,
        n_transformer_layers=cfg.n_transformer_layers,
        num_heads=cfg.num_heads,
        num_prototypes_per_class=cfg.num_prototypes_per_class,
        tabm_num_members=cfg.tabm_num_members,
        grl_lambda=cfg.grl_lambda,
        use_linear_residual=cfg.use_linear_residual,
        tabr_k_neighbors=cfg.tabr_k_neighbors,
    ).to(device)

    criterion = CompositeCriterion(
        num_classes=num_classes,
        class_weights=class_weights_tensor,
        label_smoothing=cfg.label_smoothing,
        use_ordinal_loss=cfg.use_ordinal_loss,
        ordinal_weight=cfg.ordinal_weight,
        use_ovr_loss=cfg.use_ovr_loss,
        ovr_weight=cfg.ovr_weight,
        use_hierarchical_loss=resolved_use_hierarchical_loss,
        hier_stage1_weight=cfg.hier_stage1_weight,
        hier_stage2_weight=cfg.hier_stage2_weight,
        hier_aux3_weight=cfg.hier_aux3_weight,
        use_enet_reg=cfg.use_enet_reg,
        enet_l1_weight=cfg.enet_l1_weight,
        enet_l2_weight=cfg.enet_l2_weight,
        use_linear_residual_reg=cfg.use_linear_residual_reg,
        linear_l1_weight=cfg.linear_l1_weight,
        linear_l2_weight=cfg.linear_l2_weight,
        hierarchical_class_indices=hierarchical_class_indices,
        use_dann_loss=(cfg.use_dann_loss or normalize_dg_method(cfg.dg_method) == "dann"),
        dann_weight=cfg.dann_weight,
    )

    resolved_flags = {
        "resolved_use_hierarchical_loss": resolved_use_hierarchical_loss,
        "is_hfr_compatible_3way_task": is_hfr_task,
        "hierarchical_class_indices": hierarchical_class_indices,
    }
    return model, criterion, class_weights_np, resolved_flags


def split_train_val(X_raw: np.ndarray, y: np.ndarray, sample_ids: np.ndarray, cfg: Config, domain_values: Optional[np.ndarray] = None):
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
        "train_idx": train_idx,
        "val_idx": val_idx,
        "X_train_raw": X_raw[train_idx],
        "y_train": y[train_idx],
        "train_ids": sample_ids[train_idx],
        "X_val_raw": X_raw[val_idx],
        "y_val": y[val_idx],
        "val_ids": sample_ids[val_idx],
        "train_domain_values": None if domain_values is None else domain_values[train_idx],
        "val_domain_values": None if domain_values is None else domain_values[val_idx],
    }


def load_external_test_specs(path: Optional[str], cfg: Config) -> List[DatasetSpec]:
    if path is None or str(path).strip() == "":
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    specs = []
    for item in raw:
        specs.append(
            DatasetSpec(
                name=item["name"],
                features=str((Path(path).resolve().parent / item["features"]).resolve()),
                metadata=str((Path(path).resolve().parent / item["metadata"]).resolve()),
                matrix_orientation=item.get("matrix_orientation", cfg.matrix_orientation),
                sample_col=item.get("sample_col", cfg.sample_col),
                label_col=item.get("label_col", cfg.label_col),
                label_map=item.get("label_map", None),
                domain_col=item.get("domain_col", None),
            )
        )
    return specs


def save_split_info(path: str, sample_ids: np.ndarray, split_name: str, y: np.ndarray, task: TaskDefinition):
    pd.DataFrame({
        "sample_id": sample_ids,
        "split": [split_name] * len(sample_ids),
        "label_id": y,
        "label_name": inverse_transform_labels(y, task),
    }).to_csv(path, index=False)


def fit_domain_mapping(domain_values: Optional[np.ndarray]) -> Tuple[Optional[Dict[str, int]], Optional[np.ndarray]]:
    if domain_values is None:
        return None, None
    vals = [str(v) for v in domain_values]
    uniq = sorted(set(vals))
    mapping = {v: i for i, v in enumerate(uniq)}
    encoded = np.array([mapping[v] for v in vals], dtype=np.int64)
    return mapping, encoded


def transform_domain_values(domain_values: Optional[np.ndarray], mapping: Optional[Dict[str, int]]) -> Optional[np.ndarray]:
    if domain_values is None or mapping is None:
        return None
    return np.array([mapping.get(str(v), 0) for v in domain_values], dtype=np.int64)


def select_top_variance_features(X_train: np.ndarray, max_features: int) -> np.ndarray:
    if max_features <= 0 or X_train.shape[1] <= max_features:
        return np.arange(X_train.shape[1])
    variances = np.nanvar(X_train, axis=0)
    order = np.argsort(variances)[::-1]
    return np.sort(order[:max_features])


def _auto_device_name(device_setting: str) -> str:
    if device_setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_setting


def _try_construct(cls, candidate_kwargs: List[Dict[str, Any]]):
    last_err = None
    for kwargs in candidate_kwargs:
        try:
            return cls(**kwargs)
        except TypeError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    return cls()


def fit_foundation_classifier(X_train: np.ndarray, y_train: np.ndarray, cfg: Config):
    name = cfg.model_name.lower()

    if name == "tabpfn":
        try:
            from tabpfn import TabPFNClassifier
        except Exception as e:
            raise ImportError("TabPFN is not installed. Install it with: pip install tabpfn") from e

        device = _auto_device_name(cfg.tabpfn_device)
        clf = _try_construct(
            TabPFNClassifier,
            [
                {"device": device, "random_state": cfg.seed},
                {"device": device},
                {"random_state": cfg.seed},
                {},
            ],
        )

        estimator_limit = getattr(clf, "max_num_features", None)
        if estimator_limit is not None:
            try:
                estimator_limit = int(estimator_limit)
            except Exception:
                estimator_limit = None
        if estimator_limit is not None and X_train.shape[1] > estimator_limit:
            raise ValueError(
                f"TabPFN in this environment allows at most {estimator_limit} features, "
                f"but got {X_train.shape[1]}. Re-run with --tabpfn-max-features {estimator_limit}."
            )

        clf.fit(X_train, y_train)
        return clf

    if name == "tabicl":
        if sys.version_info < (3, 9):
            raise ImportError(
                "TabICL requires Python >=3.9. Please create a separate Python 3.9/3.10 environment and install tabicl."
            )
        try:
            from tabicl import TabICLClassifier
        except Exception as e:
            raise ImportError("TabICL is not installed. Install it with Python >=3.9 using: pip install tabicl") from e

        device = None if cfg.tabicl_device == "auto" else cfg.tabicl_device
        clf = _try_construct(
            TabICLClassifier,
            [
                {
                    "n_estimators": cfg.tabicl_n_estimators,
                    "batch_size": cfg.tabicl_batch_size,
                    "kv_cache": cfg.tabicl_kv_cache,
                    "device": device,
                    "random_state": cfg.seed,
                },
                {
                    "n_estimators": cfg.tabicl_n_estimators,
                    "batch_size": cfg.tabicl_batch_size,
                    "kv_cache": cfg.tabicl_kv_cache,
                    "random_state": cfg.seed,
                },
                {"random_state": cfg.seed},
                {},
            ],
        )
        clf.fit(X_train, y_train)
        return clf

    raise ValueError(f"Unsupported foundation model_name={cfg.model_name!r}")


def foundation_predict_proba(clf, X: np.ndarray, num_classes: int) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X)
    elif hasattr(clf, "predict"):
        pred = np.asarray(clf.predict(X), dtype=np.int64)
        prob = np.zeros((len(pred), num_classes), dtype=np.float64)
        prob[np.arange(len(pred)), pred] = 1.0
    else:
        raise AttributeError("Foundation estimator has neither predict_proba nor predict.")

    prob = np.asarray(prob, dtype=np.float64)
    if prob.ndim == 1:
        prob = np.stack([1.0 - prob, prob], axis=1)
    if prob.shape[1] != num_classes:
        dense = np.zeros((prob.shape[0], num_classes), dtype=np.float64)
        classes = getattr(clf, "classes_", np.arange(prob.shape[1]))
        for j, cls in enumerate(classes):
            cls_id = int(cls)
            if 0 <= cls_id < num_classes:
                dense[:, cls_id] = prob[:, j]
        row_sum = dense.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        prob = dense / row_sum
    return prob.astype(np.float32)


def _foundation_max_features(cfg: Config) -> int:
    name = cfg.model_name.lower()
    if name == "tabpfn":
        requested = int(cfg.tabpfn_max_features)
        if requested <= 0:
            return TABPFN_CONSERVATIVE_MAX_FEATURES
        effective = min(requested, TABPFN_CONSERVATIVE_MAX_FEATURES)
        if requested > TABPFN_CONSERVATIVE_MAX_FEATURES:
            warnings.warn(
                f"Current TabPFN-compatible pipeline caps features at {TABPFN_CONSERVATIVE_MAX_FEATURES}. "
                f"Requested --tabpfn-max-features={requested}, using {effective} instead.",
                UserWarning,
            )
        return effective
    if name == "tabicl":
        return cfg.tabicl_max_features
    raise ValueError(f"Unsupported foundation model_name={cfg.model_name!r}")


def run_foundation_pipeline(
    cfg: Config,
    task: TaskDefinition,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_ids: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_ids: np.ndarray,
    train_pack: Dict[str, Any],
    preproc: Dict[str, Any],
    val_overlap_info: Optional[Dict[str, Any]],
    val_filter_info: Optional[Dict[str, Any]],
):
    model_tag = cfg.model_name.lower()
    max_features = _foundation_max_features(cfg)
    selected_idx = select_top_variance_features(X_train, max_features)
    selected_names = [train_pack["feature_names"][i] for i in selected_idx]
    pd.DataFrame({"feature_name": selected_names, "feature_index": selected_idx}).to_csv(
        os.path.join(cfg.outdir, f"{model_tag}_selected_features.csv"), index=False
    )

    X_train_sel = X_train[:, selected_idx]
    X_val_sel = X_val[:, selected_idx]
    clf = fit_foundation_classifier(X_train_sel, y_train, cfg)

    try:
        with open(os.path.join(cfg.outdir, "best_model.pkl"), "wb") as f:
            pickle.dump(clf, f)
    except Exception as e:
        warnings.warn(f"Could not pickle {model_tag} estimator: {e}", UserWarning)

    train_probs = foundation_predict_proba(clf, X_train_sel, task.num_classes)
    train_preds = train_probs.argmax(axis=1)
    train_metrics = compute_metrics(y_train, train_probs, train_preds, task.class_names)

    val_probs = foundation_predict_proba(clf, X_val_sel, task.num_classes)
    val_preds = val_probs.argmax(axis=1)
    val_metrics = compute_metrics(y_val, val_probs, val_preds, task.class_names)

    save_evaluation_outputs(os.path.join(cfg.outdir, "eval_train"), "train", task.class_names, train_metrics, train_probs, train_preds, y_train, train_ids, None, train_pack["filter_info"])
    save_evaluation_outputs(os.path.join(cfg.outdir, "eval_val"), "val", task.class_names, val_metrics, val_probs, val_preds, y_val, val_ids, val_overlap_info, val_filter_info)

    external_specs = load_external_test_specs(cfg.external_tests_json, cfg)
    external_rows = []
    for spec in external_specs:
        pack = load_dataset_as_arrays(
            spec,
            task=task,
            expected_feature_order=train_pack["feature_names"],
            external_feature_policy=cfg.external_feature_policy,
            min_feature_overlap_ratio=cfg.min_feature_overlap_ratio,
        )
        X_ext = transform_with_preprocessor(pack["X_raw"], preproc)[:, selected_idx]
        y_ext = pack["y"]
        ids_ext = pack["sample_ids"]
        ext_probs = foundation_predict_proba(clf, X_ext, task.num_classes)
        ext_preds = ext_probs.argmax(axis=1)
        ext_metrics = compute_metrics(y_ext, ext_probs, ext_preds, task.class_names)

        save_evaluation_outputs(
            save_dir=os.path.join(cfg.outdir, "external_tests", spec.name),
            split_name=f"external_test__{spec.name}",
            class_names=task.class_names,
            metrics=ext_metrics,
            probs=ext_probs,
            preds=ext_preds,
            y_true=y_ext,
            sample_ids=ids_ext,
            overlap_info=pack["overlap_info"],
            filter_info=pack["filter_info"],
        )
        external_rows.append(make_external_summary_row(spec.name, y_ext, ext_metrics, pack))

    pd.DataFrame(external_rows).to_csv(os.path.join(cfg.outdir, "external_test_summary.csv"), index=False)
    val_metric_name, val_selection_score = select_validation_score(val_metrics, task)
    summary = {
        "task_type": task.task_type,
        "class_names": task.class_names,
        "binary_positive_class": task.class_names[1] if task.num_classes == 2 else None,
        "binary_negative_class": task.class_names[0] if task.num_classes == 2 else None,
        "model_name": cfg.model_name,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_features_original": int(X_train.shape[1]),
        "n_features_selected": int(len(selected_idx)),
        "feature_selection": "top_train_variance" if len(selected_idx) < X_train.shape[1] else "none",
        "val_selection_metric": val_metric_name,
        "best_val_selection_score": float(val_selection_score),
        "selection_rule": "binary: auc_binary -> f1 -> bal_acc; multiclass: macro_f1 -> auc_ovr_macro -> bal_acc",
        "train_metrics": _metrics_to_jsonable(train_metrics),
        "val_metrics": _metrics_to_jsonable(val_metrics),
        "n_external_tests": int(len(external_rows)),
        "external_tests": external_rows,
    }
    save_json(summary, os.path.join(cfg.outdir, "summary.json"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# Backward-compatible aliases

def fit_tabpfn_classifier(X_train: np.ndarray, y_train: np.ndarray, cfg: Config):
    old = cfg.model_name
    cfg.model_name = "tabpfn"
    try:
        return fit_foundation_classifier(X_train, y_train, cfg)
    finally:
        cfg.model_name = old


def tabpfn_predict_proba(clf, X: np.ndarray, num_classes: int) -> np.ndarray:
    return foundation_predict_proba(clf, X, num_classes)


def run_tabpfn_pipeline(*args, **kwargs):
    return run_foundation_pipeline(*args, **kwargs)


def estimator_predict_proba(clf, X: np.ndarray, num_classes: int) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X)
    elif hasattr(clf, "decision_function"):
        scores = np.asarray(clf.decision_function(X), dtype=np.float64)
        if scores.ndim == 1:
            prob_pos = 1.0 / (1.0 + np.exp(-scores))
            prob = np.stack([1.0 - prob_pos, prob_pos], axis=1)
        else:
            scores = scores - np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    else:
        pred = np.asarray(clf.predict(X), dtype=np.int64)
        prob = np.zeros((len(pred), num_classes), dtype=np.float64)
        prob[np.arange(len(pred)), pred] = 1.0

    prob = np.asarray(prob, dtype=np.float64)
    if prob.ndim == 1:
        prob = np.stack([1.0 - prob, prob], axis=1)

    if prob.shape[1] != num_classes:
        dense = np.zeros((prob.shape[0], num_classes), dtype=np.float64)
        classes = getattr(clf, "classes_", np.arange(prob.shape[1]))
        for j, cls in enumerate(classes):
            cls_id = int(cls)
            if 0 <= cls_id < num_classes:
                dense[:, cls_id] = prob[:, j]
        row_sum = dense.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        prob = dense / row_sum

    return prob.astype(np.float32)


def fit_sklearn_classifier(X_train: np.ndarray, y_train: np.ndarray, cfg: Config, task: TaskDefinition):
    if cfg.model_name.lower() == "xgboost" and _EARLY_XGB_IMPORT_ERROR is not None:
        raise ImportError(
            "xgboost failed during the early import stage before torch/sklearn were imported. "
            "This is usually an OpenMP/libgomp static TLS issue. "
            f"Original error: {repr(_EARLY_XGB_IMPORT_ERROR)}"
        )
    clf = build_model(
        model_name=cfg.model_name,
        num_features=X_train.shape[1],
        num_classes=task.num_classes,
        seed=cfg.seed,
        enet_C=cfg.enet_C,
        enet_l1_ratio=cfg.enet_l1_ratio,
        enet_max_iter=cfg.enet_max_iter,
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        max_features=cfg.max_features,
        svm_C=cfg.svm_C,
        svm_gamma=cfg.svm_gamma,
        gb_learning_rate=cfg.gb_learning_rate,
        gb_max_depth=cfg.gb_max_depth,
        gb_num_leaves=cfg.gb_num_leaves,
        gb_subsample=cfg.gb_subsample,
        gb_colsample_bytree=cfg.gb_colsample_bytree,
        gb_min_child_samples=cfg.gb_min_child_samples,
        gb_reg_lambda=cfg.gb_reg_lambda,
    )
    clf.fit(X_train, y_train)
    return clf


def run_sklearn_pipeline(
    cfg: Config,
    task: TaskDefinition,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_ids: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_ids: np.ndarray,
    train_pack: Dict[str, Any],
    preproc: Dict[str, Any],
    val_overlap_info: Optional[Dict[str, Any]],
    val_filter_info: Optional[Dict[str, Any]],
):
    clf = fit_sklearn_classifier(X_train, y_train, cfg, task)

    with open(os.path.join(cfg.outdir, "best_model.pkl"), "wb") as f:
        pickle.dump(clf, f)

    train_probs = estimator_predict_proba(clf, X_train, task.num_classes)
    train_preds = train_probs.argmax(axis=1)
    train_metrics = compute_metrics(y_train, train_probs, train_preds, task.class_names)

    val_probs = estimator_predict_proba(clf, X_val, task.num_classes)
    val_preds = val_probs.argmax(axis=1)
    val_metrics = compute_metrics(y_val, val_probs, val_preds, task.class_names)

    save_evaluation_outputs(os.path.join(cfg.outdir, "eval_train"), "train", task.class_names, train_metrics, train_probs, train_preds, y_train, train_ids, None, train_pack["filter_info"])
    save_evaluation_outputs(os.path.join(cfg.outdir, "eval_val"), "val", task.class_names, val_metrics, val_probs, val_preds, y_val, val_ids, val_overlap_info, val_filter_info)

    external_specs = load_external_test_specs(cfg.external_tests_json, cfg)
    external_rows = []
    for spec in external_specs:
        pack = load_dataset_as_arrays(
            spec,
            task=task,
            expected_feature_order=train_pack["feature_names"],
            external_feature_policy=cfg.external_feature_policy,
            min_feature_overlap_ratio=cfg.min_feature_overlap_ratio,
        )
        X_ext = transform_with_preprocessor(pack["X_raw"], preproc)
        y_ext = pack["y"]
        ids_ext = pack["sample_ids"]

        ext_probs = estimator_predict_proba(clf, X_ext, task.num_classes)
        ext_preds = ext_probs.argmax(axis=1)
        ext_metrics = compute_metrics(y_ext, ext_probs, ext_preds, task.class_names)

        save_evaluation_outputs(
            save_dir=os.path.join(cfg.outdir, "external_tests", spec.name),
            split_name=f"external_test__{spec.name}",
            class_names=task.class_names,
            metrics=ext_metrics,
            probs=ext_probs,
            preds=ext_preds,
            y_true=y_ext,
            sample_ids=ids_ext,
            overlap_info=pack["overlap_info"],
            filter_info=pack["filter_info"],
        )
        external_rows.append(make_external_summary_row(spec.name, y_ext, ext_metrics, pack))

    pd.DataFrame(external_rows).to_csv(os.path.join(cfg.outdir, "external_test_summary.csv"), index=False)
    val_metric_name, val_selection_score = select_validation_score(val_metrics, task)
    summary = {
        "task_type": task.task_type,
        "class_names": task.class_names,
        "binary_positive_class": task.class_names[1] if task.num_classes == 2 else None,
        "binary_negative_class": task.class_names[0] if task.num_classes == 2 else None,
        "model_name": cfg.model_name,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_features": int(X_train.shape[1]),
        "val_selection_metric": val_metric_name,
        "best_val_selection_score": float(val_selection_score),
        "selection_rule": "binary: auc_binary -> f1 -> bal_acc; multiclass: macro_f1 -> auc_ovr_macro -> bal_acc",
        "train_metrics": _metrics_to_jsonable(train_metrics),
        "val_metrics": _metrics_to_jsonable(val_metrics),
        "n_external_tests": int(len(external_rows)),
        "external_tests": external_rows,
    }
    save_json(summary, os.path.join(cfg.outdir, "summary.json"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# ============================================================
# CLI / Main
# ============================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--train-features", type=str, default=str(project_root / "data/GSE243529_intersection_CpGs_after_com.csv.gz"))
    parser.add_argument("--train-metadata", type=str, default=str(project_root / "data/GSE243529_pheno.csv"))
    parser.add_argument("--val-features", type=str, default=None)
    parser.add_argument("--val-metadata", type=str, default=None)
    parser.add_argument("--external-tests-json", type=str, default=str(project_root / "data/external_tests.json"))

    parser.add_argument("--outdir", type=str, default=str(project_root / "runs/benchmark"))
    parser.add_argument("--matrix-orientation", type=str, default="cpg_rows", choices=["sample_rows", "cpg_rows"])
    parser.add_argument("--sample-col", type=str, default="SampleID")
    parser.add_argument("--label-col", type=str, default="disease.state")
    parser.add_argument(
        "--class-order",
        type=str,
        default="unaffected, affected",
        help="For binary task, the second class is treated as the positive class. Example: 'unaffected, affected'.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--no-stratify-val", action="store_true")

    parser.add_argument("--external-feature-policy", type=str, default="strict", choices=["fill_train_median", "strict"])
    parser.add_argument("--min-feature-overlap-ratio", type=float, default=0.98)

    parser.add_argument(
        "--model-name", type=str, default="pref_ft",
        choices=[
            "mlp", "cnn", "ms_mamba", "hybrid_ms_mamba", "auc_boost_tabular",
            "subspace_ensemble", 
            # "hfr_net", "rank_hfr_mlp", "sparse_rank_hfr_mlp",
            # "linear_residual_rank_hfr_mlp", "cross_cohort_linear_residual_rank_hfr_mlp",
            # "cross_cohort_dg_linear_residual_rank_hfr_mlp",
            "ft_transformer", "modernnca",
            "tabm", "dann_ft_transformer", 
            "enhanced_dann_ft_transformer", 
            "enhanced_ft_transformer", 
            "pref_ft", "binary_positive_aware_ft", "binary_positive_aware_ft_2",
            "tabr", "tabr_lite", "excelformer", 
            "elastic_net_lr", "random_forest", "extra_trees", "svm_rbf", "xgboost", "lightgbm", "stable_methyl_res_mlp"
        ]
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patch-scales", type=str, default="16,32,64")
    parser.add_argument("--n-mamba-layers", type=int, default=2)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)

    parser.add_argument("--n-transformer-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-prototypes-per-class", type=int, default=4)
    parser.add_argument("--tabm-num-members", type=int, default=4)
    parser.add_argument("--disable-linear-residual", action="store_true")

    parser.add_argument("--domain-col", type=str, default=None)
    parser.add_argument("--use-dann-loss", action="store_true")
    parser.add_argument("--dann-weight", type=float, default=0.05)
    parser.add_argument("--grl-lambda", type=float, default=0.1)
    parser.add_argument("--num-domains", type=int, default=1)

    parser.add_argument("--dg-method", type=str, default="erm", choices=sorted(DG_METHODS))
    parser.add_argument("--dg-weight", type=float, default=0.1)
    parser.add_argument("--groupdro-eta", type=float, default=0.01)
    parser.add_argument("--irm-weight", type=float, default=0.1)
    parser.add_argument("--vrex-weight", type=float, default=0.1)
    parser.add_argument("--coral-weight", type=float, default=0.1)
    parser.add_argument("--mmd-weight", type=float, default=0.1)
    parser.add_argument("--cdan-weight", type=float, default=0.1)
    parser.add_argument("--cdan-hidden-dim", type=int, default=128)
    parser.add_argument("--pseudo-unlabeled-json", type=str, default=None)
    parser.add_argument("--pseudo-threshold", type=float, default=0.9)
    parser.add_argument("--pseudo-weight", type=float, default=0.2)
    parser.add_argument("--pseudo-warmup-epochs", type=int, default=10)
    parser.add_argument("--pseudo-max-steps-per-epoch", type=int, default=0)

    parser.add_argument("--mixup-alpha", type=float, default=0.0)

    parser.add_argument("--tabpfn-max-features", type=int, default=100)
    parser.add_argument("--tabpfn-device", type=str, default="auto")
    parser.add_argument("--tabicl-max-features", type=int, default=512)
    parser.add_argument("--tabicl-device", type=str, default="auto")
    parser.add_argument("--tabicl-n-estimators", type=int, default=8)
    parser.add_argument("--tabicl-batch-size", type=int, default=8)
    parser.add_argument("--tabicl-kv-cache", action="store_true")
    parser.add_argument("--tabr-k-neighbors", type=int, default=8)

    parser.add_argument("--enet-c", dest="enet_C", type=float, default=1.0)
    parser.add_argument("--enet-l1-ratio", type=float, default=0.5)
    parser.add_argument("--enet-max-iter", type=int, default=5000)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=0, help="0 means unlimited/None for sklearn trees")
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-features", type=str, default="sqrt")
    parser.add_argument("--svm-c", dest="svm_C", type=float, default=1.0)
    parser.add_argument("--svm-gamma", type=str, default="scale")
    parser.add_argument("--gb-learning-rate", type=float, default=0.03)
    parser.add_argument("--gb-max-depth", type=int, default=6)
    parser.add_argument("--gb-num-leaves", type=int, default=31)
    parser.add_argument("--gb-subsample", type=float, default=0.9)
    parser.add_argument("--gb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--gb-min-child-samples", type=int, default=20)
    parser.add_argument("--gb-reg-lambda", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)

    parser.add_argument("--use-hierarchical-loss", action="store_true")
    parser.add_argument("--hier-stage1-weight", type=float, default=0.6)
    parser.add_argument("--hier-stage2-weight", type=float, default=0.9)
    parser.add_argument("--hier-aux3-weight", type=float, default=0.2)

    parser.add_argument("--use-class-weight", action="store_true")
    parser.add_argument("--use-ordinal-loss", action="store_true")
    parser.add_argument("--ordinal-weight", type=float, default=0.3)

    parser.add_argument("--use-enet-reg", action="store_true")
    parser.add_argument("--enet-l1-weight", type=float, default=1e-4)
    parser.add_argument("--enet-l2-weight", type=float, default=1e-4)

    parser.add_argument("--use-linear-residual-reg", action="store_true")
    parser.add_argument("--linear-l1-weight", type=float, default=1e-5)
    parser.add_argument("--linear-l2-weight", type=float, default=1e-4)

    parser.add_argument("--use-ovr-loss", action="store_true")
    parser.add_argument("--ovr-weight", type=float, default=0.3)
    parser.add_argument("--save-logits", action="store_true")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args_dict = vars(args).copy()
    args_dict["stratify_val"] = (not args_dict.pop("no_stratify_val"))
    args_dict["use_linear_residual"] = (not args_dict.pop("disable_linear_residual"))
    cfg = Config(**args_dict)
    cfg.dg_method = normalize_dg_method(cfg.dg_method)
    if cfg.dg_method == "dann":
        cfg.use_dann_loss = True

    # Deterministic environment before any random operation
    set_seed(cfg.seed)

    task = parse_class_order_to_task(cfg.class_order)
    validate_task_and_model_compatibility(cfg, task)

    os.makedirs(cfg.outdir, exist_ok=True)

    config_to_save = asdict(cfg)
    config_to_save["resolved_task_type"] = task.task_type
    config_to_save["resolved_class_names"] = task.class_names
    config_to_save["binary_positive_class"] = task.class_names[1] if task.num_classes == 2 else None
    config_to_save["binary_negative_class"] = task.class_names[0] if task.num_classes == 2 else None
    save_json(config_to_save, os.path.join(cfg.outdir, "config.json"))
    save_json(
        {"class_names": task.class_names, "label_to_id": task.label_to_id, "id_to_label": task.id_to_label},
        os.path.join(cfg.outdir, "label_mapping.json"),
    )

    # --------------------------------------------------------
    # Step 1. load main training cohort
    # --------------------------------------------------------
    train_spec = DatasetSpec(
        name="train_cohort",
        features=cfg.train_features,
        metadata=cfg.train_metadata,
        matrix_orientation=cfg.matrix_orientation,
        sample_col=cfg.sample_col,
        label_col=cfg.label_col,
        label_map=None,
        domain_col=cfg.domain_col,
    )
    train_pack = load_dataset_as_arrays(train_spec, task=task)

    train_label_counts = np.bincount(train_pack["y"], minlength=task.num_classes)
    missing_train_classes = [task.id_to_label[i] for i, c in enumerate(train_label_counts) if c == 0]
    if missing_train_classes:
        raise ValueError(
            f"Training cohort does not contain required class(es) from class-order after filtering: {missing_train_classes}."
        )

    # --------------------------------------------------------
    # Step 2. define validation set
    # --------------------------------------------------------
    if cfg.val_features is not None and cfg.val_metadata is not None:
        save_split_info(
            os.path.join(cfg.outdir, "train_samples.csv"),
            train_pack["sample_ids"],
            "train",
            train_pack["y"],
            task,
        )
        save_json(train_pack["filter_info"], os.path.join(cfg.outdir, "train_label_filter_info.json"))

        val_spec = DatasetSpec(
            name="explicit_val",
            features=cfg.val_features,
            metadata=cfg.val_metadata,
            matrix_orientation=cfg.matrix_orientation,
            sample_col=cfg.sample_col,
            label_col=cfg.label_col,
            label_map=None,
            domain_col=cfg.domain_col,
        )
        val_pack = load_dataset_as_arrays(
            val_spec,
            task=task,
            expected_feature_order=train_pack["feature_names"],
            external_feature_policy=cfg.external_feature_policy,
            min_feature_overlap_ratio=cfg.min_feature_overlap_ratio,
        )

        X_train_raw = train_pack["X_raw"]
        y_train = train_pack["y"]
        train_ids = train_pack["sample_ids"]
        train_domain_values = train_pack.get("domain_values")
        X_val_raw = val_pack["X_raw"]
        y_val = val_pack["y"]
        val_ids = val_pack["sample_ids"]
        val_domain_values = val_pack.get("domain_values")

        save_split_info(
            os.path.join(cfg.outdir, "val_samples.csv"),
            val_ids,
            "val",
            y_val,
            task,
        )

        val_overlap_info = val_pack["overlap_info"]
        val_filter_info = val_pack["filter_info"]
    else:
        split_pack = split_train_val(
            train_pack["X_raw"],
            train_pack["y"],
            train_pack["sample_ids"],
            cfg,
            domain_values=train_pack.get("domain_values"),
        )
        X_train_raw = split_pack["X_train_raw"]
        y_train = split_pack["y_train"]
        train_ids = split_pack["train_ids"]
        train_domain_values = split_pack.get("train_domain_values")
        X_val_raw = split_pack["X_val_raw"]
        y_val = split_pack["y_val"]
        val_ids = split_pack["val_ids"]
        val_domain_values = split_pack.get("val_domain_values")

        split_df = pd.DataFrame({
            "sample_id": np.concatenate([train_ids, val_ids]),
            "split": ["train"] * len(train_ids) + ["val"] * len(val_ids),
            "label_id": np.concatenate([y_train, y_val]),
            "label_name": inverse_transform_labels(np.concatenate([y_train, y_val]), task),
        })
        split_df.to_csv(os.path.join(cfg.outdir, "train_val_split_samples.csv"), index=False)
        save_json(train_pack["filter_info"], os.path.join(cfg.outdir, "train_label_filter_info.json"))
        val_overlap_info = None
        val_filter_info = None

    # save train feature order
    pd.DataFrame({"feature_name": train_pack["feature_names"]}).to_csv(
        os.path.join(cfg.outdir, "feature_order.csv"), index=False
    )

    # --------------------------------------------------------
    # Step 3. fit preprocessor on train only
    # --------------------------------------------------------
    preproc = fit_preprocessor_on_train(X_train_raw)
    X_train = preproc["X_train_scaled"]
    X_val = transform_with_preprocessor(X_val_raw, preproc)

    np.save(os.path.join(cfg.outdir, "train_feature_medians.npy"), preproc["train_medians"])
    np.save(os.path.join(cfg.outdir, "scaler_mean.npy"), preproc["scaler"].mean_.astype(np.float32))
    np.save(os.path.join(cfg.outdir, "scaler_scale.npy"), preproc["scaler"].scale_.astype(np.float32))

    # --------------------------------------------------------
    # Optional domain encoding for DANN / DG methods
    # --------------------------------------------------------
    cfg.dg_method = normalize_dg_method(cfg.dg_method)
    if cfg.dg_method == "dann":
        cfg.use_dann_loss = True

    if dg_requires_domain_labels(cfg) and train_domain_values is None:
        warnings.warn(
            f"dg_method={cfg.dg_method!r} requires --domain-col for meaningful domain generalization/adaptation, "
            "but --domain-col was not provided. All samples are assigned to one dummy domain; "
            "the domain-invariance penalty will be ineffective.",
            UserWarning,
        )
        train_domain_values = np.array(["domain0"] * len(y_train))
        val_domain_values = np.array(["domain0"] * len(y_val))

    domain_mapping, train_domains = fit_domain_mapping(train_domain_values)
    val_domains = transform_domain_values(val_domain_values, domain_mapping)
    if domain_mapping is not None:
        cfg.num_domains = max(cfg.num_domains, len(domain_mapping))
        save_json(domain_mapping, os.path.join(cfg.outdir, "domain_mapping.json"))

    # --------------------------------------------------------
    # TabPFN / TabICL branch
    # --------------------------------------------------------
    if cfg.model_name.lower() in FOUNDATION_MODEL_NAMES:
        run_foundation_pipeline(
            cfg=cfg,
            task=task,
            X_train=X_train,
            y_train=y_train,
            train_ids=train_ids,
            X_val=X_val,
            y_val=y_val,
            val_ids=val_ids,
            train_pack=train_pack,
            preproc=preproc,
            val_overlap_info=val_overlap_info,
            val_filter_info=val_filter_info,
        )
        return

    if cfg.model_name.lower() in SKLEARN_MODEL_NAMES:
        run_sklearn_pipeline(
            cfg=cfg,
            task=task,
            X_train=X_train,
            y_train=y_train,
            train_ids=train_ids,
            X_val=X_val,
            y_val=y_val,
            val_ids=val_ids,
            train_pack=train_pack,
            preproc=preproc,
            val_overlap_info=val_overlap_info,
            val_filter_info=val_filter_info,
        )
        return

    # --------------------------------------------------------
    # Step 4. build model / criterion / optimizer
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, criterion, class_weights_np, resolved_flags = build_model_and_criterion(
        cfg=cfg,
        task=task,
        num_features=X_train.shape[1],
        y_train=y_train,
        device=device,
    )

    save_json(resolved_flags, os.path.join(cfg.outdir, "resolved_flags.json"))

    if class_weights_np is not None:
        pd.DataFrame({"class_name": task.class_names, "weight": class_weights_np}).to_csv(
            os.path.join(cfg.outdir, "class_weights.csv"), index=False
        )

    dg_state = build_dg_state(cfg, model, X_train, task, device)
    save_json({
        "dg_method": cfg.dg_method,
        "dg_weight": cfg.dg_weight,
        "num_domains": cfg.num_domains,
        "has_cdan_head": dg_state.get("cdan_head") is not None,
        "note": "For CORAL/MMD/CDAN, aux['features'] from the model is preferred; otherwise logits are used as a fallback.",
    }, os.path.join(cfg.outdir, "domain_generalization_config.json"))

    optimizer_params = list(model.parameters()) + dg_parameters(dg_state)
    optimizer = torch.optim.AdamW(optimizer_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    early = EarlyStopping(patience=cfg.patience, mode="max")

    train_loader = build_loader(X_train, y_train, cfg, shuffle=True, domains=train_domains)
    val_loader = build_loader(X_val, y_val, cfg, shuffle=False, domains=val_domains)
    pseudo_loader = build_pseudo_loader(cfg, train_pack, preproc)

    # --------------------------------------------------------
    # Step 5. train and early-stop by task-aware validation score
    # --------------------------------------------------------
    history = []
    last_selection_metric_name = None
    for epoch in range(1, cfg.epochs + 1):
        train_metrics, _, _, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            amp_enabled=cfg.amp,
            is_train=True,
            criterion=criterion,
            class_names=task.class_names,
            max_grad_norm=cfg.max_grad_norm,
            mixup_alpha=cfg.mixup_alpha if cfg.model_name.lower() in MIXUP_MODEL_NAMES else 0.0,
            cfg=cfg,
            dg_state=dg_state,
        )
        pseudo_metrics = {"pseudo_loss": 0.0, "pseudo_kept": 0, "pseudo_total": 0, "pseudo_keep_rate": 0.0}
        if normalize_dg_method(cfg.dg_method) == "self_training" and epoch >= int(cfg.pseudo_warmup_epochs):
            pseudo_metrics = run_pseudo_label_epoch(
                model=model,
                loader=pseudo_loader,
                optimizer=optimizer,
                device=device,
                amp_enabled=cfg.amp,
                cfg=cfg,
                max_grad_norm=cfg.max_grad_norm,
            )

        val_metrics, _, _, _, _ = predict(
            model=model,
            loader=val_loader,
            device=device,
            amp_enabled=cfg.amp,
            criterion=criterion,
            class_names=task.class_names,
        )

        scheduler.step()

        selection_metric_name, selection_score = select_validation_score(val_metrics, task)
        last_selection_metric_name = selection_metric_name

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"pseudo_{k}": v for k, v in pseudo_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row["val_selection_metric"] = selection_metric_name
        row["val_selection_score"] = selection_score
        history.append(row)

        if early.step(selection_score, model):
            break

    if early.best_state is not None:
        model.load_state_dict(early.best_state)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(cfg.outdir, "history.csv"), index=False)
    plot_history(hist_df, cfg.outdir)

    torch.save(model.state_dict(), os.path.join(cfg.outdir, "best_model.pt"))

    # --------------------------------------------------------
    # Step 6. evaluate best model on train / val
    # --------------------------------------------------------
    train_final, train_probs, train_preds, train_true, _ = predict(
        model, train_loader, device, cfg.amp, criterion, task.class_names
    )
    val_final, val_probs, val_preds, val_true, _ = predict(
        model, val_loader, device, cfg.amp, criterion, task.class_names
    )

    save_evaluation_outputs(
        save_dir=os.path.join(cfg.outdir, "eval_train"),
        split_name="train",
        class_names=task.class_names,
        metrics=train_final,
        probs=train_probs,
        preds=train_preds,
        y_true=train_true,
        sample_ids=train_ids,
        overlap_info=None,
        filter_info=train_pack["filter_info"],
    )
    save_evaluation_outputs(
        save_dir=os.path.join(cfg.outdir, "eval_val"),
        split_name="val",
        class_names=task.class_names,
        metrics=val_final,
        probs=val_probs,
        preds=val_preds,
        y_true=val_true,
        sample_ids=val_ids,
        overlap_info=val_overlap_info,
        filter_info=val_filter_info,
    )

    # --------------------------------------------------------
    # Step 7. evaluate external independent test cohorts
    # --------------------------------------------------------
    external_specs = load_external_test_specs(cfg.external_tests_json, cfg)
    external_rows = []

    for spec in external_specs:
        pack = load_dataset_as_arrays(
            spec,
            task=task,
            expected_feature_order=train_pack["feature_names"],
            external_feature_policy=cfg.external_feature_policy,
            min_feature_overlap_ratio=cfg.min_feature_overlap_ratio,
        )
        X_ext = transform_with_preprocessor(pack["X_raw"], preproc)
        y_ext = pack["y"]
        ids_ext = pack["sample_ids"]

        ext_loader = build_loader(X_ext, y_ext, cfg, shuffle=False)
        ext_metrics, ext_probs, ext_preds, ext_true, _ = predict(
            model, ext_loader, device, cfg.amp, criterion, task.class_names
        )

        save_evaluation_outputs(
            save_dir=os.path.join(cfg.outdir, "external_tests", spec.name),
            split_name=f"external_test__{spec.name}",
            class_names=task.class_names,
            metrics=ext_metrics,
            probs=ext_probs,
            preds=ext_preds,
            y_true=ext_true,
            sample_ids=ids_ext,
            overlap_info=pack["overlap_info"],
            filter_info=pack["filter_info"],
        )

        external_rows.append(make_external_summary_row(spec.name, y_ext, ext_metrics, pack))

    if len(external_rows) > 0:
        external_df = pd.DataFrame(external_rows)
    else:
        external_df = pd.DataFrame(columns=[
            "dataset", "n_samples", "acc", "bal_acc",
            "precision", "recall", "sensitivity", "specificity", "f1", "mcc",
            "auc_binary", "auprc_binary",
            "macro_f1", "weighted_f1", "auc_ovr_macro", "auc_ovr_weighted",
        ])
    external_df.to_csv(os.path.join(cfg.outdir, "external_test_summary.csv"), index=False)

    # --------------------------------------------------------
    # Step 8. final summary
    # --------------------------------------------------------
    final_selection_metric_name, final_selection_score = select_validation_score(val_final, task)
    summary = {
        "task_type": task.task_type,
        "class_names": task.class_names,
        "binary_positive_class": task.class_names[1] if task.num_classes == 2 else None,
        "binary_negative_class": task.class_names[0] if task.num_classes == 2 else None,
        "model_name": cfg.model_name,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_features": int(X_train.shape[1]),
        "best_val_selection_score": float(early.best) if early.best is not None else np.nan,
        "best_val_selection_metric": last_selection_metric_name,
        "final_val_selection_metric": final_selection_metric_name,
        "final_val_selection_score": float(final_selection_score),
        "selection_rule": "binary: auc_binary -> f1 -> bal_acc; multiclass: macro_f1 -> auc_ovr_macro -> bal_acc",
        "train_metrics": _metrics_to_jsonable(train_final),
        "val_metrics": _metrics_to_jsonable(val_final),
        "resolved_flags": resolved_flags,
        "n_external_tests": int(len(external_rows)),
        "external_tests": external_rows,
    }
    save_json(summary, os.path.join(cfg.outdir, "summary.json"))

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
