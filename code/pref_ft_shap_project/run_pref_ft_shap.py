#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explain PREF-FT predictions using the affected-class logit margin.

The canonical model is shared with training. Training feature order and saved
imputation/scaling parameters are reused. SHAP is an optional plotting dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyTorch is required. Install it first, e.g. pip install torch") from e

try:
    import shap
except ImportError:  # plotting is optional until the CLI is run
    shap = None

try:
    if __package__:
        from .pref_ft_model import PREF_FT
    else:
        from pref_ft_model import PREF_FT
except Exception as e:  # pragma: no cover
    raise RuntimeError("Could not import PREF_FT. Put pref_ft_model.py next to this script.") from e


# ============================================================
# Defaults copied from the provided training script
# ============================================================

DEFAULT_MATRIX_ORIENTATION = "cpg_rows"
DEFAULT_SAMPLE_COL = "SampleID"
DEFAULT_LABEL_COL = "disease.state"
DEFAULT_CLASS_ORDER = "unaffected, affected"

DEFAULT_D_MODEL = 128
DEFAULT_DROPOUT = 0.2
DEFAULT_N_TRANSFORMER_LAYERS = 3
DEFAULT_NUM_HEADS = 4
DEFAULT_TABM_NUM_MEMBERS = 4
DEFAULT_USE_LINEAR_RESIDUAL = True
DEFAULT_GATE_DROPOUT = 0.05
DEFAULT_LINEAR_DROPOUT = 0.1
DEFAULT_USE_POSITIVE_OVR = True
DEFAULT_USE_PROTOTYPE_RESIDUAL = True


# ============================================================
# General utilities
# ============================================================

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_table(path: str | Path) -> pd.DataFrame:
    """Read csv/tsv/txt/csv.gz/xlsx files robustly."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    suffixes = "".join(p.suffixes).lower()
    if suffixes.endswith(".xlsx") or suffixes.endswith(".xls"):
        return pd.read_excel(p)
    if suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz"):
        return pd.read_csv(p, sep="\t")

    encodings = ["utf-8", "utf-8-sig", "gb18030", "gbk", "cp1252", "latin1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(p, sep=None, engine="python", encoding=enc)
        except Exception as e:
            last_err = e
    raise ValueError(f"Failed to read table: {p}. Last error: {last_err}")


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


def parse_class_order(class_order: str) -> List[str]:
    class_names = [x.strip() for x in str(class_order).split(",") if x.strip()]
    if len(class_names) != 2:
        raise ValueError(f"This SHAP script only supports binary tasks. Got class_order={class_names}")
    if len(set(class_names)) != 2:
        raise ValueError(f"class_order contains duplicate labels: {class_names}")
    return class_names


def read_label_map(label_map_json: Optional[str]) -> Optional[Dict[str, str]]:
    if not label_map_json:
        return None
    raw = label_map_json.strip()
    if Path(raw).exists():
        return json.loads(Path(raw).read_text(encoding="utf-8"))
    return json.loads(raw)


def canonicalize_label_series(series: pd.Series, label_map: Optional[Dict[str, str]] = None) -> pd.Series:
    out = series.astype(str).copy()
    if label_map is not None:
        out = out.map(lambda x: label_map.get(x, x))
    return out.astype(str)


def read_feature_list(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"feature_list not found: {p}")
    if p.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(p, sep=sep)
        preferred = ["feature_name", "feature", "CpG", "cpg", "probe", "probe_id"]
        col = next((c for c in preferred if c in df.columns), df.columns[0])
        features = df[col].astype(str).tolist()
    else:
        features = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    features = [f for f in features if f and f.lower() != "nan"]
    if len(features) == 0:
        raise ValueError(f"feature_list is empty: {p}")
    return features


def try_auto_training_files(args: argparse.Namespace) -> argparse.Namespace:
    """Use checkpoint, feature_order and scaler files from --training_outdir when present."""
    if not args.training_outdir:
        return args
    td = Path(args.training_outdir)
    if not td.exists():
        raise FileNotFoundError(f"--training_outdir not found: {td}")

    config_path = td / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        supplied_options = {item.split("=", 1)[0].lstrip("-").replace("-", "_")
                            for item in sys.argv[1:] if item.startswith("--")}
        for key in ["d_model", "dropout", "n_transformer_layers", "num_heads", "tabm_num_members",
                    "use_linear_residual", "gate_dropout", "linear_dropout", "use_positive_ovr",
                    "use_prototype_residual"]:
            if key in config and key not in supplied_options:
                setattr(args, key, config[key])

    if not args.checkpoint:
        candidate = td / "best_model.pt"
        if candidate.exists():
            args.checkpoint = str(candidate)

    if not args.feature_list:
        candidate = td / "feature_order.csv"
        if candidate.exists():
            args.feature_list = str(candidate)

    if not args.preprocess_dir and not args.already_standardized:
        needed = [td / "train_feature_medians.npy", td / "scaler_mean.npy", td / "scaler_scale.npy"]
        if all(p.exists() for p in needed):
            args.preprocess_dir = str(td)

    label_mapping = td / "label_mapping.json"
    if label_mapping.exists() and args.class_order == DEFAULT_CLASS_ORDER:
        try:
            obj = json.loads(label_mapping.read_text(encoding="utf-8"))
            if "class_names" in obj and len(obj["class_names"]) == 2:
                args.class_order = ", ".join(obj["class_names"])
        except Exception:
            pass
    return args


def restrict_background_to_training_split(pack: Dict[str, Any], training_outdir: Optional[str],
                                          allow_all: bool = False) -> Dict[str, Any]:
    """Use saved train IDs to prevent validation samples entering the background."""
    if not training_outdir or allow_all:
        return pack
    split_path = Path(training_outdir) / "train_val_split_samples.csv"
    if not split_path.exists():
        return pack
    split = pd.read_csv(split_path)
    train_ids = set(split.loc[split["split"] == "train", "sample_id"].astype(str))
    mask = np.array([str(x) in train_ids for x in pack["sample_ids"]])
    if not mask.any():
        raise ValueError("Background has no samples from the saved training split.")
    restricted = dict(pack)
    restricted["X_raw"] = pack["X_raw"][mask]
    restricted["sample_ids"] = pack["sample_ids"][mask]
    if pack.get("y") is not None:
        restricted["y"] = pack["y"][mask]
    restricted["filter_info"] = {**pack.get("filter_info", {}),
                                  "restricted_to_saved_train_split": True,
                                  "background_n_before_split_filter": int(len(mask)),
                                  "background_n_after_split_filter": int(mask.sum())}
    return restricted


# ============================================================
# Separated feature matrix + metadata loading
# ============================================================

def numeric_feature_block(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    block = df.loc[:, feature_cols].copy()
    block = block.apply(pd.to_numeric, errors="coerce")
    if block.columns.duplicated().any():
        block = block.T.groupby(level=0, sort=False).mean().T
    return block


def matrix_to_sample_feature_df(
    feat: pd.DataFrame,
    meta: pd.DataFrame,
    matrix_orientation: str,
    sample_col: str,
    dataset_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Return sample x CpG dataframe aligned with metadata order."""
    if matrix_orientation == "sample_rows":
        feat = ensure_sample_id_column(feat, sample_col)
        meta_ids = meta[sample_col].astype(str).tolist()
        feat_ids = set(feat[sample_col].astype(str))
        common = [s for s in meta_ids if s in feat_ids]
        if not common:
            raise ValueError(f"[{dataset_name}] No overlapping samples between feature file and metadata file.")
        feat = feat[feat[sample_col].astype(str).isin(common)].copy()
        meta = meta[meta[sample_col].astype(str).isin(common)].copy()
        feat = feat.set_index(sample_col).loc[common].reset_index()
        meta = meta.set_index(sample_col).loc[common].reset_index()
        feature_cols = [c for c in feat.columns if c != sample_col]
        block = numeric_feature_block(feat, feature_cols)
        feat_out = pd.concat([feat[[sample_col]].reset_index(drop=True), block.reset_index(drop=True)], axis=1)
        return feat_out, meta, block.columns.astype(str).tolist()

    if matrix_orientation == "cpg_rows":
        feat = feat.copy()
        cpg_col = feat.columns[0]
        feat[cpg_col] = feat[cpg_col].astype(str)
        feat = feat.set_index(cpg_col)
        feat.columns = feat.columns.astype(str)
        meta_ids = meta[sample_col].astype(str).tolist()
        common = [s for s in meta_ids if s in set(feat.columns)]
        if not common:
            raise ValueError(f"[{dataset_name}] No overlapping samples between cpg_rows matrix columns and metadata {sample_col}.")

        feat = feat.loc[:, common].apply(pd.to_numeric, errors="coerce")
        if feat.index.duplicated().any():
            feat = feat.groupby(level=0, sort=False).mean()

        feat_t = feat.T.reset_index().rename(columns={"index": sample_col})
        meta = meta[meta[sample_col].astype(str).isin(common)].copy()
        feat_t = feat_t.set_index(sample_col).loc[common].reset_index()
        meta = meta.set_index(sample_col).loc[common].reset_index()
        feature_cols = [c for c in feat_t.columns if c != sample_col]
        return feat_t, meta, [str(c) for c in feature_cols]

    raise ValueError("matrix_orientation must be 'sample_rows' or 'cpg_rows'.")


def align_feature_order(
    numeric_df: pd.DataFrame,
    expected_feature_order: Optional[List[str]],
    dataset_name: str,
    external_feature_policy: str,
    min_feature_overlap_ratio: float,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    raw_features = [str(c) for c in numeric_df.columns]
    numeric_df.columns = raw_features
    overlap_info: Dict[str, Any] = {
        "n_raw_features": int(len(raw_features)),
        "n_expected_features": int(len(expected_feature_order)) if expected_feature_order is not None else None,
        "n_overlap_with_expected": None,
        "overlap_ratio_against_expected": None,
        "missing_expected_features": None,
        "extra_features": None,
    }

    if expected_feature_order is None:
        return numeric_df, raw_features, overlap_info

    expected_feature_order = [str(f) for f in expected_feature_order]
    present = [f for f in expected_feature_order if f in numeric_df.columns]
    missing = [f for f in expected_feature_order if f not in numeric_df.columns]
    extra = [f for f in numeric_df.columns if f not in set(expected_feature_order)]
    overlap_ratio = len(present) / max(1, len(expected_feature_order))
    overlap_info.update({
        "n_overlap_with_expected": int(len(present)),
        "overlap_ratio_against_expected": float(overlap_ratio),
        "missing_expected_features": int(len(missing)),
        "extra_features": int(len(extra)),
        "missing_examples": missing[:20],
        "extra_examples": extra[:20],
    })

    if external_feature_policy == "strict" and missing:
        raise ValueError(
            f"[{dataset_name}] strict feature policy failed: {len(missing)} expected training features are missing. "
            f"Examples: {missing[:10]}"
        )
    if overlap_ratio < min_feature_overlap_ratio:
        raise ValueError(
            f"[{dataset_name}] feature overlap ratio too low: {overlap_ratio:.4f} < {min_feature_overlap_ratio:.4f}."
        )
    aligned = numeric_df.reindex(columns=expected_feature_order)
    return aligned, expected_feature_order, overlap_info


def load_separated_dataset(
    feature_path: str,
    metadata_path: str,
    dataset_name: str,
    matrix_orientation: str,
    sample_col: str,
    label_col: str,
    class_names: List[str],
    label_map: Optional[Dict[str, str]],
    expected_feature_order: Optional[List[str]],
    external_feature_policy: str,
    min_feature_overlap_ratio: float,
) -> Dict[str, Any]:
    feat = read_table(feature_path)
    meta = read_table(metadata_path)
    meta = ensure_sample_id_column(meta, sample_col)
    if label_col not in meta.columns:
        raise ValueError(f"[{dataset_name}] label_col={label_col} not found in metadata columns: {meta.columns.tolist()[:30]}")
    meta[sample_col] = meta[sample_col].astype(str)
    meta[label_col] = canonicalize_label_series(meta[label_col], label_map)

    feat_df, meta_df, feature_cols = matrix_to_sample_feature_df(
        feat=feat,
        meta=meta,
        matrix_orientation=matrix_orientation,
        sample_col=sample_col,
        dataset_name=dataset_name,
    )

    labels_before = meta_df[label_col].astype(str)
    keep_mask = labels_before.isin(set(class_names)).to_numpy()
    dropped_counts = labels_before.loc[~pd.Series(keep_mask, index=labels_before.index)].value_counts().to_dict()
    feat_df = feat_df.loc[keep_mask].reset_index(drop=True)
    meta_df = meta_df.loc[keep_mask].reset_index(drop=True)
    if len(meta_df) == 0:
        raise ValueError(f"[{dataset_name}] no samples remain after filtering to class_order={class_names}")

    # Keep metadata order after filtering.
    feat_df = feat_df.set_index(sample_col).loc[meta_df[sample_col].astype(str).tolist()].reset_index()
    y = np.array([class_names.index(x) for x in meta_df[label_col].astype(str).tolist()], dtype=np.int64)
    sample_ids = meta_df[sample_col].astype(str).to_numpy()

    numeric_df = feat_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    numeric_df, final_features, overlap_info = align_feature_order(
        numeric_df=numeric_df,
        expected_feature_order=expected_feature_order,
        dataset_name=dataset_name,
        external_feature_policy=external_feature_policy,
        min_feature_overlap_ratio=min_feature_overlap_ratio,
    )

    X = numeric_df.to_numpy(dtype=np.float32)
    filter_info = {
        "n_samples_after_alignment_before_label_filter": int(len(labels_before)),
        "n_samples_after_label_filter": int(len(meta_df)),
        "n_samples_dropped_by_label_filter": int((~keep_mask).sum()),
        "allowed_labels": class_names,
        "dropped_label_counts": dropped_counts,
        "kept_label_counts": meta_df[label_col].astype(str).value_counts().to_dict(),
    }
    return {
        "X_raw": X,
        "y": y,
        "sample_ids": sample_ids,
        "feature_names": final_features,
        "metadata": meta_df,
        "filter_info": filter_info,
        "overlap_info": overlap_info,
    }


# ============================================================
# Legacy single CSV loading
# ============================================================

def infer_feature_columns(
    df: pd.DataFrame,
    label_col: Optional[str],
    id_col: Optional[str],
    split_col: Optional[str],
    extra_exclude_cols: Sequence[str],
) -> List[str]:
    exclude = set(c for c in [label_col, id_col, split_col] if c)
    exclude.update([c for c in extra_exclude_cols if c])
    candidate_cols = [c for c in df.columns if c not in exclude]
    numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 0:
        raise ValueError("No numeric feature columns were found. Provide --feature_list or check the CSV.")
    return [str(c) for c in numeric_cols]


def map_labels_with_class_order(y_raw: pd.Series, class_names: List[str], positive_label: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    y = y_raw.copy()
    if pd.api.types.is_numeric_dtype(y):
        vals = sorted(pd.unique(y.dropna()))
        if set(vals).issubset({0, 1, 0.0, 1.0}):
            return y.astype(int).to_numpy(), {"0": 0, "1": 1}
        if len(vals) == 2 and positive_label is not None:
            pos = float(positive_label)
            return (y.astype(float) == pos).astype(int).to_numpy(), {str(v): int(float(v) == pos) for v in vals}
        raise ValueError(f"Numeric labels must be 0/1, or provide --positive_label. Found values: {vals[:10]}")

    y_str = y.astype(str)
    vals = sorted(pd.unique(y_str.dropna()))
    if set(vals).issubset(set(class_names)):
        mapping = {class_names[0]: 0, class_names[1]: 1}
        return y_str.map(mapping).astype(int).to_numpy(), mapping
    if len(vals) == 2 and positive_label is not None:
        mapping = {v: int(v == str(positive_label)) for v in vals}
        return y_str.map(mapping).astype(int).to_numpy(), mapping
    raise ValueError(f"Cannot map labels {vals}. Use --class_order or --positive_label.")


def load_combined_csv(
    csv_path: str,
    dataset_name: str,
    feature_cols: Optional[List[str]],
    label_col: Optional[str],
    id_col: Optional[str],
    split_col: Optional[str],
    extra_exclude_cols: Sequence[str],
    class_names: List[str],
    positive_label: Optional[str],
    expected_feature_order: Optional[List[str]],
    external_feature_policy: str,
    min_feature_overlap_ratio: float,
) -> Dict[str, Any]:
    df = read_table(csv_path)
    if feature_cols is None:
        feature_cols = infer_feature_columns(df, label_col, id_col, split_col, extra_exclude_cols)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{dataset_name}] {len(missing)} feature columns are missing. Examples: {missing[:10]}")
    numeric_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    numeric_df, final_features, overlap_info = align_feature_order(
        numeric_df=numeric_df,
        expected_feature_order=expected_feature_order,
        dataset_name=dataset_name,
        external_feature_policy=external_feature_policy,
        min_feature_overlap_ratio=min_feature_overlap_ratio,
    )

    y_np = None
    label_map = None
    if label_col and label_col in df.columns:
        y_np, label_map = map_labels_with_class_order(df[label_col], class_names, positive_label)
    ids = df[id_col].astype(str).to_numpy() if id_col and id_col in df.columns else np.array([f"sample_{i}" for i in range(len(df))])
    return {
        "X_raw": numeric_df.to_numpy(dtype=np.float32),
        "y": y_np,
        "sample_ids": ids,
        "feature_names": final_features,
        "metadata": df,
        "filter_info": {"label_map": label_map},
        "overlap_info": overlap_info,
    }


# ============================================================
# Preprocessing
# ============================================================

def impute_with_medians(X: np.ndarray, medians: np.ndarray) -> np.ndarray:
    X_imp = X.astype(np.float32, copy=True)
    inds = np.where(np.isnan(X_imp))
    if len(inds[0]) > 0:
        X_imp[inds] = np.take(medians.astype(np.float32), inds[1])
    return X_imp


def apply_training_preprocessor(X: np.ndarray, preprocess_dir: Optional[str]) -> np.ndarray:
    """Apply the exact npy preprocessor saved by the provided training script."""
    if not preprocess_dir:
        # At minimum fill NaN to avoid SHAP failure.
        if np.isnan(X).any():
            med = np.nanmedian(X, axis=0)
            med = np.where(np.isnan(med), 0.0, med).astype(np.float32)
            warnings.warn("No --preprocess_dir was provided; filling NaNs with medians computed from this input matrix.")
            return impute_with_medians(X, med)
        return X.astype(np.float32)

    p = Path(preprocess_dir)
    med_path = p / "train_feature_medians.npy"
    mean_path = p / "scaler_mean.npy"
    scale_path = p / "scaler_scale.npy"
    for f in [med_path, mean_path, scale_path]:
        if not f.exists():
            raise FileNotFoundError(f"Preprocessor file missing: {f}")
    med = np.load(med_path).astype(np.float32)
    mean = np.load(mean_path).astype(np.float32)
    scale = np.load(scale_path).astype(np.float32)
    if X.shape[1] != len(mean):
        raise ValueError(f"Preprocessor feature count mismatch: X has {X.shape[1]}, scaler has {len(mean)}.")
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    X_imp = impute_with_medians(X, med)
    return ((X_imp - mean) / scale).astype(np.float32)


# ============================================================
# Sampling and checkpoint loading
# ============================================================

def stratified_sample_indices(y: Optional[np.ndarray], n: int, seed: int, prefer_balanced: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    total = len(y) if y is not None else 0
    if y is None or not prefer_balanced:
        size = min(int(n), int(total))
        return np.sort(rng.choice(np.arange(total), size=size, replace=False))
    indices: List[int] = []
    classes = sorted(np.unique(y))
    per_class = max(1, int(np.ceil(n / max(1, len(classes)))))
    for c in classes:
        idx_c = np.where(y == c)[0]
        if len(idx_c) == 0:
            continue
        k = min(per_class, len(idx_c))
        indices.extend(rng.choice(idx_c, size=k, replace=False).tolist())
    indices_np = np.array(indices, dtype=int)
    if len(indices_np) > n:
        indices_np = rng.choice(indices_np, size=n, replace=False)
    return np.sort(indices_np)


def random_sample_indices(num_samples: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = min(int(n), int(num_samples))
    return np.sort(rng.choice(np.arange(num_samples), size=size, replace=False))


def is_plain_state_dict(obj: object) -> bool:
    return isinstance(obj, dict) and len(obj) > 0 and all(torch.is_tensor(v) for v in obj.values())


def extract_state_dict(ckpt: object) -> Dict[str, torch.Tensor]:
    if is_plain_state_dict(ckpt):
        return ckpt  # type: ignore[return-value]
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a state_dict or a dict containing a state_dict.")
    for key in ["model_state_dict", "state_dict", "model", "net", "ema_state_dict"]:
        if key in ckpt and is_plain_state_dict(ckpt[key]):
            return ckpt[key]
    for key, value in ckpt.items():
        if is_plain_state_dict(value):
            warnings.warn(f"Using nested state_dict under checkpoint key: {key}")
            return value
    raise ValueError(f"Could not find a valid state_dict in checkpoint keys: {list(ckpt.keys())[:20]}")


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ["module.", "model.", "net.", "_orig_mod."]
    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_k = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if new_k.startswith(p):
                    new_k = new_k[len(p):]
                    changed = True
        cleaned[new_k] = v
    return cleaned


def build_model(args: argparse.Namespace, num_features: int, device: torch.device) -> nn.Module:
    model = PREF_FT(
        num_features=num_features,
        num_classes=2,
        d_model=args.d_model,
        n_transformer_layers=args.n_transformer_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        tabm_num_members=args.tabm_num_members,
        use_linear_residual=args.use_linear_residual,
        gate_dropout=args.gate_dropout,
        linear_dropout=args.linear_dropout,
        use_positive_ovr=args.use_positive_ovr,
        use_prototype_residual=args.use_prototype_residual,
    )
    if not args.checkpoint:
        raise ValueError("Please provide --checkpoint, or provide --training_outdir containing best_model.pt.")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = clean_state_dict_keys(extract_state_dict(ckpt))
    load_info = model.load_state_dict(state_dict, strict=not args.non_strict_load)
    if args.non_strict_load:
        print("[load] missing keys:", getattr(load_info, "missing_keys", []))
        print("[load] unexpected keys:", getattr(load_info, "unexpected_keys", []))
    model.to(device)
    model.eval()
    return model


class BinaryTargetWrapper(nn.Module):
    """Wrap model so SHAP receives one scalar output per sample."""

    def __init__(self, model: nn.Module, target: str = "logit_margin", positive_class: int = 1):
        super().__init__()
        self.model = model
        self.target = target
        self.positive_class = int(positive_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(x)
        pos = self.positive_class
        neg = 1 - pos
        if self.target == "logit_margin":
            y = logits[:, pos] - logits[:, neg]
        elif self.target == "positive_logit":
            y = logits[:, pos]
        elif self.target == "positive_prob":
            y = torch.softmax(logits, dim=1)[:, pos]
        else:
            raise ValueError(f"Unknown explain_target: {self.target}")
        return y.view(-1, 1)


@torch.no_grad()
def predict_batches(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> pd.DataFrame:
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    logits_list: List[np.ndarray] = []
    probs_list: List[np.ndarray] = []
    for (xb,) in loader:
        xb = xb.to(device)
        logits, _ = model(xb)
        probs = torch.softmax(logits, dim=1)
        logits_list.append(logits.detach().cpu().numpy())
        probs_list.append(probs.detach().cpu().numpy())
    logits_np = np.concatenate(logits_list, axis=0)
    probs_np = np.concatenate(probs_list, axis=0)
    return pd.DataFrame({
        "logit_class0_negative": logits_np[:, 0],
        "logit_class1_positive": logits_np[:, 1],
        "logit_margin_positive_minus_negative": logits_np[:, 1] - logits_np[:, 0],
        "prob_class0_negative": probs_np[:, 0],
        "prob_class1_positive": probs_np[:, 1],
        "pred_class": probs_np.argmax(axis=1),
    })


# ============================================================
# SHAP computation and plotting
# ============================================================

def make_gradient_explainer(wrapper: nn.Module, background_t: torch.Tensor, batch_size: int):
    try:
        return shap.GradientExplainer(wrapper, background_t, batch_size=batch_size)
    except TypeError:
        return shap.GradientExplainer(wrapper, background_t)


def unwrap_shap_values(shap_values: object) -> np.ndarray:
    if isinstance(shap_values, list):
        arr = shap_values[0]
    else:
        arr = shap_values
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr[:, :, -1]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape: {arr.shape}")
    return arr.astype(np.float32)


def compute_shap_values(
    wrapper: nn.Module,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    device: torch.device,
    batch_size: int,
    nsamples: int,
    chunk_size: int,
) -> np.ndarray:
    background_t = torch.from_numpy(X_background.astype(np.float32)).to(device)
    explainer = make_gradient_explainer(wrapper, background_t, batch_size=batch_size)
    chunks: List[np.ndarray] = []
    for start in range(0, len(X_explain), int(chunk_size)):
        end = min(start + int(chunk_size), len(X_explain))
        xb = torch.from_numpy(X_explain[start:end].astype(np.float32)).to(device)
        try:
            vals = explainer.shap_values(xb, nsamples=nsamples)
        except TypeError:
            vals = explainer.shap_values(xb)
        chunks.append(unwrap_shap_values(vals))
        print(f"[shap] explained samples {start}:{end}")
    return np.concatenate(chunks, axis=0)




def compute_expected_gradients_shap_values(
    wrapper: nn.Module,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    device: torch.device,
    nsamples: int,
    chunk_size: int,
    seed: int,
) -> np.ndarray:
    """
    Fast manual Expected Gradients estimator.

    SHAP's GradientExplainer is also based on expected gradients for differentiable
    models. This implementation avoids version-specific SHAP/PyTorch slowdowns and
    returns feature attributions with the same practical interpretation: positive
    values increase the selected model output relative to the background baseline.
    """
    wrapper.eval()
    rng = np.random.default_rng(seed)
    background_t_all = torch.from_numpy(X_background.astype(np.float32)).to(device)
    chunks: List[np.ndarray] = []
    nsamples = max(1, int(nsamples))
    for start in range(0, len(X_explain), int(chunk_size)):
        end = min(start + int(chunk_size), len(X_explain))
        x = torch.from_numpy(X_explain[start:end].astype(np.float32)).to(device)
        bsz = x.shape[0]
        acc = torch.zeros_like(x)
        for _ in range(nsamples):
            bg_idx = torch.as_tensor(
                rng.integers(0, len(X_background), size=bsz),
                dtype=torch.long,
                device=device,
            )
            b = background_t_all.index_select(0, bg_idx)
            alpha = torch.rand((bsz, 1), device=device)
            x_interp = (b + alpha * (x - b)).detach().requires_grad_(True)
            out = wrapper(x_interp).sum()
            grad = torch.autograd.grad(out, x_interp, retain_graph=False, create_graph=False)[0]
            acc = acc + grad.detach() * (x - b)
        vals = (acc / float(nsamples)).detach().cpu().numpy().astype(np.float32)
        chunks.append(vals)
        print(f"[shap/expected_gradients] explained samples {start}:{end}", flush=True)
    return np.concatenate(chunks, axis=0)


def get_base_value(wrapper: nn.Module, X_background: np.ndarray, device: torch.device, batch_size: int) -> float:
    values: List[np.ndarray] = []
    ds = TensorDataset(torch.from_numpy(X_background.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            y = wrapper(xb.to(device)).detach().cpu().numpy().reshape(-1)
            values.append(y)
    return float(np.concatenate(values).mean())


def save_summary_plots(shap_values: np.ndarray, X_explain: np.ndarray, feature_names: List[str], out_dir: Path, top_n: int) -> None:
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        max_display=top_n,
        show=False,
        plot_size=(9, max(5, min(18, 0.35 * top_n + 2))),
    )
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "shap_summary_beeswarm.pdf", bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        plot_type="bar",
        max_display=top_n,
        show=False,
        plot_size=(8, max(5, min(18, 0.32 * top_n + 2))),
    )
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_bar.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "shap_summary_bar.pdf", bbox_inches="tight")
    plt.close()


def save_top_feature_tables(shap_values: np.ndarray, X_explain: np.ndarray, feature_names: List[str], out_dir: Path, top_n: int) -> pd.DataFrame:
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    mean_shap = np.mean(shap_values, axis=0)
    std_shap = np.std(shap_values, axis=0)
    mean_value = np.mean(X_explain, axis=0)
    corr: List[float] = []
    for j in range(shap_values.shape[1]):
        sj = shap_values[:, j]
        xj = X_explain[:, j]
        if np.std(sj) < 1e-12 or np.std(xj) < 1e-12:
            corr.append(np.nan)
        else:
            corr.append(float(np.corrcoef(sj, xj)[0, 1]))
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
        "mean_shap": mean_shap,
        "std_shap": std_shap,
        "mean_feature_value_in_explained_samples": mean_value,
        "corr_shap_feature_value": corr,
    }).sort_values("mean_abs_shap", ascending=False)
    df_imp.to_csv(out_dir / "shap_feature_importance_all.csv", index=False)
    df_imp.head(top_n).to_csv(out_dir / f"shap_feature_importance_top{top_n}.csv", index=False)
    return df_imp


def save_top_shap_value_matrix(
    shap_values: np.ndarray,
    X_explain: np.ndarray,
    sample_ids: np.ndarray,
    feature_names: List[str],
    top_df: pd.DataFrame,
    out_dir: Path,
    top_n: int,
) -> None:
    if top_n <= 0:
        return
    top_features = top_df.head(top_n)["feature"].tolist()
    index_map = {f: i for i, f in enumerate(feature_names)}
    top_idx = [index_map[f] for f in top_features]
    shap_df = pd.DataFrame(shap_values[:, top_idx], columns=[f"SHAP__{f}" for f in top_features])
    value_df = pd.DataFrame(X_explain[:, top_idx], columns=[f"VALUE__{f}" for f in top_features])
    out = pd.concat([pd.Series(sample_ids, name="sample_id"), shap_df, value_df], axis=1)
    out.to_csv(out_dir / f"per_sample_top{top_n}_shap_values.csv", index=False)


def save_waterfall_plot(
    shap_values: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    base_value: float,
    sample_idx: int,
    out_dir: Path,
    max_display: int,
) -> None:
    sample_idx = int(np.clip(sample_idx, 0, len(X_explain) - 1))
    exp = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=base_value,
        data=X_explain[sample_idx],
        feature_names=feature_names,
    )
    plt.figure(figsize=(9, max(5, 0.35 * max_display + 2)))
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_waterfall_sample{sample_idx}.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"shap_waterfall_sample{sample_idx}.pdf", bbox_inches="tight")
    plt.close()

    try:
        force = shap.plots.force(base_value, shap_values[sample_idx], X_explain[sample_idx], feature_names=feature_names, matplotlib=False)
        shap.save_html(str(out_dir / f"shap_force_sample{sample_idx}.html"), force)
    except Exception as e:
        warnings.warn(f"Failed to save SHAP force HTML for sample {sample_idx}: {e}")


def save_dependence_plots(shap_values: np.ndarray, X_explain: np.ndarray, feature_names: List[str], selected_features: List[str], out_dir: Path) -> None:
    dep_dir = ensure_dir(out_dir / "dependence_plots")
    for feat in selected_features:
        if feat not in feature_names:
            continue
        try:
            plt.figure()
            shap.dependence_plot(feat, shap_values, X_explain, feature_names=feature_names, show=False)
            plt.tight_layout()
            safe = str(feat).replace("/", "_").replace("\\", "_").replace(":", "_")
            plt.savefig(dep_dir / f"dependence__{safe}.png", dpi=300, bbox_inches="tight")
            plt.savefig(dep_dir / f"dependence__{safe}.pdf", bbox_inches="tight")
            plt.close()
        except Exception as e:
            warnings.warn(f"Failed to plot dependence for {feat}: {e}")


def save_classwise_heatmap(
    shap_values: np.ndarray,
    y: Optional[np.ndarray],
    feature_names: List[str],
    class_names: List[str],
    out_dir: Path,
    top_df: pd.DataFrame,
    top_n: int,
) -> None:
    if y is None:
        return
    classes = sorted(np.unique(y))
    if len(classes) < 2:
        return
    top_features = top_df.head(top_n)["feature"].tolist()
    index_map = {f: i for i, f in enumerate(feature_names)}
    top_idx = [index_map[f] for f in top_features]
    rows = []
    mat = []
    for c in classes:
        mask = y == c
        if not mask.any():
            continue
        label = class_names[int(c)] if int(c) < len(class_names) else f"class{c}"
        rows.append(label)
        mat.append(np.mean(np.abs(shap_values[mask][:, top_idx]), axis=0))
    mat_np = np.asarray(mat, dtype=float)
    df = pd.DataFrame(mat_np, index=rows, columns=top_features)
    df.to_csv(out_dir / f"classwise_mean_abs_shap_top{top_n}.csv")

    plt.figure(figsize=(max(8, 0.35 * top_n + 2), max(3.5, 0.7 * len(rows) + 1.5)))
    im = plt.imshow(mat_np, aspect="auto")
    plt.colorbar(im, label="Mean |SHAP|")
    plt.yticks(np.arange(len(rows)), rows)
    plt.xticks(np.arange(len(top_features)), top_features, rotation=70, ha="right")
    plt.title("Class-wise mean absolute SHAP")
    plt.tight_layout()
    plt.savefig(out_dir / f"classwise_mean_abs_shap_heatmap_top{top_n}.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / f"classwise_mean_abs_shap_heatmap_top{top_n}.pdf", bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def load_background_and_explain(args: argparse.Namespace, class_names: List[str], feature_order: Optional[List[str]]) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    label_map = read_label_map(args.label_map_json)

    # Separated feature + metadata mode, preferred for the provided training script.
    if args.data_features and args.data_metadata:
        bg_pack = load_separated_dataset(
            feature_path=args.data_features,
            metadata_path=args.data_metadata,
            dataset_name="background",
            matrix_orientation=args.matrix_orientation,
            sample_col=args.sample_col,
            label_col=args.label_col,
            class_names=class_names,
            label_map=label_map,
            expected_feature_order=feature_order,
            external_feature_policy=args.external_feature_policy,
            min_feature_overlap_ratio=args.min_feature_overlap_ratio,
        )
        final_features = bg_pack["feature_names"]

        if args.test_features and args.test_metadata:
            ex_pack = load_separated_dataset(
                feature_path=args.test_features,
                metadata_path=args.test_metadata,
                dataset_name="explain",
                matrix_orientation=args.test_matrix_orientation or args.matrix_orientation,
                sample_col=args.test_sample_col or args.sample_col,
                label_col=args.test_label_col or args.label_col,
                class_names=class_names,
                label_map=label_map,
                expected_feature_order=final_features,
                external_feature_policy=args.external_feature_policy,
                min_feature_overlap_ratio=args.min_feature_overlap_ratio,
            )
        else:
            ex_pack = bg_pack
        return bg_pack, ex_pack, final_features

    # Legacy combined CSV mode.
    if args.data_csv:
        bg_pack = load_combined_csv(
            csv_path=args.data_csv,
            dataset_name="background",
            feature_cols=feature_order,
            label_col=args.label_col if args.label_col else None,
            id_col=args.id_col if args.id_col else None,
            split_col=args.split_col if args.split_col else None,
            extra_exclude_cols=args.exclude_cols,
            class_names=class_names,
            positive_label=args.positive_label,
            expected_feature_order=feature_order,
            external_feature_policy=args.external_feature_policy,
            min_feature_overlap_ratio=args.min_feature_overlap_ratio,
        )
        final_features = bg_pack["feature_names"]
        if args.test_csv:
            ex_pack = load_combined_csv(
                csv_path=args.test_csv,
                dataset_name="explain",
                feature_cols=final_features,
                label_col=args.label_col if args.label_col else None,
                id_col=args.id_col if args.id_col else None,
                split_col=args.split_col if args.split_col else None,
                extra_exclude_cols=args.exclude_cols,
                class_names=class_names,
                positive_label=args.positive_label,
                expected_feature_order=final_features,
                external_feature_policy=args.external_feature_policy,
                min_feature_overlap_ratio=args.min_feature_overlap_ratio,
            )
        else:
            ex_pack = bg_pack
            if args.split_col and args.split_col in bg_pack["metadata"].columns:
                df = bg_pack["metadata"]
                test_mask = df[args.split_col].astype(str).eq(str(args.test_split_value)).to_numpy()
                bg_values = [s.strip() for s in args.background_split_values.split(",") if s.strip()]
                bg_mask = df[args.split_col].astype(str).isin(bg_values).to_numpy()
                if test_mask.any():
                    ex_pack = {**bg_pack}
                    ex_pack["X_raw"] = bg_pack["X_raw"][test_mask]
                    ex_pack["y"] = bg_pack["y"][test_mask] if bg_pack["y"] is not None else None
                    ex_pack["sample_ids"] = bg_pack["sample_ids"][test_mask]
                    ex_pack["metadata"] = df.loc[test_mask].reset_index(drop=True)
                if bg_mask.any():
                    bg_pack = {**bg_pack}
                    bg_pack["X_raw"] = bg_pack["X_raw"][bg_mask]
                    bg_pack["y"] = bg_pack["y"][bg_mask] if bg_pack["y"] is not None else None
                    bg_pack["sample_ids"] = bg_pack["sample_ids"][bg_mask]
                    bg_pack["metadata"] = df.loc[bg_mask].reset_index(drop=True)
        return bg_pack, ex_pack, final_features

    raise ValueError("Provide either --data_features and --data_metadata, or legacy --data_csv.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SHAP analysis for binary Positive-Aware FT model with separated methylation feature/metadata files."
    )

    # Project shortcuts.
    parser.add_argument("--training_outdir", type=str, default=None, help="Training output folder containing best_model.pt, feature_order.csv and scaler npy files.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained best_model.pt or state_dict checkpoint.")
    parser.add_argument("--feature_list", type=str, default=None, help="Feature order file. Defaults to training_outdir/feature_order.csv when available.")
    parser.add_argument("--preprocess_dir", type=str, default=None, help="Folder containing train_feature_medians.npy, scaler_mean.npy, scaler_scale.npy.")
    parser.add_argument("--already_standardized", action="store_true", help="Set this if input features have already been standardized exactly as in training.")

    # Preferred separated data layout.
    parser.add_argument("--data_features", "--train-features", type=str, default=None, help="Background/training feature matrix file.")
    parser.add_argument("--data_metadata", "--train-metadata", type=str, default=None, help="Background/training metadata/label file.")
    parser.add_argument("--test_features", "--test-features", type=str, default=None, help="Optional explain/test feature matrix file.")
    parser.add_argument("--test_metadata", "--test-metadata", type=str, default=None, help="Optional explain/test metadata/label file.")

    parser.add_argument("--matrix_orientation", "--matrix-orientation", type=str, default=DEFAULT_MATRIX_ORIENTATION, choices=["cpg_rows", "sample_rows"])
    parser.add_argument("--test_matrix_orientation", "--test-matrix-orientation", type=str, default=None, choices=["cpg_rows", "sample_rows"])
    parser.add_argument("--sample_col", "--sample-col", type=str, default=DEFAULT_SAMPLE_COL)
    parser.add_argument("--test_sample_col", "--test-sample-col", type=str, default=None)
    parser.add_argument("--label_col", "--label-col", type=str, default=DEFAULT_LABEL_COL)
    parser.add_argument("--test_label_col", "--test-label-col", type=str, default=None)
    parser.add_argument("--class_order", "--class-order", type=str, default=DEFAULT_CLASS_ORDER)
    parser.add_argument("--label_map_json", type=str, default=None, help="Optional JSON string or file mapping raw labels to class_order labels.")
    parser.add_argument("--external_feature_policy", "--external-feature-policy", type=str, default="strict", choices=["strict", "fill_train_median"])
    parser.add_argument("--min_feature_overlap_ratio", "--min-feature-overlap-ratio", type=float, default=0.98)

    # Legacy combined CSV layout.
    parser.add_argument("--data_csv", type=str, default=None, help="Legacy: one CSV containing features and optional labels.")
    parser.add_argument("--test_csv", type=str, default=None, help="Legacy: optional test CSV containing same feature columns.")
    parser.add_argument("--id_col", type=str, default=None, help="Legacy combined CSV sample ID column.")
    parser.add_argument("--split_col", type=str, default=None, help="Legacy combined CSV split column.")
    parser.add_argument("--test_split_value", type=str, default="test")
    parser.add_argument("--background_split_values", type=str, default="train,val,background")
    parser.add_argument("--exclude_cols", type=str, default="", help="Legacy combined CSV columns to exclude from features, comma-separated.")
    parser.add_argument("--positive_label", type=str, default=None, help="Legacy label mapping fallback. With class_order, class 1 is positive.")

    # Model defaults are copied from the provided training Config/build_model.
    parser.add_argument("--d_model", "--d-model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--n_transformer_layers", "--n-transformer-layers", type=int, default=DEFAULT_N_TRANSFORMER_LAYERS)
    parser.add_argument("--num_heads", "--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--tabm_num_members", "--tabm-num-members", type=int, default=DEFAULT_TABM_NUM_MEMBERS)
    parser.add_argument("--use_linear_residual", type=int, default=int(DEFAULT_USE_LINEAR_RESIDUAL))
    parser.add_argument("--gate_dropout", "--gate-dropout", type=float, default=DEFAULT_GATE_DROPOUT)
    parser.add_argument("--linear_dropout", "--linear-dropout", type=float, default=DEFAULT_LINEAR_DROPOUT)
    parser.add_argument("--use_positive_ovr", type=int, default=int(DEFAULT_USE_POSITIVE_OVR))
    parser.add_argument("--use_prototype_residual", type=int, default=int(DEFAULT_USE_PROTOTYPE_RESIDUAL))
    parser.add_argument("--non_strict_load", action="store_true")

    # SHAP settings.
    parser.add_argument("--out_dir", "--outdir", type=str, default="shap_outputs")
    parser.add_argument("--explain_target", type=str, default="logit_margin", choices=["logit_margin", "positive_logit", "positive_prob"])
    parser.add_argument("--explainer", type=str, default="expected_gradients", choices=["expected_gradients", "shap_gradient"], help="expected_gradients is a fast SHAP/GradientSHAP-style estimator; shap_gradient uses shap.GradientExplainer.")
    parser.add_argument("--background_size", type=int, default=50)
    parser.add_argument("--background-all-samples", action="store_true",
                        help="Allow validation samples in the background instead of using saved train IDs.")
    parser.add_argument("--explain_size", type=int, default=100)
    parser.add_argument("--stratified_sample", action="store_true")
    parser.add_argument("--shap_batch_size", type=int, default=8)
    parser.add_argument("--shap_chunk_size", type=int, default=16)
    parser.add_argument("--nsamples", type=int, default=200)
    parser.add_argument("--predict_batch_size", type=int, default=64)
    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument("--save_top_shap_values", type=int, default=100)
    parser.add_argument("--waterfall_sample", type=int, default=0)
    parser.add_argument("--dependence_top_k", type=int, default=5)

    # Runtime.
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    args.exclude_cols = [c.strip() for c in args.exclude_cols.split(",") if c.strip()]
    args.use_linear_residual = bool(int(args.use_linear_residual))
    args.use_positive_ovr = bool(int(args.use_positive_ovr))
    args.use_prototype_residual = bool(int(args.use_prototype_residual))
    args = try_auto_training_files(args)
    if args.already_standardized:
        args.preprocess_dir = None
    return args


def main() -> None:
    args = parse_args()
    if shap is None:
        raise RuntimeError("Install explanation dependencies: python -m pip install -r requirements-shap.txt")
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    class_names = parse_class_order(args.class_order)
    print(f"[task] class 0 = {class_names[0]} | class 1 / positive = {class_names[1]}")
    print(
        "[model defaults] "
        f"d_model={args.d_model}, layers={args.n_transformer_layers}, heads={args.num_heads}, "
        f"dropout={args.dropout}, tabm_members={args.tabm_num_members}, "
        f"linear_residual={args.use_linear_residual}, positive_ovr={args.use_positive_ovr}, "
        f"prototype_residual={args.use_prototype_residual}"
    )

    feature_order = read_feature_list(args.feature_list)
    bg_pack, ex_pack, feature_cols = load_background_and_explain(args, class_names=class_names, feature_order=feature_order)
    bg_pack = restrict_background_to_training_split(bg_pack, args.training_outdir, args.background_all_samples)

    # Apply preprocessing exactly as in training unless the user says the matrix is already standardized.
    X_bg_all = apply_training_preprocessor(bg_pack["X_raw"], args.preprocess_dir)
    X_ex_all = apply_training_preprocessor(ex_pack["X_raw"], args.preprocess_dir)
    y_bg_all = bg_pack["y"]
    y_ex_all = ex_pack["y"]
    ids_ex_all = ex_pack["sample_ids"]

    num_features = len(feature_cols)
    print(f"[data] background source: {X_bg_all.shape}, explain source: {X_ex_all.shape}, features: {num_features}")
    print(f"[preprocess] {'already standardized / no scaler applied' if not args.preprocess_dir else 'applied scaler from ' + str(args.preprocess_dir)}")

    # Sampling.
    if args.stratified_sample and y_bg_all is not None:
        bg_idx = stratified_sample_indices(y_bg_all, args.background_size, args.seed, prefer_balanced=True)
    else:
        bg_idx = random_sample_indices(len(X_bg_all), args.background_size, args.seed)
    if args.stratified_sample and y_ex_all is not None:
        ex_idx = stratified_sample_indices(y_ex_all, args.explain_size, args.seed + 1, prefer_balanced=True)
    else:
        ex_idx = random_sample_indices(len(X_ex_all), args.explain_size, args.seed + 1)

    X_background = X_bg_all[bg_idx]
    X_explain = X_ex_all[ex_idx]
    y_explain = y_ex_all[ex_idx] if y_ex_all is not None else None
    ids_explain = ids_ex_all[ex_idx]
    print(f"[sample] background: {X_background.shape}, explain: {X_explain.shape}")

    # Save data audit information.
    pd.Series(feature_cols, name="feature").to_csv(out_dir / "feature_names.csv", index=False)
    save_json(bg_pack.get("filter_info", {}), out_dir / "background_label_filter_info.json")
    save_json(ex_pack.get("filter_info", {}), out_dir / "explain_label_filter_info.json")
    save_json(bg_pack.get("overlap_info", {}), out_dir / "background_feature_overlap_info.json")
    save_json(ex_pack.get("overlap_info", {}), out_dir / "explain_feature_overlap_info.json")

    # Build and load model.
    model = build_model(args, num_features=num_features, device=device)
    wrapper = BinaryTargetWrapper(model, target=args.explain_target, positive_class=1).to(device).eval()

    # Predictions for explained samples.
    pred_df = predict_batches(model, X_explain, device=device, batch_size=args.predict_batch_size)
    pred_df.insert(0, "sample_id", ids_explain)
    if y_explain is not None:
        pred_df.insert(1, "true_label_binary", y_explain)
        pred_df.insert(2, "true_label_name", [class_names[int(i)] for i in y_explain])
    pred_df.to_csv(out_dir / "predictions_for_explained_samples.csv", index=False)

    # SHAP.
    print(f"[shap] target={args.explain_target}; positive SHAP supports class 1 / {class_names[1]}.")
    if args.explainer == "expected_gradients":
        shap_values = compute_expected_gradients_shap_values(
            wrapper=wrapper,
            X_background=X_background,
            X_explain=X_explain,
            device=device,
            nsamples=args.nsamples,
            chunk_size=args.shap_chunk_size,
            seed=args.seed,
        )
    else:
        shap_values = compute_shap_values(
            wrapper=wrapper,
            X_background=X_background,
            X_explain=X_explain,
            device=device,
            batch_size=args.shap_batch_size,
            nsamples=args.nsamples,
            chunk_size=args.shap_chunk_size,
        )
    np.save(out_dir / "shap_values.npy", shap_values)
    np.save(out_dir / "X_explain.npy", X_explain)
    np.save(out_dir / "X_background.npy", X_background)

    base_value = get_base_value(wrapper, X_background, device=device, batch_size=args.predict_batch_size)
    meta = {
        "class_order": class_names,
        "negative_class_index": 0,
        "positive_class_index": 1,
        "positive_class_name": class_names[1],
        "explain_target": args.explain_target,
        "explainer": args.explainer,
        "base_value_mean_background_output": base_value,
        "num_features": num_features,
        "background_size": int(len(X_background)),
        "explain_size": int(len(X_explain)),
        "matrix_orientation": args.matrix_orientation,
        "sample_col": args.sample_col,
        "label_col": args.label_col,
        "checkpoint": args.checkpoint,
        "feature_list": args.feature_list,
        "preprocess_dir": args.preprocess_dir,
        "model_params": {
            "d_model": args.d_model,
            "dropout": args.dropout,
            "n_transformer_layers": args.n_transformer_layers,
            "num_heads": args.num_heads,
            "tabm_num_members": args.tabm_num_members,
            "use_linear_residual": args.use_linear_residual,
            "gate_dropout": args.gate_dropout,
            "linear_dropout": args.linear_dropout,
            "use_positive_ovr": args.use_positive_ovr,
            "use_prototype_residual": args.use_prototype_residual,
        },
        "interpretation_note": "For logit_margin, positive SHAP supports class 1 / positive class and negative SHAP supports class 0 / negative class.",
    }
    save_json(meta, out_dir / "shap_metadata.json")

    top_df = save_top_feature_tables(shap_values, X_explain, feature_cols, out_dir, args.top_n)
    save_top_shap_value_matrix(shap_values, X_explain, ids_explain, feature_cols, top_df, out_dir, args.save_top_shap_values)
    save_summary_plots(shap_values, X_explain, feature_cols, out_dir, args.top_n)
    save_waterfall_plot(shap_values, X_explain, feature_cols, base_value, args.waterfall_sample, out_dir, max_display=args.top_n)
    dep_features = top_df.head(args.dependence_top_k)["feature"].tolist()
    save_dependence_plots(shap_values, X_explain, feature_cols, dep_features, out_dir)
    save_classwise_heatmap(shap_values, y_explain, feature_cols, class_names, out_dir, top_df, min(args.top_n, 50))

    print("\n[done] SHAP analysis finished. Files saved to:", out_dir.resolve())
    print("[main outputs]")
    for name in [
        "shap_summary_beeswarm.png",
        "shap_summary_bar.png",
        f"shap_feature_importance_top{args.top_n}.csv",
        "predictions_for_explained_samples.csv",
        f"shap_waterfall_sample{args.waterfall_sample}.png",
        f"shap_force_sample{args.waterfall_sample}.html",
    ]:
        print(" -", out_dir / name)


if __name__ == "__main__":
    main()
