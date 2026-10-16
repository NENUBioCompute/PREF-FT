"""Numerical helpers extracted unchanged from the original single-cohort screen."""

from typing import Tuple

import numpy as np

import pandas as pd

from scipy import stats

def beta_to_mvalue(beta_df: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    """Convert beta values to M-values."""
    beta = beta_df.clip(lower=eps, upper=1 - eps)
    mval = np.log2(beta / (1 - beta))
    return mval

def compute_mad(df: pd.DataFrame) -> pd.Series:
    """Median absolute deviation per row."""
    med = df.median(axis=1, skipna=True)
    mad = (df.sub(med, axis=0)).abs().median(axis=1, skipna=True)
    return mad

def vectorized_welch_ttest(
    matrix: np.ndarray,
    labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Row-wise Welch's t-test.
    matrix: shape [n_features, n_samples]
    labels: binary array [n_samples], values in {0,1}

    Returns:
        pvals, mean_group0, mean_group1, delta(group1 - group0)
    """
    group0 = matrix[:, labels == 0]
    group1 = matrix[:, labels == 1]

    n0 = np.sum(~np.isnan(group0), axis=1)
    n1 = np.sum(~np.isnan(group1), axis=1)

    mean0 = np.nanmean(group0, axis=1)
    mean1 = np.nanmean(group1, axis=1)

    var0 = np.nanvar(group0, axis=1, ddof=1)
    var1 = np.nanvar(group1, axis=1, ddof=1)

    se2 = var0 / np.maximum(n0, 1) + var1 / np.maximum(n1, 1)

    # Avoid division errors
    invalid = (n0 < 2) | (n1 < 2) | (se2 <= 0) | np.isnan(se2)
    t_stat = np.full(matrix.shape[0], np.nan, dtype=float)
    dfree = np.full(matrix.shape[0], np.nan, dtype=float)

    valid_idx = ~invalid
    if np.any(valid_idx):
        t_stat[valid_idx] = (mean1[valid_idx] - mean0[valid_idx]) / np.sqrt(se2[valid_idx])

        numerator = se2[valid_idx] ** 2
        denominator = (
            (var0[valid_idx] / np.maximum(n0[valid_idx], 1)) ** 2 / np.maximum(n0[valid_idx] - 1, 1)
            + (var1[valid_idx] / np.maximum(n1[valid_idx], 1)) ** 2 / np.maximum(n1[valid_idx] - 1, 1)
        )
        dfree[valid_idx] = numerator / np.maximum(denominator, 1e-12)

    pvals = np.ones(matrix.shape[0], dtype=float)
    good = valid_idx & np.isfinite(t_stat) & np.isfinite(dfree) & (dfree > 0)
    if np.any(good):
        pvals[good] = 2 * stats.t.sf(np.abs(t_stat[good]), df=dfree[good])

    delta = mean1 - mean0
    return pvals, mean0, mean1, delta

def align_beta_and_meta(
    beta_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    sample_col: str,
    label_col: str,
    case_label: str
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Align beta matrix columns with metadata sample order."""
    if sample_col not in meta_df.columns:
        raise ValueError(f"'{sample_col}' not found in metadata.")
    if label_col not in meta_df.columns:
        raise ValueError(f"'{label_col}' not found in metadata.")

    meta_df = meta_df.copy()
    meta_df[sample_col] = meta_df[sample_col].astype(str)
    beta_df.columns = beta_df.columns.astype(str)

    common_samples = [s for s in meta_df[sample_col].tolist() if s in beta_df.columns]
    if len(common_samples) < 4:
        raise ValueError("Too few overlapping samples between beta matrix and metadata.")

    meta_df = meta_df[meta_df[sample_col].isin(common_samples)].copy()
    beta_df = beta_df.loc[:, common_samples].copy()
    meta_df = meta_df.set_index(sample_col).loc[common_samples].reset_index()

    # Binary labels
    unique_labels = meta_df[label_col].dropna().astype(str).unique().tolist()
    if len(unique_labels) != 2:
        raise ValueError(f"Binary classification required, but got labels: {unique_labels}")

    y = (meta_df[label_col].astype(str) == str(case_label)).astype(int).values
    if len(np.unique(y)) != 2:
        raise ValueError(
            f"case_label='{case_label}' does not form a valid binary split. "
            f"Metadata labels found: {unique_labels}"
        )

    return beta_df, meta_df, y

def sample_and_probe_missing_filter(
    beta_df: pd.DataFrame,
    sample_missing_thresh: float,
    probe_missing_thresh: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter by sample/probe missing rates."""
    sample_missing = beta_df.isna().mean(axis=0)
    sample_qc = pd.DataFrame({
        "sample_id": beta_df.columns,
        "missing_rate": sample_missing.values
    })

    keep_samples = sample_missing <= sample_missing_thresh
    beta_df = beta_df.loc[:, keep_samples]

    probe_missing = beta_df.isna().mean(axis=1)
    probe_qc = pd.DataFrame({
        "cpg_id": beta_df.index,
        "missing_rate": probe_missing.values
    }).set_index("cpg_id")

    keep_probes = probe_missing <= probe_missing_thresh
    beta_df = beta_df.loc[keep_probes, :]

    probe_qc["kept_after_missing_filter"] = probe_qc.index.isin(beta_df.index)

    return beta_df, sample_qc, probe_qc.reset_index()

def low_variance_filter(
    beta_df: pd.DataFrame,
    method: str,
    threshold: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter low-variance probes using variance or MAD."""
    if method == "var":
        metric = beta_df.var(axis=1, skipna=True)
        metric_name = "variance"
    elif method == "mad":
        metric = compute_mad(beta_df)
        metric_name = "mad"
    else:
        raise ValueError("method must be 'var' or 'mad'")

    keep = metric >= threshold
    metrics_df = pd.DataFrame({
        "cpg_id": beta_df.index,
        metric_name: metric.values,
        "kept_after_low_variance_filter": keep.values
    })
    beta_filtered = beta_df.loc[keep, :]
    return beta_filtered, metrics_df
