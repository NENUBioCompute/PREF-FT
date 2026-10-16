#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-dataset CpG screening pipeline for beta matrix.

Steps:
1) Sample missing-rate filtering
2) Probe missing-rate filtering
3) Low-variance probe filtering
4) Outlier sample check (PCA + sample correlation)
5) M-value based statistical prefilter (Welch's t-test)
6) Univariate CpG analysis on beta values (Welch's t-test)
7) Save selected CpG IDs

Input assumptions:
- Beta matrix: rows = CpGs, columns = samples, first column = CpG ID
- Metadata: must contain sample_id column and label column
- Binary classification only

Example:
python cpg_screen_single_dataset.py \
    --beta beta_matrix.csv \
    --meta metadata.csv \
    --sample-col sample_id \
    --label-col label \
    --case-label disease \
    --outdir results_single

Historical experiment screening. See docs/data.md for the paper protocol.
The original variance and FDR-OR defaults are preserved here.
"""

import os
import json
import argparse
import warnings
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def read_table(path: str) -> pd.DataFrame:
    """Read csv/tsv/txt automatically."""
    return pd.read_csv(path, sep=None, engine="python")


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked_p = pvals[order]
    fdr = ranked_p * n / (np.arange(1, n + 1))
    fdr = np.minimum.accumulate(fdr[::-1])[::-1]
    fdr = np.clip(fdr, 0, 1)
    out = np.empty_like(fdr)
    out[order] = fdr
    return out


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


def detect_outlier_samples(
    beta_df: pd.DataFrame,
    max_pca_probes: int = 5000,
    pca_z_thresh: float = 3.5,
    corr_iqr_mult: float = 3.0,
    outdir: str = "."
) -> pd.DataFrame:
    """
    Detect outlier samples by:
    1) PCA robust z-score on PC1/PC2
    2) low median sample-sample correlation

    Returns dataframe with flags. Does not remove automatically.
    """
    n_probes = beta_df.shape[0]
    if n_probes == 0:
        raise ValueError("No probes left for outlier detection.")

    # Select top variable probes for PCA
    variances = beta_df.var(axis=1, skipna=True)
    top_n = min(max_pca_probes, n_probes)
    top_idx = variances.sort_values(ascending=False).head(top_n).index
    sub = beta_df.loc[top_idx].copy()

    # Impute row median
    row_medians = sub.median(axis=1, skipna=True)
    sub = sub.T
    sub = sub.fillna(row_medians)
    X = sub.values  # [n_samples, n_features]

    # PCA
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X)

    def robust_z(x: np.ndarray) -> np.ndarray:
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad < 1e-12:
            return np.zeros_like(x)
        return 0.6745 * (x - med) / mad

    z1 = robust_z(pcs[:, 0])
    z2 = robust_z(pcs[:, 1])
    pca_flag = (np.abs(z1) > pca_z_thresh) | (np.abs(z2) > pca_z_thresh)

    # Correlation-based outliers
    corr = np.corrcoef(X)
    np.fill_diagonal(corr, np.nan)
    median_corr = np.nanmedian(corr, axis=1)

    q1 = np.nanpercentile(median_corr, 25)
    q3 = np.nanpercentile(median_corr, 75)
    iqr = q3 - q1
    corr_thresh = q1 - corr_iqr_mult * iqr
    corr_flag = median_corr < corr_thresh

    sample_qc = pd.DataFrame({
        "sample_id": sub.index,
        "pc1": pcs[:, 0],
        "pc2": pcs[:, 1],
        "pc1_robust_z": z1,
        "pc2_robust_z": z2,
        "pca_outlier": pca_flag,
        "median_sample_corr": median_corr,
        "corr_outlier": corr_flag,
    })
    sample_qc["outlier_flag"] = sample_qc["pca_outlier"] | sample_qc["corr_outlier"]

    # Save PCA plot
    plt.figure(figsize=(6, 5))
    for _, row in sample_qc.iterrows():
        color = "red" if row["outlier_flag"] else "black"
        plt.scatter(row["pc1"], row["pc2"], c=color, s=35)
        plt.text(row["pc1"], row["pc2"], str(row["sample_id"]), fontsize=7, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Outlier Check")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sample_pca_outlier_check.png"), dpi=300)
    plt.close()

    return sample_qc


def main():
    parser = argparse.ArgumentParser(description="Single-dataset CpG screening pipeline")

    parser.add_argument("--beta", type=str, required=True, help="Beta matrix file (rows=CpGs, cols=samples)")
    parser.add_argument("--meta", type=str, required=True, help="Metadata file")
    parser.add_argument("--sample-col", type=str, required=True, help="Sample ID column in metadata")
    parser.add_argument("--label-col", type=str, required=True, help="Binary label column in metadata")
    parser.add_argument("--case-label", type=str, required=True, help="Positive/case label value")

    parser.add_argument("--outdir", type=str, default="cpg_screen_results", help="Output directory")

    # QC thresholds
    parser.add_argument("--sample-missing-thresh", type=float, default=0.10, help="Max allowed sample missing rate")
    parser.add_argument("--probe-missing-thresh", type=float, default=0.10, help="Max allowed probe missing rate")

    parser.add_argument("--lowvar-method", type=str, choices=["var", "mad"], default="var",
                        help="Low-variance filter metric")
    parser.add_argument("--lowvar-thresh", type=float, default=0.001,
                        help="Threshold for low-variance filter; var default=0.001, mad can use ~0.01-0.02")

    # Outlier settings
    parser.add_argument("--drop-outliers", action="store_true",
                        help="If set, remove samples flagged as outliers before statistical tests")
    parser.add_argument("--max-pca-probes", type=int, default=5000, help="Max probes used for PCA outlier check")

    # Stage 1: M-value prefilter
    parser.add_argument("--mvalue-prefilter-p", type=float, default=0.05,
                        help="Nominal p cutoff for M-value prefilter")
    parser.add_argument("--mvalue-prefilter-fdr", type=float, default=1.0,
                        help="Optional FDR cutoff for M-value prefilter; default 1.0 means not used effectively")

    # Stage 2: final univariate filter
    parser.add_argument("--final-fdr", type=float, default=0.10, help="Final FDR threshold")
    parser.add_argument("--final-p", type=float, default=0.01, help="Final nominal p threshold")
    parser.add_argument("--min-delta-beta", type=float, default=0.02, help="Minimum absolute delta beta")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # -----------------------------
    # 1. Read input
    # -----------------------------
    print("[1/7] Reading beta matrix and metadata ...")
    beta_raw = read_table(args.beta)
    meta = read_table(args.meta)

    # First column as CpG ID
    beta_raw = beta_raw.copy()
    beta_raw.iloc[:, 0] = beta_raw.iloc[:, 0].astype(str)
    beta = beta_raw.set_index(beta_raw.columns[0])

    # Ensure numeric
    for col in beta.columns:
        beta[col] = pd.to_numeric(beta[col], errors="coerce")

    # Keep only values in [0,1], otherwise set as NA
    beta = beta.mask((beta < 0) | (beta > 1), np.nan)

    beta, meta, y = align_beta_and_meta(
        beta_df=beta,
        meta_df=meta,
        sample_col=args.sample_col,
        label_col=args.label_col,
        case_label=args.case_label
    )

    # Save aligned sample list
    meta.to_csv(os.path.join(args.outdir, "aligned_metadata.csv"), index=False)

    summary = {
        "n_cpg_raw": int(beta.shape[0]),
        "n_samples_raw": int(beta.shape[1]),
    }

    # -----------------------------
    # 2. Missing-rate filtering
    # -----------------------------
    print("[2/7] Filtering samples/probes by missing rate ...")
    beta, sample_missing_df, probe_missing_df = sample_and_probe_missing_filter(
        beta_df=beta,
        sample_missing_thresh=args.sample_missing_thresh,
        probe_missing_thresh=args.probe_missing_thresh
    )

    kept_samples = beta.columns.tolist()
    meta = meta[meta[args.sample_col].astype(str).isin(kept_samples)].copy()
    meta = meta.set_index(args.sample_col).loc[kept_samples].reset_index()
    y = (meta[args.label_col].astype(str) == str(args.case_label)).astype(int).values

    sample_missing_df["kept_after_missing_filter"] = sample_missing_df["sample_id"].isin(kept_samples)
    sample_missing_df.to_csv(os.path.join(args.outdir, "sample_missing_qc.csv"), index=False)
    probe_missing_df.to_csv(os.path.join(args.outdir, "probe_missing_qc.csv"), index=False)

    summary["n_samples_after_missing_filter"] = int(beta.shape[1])
    summary["n_cpg_after_missing_filter"] = int(beta.shape[0])

    if len(np.unique(y)) != 2:
        raise ValueError("After sample missing-rate filtering, only one class remains.")

    # -----------------------------
    # 3. Low-variance probe filtering
    # -----------------------------
    print("[3/7] Filtering low-variance probes ...")
    beta, lowvar_df = low_variance_filter(
        beta_df=beta,
        method=args.lowvar_method,
        threshold=args.lowvar_thresh
    )
    lowvar_df.to_csv(os.path.join(args.outdir, "probe_low_variance_qc.csv"), index=False)

    summary["n_cpg_after_low_variance_filter"] = int(beta.shape[0])

    if beta.shape[0] == 0:
        raise ValueError("No CpGs left after low-variance filtering. Please loosen threshold.")

    # -----------------------------
    # 4. Outlier sample check
    # -----------------------------
    print("[4/7] Checking outlier samples ...")
    outlier_df = detect_outlier_samples(
        beta_df=beta,
        max_pca_probes=args.max_pca_probes,
        outdir=args.outdir
    )
    outlier_df.to_csv(os.path.join(args.outdir, "sample_outlier_report.csv"), index=False)

    flagged_samples = outlier_df.loc[outlier_df["outlier_flag"], "sample_id"].astype(str).tolist()
    summary["n_flagged_outlier_samples"] = int(len(flagged_samples))

    if args.drop_outliers and len(flagged_samples) > 0:
        print(f"    Removing {len(flagged_samples)} flagged outlier samples ...")
        keep_samples = [s for s in beta.columns if s not in flagged_samples]
        beta = beta.loc[:, keep_samples]
        meta = meta[meta[args.sample_col].astype(str).isin(keep_samples)].copy()
        meta = meta.set_index(args.sample_col).loc[keep_samples].reset_index()
        y = (meta[args.label_col].astype(str) == str(args.case_label)).astype(int).values

        if len(np.unique(y)) != 2:
            raise ValueError("After removing outliers, only one class remains.")

    summary["n_samples_after_outlier_step"] = int(beta.shape[1])

    # -----------------------------
    # 5. M-value statistical prefilter
    # -----------------------------
    print("[5/7] M-value prefilter ...")
    mval = beta_to_mvalue(beta)
    m_pvals, m_mean0, m_mean1, m_delta = vectorized_welch_ttest(mval.values, y)
    m_fdr = bh_fdr(m_pvals)

    mvalue_stats = pd.DataFrame({
        "cpg_id": beta.index,
        "mvalue_p": m_pvals,
        "mvalue_fdr": m_fdr,
        "mvalue_mean_control": m_mean0,
        "mvalue_mean_case": m_mean1,
        "mvalue_delta": m_delta,
    })

    prefilter_mask = (mvalue_stats["mvalue_p"] < args.mvalue_prefilter_p) | (
        mvalue_stats["mvalue_fdr"] < args.mvalue_prefilter_fdr
    )
    mvalue_stats["kept_after_mvalue_prefilter"] = prefilter_mask.values
    mvalue_stats.to_csv(os.path.join(args.outdir, "mvalue_prefilter_stats.csv"), index=False)

    kept_cpg_stage1 = mvalue_stats.loc[prefilter_mask, "cpg_id"].tolist()
    beta_stage1 = beta.loc[kept_cpg_stage1].copy()

    summary["n_cpg_after_mvalue_prefilter"] = int(beta_stage1.shape[0])

    if beta_stage1.shape[0] == 0:
        raise ValueError("No CpGs left after M-value prefilter. Please loosen thresholds.")

    # -----------------------------
    # 6. Final univariate CpG analysis on beta
    # -----------------------------
    print("[6/7] Final univariate CpG analysis on beta values ...")
    pvals, mean0, mean1, delta_beta = vectorized_welch_ttest(beta_stage1.values, y)
    fdr = bh_fdr(pvals)

    final_stats = pd.DataFrame({
        "cpg_id": beta_stage1.index,
        "mean_beta_control": mean0,
        "mean_beta_case": mean1,
        "delta_beta_case_minus_control": delta_beta,
        "p_value": pvals,
        "fdr": fdr,
        "abs_delta_beta": np.abs(delta_beta)
    })

    final_stats["selected"] = (
        ((final_stats["fdr"] < args.final_fdr) | (final_stats["p_value"] < args.final_p))
        & (final_stats["abs_delta_beta"] >= args.min_delta_beta)
    )

    final_stats = final_stats.sort_values(
        by=["selected", "fdr", "p_value", "abs_delta_beta"],
        ascending=[False, True, True, False]
    )

    final_stats.to_csv(os.path.join(args.outdir, "final_cpg_univariate_stats.csv"), index=False)

    selected_df = final_stats.loc[final_stats["selected"]].copy()
    selected_df.to_csv(os.path.join(args.outdir, "selected_cpg_stats.csv"), index=False)

    selected_ids = selected_df["cpg_id"].astype(str).tolist()
    with open(os.path.join(args.outdir, "selected_cpg_ids.txt"), "w", encoding="utf-8") as f:
        for cpg in selected_ids:
            f.write(f"{cpg}\n")

    pd.DataFrame({"cpg_id": selected_ids}).to_csv(
        os.path.join(args.outdir, "selected_cpg_ids.csv"),
        index=False
    )

    summary["n_selected_cpg"] = int(len(selected_ids))

    # -----------------------------
    # 7. Save filtered beta matrix (selected CpGs)
    # -----------------------------
    print("[7/7] Saving filtered beta matrix and summary ...")
    if len(selected_ids) > 0:
        beta_selected = beta.loc[selected_ids].copy()
        beta_selected.to_csv(os.path.join(args.outdir, "beta_matrix_selected_cpgs.csv"))
    else:
        print("Warning: No CpGs passed final selection criteria.")

    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()