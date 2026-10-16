"""Paper II.B: common CpGs -> per-cohort QC/MAD/Welch -> selected intersection.

This is a new orchestration of the original numerical helpers. It explicitly
implements the paper's p-value conditions instead of the source FDR-OR defaults.
It uses labels from BOTH cohorts, as described in the supplied paper; the
external cohort therefore participates in supervised feature discovery.
The paper does not give a MAD cutoff: --mad-threshold is deliberately required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from screening_utils import (
    align_beta_and_meta, beta_to_mvalue, low_variance_filter,
    sample_and_probe_missing_filter, vectorized_welch_ttest,
)


def load_beta(path: str) -> pd.DataFrame:
    beta = pd.read_csv(path, index_col=0)
    beta.index = beta.index.astype(str)
    beta.columns = beta.columns.astype(str)
    if beta.index.has_duplicates or beta.columns.has_duplicates:
        raise ValueError(f"Duplicate CpG/sample IDs in {path}; resolve duplicates before screening.")
    beta = beta.apply(pd.to_numeric, errors="coerce")
    return beta.mask((beta < 0) | (beta > 1), np.nan)


def screen_cohort(beta: pd.DataFrame, meta: pd.DataFrame, args, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    if meta[args.sample_col].astype(str).duplicated().any():
        raise ValueError("Metadata sample IDs must be unique.")
    if meta[args.label_col].isna().any():
        raise ValueError("Metadata contains missing labels.")
    beta, meta, _ = align_beta_and_meta(
        beta, meta, args.sample_col, args.label_col, args.case_label)
    before = beta.shape
    beta, sample_qc, probe_qc = sample_and_probe_missing_filter(beta, 0.10, 0.10)
    meta = (meta.set_index(args.sample_col).loc[beta.columns.tolist()]
            .rename_axis(args.sample_col).reset_index())
    y = (meta[args.label_col].astype(str) == args.case_label).astype(int).to_numpy()
    if min(np.bincount(y, minlength=2)) < 2:
        raise ValueError("Welch testing needs at least two samples in each class after QC.")
    after_missing = beta.shape
    beta, mad_qc = low_variance_filter(beta, "mad", args.mad_threshold)
    if beta.empty:
        raise ValueError("No CpGs survive the chosen MAD threshold.")
    m_p, *_ = vectorized_welch_ttest(beta_to_mvalue(beta).to_numpy(), y)
    m_stats = pd.DataFrame({"cpg_id": beta.index, "mvalue_p": m_p,
                            "selected": m_p < 0.05})
    candidates = beta.loc[m_p < 0.05]
    if candidates.empty:
        raise ValueError("No CpGs survive M-value p < 0.05.")
    p, mean0, mean1, delta = vectorized_welch_ttest(candidates.to_numpy(), y)
    selected = (p < 0.01) & (np.abs(delta) >= 0.02)
    final = pd.DataFrame({
        "cpg_id": candidates.index, "p_value": p,
        "mean_beta_control": mean0, "mean_beta_case": mean1,
        "delta_beta_case_minus_control": delta, "selected": selected,
    })
    ids = candidates.index[selected].tolist()
    sample_qc.to_csv(outdir / "sample_missing_qc.csv", index=False)
    probe_qc.to_csv(outdir / "probe_missing_qc.csv", index=False)
    mad_qc.to_csv(outdir / "probe_mad_qc.csv", index=False)
    m_stats.to_csv(outdir / "mvalue_prefilter_stats.csv", index=False)
    final.to_csv(outdir / "final_cpg_univariate_stats.csv", index=False)
    (outdir / "selected_cpg_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    summary = dict(n_common_input=before[0], n_samples_input=before[1],
                   n_cpg_after_missing=after_missing[0], n_samples_after_missing=after_missing[1],
                   n_cpg_after_mad=beta.shape[0], n_mvalue_candidates=len(candidates),
                   n_selected=len(ids), class_counts=np.bincount(y, minlength=2).tolist())
    # Return all post-QC rows so the selected intersection can be extracted.
    return beta, meta, ids, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-beta", required=True)
    parser.add_argument("--train-meta", required=True)
    parser.add_argument("--external-beta", required=True)
    parser.add_argument("--external-meta", required=True)
    parser.add_argument("--mad-threshold", type=float, required=True,
                        help="Paper omits this cutoff; supply an explicitly chosen value.")
    parser.add_argument("--sample-col", default="SampleID")
    parser.add_argument("--label-col", default="disease.state")
    parser.add_argument("--case-label", default="affected")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    if not np.isfinite(args.mad_threshold) or args.mad_threshold < 0:
        parser.error("--mad-threshold must be finite and non-negative.")
    out = Path(args.outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    train, external = load_beta(args.train_beta), load_beta(args.external_beta)
    # Preserve training order; set ordering must never determine model feature order.
    common = train.index[train.index.isin(external.index)]
    if common.empty:
        raise ValueError("No common CpGs.")
    (out / "common_cpg_ids.txt").write_text("\n".join(common) + "\n", encoding="utf-8")
    train, tm, tids, ts = screen_cohort(
        train.loc[common], pd.read_csv(args.train_meta), args, out / "train_screen")
    external, em, eids, es = screen_cohort(
        external.loc[common], pd.read_csv(args.external_meta), args, out / "external_screen")
    external_selected = set(eids)
    final = [x for x in tids if x in external_selected]
    if not final:
        raise ValueError("No shared selected CpGs; inspect the two screen reports.")
    train.loc[final].to_csv(out / "train_selected_beta.csv.gz")
    external.loc[final].to_csv(out / "external_selected_beta.csv.gz")
    tm.to_csv(out / "train_metadata.csv", index=False)
    em.to_csv(out / "external_metadata.csv", index=False)
    pd.DataFrame({"feature_name": final}).to_csv(out / "feature_order.csv", index=False)
    specs = [dict(name="GSE104942", features=str(out / "external_selected_beta.csv.gz"),
                  metadata=str(out / "external_metadata.csv"), matrix_orientation="cpg_rows",
                  sample_col=args.sample_col, label_col=args.label_col)]
    (out / "external_tests.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")
    report = dict(n_common=len(common), n_selected_intersection=len(final), train=ts, external=es,
                  mad_threshold=args.mad_threshold, thresholds=dict(m_p=0.05,beta_p=0.01,abs_delta_beta=0.02),
                  uses_external_labels_for_selection=True, uses_train_labels_before_internal_split=True,
                  source="New orchestration; original Welch/QC/MAD helpers preserved.",
                  expected_counts_in_paper=dict(common=30325,selected=3525),
                  note="Counts are measured, never forced to match the paper; this is supervised cross-cohort discovery.")
    (out / "feature_selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
