# PREF-FT

**Positive-Aware Residual Evidence Fusion Transformer for PBMC DNA methylation-based breast cancer prediction.**

PREF-FT combines contextual CpG representations with direct linear, positive-class and prototype evidence. 


## Architecture

<img src="/home/qujing/PREF_FT/Figure/Fig1.png" width="1000"/>

## Installation

Python 3.10 or 3.11 is recommended. Install PyTorch for the intended CPU/CUDA environment, then install the core dependencies:

```bash
python -m pip install -r requirements.txt
```

For interpretation, add `requirements-shap.txt`. For optional comparison models, use `requirements-benchmarks.txt`. `mamba-ssm` is needed only for the Mamba comparisons and should be installed separately in a compatible environment. Tests use `requirements-dev.txt`.

## Data

Place these files under `data/`, or pass explicit paths:

```text
data/
  GSE243529_intersection_CpGs_after_com.csv.gz
  GSE243529_pheno.csv
  GSE104942_intersection_CpGs_after_com.csv.gz
  GSE104942_pheno.csv
  external_tests.json
```

Feature matrices have CpGs in rows, samples in columns, and CpG IDs in the first column. Metadata must contain `SampleID` and `disease.state`. Labels are `unaffected` (0) and `affected` (1). 


## Train

Run commands from the repository root:

```bash
python train.py --outdir runs/pref_ft
```

Or provide another dataset:

```bash
python train.py --train-features path/to/train.csv.gz --train-metadata path/to/metadata.csv --external-tests-json path/to/external_tests.json --outdir runs/pref_ft
```


## Predict from an unlabeled matrix

```bash
python predict.py --training-outdir runs/pref_ft --features path/to/new_samples.csv.gz --output runs/predictions.csv
```

Prediction requires the saved feature order, preprocessing arrays, configuration and class mapping. Extra CpGs are ignored; missing required CpGs are rejected. The original training preprocessing is applied. The default prediction batch size is one to limit memory use with long CpG token sequences.

## Ablation and repeated runs

```bash
python ablate.py --variants baseline,no_linear_residual,no_positive_ovr --outdir runs/ablation
python scripts/run_repeated.py --seeds 1 2 3 4 5 --out-root runs/five_runs -- --label-smoothing 0.05
```


## SHAP interpretation

```bash
python -m pip install -r requirements-shap.txt
python explain.py --training_outdir runs/pref_ft --data_features data/GSE243529_intersection_CpGs_after_com.csv.gz --data_metadata data/GSE243529_pheno.csv --test_features data/GSE104942_intersection_CpGs_after_com.csv.gz --test_metadata data/GSE104942_pheno.csv --explain_target logit_margin --explainer expected_gradients --out_dir runs/shap_external
```


## Comparison models

```bash
python benchmark.py --model-name elastic_net_lr --train-features data/GSE243529_intersection_CpGs_after_com.csv.gz --train-metadata data/GSE243529_pheno.csv --external-tests-json data/external_tests.json --matrix-orientation cpg_rows --sample-col SampleID --label-col disease.state --class-order "unaffected, affected" --outdir runs/elastic_net_lr
```


## Project layout

```text
train.py / ablate.py / explain.py / predict.py / benchmark.py
prepare_features.py                Paper-protocol feature discovery
code/pref_ft_project/              Canonical model, training and ablation
code/pref_ft_shap_project/          Shared model exports and interpretation
code/preprocessing/                QC, MAD, Welch screening and intersection
code/methylation_*.py              Retained comparison implementations
scripts/                          Repeated runs and metric aggregation
examples/                         Portable external-dataset configuration
tests/                            Model contracts and synthetic workflow checks
data/                             Local data; ignored except documentation/config
runs/                             New outputs; ignored
```

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Manuscript and licensing

Associated manuscript: *PREF-FT: A Positive-Aware Residual Evidence Fusion Transformer for PBMC DNA Methylation-Based Breast Cancer Prediction*, Jing Qu, Han Wang and Guixia Liu. 


