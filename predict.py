"""Predict from an unlabeled methylation matrix using saved PREF-FT artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from pref_ft_project import PREF_FT


def load_model(training_outdir: Path, device: torch.device):
    cfg = json.loads((training_outdir / "config.json").read_text(encoding="utf-8"))
    order = pd.read_csv(training_outdir / "feature_order.csv").iloc[:, 0].astype(str).tolist()
    options = {key: cfg[key] for key in [
        "d_model", "dropout", "n_transformer_layers", "num_heads", "tabm_num_members",
        "use_linear_residual", "gate_dropout", "linear_dropout", "use_positive_ovr",
        "use_prototype_residual"] if key in cfg}
    model = PREF_FT(num_features=len(order), num_classes=2, **options).to(device)
    state = torch.load(training_outdir / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    labels = json.loads((training_outdir / "label_mapping.json").read_text(encoding="utf-8"))["class_names"]
    if len(order) != len(set(order)) or len(labels) != 2:
        raise ValueError("Invalid saved feature order or class mapping.")
    return model, order, labels


def prepare_matrix(path: Path, orientation: str, order, training_outdir: Path):
    data = pd.read_csv(path, index_col=0)
    if orientation == "cpg_rows":
        data = data.T
    data.index = data.index.astype(str)
    data.columns = data.columns.astype(str)
    if data.index.has_duplicates or data.columns.has_duplicates:
        raise ValueError("Sample IDs and CpG IDs must be unique.")
    missing = [f for f in order if f not in data.columns]
    if missing:
        raise ValueError(f"Input is missing {len(missing)} required CpGs; first IDs: {missing[:5]}")
    matrix = data.loc[:, order].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    medians = np.load(training_outdir / "train_feature_medians.npy")
    mean = np.load(training_outdir / "scaler_mean.npy")
    scale = np.load(training_outdir / "scaler_scale.npy")
    if any(x.shape != (len(order),) for x in [medians, mean, scale]):
        raise ValueError("Saved preprocessing arrays do not match feature_order.csv.")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid saved standardization scale.")
    matrix = np.where(np.isnan(matrix), medians, matrix)
    matrix = ((matrix - mean) / scale).astype(np.float32)
    if not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError("Input must contain samples with finite values after preprocessing.")
    return matrix, data.index.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-outdir", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix-orientation", choices=["cpg_rows", "sample_rows"], default="cpg_rows")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    model, order, labels = load_model(args.training_outdir, device)
    matrix, sample_ids = prepare_matrix(args.features, args.matrix_orientation, order, args.training_outdir)
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(matrix), args.batch_size):
            logits, _ = model(torch.from_numpy(matrix[start:start+args.batch_size]).to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    prob = np.concatenate(probabilities)
    predicted = prob.argmax(1)
    output = pd.DataFrame({"sample_id": sample_ids, "prob_negative": prob[:, 0],
                           "prob_positive": prob[:, 1], "predicted_class_id": predicted,
                           "predicted_label": [labels[i] for i in predicted]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output)} predictions to {args.output}")


if __name__ == "__main__":
    main()
