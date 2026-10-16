"""Run five explicitly supplied seeds and aggregate validation/external metrics.

Example: python run_repeated.py --seeds 1 2 3 4 5 --out-root results/repeated --
         --train-features ... --train-metadata ... --external-tests-json ...
The example seeds are illustrative, not recovered from the paper.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs=5, type=int, required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("train_args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if len(set(args.seeds)) != 5:
        p.error("Supply five distinct seeds; repeated copies are not independent runs.")
    extra = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    if any(x.split("=", 1)[0] in {"--seed", "--outdir"} for x in extra):
        p.error("Specify --seed/--outdir through --seeds/--out-root only.")
    out = Path(args.out_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        dest = out / f"seed_{seed}"
        if dest.exists() and any(dest.iterdir()):
            raise FileExistsError(f"Refusing to overwrite a populated run: {dest}")
        cmd = [sys.executable, str(Path(__file__).resolve().parents[1] / "train.py"),
               *extra, "--seed", str(seed), "--outdir", str(dest)]
        subprocess.run(cmd, check=True)
        result = json.loads((dest / "summary.json").read_text(encoding="utf-8"))
        for cohort, metrics in [("validation", result["val_metrics"]),
                                *[(x["dataset"], x) for x in result["external_tests"]]]:
            row = dict(seed=seed, cohort=cohort)
            for key in ["acc", "bal_acc", "sensitivity", "specificity", "mcc", "auc_binary"]:
                row[key] = metrics[key]
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_run_metrics.csv", index=False)
    metrics = ["acc", "bal_acc", "sensitivity", "specificity", "mcc", "auc_binary"]
    df.groupby("cohort")[metrics].agg(["mean", "std", "count"]).to_csv(out / "five_run_summary.csv")


if __name__ == "__main__":
    main()
