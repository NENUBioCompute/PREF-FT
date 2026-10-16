import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from pref_ft_project import PREF_FT
from pref_ft_shap_project import run_pref_ft_shap as interpretation
from pref_ft_project.train_pref_ft import load_external_test_specs, Config
from predict import load_model, prepare_matrix

ROOT = Path(__file__).resolve().parents[1]


def run_cli(script, args):
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, str(ROOT/script), *args], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8", env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_external_paths_resolve_against_json(tmp_path):
    path = tmp_path / "external.json"
    path.write_text(json.dumps([dict(name="cohort", features="matrix.csv", metadata="meta.csv")]))
    spec = load_external_test_specs(str(path), Config())[0]
    assert spec.features == str((tmp_path/"matrix.csv").resolve())
    assert spec.metadata == str((tmp_path/"meta.csv").resolve())


def test_background_respects_saved_training_ids(tmp_path):
    pd.DataFrame({"sample_id":["a","b","c"],"split":["train","val","train"]}).to_csv(
        tmp_path/"train_val_split_samples.csv",index=False)
    pack = dict(X_raw=np.arange(6).reshape(3,2),sample_ids=np.array(["a","b","c"]),y=np.array([0,1,1]))
    actual = interpretation.restrict_background_to_training_split(pack,str(tmp_path))
    assert actual["sample_ids"].tolist() == ["a","c"]
    assert actual["y"].tolist() == [0,1]


def test_explanation_inherits_saved_model_parameters(tmp_path, monkeypatch):
    from types import SimpleNamespace
    (tmp_path/"config.json").write_text(json.dumps(dict(d_model=32,num_heads=2,dropout=0.1)))
    args=SimpleNamespace(training_outdir=str(tmp_path),d_model=128,num_heads=4,dropout=0.2,
                         checkpoint=None,feature_list=None,preprocess_dir=None,
                         already_standardized=False,class_order="unaffected, affected")
    monkeypatch.setattr(sys,"argv",["explain.py","--dropout","0.3"])
    args.dropout=0.3
    actual=interpretation.try_auto_training_files(args)
    assert actual.d_model==32 and actual.num_heads==2 and actual.dropout==0.3


def test_ablation_variants_produce_binary_logits():
    from pref_ft_project.ablation_study import AblationConfig, ABLATION_VARIANTS, build_ablation_model
    cfg=AblationConfig(d_model=16,n_transformer_layers=1,num_heads=4)
    for variant in ABLATION_VARIANTS:
        model=build_ablation_model(cfg,variant,num_features=10).eval()
        logits,_=model(torch.randn(2,10))
        assert logits.shape==(2,2) and torch.isfinite(logits).all(),variant.name


def test_synthetic_train_predict_and_feature_order(tmp_path):
    rng=np.random.default_rng(314)
    n=32
    y=np.tile([0,1],n//2)
    values=np.clip(0.3+0.3*y[None,:]+rng.normal(0,.03,(10,n)),0,1)
    ids=[f"sample{i}" for i in range(n)]
    beta=pd.DataFrame(values,index=[f"cg{i}" for i in range(10)],columns=ids)
    beta.iloc[0,0]=np.nan
    beta.to_csv(tmp_path/"beta.csv")
    pd.DataFrame({"SampleID":ids,"disease.state":np.where(y,"affected","unaffected")}).to_csv(tmp_path/"meta.csv",index=False)
    (tmp_path/"external.json").write_text(json.dumps([dict(name="synthetic",features="beta.csv",metadata="meta.csv")]))
    out=tmp_path/"trained"
    run_cli("train.py",["--train-features",str(tmp_path/"beta.csv"),"--train-metadata",str(tmp_path/"meta.csv"),
                         "--external-tests-json",str(tmp_path/"external.json"),"--outdir",str(out),"--epochs","1",
                         "--d-model","16","--n-transformer-layers","1","--num-heads","4","--batch-size","8"])
    summary=json.loads((out/"summary.json").read_text())
    assert summary["model_name"]=="pref_ft" and summary["n_features"]==10
    assert summary["external_tests"][0]["n_samples"]==n
    exported=pd.read_csv(out/"eval_train/predictions.csv")
    split=pd.read_csv(out/"train_val_split_samples.csv")
    expected=split[split["split"]=="train"]
    assert exported["sample_id"].tolist()==expected["sample_id"].tolist()
    assert exported["y_true"].tolist()==expected["label_id"].tolist()
    # Input ordering is independent of saved training feature ordering.
    beta.iloc[::-1].to_csv(tmp_path/"shuffled.csv")
    prediction=tmp_path/"predictions.csv"
    run_cli("predict.py",["--training-outdir",str(out),"--features",str(tmp_path/"shuffled.csv"),
                           "--output",str(prediction),"--batch-size","8","--device","cpu"])
    actual=pd.read_csv(prediction)
    external=pd.read_csv(out/"external_tests/synthetic/predictions.csv")
    np.testing.assert_allclose(actual["prob_positive"],external["prob__affected"],atol=1e-6)
    model,order,_=load_model(out,torch.device("cpu"))
    a,_=prepare_matrix(tmp_path/"beta.csv","cpg_rows",order,out)
    b,_=prepare_matrix(tmp_path/"shuffled.csv","cpg_rows",order,out)
    np.testing.assert_array_equal(a,b)
    beta.iloc[1:].to_csv(tmp_path/"missing.csv")
    with pytest.raises(ValueError,match="required CpGs"):
        prepare_matrix(tmp_path/"missing.csv","cpg_rows",order,out)


def test_paper_screening_pipeline(tmp_path):
    rng=np.random.default_rng(17)
    for cohort in ["train","external"]:
        y=np.tile([0,1],16)
        ids=[f"{cohort}{i}" for i in range(32)]
        beta=pd.DataFrame(np.clip(.2+.5*y[None,:]+rng.normal(0,.02,(8,32)),0,1),
                          index=[f"cg{i}" for i in range(8)],columns=ids)
        beta.to_csv(tmp_path/f"{cohort}.csv")
        pd.DataFrame({"SampleID":ids,"disease.state":np.where(y,"affected","unaffected")}).to_csv(
            tmp_path/f"{cohort}_meta.csv",index=False)
    out=tmp_path/"prepared"
    run_cli("prepare_features.py",["--train-beta",str(tmp_path/"train.csv"),"--train-meta",str(tmp_path/"train_meta.csv"),
        "--external-beta",str(tmp_path/"external.csv"),"--external-meta",str(tmp_path/"external_meta.csv"),
        "--mad-threshold","0.01","--outdir",str(out)])
    report=json.loads((out/"feature_selection_report.json").read_text())
    assert report["n_selected_intersection"]==8 and report["uses_external_labels_for_selection"]
    assert "SampleID" in pd.read_csv(out/"train_metadata.csv").columns
