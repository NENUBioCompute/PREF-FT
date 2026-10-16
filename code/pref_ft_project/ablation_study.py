#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PREF-FT component ablations.

The baseline contains CERM, CCEM, REPM, SMEM and the four-branch PREF Head.
Run python ablate.py --help for data, model and variant options.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import copy
import json
import os
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
from dataclasses import asdict, dataclass, field
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 复用现有模型组件
from pref_ft_project.pref_ft_model import (
    PREF_FT,
    CpGEvidenceRecalibrationModule,
    FeatureTokenizer,
    TransformerEncoderBlock,
    ResidualEvidencePoolingModule,
    MethylationEvidenceEncoder,
    EnsembleContextualEvidenceBranch,
    LinearCpGEvidenceBranch,
    PositiveEvidenceAmplificationBranch,
    PrototypeAnchoredEvidenceBranch,
    PositiveAwareResidualEvidenceFusionHead,
)


# ============================================================
# 1. 消融模型变体定义
# ============================================================

class EncoderNoGate(nn.Module):
    """移除 CpGEvidenceRecalibrationModule — 特征直通."""

    def __init__(self, num_features: int, d_model: int = 128, num_layers: int = 3,
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features=num_features, token_dim=d_model, dropout=dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(int(num_layers))
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.pool = ResidualEvidencePoolingModule(d_model=d_model, dropout=dropout, init_alpha=-2.5)
        self.stable_bottleneck = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        tokens = self.tokenizer(x)  # 无门控
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)
        z_raw, pool_aux = self.pool(tokens)
        z = self.stable_bottleneck(z_raw)
        aux: Dict[str, torch.Tensor] = {
            "features": z, "shared_embedding": z,
            "x_gated": x, "z_raw": z_raw.detach(),
        }
        aux.update(pool_aux)
        return z, aux


class SimpleLinearTokenizer(nn.Module):
    """将 FeatureTokenizer 替换为单层 Linear + [CLS] token."""

    def __init__(self, num_features: int, token_dim: int, dropout: float = 0.2):
        super().__init__()
        self.num_features = int(num_features)
        self.token_dim = int(token_dim)
        self.linear = nn.Linear(1, token_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F] → [B, F, 1] → [B, F, D]
        tokens = self.linear(x.unsqueeze(-1))
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        return self.dropout(torch.cat([cls, tokens], dim=1))


class EncoderSimpleLinearTokenizer(nn.Module):
    """使用 SimpleLinearTokenizer 替代 FeatureTokenizer 的编码器."""

    def __init__(self, num_features: int, d_model: int = 128, num_layers: int = 3,
                 num_heads: int = 4, dropout: float = 0.2, gate_dropout: float = 0.05):
        super().__init__()
        self.feature_gate = CpGEvidenceRecalibrationModule(num_features=num_features, init_logit=4.0, gate_dropout=gate_dropout)
        self.tokenizer = SimpleLinearTokenizer(num_features=num_features, token_dim=d_model, dropout=dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(int(num_layers))
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.pool = ResidualEvidencePoolingModule(d_model=d_model, dropout=dropout, init_alpha=-2.5)
        self.stable_bottleneck = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_gated, gate_aux = self.feature_gate(x)
        tokens = self.tokenizer(x_gated)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)
        z_raw, pool_aux = self.pool(tokens)
        z = self.stable_bottleneck(z_raw)
        aux: Dict[str, torch.Tensor] = {
            "features": z, "shared_embedding": z,
            "x_gated": x_gated, "z_raw": z_raw.detach(),
        }
        aux.update(gate_aux)
        aux.update(pool_aux)
        return z, aux


class MeanPooling(nn.Module):
    """Mean pooling over all CpG tokens (替代 CLS+Attention pooling)."""

    def __init__(self, d_model: int = 128, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cpg_tokens = tokens[:, 1:]  # 丢弃 CLS
        pooled = cpg_tokens.mean(dim=1)
        z = pooled + self.proj(pooled)  # 残差投影
        return z, {
            "token_pool_alpha": torch.tensor(0.5),
            "token_pool_attention": torch.ones(tokens.shape[0], cpg_tokens.shape[1]) / cpg_tokens.shape[1],
            "token_pool_embedding": pooled.detach(),
        }


class CLSOnlyPooling(nn.Module):
    """仅使用 CLS token (不使用 attention pooling over CpG tokens)."""

    def __init__(self, d_model: int = 128):
        super().__init__()

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cls = tokens[:, 0]
        return cls, {
            "token_pool_alpha": torch.tensor(0.0),
            "token_pool_attention": torch.zeros(tokens.shape[0], tokens.shape[1] - 1),
            "token_pool_embedding": cls.detach(),
        }


class EncoderMeanPooling(nn.Module):
    """使用 MeanPooling 的编码器."""

    def __init__(self, num_features: int, d_model: int = 128, num_layers: int = 3,
                 num_heads: int = 4, dropout: float = 0.2, gate_dropout: float = 0.05):
        super().__init__()
        self.feature_gate = CpGEvidenceRecalibrationModule(num_features=num_features, init_logit=4.0, gate_dropout=gate_dropout)
        self.tokenizer = FeatureTokenizer(num_features=num_features, token_dim=d_model, dropout=dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(int(num_layers))
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.pool = MeanPooling(d_model=d_model, dropout=dropout)
        self.stable_bottleneck = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_gated, gate_aux = self.feature_gate(x)
        tokens = self.tokenizer(x_gated)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)
        z_raw, pool_aux = self.pool(tokens)
        z = self.stable_bottleneck(z_raw)
        aux: Dict[str, torch.Tensor] = {
            "features": z, "shared_embedding": z,
            "x_gated": x_gated, "z_raw": z_raw.detach(),
        }
        aux.update(gate_aux)
        aux.update(pool_aux)
        return z, aux


class EncoderCLSOnlyPooling(nn.Module):
    """使用 CLS-only Pooling 的编码器."""

    def __init__(self, num_features: int, d_model: int = 128, num_layers: int = 3,
                 num_heads: int = 4, dropout: float = 0.2, gate_dropout: float = 0.05):
        super().__init__()
        self.feature_gate = CpGEvidenceRecalibrationModule(num_features=num_features, init_logit=4.0, gate_dropout=gate_dropout)
        self.tokenizer = FeatureTokenizer(num_features=num_features, token_dim=d_model, dropout=dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(int(num_layers))
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.pool = CLSOnlyPooling(d_model=d_model)
        self.stable_bottleneck = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_gated, gate_aux = self.feature_gate(x)
        tokens = self.tokenizer(x_gated)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)
        z_raw, pool_aux = self.pool(tokens)
        z = self.stable_bottleneck(z_raw)
        aux: Dict[str, torch.Tensor] = {
            "features": z, "shared_embedding": z,
            "x_gated": x_gated, "z_raw": z_raw.detach(),
        }
        aux.update(gate_aux)
        aux.update(pool_aux)
        return z, aux


class EncoderNoBottleneck(nn.Module):
    """移除 StableBottleneck 的编码器."""

    def __init__(self, num_features: int, d_model: int = 128, num_layers: int = 3,
                 num_heads: int = 4, dropout: float = 0.2, gate_dropout: float = 0.05):
        super().__init__()
        self.feature_gate = CpGEvidenceRecalibrationModule(num_features=num_features, init_logit=4.0, gate_dropout=gate_dropout)
        self.tokenizer = FeatureTokenizer(num_features=num_features, token_dim=d_model, dropout=dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(int(num_layers))
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.pool = ResidualEvidencePoolingModule(d_model=d_model, dropout=dropout, init_alpha=-2.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_gated, gate_aux = self.feature_gate(x)
        tokens = self.tokenizer(x_gated)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)
        z, pool_aux = self.pool(tokens)
        aux: Dict[str, torch.Tensor] = {
            "features": z, "shared_embedding": z,
            "x_gated": x_gated, "z_raw": z.detach(),
        }
        aux.update(gate_aux)
        aux.update(pool_aux)
        return z, aux


# ============================================================
# 2. 消融变体注册表
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class AblationVariant:
    """描述一个消融变体."""
    name: str                                # 唯一标识符
    description: str                         # 中文描述
    category: str                            # 分类: encoder / pooling / head / residual
    uses_custom_encoder: bool = False        # 是否使用自定义编码器
    encoder_factory: Optional[callable] = None  # 自定义编码器构造器
    # 传给 PREF_FT / PositiveAwareResidualEvidenceFusionHead 的参数覆盖
    overrides: Dict[str, Any] = field(default_factory=dict)


# 所有消融变体定义
ABLATION_VARIANTS: List[AblationVariant] = [
    # ---- Baseline ----
    AblationVariant(
        name="baseline",
        description="完整模型 (Baseline, 不做任何消融)",
        category="baseline",
    ),
    # ---- 编码器消融 ----
    AblationVariant(
        name="no_feature_gate",
        description="移除 CpGEvidenceRecalibrationModule, 特征直通输入",
        category="encoder",
        uses_custom_encoder=True,
        encoder_factory=lambda nf, cfg: EncoderNoGate(
            num_features=nf, d_model=cfg.d_model, num_layers=cfg.n_transformer_layers,
            num_heads=cfg.num_heads, dropout=cfg.dropout,
        ),
    ),
    AblationVariant(
        name="simple_linear_tokenizer",
        description="FeatureTokenizer → 单层 Linear 投影",
        category="encoder",
        uses_custom_encoder=True,
        encoder_factory=lambda nf, cfg: EncoderSimpleLinearTokenizer(
            num_features=nf, d_model=cfg.d_model, num_layers=cfg.n_transformer_layers,
            num_heads=cfg.num_heads, dropout=cfg.dropout, gate_dropout=cfg.gate_dropout,
        ),
    ),
    AblationVariant(
        name="transformer_layers_0",
        description="Transformer 层数 3 → 0 (仅 tokenizer + pooling)",
        category="encoder",
        overrides={"n_transformer_layers": 0},
    ),
    AblationVariant(
        name="transformer_layers_1",
        description="Transformer 层数 3 → 1",
        category="encoder",
        overrides={"n_transformer_layers": 1},
    ),
    AblationVariant(
        name="transformer_layers_2",
        description="Transformer 层数 3 → 2",
        category="encoder",
        overrides={"n_transformer_layers": 2},
    ),
    # ---- 池化消融 ----
    AblationVariant(
        name="mean_pooling",
        description="CLS+Attention Pooling → Mean Pooling",
        category="pooling",
        uses_custom_encoder=True,
        encoder_factory=lambda nf, cfg: EncoderMeanPooling(
            num_features=nf, d_model=cfg.d_model, num_layers=cfg.n_transformer_layers,
            num_heads=cfg.num_heads, dropout=cfg.dropout, gate_dropout=cfg.gate_dropout,
        ),
    ),
    AblationVariant(
        name="cls_only_pooling",
        description="CLS+Attention Pooling → 仅使用 CLS token",
        category="pooling",
        uses_custom_encoder=True,
        encoder_factory=lambda nf, cfg: EncoderCLSOnlyPooling(
            num_features=nf, d_model=cfg.d_model, num_layers=cfg.n_transformer_layers,
            num_heads=cfg.num_heads, dropout=cfg.dropout, gate_dropout=cfg.gate_dropout,
        ),
    ),
    # ---- 瓶颈消融 ----
    AblationVariant(
        name="no_stable_bottleneck",
        description="移除 StableBottleneck (LayerNorm+Linear+GELU+Dropout+LayerNorm)",
        category="encoder",
        uses_custom_encoder=True,
        encoder_factory=lambda nf, cfg: EncoderNoBottleneck(
            num_features=nf, d_model=cfg.d_model, num_layers=cfg.n_transformer_layers,
            num_heads=cfg.num_heads, dropout=cfg.dropout, gate_dropout=cfg.gate_dropout,
        ),
    ),
    # ---- 分类头消融 ----
    AblationVariant(
        name="single_member_head",
        description="EnsembleContextualEvidenceBranch (K=4) → 单成员头 (K=1)",
        category="head",
        overrides={"tabm_num_members": 1},
    ),
    # ---- 残差路径消融 ----
    AblationVariant(
        name="no_linear_residual",
        description="禁用 LinearCpGEvidenceBranch (CpG→logit 直连)",
        category="residual",
        overrides={"use_linear_residual": False},
    ),
    AblationVariant(
        name="no_positive_ovr",
        description="禁用 PositiveEvidenceAmplificationBranch (正类一对多)",
        category="residual",
        overrides={"use_positive_ovr": False},
    ),
    AblationVariant(
        name="no_prototype_residual",
        description="禁用 PrototypeAnchoredEvidenceBranch (原型相似度)",
        category="residual",
        overrides={"use_prototype_residual": False},
    ),
    AblationVariant(
        name="no_all_residuals",
        description="禁用全部三个残差路径 (仅保留 EnsembleContextualEvidenceBranch)",
        category="residual",
        overrides={
            "use_linear_residual": False,
            "use_positive_ovr": False,
            "use_prototype_residual": False,
        },
    ),
]


def get_variant_by_name(name: str) -> Optional[AblationVariant]:
    for v in ABLATION_VARIANTS:
        if v.name == name:
            return v
    return None


# ============================================================
# 3. 模型构建 (支持消融)
# ============================================================

def build_ablation_model(cfg: "AblationConfig", variant: AblationVariant,
                         num_features: int) -> nn.Module:
    """根据消融变体构建模型.

    对于仅修改参数即可实现的消融 (如 disable 某个残差), 直接使用 PREF_FT。
    对于需要替换编码器的消融, 组装自定义 encoder + 标准 PositiveAwareResidualEvidenceFusionHead。
    """
    # 合并参数: 默认 cfg → variant.overrides
    overrides = variant.overrides

    head_kwargs = dict(
        embed_dim=cfg.d_model,
        num_features=num_features,
        num_classes=2,
        tabm_num_members=overrides.get("tabm_num_members", cfg.tabm_num_members),
        dropout=cfg.dropout,
        linear_dropout=overrides.get("linear_dropout", cfg.linear_dropout),
        use_linear_residual=overrides.get("use_linear_residual", cfg.use_linear_residual),
        use_positive_ovr=overrides.get("use_positive_ovr", cfg.use_positive_ovr),
        use_prototype_residual=overrides.get("use_prototype_residual", cfg.use_prototype_residual),
    )

    if not variant.uses_custom_encoder:
        # 使用原生 PREF_FT, 仅参数覆盖
        return PREF_FT(
            num_features=num_features,
            num_classes=2,
            d_model=cfg.d_model,
            n_transformer_layers=overrides.get("n_transformer_layers", cfg.n_transformer_layers),
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            tabm_num_members=overrides.get("tabm_num_members", cfg.tabm_num_members),
            use_linear_residual=overrides.get("use_linear_residual", cfg.use_linear_residual),
            gate_dropout=cfg.gate_dropout,
            linear_dropout=overrides.get("linear_dropout", cfg.linear_dropout),
            use_positive_ovr=overrides.get("use_positive_ovr", cfg.use_positive_ovr),
            use_prototype_residual=overrides.get("use_prototype_residual", cfg.use_prototype_residual),
        )

    # 自定义编码器 + 标准分类头
    class CustomEncoderModel(nn.Module):
        def __init__(self, encoder, head):
            super().__init__()
            self.encoder = encoder
            self.head = head

        def forward(self, x):
            z, aux = self.encoder(x)
            logits, head_aux = self.head(z, x)
            aux.update(head_aux)
            return logits, aux

    encoder = variant.encoder_factory(num_features, cfg)
    head = PositiveAwareResidualEvidenceFusionHead(**head_kwargs)
    return CustomEncoderModel(encoder, head)


# ============================================================
# 4. 消融实验配置
# ============================================================

@dataclass
class AblationConfig:
    """消融实验配置 — 镜像原训练脚本 Config, 但专注于消融对比."""
    # 数据路径
    train_features: str = str(PROJECT_ROOT / "data/GSE243529_intersection_CpGs_after_com.csv.gz")
    train_metadata: str = str(PROJECT_ROOT / "data/GSE243529_pheno.csv")
    val_features: Optional[str] = None
    val_metadata: Optional[str] = None
    external_tests_json: Optional[str] = str(PROJECT_ROOT / "data/external_tests.json")

    # 输出
    outdir: str = str(PROJECT_ROOT / "runs/ablation")

    # 数据处理
    matrix_orientation: str = "cpg_rows"
    sample_col: str = "SampleID"
    label_col: str = "disease.state"
    class_order: str = "unaffected, affected"
    seed: int = 42
    val_size: float = 0.15
    stratify_val: bool = True
    external_feature_policy: str = "strict"
    min_feature_overlap_ratio: float = 0.98

    # 模型参数 (baseline)
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

    # 训练参数 (消融实验可适当缩短)
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

    # 硬件
    cpu: bool = False  # 强制使用 CPU

    # 消融特定
    variants: str = "all"  # "all" 或逗号分隔的变体名
    resume: bool = False


# ============================================================
# 5. 数据加载与预处理 (复用自 train_pref_ft.py)
# ============================================================

def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Failed to read file: {path}")


def ensure_sample_id_column(df: pd.DataFrame, sample_col: str) -> pd.DataFrame:
    df = df.copy()
    if sample_col in df.columns:
        df[sample_col] = df[sample_col].astype(str)
        return df
    if df.index.name == sample_col:
        df = df.reset_index()
        df[sample_col] = df[sample_col].astype(str)
        return df
    first_col = df.columns[0]
    df = df.rename(columns={first_col: sample_col})
    df[sample_col] = df[sample_col].astype(str)
    return df


@dataclass
class TaskDefinition:
    class_names: List[str]
    label_to_id: Dict[str, int]
    id_to_label: Dict[int, str]
    num_classes: int
    task_type: str = "binary"


def parse_class_order_to_task(class_order: str) -> TaskDefinition:
    class_names = [x.strip() for x in class_order.split(",") if x.strip()]
    if len(class_names) != 2:
        raise ValueError(f"Only binary tasks supported, got {class_names}")
    return TaskDefinition(
        class_names=class_names,
        label_to_id={name: i for i, name in enumerate(class_names)},
        id_to_label={i: name for i, name in enumerate(class_names)},
        num_classes=2,
    )


def canonicalize_label_series(series: pd.Series, label_map: Optional[dict] = None) -> pd.Series:
    out = series.astype(str).copy()
    if label_map is not None:
        out = out.map(lambda x: label_map.get(x, x))
    return out.astype(str)


@dataclass
class DatasetSpec:
    name: str
    features: str
    metadata: str
    matrix_orientation: str = "cpg_rows"
    sample_col: str = "SampleID"
    label_col: str = "disease.state"
    label_map: Optional[Dict[str, str]] = None


def matrix_to_sample_feature_df(feat, meta, matrix_orientation, sample_col, dataset_name):
    if matrix_orientation == "sample_rows":
        feat = ensure_sample_id_column(feat, sample_col)
        common = [s for s in meta[sample_col].astype(str).tolist() if s in set(feat[sample_col].astype(str))]
        if not common:
            raise ValueError(f"[{dataset_name}] No overlapping samples.")
        feat = feat[feat[sample_col].astype(str).isin(common)].copy()
        meta = meta[meta[sample_col].astype(str).isin(common)].copy()
        feat = feat.set_index(sample_col).loc[common].reset_index()
        meta = meta.set_index(sample_col).loc[common].reset_index()
        feature_cols = [c for c in feat.columns if c != sample_col]
        block = feat[feature_cols].apply(pd.to_numeric, errors="coerce")
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
            raise ValueError(f"[{dataset_name}] No overlapping samples between matrix and metadata.")
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


def load_matrix_and_metadata(spec: DatasetSpec):
    feat = read_table(spec.features)
    meta = read_table(spec.metadata)
    meta = ensure_sample_id_column(meta, spec.sample_col)
    if spec.label_col not in meta.columns:
        raise ValueError(f"[{spec.name}] label_col={spec.label_col} not found.")
    meta[spec.sample_col] = meta[spec.sample_col].astype(str)
    meta[spec.label_col] = canonicalize_label_series(meta[spec.label_col], spec.label_map)
    return feat, meta


def filter_by_task_labels(feat_df, meta_df, sample_col, label_col, task, dataset_name):
    allowed = set(task.class_names)
    labels_before = meta_df[label_col].astype(str)
    keep_mask = labels_before.isin(allowed)
    feat_kept = feat_df.loc[keep_mask.values].reset_index(drop=True)
    meta_kept = meta_df.loc[keep_mask.values].reset_index(drop=True)
    info = {
        "n_samples_before": int(len(meta_df)),
        "n_samples_after": int(len(meta_kept)),
        "n_dropped": int((~keep_mask).sum()),
        "allowed_labels": list(task.class_names),
    }
    if len(meta_kept) == 0:
        raise ValueError(f"[{dataset_name}] No samples after class_order filtering.")
    feat_kept = feat_kept.set_index(sample_col).loc[meta_kept[sample_col].tolist()].reset_index()
    return feat_kept, meta_kept, info


def encode_labels(meta, label_col, task, dataset_name):
    labels = meta[label_col].astype(str).values.tolist()
    unknown = sorted(list(set(labels) - set(task.class_names)))
    if unknown:
        raise ValueError(f"[{dataset_name}] unseen classes: {unknown}")
    return np.array([task.label_to_id[x] for x in labels], dtype=np.int64)


def load_dataset_as_arrays(spec, task, expected_feature_order=None,
                           external_feature_policy="strict", min_feature_overlap_ratio=0.98):
    feat, meta = load_matrix_and_metadata(spec)
    feat_df, meta_df, feature_cols = matrix_to_sample_feature_df(
        feat=feat, meta=meta, matrix_orientation=spec.matrix_orientation,
        sample_col=spec.sample_col, dataset_name=spec.name,
    )
    feat_df, meta_df, filter_info = filter_by_task_labels(
        feat_df=feat_df, meta_df=meta_df, sample_col=spec.sample_col,
        label_col=spec.label_col, task=task, dataset_name=spec.name,
    )
    y = encode_labels(meta_df, spec.label_col, task, spec.name)
    sample_ids = feat_df[spec.sample_col].astype(str).values
    numeric_df = feat_df[feature_cols].copy()

    overlap_info = {"n_raw_features": int(len(feature_cols))}
    if expected_feature_order is None:
        return {
            "X_raw": numeric_df.values.astype(np.float32), "y": y,
            "sample_ids": sample_ids, "feature_names": feature_cols,
            "overlap_info": overlap_info, "filter_info": filter_info,
        }

    present = [f for f in expected_feature_order if f in numeric_df.columns]
    missing = [f for f in expected_feature_order if f not in numeric_df.columns]
    overlap_ratio = len(present) / max(1, len(expected_feature_order))
    overlap_info.update({
        "n_overlap_with_train": int(len(present)),
        "overlap_ratio": float(overlap_ratio),
        "n_missing": int(len(missing)),
    })
    if external_feature_policy == "strict" and missing:
        raise ValueError(f"[{spec.name}] strict: {len(missing)} features missing.")
    if overlap_ratio < min_feature_overlap_ratio:
        raise ValueError(f"[{spec.name}] overlap too low: {overlap_ratio:.4f}")
    aligned_df = numeric_df.reindex(columns=expected_feature_order)
    return {
        "X_raw": aligned_df.values.astype(np.float32), "y": y,
        "sample_ids": sample_ids, "feature_names": list(expected_feature_order),
        "overlap_info": overlap_info, "filter_info": filter_info,
    }


def fit_preprocessor_on_train(X_train_raw: np.ndarray) -> dict:
    train_medians = np.nanmedian(X_train_raw, axis=0)
    train_medians = np.where(np.isnan(train_medians), 0.0, train_medians).astype(np.float32)
    X_imp = X_train_raw.copy()
    inds = np.where(np.isnan(X_imp))
    if len(inds[0]) > 0:
        X_imp[inds] = np.take(train_medians, inds[1])
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_imp).astype(np.float32)
    return {"train_medians": train_medians, "scaler": scaler, "X_train_scaled": X_scaled}


def transform_with_preprocessor(X_raw: np.ndarray, preproc: dict) -> np.ndarray:
    X_imp = X_raw.copy()
    inds = np.where(np.isnan(X_imp))
    if len(inds[0]) > 0:
        X_imp[inds] = np.take(preproc["train_medians"], inds[1])
    return preproc["scaler"].transform(X_imp).astype(np.float32)


def split_train_val(X_raw, y, sample_ids, cfg):
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
        "X_train_raw": X_raw[train_idx], "y_train": y[train_idx], "train_ids": sample_ids[train_idx],
        "X_val_raw": X_raw[val_idx], "y_val": y[val_idx], "val_ids": sample_ids[val_idx],
    }


# ============================================================
# 6. Dataset / DataLoader / Metrics / Training
# ============================================================

class MethylationDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(X, y, cfg, shuffle):
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    return DataLoader(
        MethylationDataset(X, y), batch_size=cfg.batch_size, shuffle=shuffle,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
        generator=generator, worker_init_fn=worker_init_fn,
    )


# ---- Metrics ----

def safe_divide(num, den):
    return float(num / den) if den != 0 else np.nan


def binary_auc(y_bin, score):
    try:
        if len(np.unique(y_bin)) < 2:
            return np.nan
        return float(roc_auc_score(y_bin, score))
    except Exception:
        return np.nan


def binary_auprc(y_bin, score):
    try:
        if len(np.unique(y_bin)) < 2:
            return np.nan
        return float(average_precision_score(y_bin, score))
    except Exception:
        return np.nan


def binary_mcc_from_counts(tp, fp, tn, fn):
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if den == 0:
        return np.nan
    return float(((tp * tn) - (fp * fn)) / np.sqrt(den))


def compute_metrics(y_true, prob, pred, class_names):
    y_true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    prob = np.asarray(prob)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    n = int(tp + tn + fp + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1 = safe_divide(2 * precision * recall, precision + recall) if not (np.isnan(precision) or np.isnan(recall)) else np.nan
    y_bin = (y_true == 1).astype(int)
    pos_score = prob[:, 1] if prob.ndim == 2 and prob.shape[1] > 1 else np.full_like(y_bin, np.nan, dtype=float)
    bal = float(np.nanmean([recall, specificity]))
    return {
        "acc": safe_divide(tp + tn, n),
        "bal_acc": bal,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": binary_mcc_from_counts(tp, fp, tn, fn),
        "auc_binary": binary_auc(y_bin, pos_score),
        "auprc_binary": binary_auprc(y_bin, pos_score),
        "tp": float(tp), "fp": float(fp), "tn": float(tn), "fn": float(fn),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "mcc_multiclass": float(matthews_corrcoef(y_true, pred)),
    }


def compute_class_weights(y, num_classes=2):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return (weights / weights.mean()).astype(np.float32)


def select_validation_score(metrics):
    for name in ["auc_binary", "f1", "bal_acc"]:
        value = metrics.get(name, np.nan)
        try:
            if pd.notna(value):
                return name, float(value)
        except Exception:
            pass
    return "none", -np.inf


# ---- Training ----

class EarlyStopping:
    def __init__(self, patience=30):
        self.patience = int(patience)
        self.best = None
        self.bad_epochs = 0
        self.best_state = None

    def step(self, score, model):
        improved = self.best is None or score > self.best
        if improved:
            self.best = float(score)
            self.bad_epochs = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def run_epoch(model, loader, criterion, device, amp_enabled, optimizer=None, max_grad_norm=1.0):
    is_train = optimizer is not None
    model.train(is_train)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and is_train and device.type == "cuda"))
    losses, probs_all, preds_all, y_all = [], [], [], []

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
        probs_all.append(prob)
        preds_all.append(prob.argmax(axis=1))
        y_all.append(yb.detach().cpu().numpy())
        losses.append(float(loss.detach().cpu().item()))

    probs = np.concatenate(probs_all, axis=0)
    preds = np.concatenate(preds_all, axis=0)
    y_true = np.concatenate(y_all, axis=0)
    metrics = compute_metrics(y_true, probs, preds, ["negative", "positive"])
    metrics["loss"] = float(np.mean(losses))
    return metrics, probs, preds, y_true


@torch.no_grad()
def predict(model, loader, criterion, device, amp_enabled):
    return run_epoch(model, loader, criterion, device, amp_enabled, optimizer=None)


def load_external_test_specs(path: Optional[str], cfg: AblationConfig) -> List[DatasetSpec]:
    """加载外部测试集配置 (与 train_pref_ft.py 逻辑一致)."""
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


def train_one_variant(variant: AblationVariant, cfg: AblationConfig,
                      train_data: dict, val_data: dict, device: torch.device,
                      variant_outdir: str,
                      external_test_specs: Optional[List[DatasetSpec]] = None,
                      train_feature_names: Optional[List[str]] = None,
                      preproc: Optional[dict] = None,
                      task: Optional[TaskDefinition] = None) -> Dict[str, Any]:
    """训练单个消融变体, 返回结果摘要 (含内部验证 + 外部测试)."""
    os.makedirs(variant_outdir, exist_ok=True)

    X_train = train_data["X_train"]
    y_train = train_data["y_train"]
    X_val = val_data["X_val"]
    y_val = val_data["y_val"]
    num_features = X_train.shape[1]

    # 构建模型
    model = build_ablation_model(cfg, variant, num_features).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    class_weights = torch.tensor(compute_class_weights(y_train), dtype=torch.float32, device=device) \
        if cfg.use_class_weight else None
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    early = EarlyStopping(patience=cfg.patience)

    train_loader = build_loader(X_train, y_train, cfg, shuffle=True)
    val_loader = build_loader(X_val, y_val, cfg, shuffle=False)

    # 保存变体配置
    save_json({
        "variant": variant.name,
        "description": variant.description,
        "category": variant.category,
        "overrides": variant.overrides,
        "uses_custom_encoder": variant.uses_custom_encoder,
        "num_parameters": param_count,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_features": int(num_features),
    }, os.path.join(variant_outdir, "variant_config.json"))

    history = []
    best_metric_name = "none"
    t_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        train_metrics, _, _, _ = run_epoch(
            model, train_loader, criterion, device, cfg.amp,
            optimizer=optimizer, max_grad_norm=cfg.max_grad_norm,
        )
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

        if early.step(score, model):
            print(f"    Early stopping at epoch {epoch}")
            break

    train_time = time.time() - t_start

    if early.best_state is not None:
        model.load_state_dict(early.best_state)

    # 最终评估
    train_final, _, _, _ = predict(model, train_loader, criterion, device, cfg.amp)
    val_final, _, _, _ = predict(model, val_loader, criterion, device, cfg.amp)

    # 保存
    pd.DataFrame(history).to_csv(os.path.join(variant_outdir, "history.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(variant_outdir, "best_model.pt"))
    final_metric_name, final_score = select_validation_score(val_final)

    result = {
        "variant": variant.name,
        "description": variant.description,
        "category": variant.category,
        "num_parameters": param_count,
        "train_time_seconds": round(train_time, 1),
        "best_epoch": len(history),
        "best_val_metric": best_metric_name,
        "best_val_score": to_jsonable(early.best),
        "final_val_metric": final_metric_name,
        "final_val_score": to_jsonable(final_score),
        "train_metrics": metrics_to_jsonable(train_final),
        "val_metrics": metrics_to_jsonable(val_final),
    }

    # ---- 外部独立测试集评估 ----
    external_results: Dict[str, Dict[str, Any]] = {}
    if external_test_specs and train_feature_names is not None and preproc is not None and task is not None:
        for ext_spec in external_test_specs:
            try:
                ext_pack = load_dataset_as_arrays(
                    ext_spec, task=task,
                    expected_feature_order=train_feature_names,
                    external_feature_policy=cfg.external_feature_policy,
                    min_feature_overlap_ratio=cfg.min_feature_overlap_ratio,
                )
                X_ext = transform_with_preprocessor(ext_pack["X_raw"], preproc)
                ext_loader = build_loader(X_ext, ext_pack["y"], cfg, shuffle=False)
                ext_metrics, ext_probs, ext_preds, ext_true = predict(
                    model, ext_loader, criterion, device, cfg.amp,
                )
                external_results[ext_spec.name] = {
                    "n_samples": int(len(ext_pack["y"])),
                    "metrics": metrics_to_jsonable(ext_metrics),
                    "overlap_info": {k: to_jsonable(v) for k, v in ext_pack.get("overlap_info", {}).items()},
                }
                print(f"    External [{ext_spec.name}]: AUC={ext_metrics.get('auc_binary', np.nan):.4f}, "
                      f"F1={ext_metrics.get('f1', np.nan):.4f}, n={len(ext_pack['y'])}")
            except Exception as e:
                print(f"    External [{ext_spec.name}]: ERROR — {e}")
                external_results[ext_spec.name] = {"error": str(e)}
        result["external_results"] = external_results

    save_json(result, os.path.join(variant_outdir, "result.json"))
    return result


# ============================================================
# 7. 结果分析与可视化
# ============================================================

# 用于展示的关键指标
DISPLAY_METRICS = [
    ("auc_binary", "ROC-AUC"),
    ("auprc_binary", "AUPRC"),
    ("bal_acc", "Balanced Acc"),
    ("f1", "F1 Score"),
    ("mcc", "MCC"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("specificity", "Specificity"),
    ("acc", "Accuracy"),
]

# 用于排序的主要指标
PRIMARY_METRIC = "auc_binary"


def _collect_external_dataset_names(results: List[Dict[str, Any]]) -> List[str]:
    """从结果中提取所有外部测试集名称 (按字母序)."""
    names: set = set()
    for r in results:
        for ds_name in (r.get("external_results") or {}).keys():
            if "error" not in (r["external_results"].get(ds_name) or {}):
                names.add(ds_name)
    return sorted(names)


def _ext_metric_key(dataset_name: str, metric: str) -> str:
    """构建外部测试集的列名."""
    safe_ds = dataset_name.replace(" ", "_").replace("-", "_").replace("'", "")
    return f"ext_{safe_ds}_{metric}"


MERGED_METRIC_COLUMNS = ["auc_binary", "f1", "bal_acc", "mcc", "auprc_binary"]


def generate_comparison_table(results: List[Dict[str, Any]],
                               external_datasets: Optional[List[str]] = None) -> pd.DataFrame:
    """从所有变体的结果生成对比 DataFrame (含内部验证 + 外部测试)."""
    if external_datasets is None:
        external_datasets = _collect_external_dataset_names(results)

    rows = []
    for r in results:
        row = {
            "variant": r["variant"],
            "description": r["description"],
            "category": r["category"],
            "num_params": r["num_parameters"],
            "train_time_s": r["train_time_seconds"],
            "best_epoch": r["best_epoch"],
            "best_val_score": r["best_val_score"],
        }
        val_metrics = r.get("val_metrics", {})
        for metric_key, _ in DISPLAY_METRICS:
            row[f"val_{metric_key}"] = val_metrics.get(metric_key, np.nan)
        train_metrics = r.get("train_metrics", {})
        for metric_key, _ in DISPLAY_METRICS:
            row[f"train_{metric_key}"] = train_metrics.get(metric_key, np.nan)
        # 外部测试集指标
        ext_results = r.get("external_results") or {}
        for ds_name in external_datasets:
            ds_metrics = (ext_results.get(ds_name) or {}).get("metrics") or {}
            for m in DISPLAY_METRICS:
                row[_ext_metric_key(ds_name, m[0])] = ds_metrics.get(m[0], np.nan)
            row[_ext_metric_key(ds_name, "n_samples")] = (ext_results.get(ds_name) or {}).get("n_samples", np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    # 默认按验证集主要指标降序
    primary_col = f"val_{PRIMARY_METRIC}"
    if primary_col in df.columns:
        df = df.sort_values(primary_col, ascending=False).reset_index(drop=True)
    return df


def plot_comparison_bars(df: pd.DataFrame, save_path: str,
                         col_prefix: str = "val_",
                         title_suffix: str = "Validation") -> None:
    """绘制关键指标对比柱状图 (支持内部验证或外部测试集)."""
    if col_prefix.startswith("val_"):
        metric_defs = DISPLAY_METRICS
    else:
        metric_defs = [(k, k.replace("_", " ").title()) for k in MERGED_METRIC_COLUMNS]
    cols = [f"{col_prefix}{k}" for k, _ in metric_defs]
    available_cols = [c for c in cols if c in df.columns]
    metric_labels = [label for (k, label), c in zip(metric_defs, cols) if c in df.columns]

    n_metrics = len(available_cols)
    n_variants = len(df)
    if n_metrics == 0 or n_variants == 0:
        return

    x = np.arange(n_variants)
    width = 0.8 / n_metrics
    colors = plt.cm.tab10(np.linspace(0, 1, n_metrics))

    fig, ax = plt.subplots(figsize=(max(14, n_variants * 1.2), 7))
    for i, (col, label) in enumerate(zip(available_cols, metric_labels)):
        values = df[col].values
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=colors[i], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=5.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"].values, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title(f"Ablation Study — {title_suffix} Metrics Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
    print(f"  [plot] Comparison bars ({title_suffix}) saved to {save_path}")


def plot_radar(df: pd.DataFrame, save_path: str, col_prefix: str = "val_",
               title_suffix: str = "Validation (Top-8)") -> None:
    """绘制雷达图."""
    radar_metrics = MERGED_METRIC_COLUMNS[:8]
    radar_labels = ["ROC-AUC", "F1", "Bal Acc", "MCC", "AUPRC"]
    cols = [f"{col_prefix}{k}" for k in radar_metrics]
    available_cols = [c for c in cols if c in df.columns]
    available_labels = [l for (k, l) in zip(radar_metrics, radar_labels) if f"{col_prefix}{k}" in df.columns]

    if len(available_cols) < 3:
        return
    top_df = df.head(8).copy()

    n_metrics = len(available_cols)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_df)))

    for idx, (_, row) in enumerate(top_df.iterrows()):
        values = [row.get(c, np.nan) for c in available_cols]
        values = [v if not np.isnan(v) else 0.0 for v in values]
        values += values[:1]
        ax.fill(angles, values, alpha=0.08, color=colors[idx])
        ax.plot(angles, values, "o-", linewidth=1.8, label=row["variant"], color=colors[idx], markersize=4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title(f"Ablation Study — Radar ({title_suffix})", fontsize=14, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
    print(f"  [plot] Radar ({title_suffix}) saved to {save_path}")


def plot_metric_drop(df: pd.DataFrame, save_path: str, metric_col: str,
                     metric_label: str, baseline_label: str = "baseline") -> None:
    """绘制各变体相对 Baseline 的性能变化瀑布图 (通用)."""
    baseline_row = df[df["variant"] == baseline_label]
    if baseline_row.empty:
        return
    baseline_val = baseline_row.iloc[0][metric_col]
    if np.isnan(baseline_val):
        return

    others = df[df["variant"] != baseline_label].copy()
    others["delta"] = others[metric_col] - baseline_val
    others = others.sort_values("delta")

    fig, ax = plt.subplots(figsize=(12, 6))
    variants_list = others["variant"].tolist()
    deltas = others["delta"].tolist()
    bar_colors = ["#d62728" if d < 0 else "#2ca02c" for d in deltas]

    bars = ax.barh(variants_list, deltas, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--")

    for bar, delta in zip(bars, deltas):
        x_pos = bar.get_width()
        offset = -0.012 if x_pos < 0 else 0.003
        ha = "right" if x_pos < 0 else "left"
        ax.text(x_pos + offset, bar.get_y() + bar.get_height() / 2,
                f"{delta:+.4f}", va="center", ha=ha, fontsize=9,
                color="#a00000" if delta < 0 else "#006000")

    ax.set_xlabel(f"Δ {metric_label} (vs Baseline = {baseline_val:.4f})", fontsize=11)
    ax.set_title(f"Ablation Study — {metric_label} Change vs Baseline", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
    print(f"  [plot] Metric drop ({metric_label}) saved to {save_path}")


def _print_summary_table(df: pd.DataFrame, title: str, metric_col: str,
                         metric_label: str, external_datasets: List[str]) -> None:
    """打印终端摘要表格."""
    print("\n" + "=" * 130)
    print(f"  {title}")
    print("=" * 130)

    is_ext = metric_col.startswith("ext_")
    if is_ext:
        # 按外部测试集指标排序
        sort_col = metric_col
        df_sorted = df.sort_values(sort_col, ascending=False).reset_index(drop=True) if sort_col in df.columns else df
    else:
        df_sorted = df

    baseline_auc = None
    if metric_col in df_sorted.columns:
        baseline_row = df_sorted[df_sorted["variant"] == "baseline"]
        if not baseline_row.empty:
            baseline_auc = baseline_row.iloc[0][metric_col]
            if not np.isnan(baseline_auc):
                print(f"  Baseline {metric_label}: {baseline_auc:.4f}\n")

    # 表头
    if is_ext:
        header = f"{'Variant':<30s} {'Category':<12s} {'Params':>8s} {'ValAUC':>8s} {'ExtAUC':>8s} {'ExtF1':>8s} {'ExtBal':>8s} {'ExtMCC':>8s} {'Time(s)':>8s} {'Δ ExtAUC':>10s}"
    else:
        header = f"{'Variant':<30s} {'Category':<12s} {'Params':>8s} {'AUC':>8s} {'F1':>8s} {'BalAcc':>8s} {'MCC':>8s} {'Time(s)':>8s} {'Δ AUC':>8s}"
    print(header)
    print("-" * len(header))

    for _, row in df_sorted.iterrows():
        if is_ext:
            val_auc = row.get("val_auc_binary", np.nan)
            ext_auc = row.get(metric_col, np.nan)
            ext_f1 = row.get(metric_col.replace("auc_binary", "f1"), np.nan) if "auc_binary" in metric_col else np.nan
            ext_bal = row.get(metric_col.replace("auc_binary", "bal_acc"), np.nan) if "auc_binary" in metric_col else np.nan
            ext_mcc = row.get(metric_col.replace("auc_binary", "mcc"), np.nan) if "auc_binary" in metric_col else np.nan
            delta = ext_auc - baseline_auc if (baseline_auc is not None and not np.isnan(ext_auc)) else np.nan
            delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"
            print(f"{row['variant']:<30s} {row['category']:<12s} {row['num_params']:>8,} "
                  f"{val_auc:>8.4f} {ext_auc:>8.4f} {ext_f1:>8.4f} {ext_bal:>8.4f} {ext_mcc:>8.4f} "
                  f"{row['train_time_s']:>8.1f} {delta_str:>10s}")
        else:
            delta = row.get(metric_col, np.nan) - baseline_auc if baseline_auc is not None and not np.isnan(row.get(metric_col, np.nan)) else np.nan
            delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"
            print(f"{row['variant']:<30s} {row['category']:<12s} {row['num_params']:>8,} "
                  f"{row.get('val_auc_binary', np.nan):>8.4f} {row.get('val_f1', np.nan):>8.4f} "
                  f"{row.get('val_bal_acc', np.nan):>8.4f} {row.get('val_mcc', np.nan):>8.4f} "
                  f"{row['train_time_s']:>8.1f} {delta_str:>8s}")

    print("-" * len(header))


def generate_summary_report(df: pd.DataFrame, results: List[Dict], outdir: str,
                            external_datasets: Optional[List[str]] = None) -> None:
    """生成消融实验汇总报告 (内部验证 + 外部测试双视角)."""
    if external_datasets is None:
        external_datasets = _collect_external_dataset_names(results)

    # CSV (通用)
    df.to_csv(os.path.join(outdir, "ablation_summary.csv"), index=False)

    # JSON
    save_json({
        "variants": results,
        "summary_table": df.where(pd.notna(df), None).to_dict(orient="records"),
        "primary_metric": PRIMARY_METRIC,
        "n_variants": len(results),
        "external_datasets": external_datasets,
    }, os.path.join(outdir, "ablation_summary.json"))

    # ---- 内部验证集视角 ----
    _print_summary_table(df, "RESULTS — Internal Validation (val_auc_binary)",
                         "val_auc_binary", "VAL AUC", external_datasets)
    plot_comparison_bars(df, os.path.join(outdir, "ablation_comparison_val.png"),
                         col_prefix="val_", title_suffix="Internal Validation")
    plot_radar(df, os.path.join(outdir, "ablation_radar_val.png"),
               col_prefix="val_", title_suffix="Internal Validation")
    plot_metric_drop(df, os.path.join(outdir, "ablation_metric_drop_val.png"),
                     metric_col="val_auc_binary", metric_label="VAL AUC")

    # ---- 外部独立测试集视角 ----
    for ds_name in external_datasets:
        ext_auc_col = _ext_metric_key(ds_name, "auc_binary")
        if ext_auc_col not in df.columns:
            continue
        # 按此外部测试集 AUC 排序
        df_ext_sorted = df.sort_values(ext_auc_col, ascending=False).reset_index(drop=True)

        _print_summary_table(df_ext_sorted,
                             f"RESULTS — External [{ds_name}] ({ext_auc_col})",
                             ext_auc_col, f"EXT {ds_name} AUC", external_datasets)

        safe_ds = ds_name.replace(" ", "_").replace("-", "_").replace("'", "")
        plot_comparison_bars(df_ext_sorted,
                             os.path.join(outdir, f"ablation_comparison_ext_{safe_ds}.png"),
                             col_prefix=f"ext_{safe_ds}_",
                             title_suffix=f"External [{ds_name}]")
        plot_radar(df_ext_sorted,
                   os.path.join(outdir, f"ablation_radar_ext_{safe_ds}.png"),
                   col_prefix=f"ext_{safe_ds}_",
                   title_suffix=f"External [{ds_name}]")
        plot_metric_drop(df_ext_sorted,
                         os.path.join(outdir, f"ablation_metric_drop_ext_{safe_ds}.png"),
                         metric_col=ext_auc_col,
                         metric_label=f"EXT [{ds_name}] AUC")

    # ---- 验证集 vs 外部测试集 排名对比 ----
    _plot_val_vs_ext_scatter(df, external_datasets, outdir)


def _plot_val_vs_ext_scatter(df: pd.DataFrame, external_datasets: List[str],
                              outdir: str) -> None:
    """绘制验证集 AUC vs 外部测试集 AUC 散点图 (揭示过拟合/泛化差距)."""
    for ds_name in external_datasets:
        ext_auc_col = _ext_metric_key(ds_name, "auc_binary")
        val_col = "val_auc_binary"
        if ext_auc_col not in df.columns or val_col not in df.columns:
            continue
        valid = df[[val_col, ext_auc_col, "variant"]].dropna()
        if len(valid) < 3:
            continue

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(valid[val_col], valid[ext_auc_col], s=80, c="#2c3e50", edgecolors="white", linewidth=0.5, zorder=5)

        # 标注每个点
        for _, row in valid.iterrows():
            ax.annotate(row["variant"], (row[val_col], row[ext_auc_col]),
                        textcoords="offset points", xytext=(5, 5), fontsize=6.5,
                        alpha=0.85)

        # 对角线 y=x (完美泛化)
        all_vals = np.concatenate([valid[val_col].values, valid[ext_auc_col].values])
        lo, hi = all_vals.min() - 0.02, all_vals.max() + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.4, label="y=x (perfect generalization)")

        # 标注 baseline
        baseline = valid[valid["variant"] == "baseline"]
        if not baseline.empty:
            ax.scatter(baseline[val_col], baseline[ext_auc_col],
                       s=140, c="#e74c3c", marker="*", edgecolors="black",
                       linewidth=1, zorder=10, label="baseline")

        ax.set_xlabel("Internal Validation AUC", fontsize=11)
        ax.set_ylabel(f"External [{ds_name}] AUC", fontsize=11)
        ax.set_title(f"Generalization Gap: Val AUC vs External [{ds_name}] AUC", fontsize=13, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        plt.tight_layout()
        safe_ds = ds_name.replace(" ", "_").replace("-", "_").replace("'", "")
        plt.savefig(os.path.join(outdir, f"ablation_val_vs_ext_{safe_ds}.png"), dpi=220)
        plt.close()
        print(f"  [plot] Val-vs-Ext scatter ({ds_name}) saved to {outdir}/ablation_val_vs_ext_{safe_ds}.png")


# ============================================================
# 8. 主流程
# ============================================================

def parse_args() -> AblationConfig:
    parser = argparse.ArgumentParser(
        description="PREF_FT 消融实验 — 一体化脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 运行全部 14 个变体
    python ablation_study.py --train-features data.csv --train-metadata meta.csv --outdir results/ablation

    # 仅运行部分变体
    python ablation_study.py --variants baseline,no_feature_gate,mean_pooling

    # 快速消融 (减少 epoch)
    python ablation_study.py --epochs 40 --patience 10

    # 从中断处恢复
    python ablation_study.py --resume --outdir results/ablation

    # 使用 CPU
    python ablation_study.py --cpu
        """,
    )

    # 数据
    parser.add_argument("--train-features", type=str, default=AblationConfig.train_features)
    parser.add_argument("--train-metadata", type=str, default=AblationConfig.train_metadata)
    parser.add_argument("--val-features", type=str, default=None)
    parser.add_argument("--val-metadata", type=str, default=None)
    parser.add_argument("--external-tests-json", type=str, default=AblationConfig.external_tests_json)
    parser.add_argument("--outdir", type=str, default=AblationConfig.outdir)
    parser.add_argument("--matrix-orientation", type=str, default="cpg_rows", choices=["sample_rows", "cpg_rows"])
    parser.add_argument("--sample-col", type=str, default="SampleID")
    parser.add_argument("--label-col", type=str, default="disease.state")
    parser.add_argument("--class-order", type=str, default="unaffected, affected")

    # 训练
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    # 模型
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n-transformer-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ensemble-members", "--tabm-num-members", dest="tabm_num_members", type=int, default=4)
    parser.add_argument("--gate-dropout", type=float, default=0.05)
    parser.add_argument("--linear-dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)

    # 消融特定
    parser.add_argument("--variants", type=str, default="all",
                        help="要运行的变体 (逗号分隔), 或 'all' 运行全部. "
                             "可选: " + ", ".join(v.name for v in ABLATION_VARIANTS))
    parser.add_argument("--resume", action="store_true",
                        help="从中断处恢复 — 跳过已有 result.json 的变体")

    args = parser.parse_args()

    cfg = AblationConfig()
    for k, v in vars(args).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.stratify_val = True  # ablation study always uses stratified split
    return cfg


def main():
    cfg = parse_args()

    # 确定要运行的变体
    if cfg.variants.strip().lower() == "all":
        variants_to_run = list(ABLATION_VARIANTS)
    else:
        names = [n.strip() for n in cfg.variants.split(",") if n.strip()]
        variants_to_run = []
        for name in names:
            v = get_variant_by_name(name)
            if v is None:
                print(f"WARNING: Unknown variant '{name}', skipping. Available: "
                      f"{[x.name for x in ABLATION_VARIANTS]}")
            else:
                variants_to_run.append(v)

    if not variants_to_run:
        print("ERROR: No valid variants to run.")
        sys.exit(1)

    print("=" * 80)
    print("PREF_FT — Ablation Study")
    print("=" * 80)
    print(f"Output directory: {cfg.outdir}")
    print(f"Variants to run ({len(variants_to_run)}):")
    for v in variants_to_run:
        print(f"  - {v.name:<35s} [{v.category}] {v.description}")
    print(f"Epochs: {cfg.epochs}, Patience: {cfg.patience}, Seed: {cfg.seed}")
    print("=" * 80)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")
    print(f"Device: {device}")

    # 加载数据
    set_seed(cfg.seed)
    task = parse_class_order_to_task(cfg.class_order)

    train_spec = DatasetSpec(
        name="train_cohort", features=cfg.train_features, metadata=cfg.train_metadata,
        matrix_orientation=cfg.matrix_orientation, sample_col=cfg.sample_col, label_col=cfg.label_col,
    )
    train_pack = load_dataset_as_arrays(train_spec, task=task)
    label_counts = np.bincount(train_pack["y"], minlength=2)
    missing = [task.id_to_label[i] for i, c in enumerate(label_counts) if c == 0]
    if missing:
        raise ValueError(f"Training data missing class(es): {missing}")

    split = split_train_val(train_pack["X_raw"], train_pack["y"], train_pack["sample_ids"], cfg)
    preproc = fit_preprocessor_on_train(split["X_train_raw"])
    X_train = preproc["X_train_scaled"]
    X_val = transform_with_preprocessor(split["X_val_raw"], preproc)

    train_data = {"X_train": X_train, "y_train": split["y_train"], "train_ids": split["train_ids"]}
    val_data = {"X_val": X_val, "y_val": split["y_val"], "val_ids": split["val_ids"]}

    print(f"\nTrain samples: {len(split['y_train'])}, Val samples: {len(split['y_val'])}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Class distribution — Train: {np.bincount(split['y_train'])}, Val: {np.bincount(split['y_val'])}")

    # 保存全局配置
    os.makedirs(cfg.outdir, exist_ok=True)
    save_json({
        "config": {k: str(v) if isinstance(v, (np.ndarray,)) else v for k, v in asdict(cfg).items()},
        "variants_requested": [v.name for v in variants_to_run],
        "device": str(device),
        "label_mapping": {"class_names": task.class_names, "label_to_id": task.label_to_id},
    }, os.path.join(cfg.outdir, "ablation_config.json"))

    # 保存预处理 (仅一次)
    np.save(os.path.join(cfg.outdir, "train_feature_medians.npy"), preproc["train_medians"])
    np.save(os.path.join(cfg.outdir, "scaler_mean.npy"), preproc["scaler"].mean_.astype(np.float32))
    np.save(os.path.join(cfg.outdir, "scaler_scale.npy"), preproc["scaler"].scale_.astype(np.float32))

    # 加载外部独立测试集
    external_test_specs = load_external_test_specs(cfg.external_tests_json, cfg)
    if external_test_specs:
        print(f"\nExternal test sets ({len(external_test_specs)}):")
        for s in external_test_specs:
            print(f"  - {s.name}: {os.path.basename(s.features)}")

    # 逐变体训练
    all_results = []
    for idx, variant in enumerate(variants_to_run):
        variant_outdir = os.path.join(cfg.outdir, variant.name)
        result_path = os.path.join(variant_outdir, "result.json")

        if cfg.resume and os.path.exists(result_path):
            print(f"\n[{idx+1}/{len(variants_to_run)}] {variant.name} — SKIPPED (result.json exists)")
            with open(result_path, "r", encoding="utf-8") as f:
                all_results.append(json.load(f))
            continue

        print(f"\n[{idx+1}/{len(variants_to_run)}] {variant.name}: {variant.description}")
        print(f"    Category: {variant.category} | "
              f"Custom encoder: {variant.uses_custom_encoder} | "
              f"Overrides: {variant.overrides if variant.overrides else 'none'}")

        try:
            result = train_one_variant(
                variant, cfg, train_data, val_data, device, variant_outdir,
                external_test_specs=external_test_specs,
                train_feature_names=train_pack["feature_names"],
                preproc=preproc,
                task=task,
            )
            all_results.append(result)
            print(f"    Done. Best val {result['best_val_metric']}={result['best_val_score']:.4f}, "
                  f"Time={result['train_time_seconds']:.0f}s")
        except Exception as e:
            print(f"    ERROR training {variant.name}: {e}")
            import traceback
            traceback.print_exc()
            # 记录失败
            all_results.append({
                "variant": variant.name,
                "description": variant.description,
                "category": variant.category,
                "num_parameters": 0,
                "train_time_seconds": 0,
                "best_val_score": np.nan,
                "train_metrics": {},
                "val_metrics": {},
                "error": str(e),
            })

    # 汇总
    external_datasets = _collect_external_dataset_names(all_results)
    df = generate_comparison_table(all_results, external_datasets)
    generate_summary_report(df, all_results, cfg.outdir, external_datasets)

    print(f"\nAblation study complete. All results in: {cfg.outdir}")


if __name__ == "__main__":
    main()
