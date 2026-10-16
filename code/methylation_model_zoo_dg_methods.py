#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model zoo for DNA methylation classification.

Designed for plug-and-play replacement in the training pipeline.

Included models
---------------
1) ResidualMLPClassifier
   Strong tabular baseline. Recommended because CpG ordering may be weak.
2) MultiScaleCNNClassifier
   Order-aware baseline when CpGs are stably ordered.
3) MultiScaleBiMambaClassifier
   Original-ish Mamba model for comparison.
4) HybridMultiScaleMambaClassifier
   Improved architecture:
   - local multi-scale convolutional stem
   - bidirectional Mamba blocks
   - gated cross-scale fusion
   - auxiliary ordinal head for ordered labels (HC -> MCI -> AD)

Usage
-----
from methylation_model_zoo import build_model
model = build_model(
    model_name="hybrid_ms_mamba",
    num_features=5804,
    num_classes=3,
    d_model=128,
)

Integration notes
-----------------
- Main training script should call model(x) and may receive:
  logits, aux
- If aux contains 'ordinal_logits', you can add auxiliary ordinal loss.
"""

from typing import Dict, Tuple, Optional
import importlib

# Import optional OpenMP-heavy GBDT libraries before torch when possible.  This
# reduces the chance of Linux static TLS errors such as:
# "libgomp... cannot allocate memory in static TLS block".
_XGB_IMPORT_ERROR = None
try:
    import xgboost as xgb
except Exception as e:
    xgb = None
    _XGB_IMPORT_ERROR = e

_LGB_IMPORT_ERROR = None
try:
    import lightgbm as lgb
except Exception as e:
    lgb = None
    _LGB_IMPORT_ERROR = e

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder

def _require_xgboost():
    """Import xgboost lazily and preserve the real import error if it fails."""
    global xgb, _XGB_IMPORT_ERROR
    if xgb is not None:
        return xgb
    try:
        xgb = importlib.import_module("xgboost")
        _XGB_IMPORT_ERROR = None
        return xgb
    except Exception as e:
        msg = (
            "xgboost is installed incorrectly or cannot be imported in the active environment. "
            "First verify with: python -c \"import xgboost; print(xgboost.__version__)\". "
            f"Initial import error: {repr(_XGB_IMPORT_ERROR)}; retry import error: {repr(e)}"
        )
        raise ImportError(msg) from e


def _require_lightgbm():
    """Import lightgbm lazily and preserve the real import error if it fails."""
    global lgb, _LGB_IMPORT_ERROR
    if lgb is not None:
        return lgb
    try:
        lgb = importlib.import_module("lightgbm")
        _LGB_IMPORT_ERROR = None
        return lgb
    except Exception as e:
        msg = (
            "lightgbm is installed incorrectly or cannot be imported in the active environment. "
            "First verify with: python -c \"import lightgbm; print(lightgbm.__version__)\". "
            f"Initial import error: {repr(_LGB_IMPORT_ERROR)}; retry import error: {repr(e)}"
        )
        raise ImportError(msg) from e

try:
    from methylation_auc_boost_model import AUCBoostTabularNet
except Exception:
    AUCBoostTabularNet = None
try:
    from methylation_subspace_ensemble_model import SubspaceEnsembleTabularNet
except Exception:
    SubspaceEnsembleTabularNet = None
try:
    from methylation_hfr_model import HFRNet
except Exception:
    HFRNet = None
try:
    from methylation_rank_hfr_mlp import RankHFRMLP
except Exception:
    RankHFRMLP = None
try:
    from methylation_sparse_rank_hfr_mlp import SparseRankHFRMLP
except Exception:
    SparseRankHFRMLP = None
try:
    from methylation_linear_residual_rank_hfr_mlp import LinearResidualRankHFRMLP
except Exception:
    LinearResidualRankHFRMLP = None
try:
    from CrossCohortLinearResidualRankHFRMLP import CrossCohortLinearResidualRankHFRMLP
except Exception:
    CrossCohortLinearResidualRankHFRMLP = None
try:
    from methylation_cross_cohort_dg_lr_rank_hfr_mlp import CrossCohortDGLinearResidualRankHFRMLP
except Exception:
    CrossCohortDGLinearResidualRankHFRMLP = None
# from methylation_cross_cohort_hodg_joint_lr_rank_hfr_mlp import CrossCohortHODGJointLRRankHFRMLP

try:
    from cross_cohort_lr_gate_hfr_mlp import CrossCohortLRGateHFRMLP
except ImportError:
    CrossCohortLRGateHFRMLP = None

try:
    from methylation_advanced_tabular_models import (
        FTTransformerClassifier,
        ModernNCAClassifier,
        TabMClassifier,
        DANNFTTransformerClassifier,
        DANNFTTransformerClassifier_v1,
        ImprovedFTTransformerClassifier,
        MethylationFTTransformerClassifier
    )
except Exception:
    FTTransformerClassifier = None
    ModernNCAClassifier = None
    TabMClassifier = None
    DANNFTTransformerClassifier = None
    DANNFTTransformerClassifier_v1 = None
    ImprovedFTTransformerClassifier = None
    MethylationFTTransformerClassifier = None
# ============================================================
# Small utilities
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight
 

class GLUBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        x = a * torch.sigmoid(b)
        x = self.out(self.drop(x))
        return x


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2, expansion: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.ffn(self.norm(x))


class BiMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.2):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba-ssm is required only for ms_mamba/hybrid_ms_mamba. Install with: pip install mamba-ssm")
        self.norm = RMSNorm(d_model)
        self.fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm(x)
        y_fwd = self.fwd(h)
        y_bwd = torch.flip(self.bwd(torch.flip(h, dims=[1])), dims=[1])
        y = self.drop(self.proj(0.5 * (y_fwd + y_bwd)))
        x = x + y
        x = x + self.ffn(x)
        return x


class AttnPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # x: [B, L, D]
        w = torch.softmax(self.net(x).squeeze(-1), dim=1)
        y = torch.sum(x * w.unsqueeze(-1), dim=1)
        return y, w


# ============================================================
# 1) Strong tabular baseline
# ============================================================

class ResidualMLPClassifier(nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden_dim: int = 512, depth: int = 4, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim, dropout=dropout) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        logits = self.head(h)
        return logits, {}


class StableMethylationResMLP(nn.Module):
    """Stable MLP with feature gating, linear residual, and hierarchical aux heads."""
    def __init__(
        self,
        num_features: int,
        num_classes: int = 3,
        hidden_dim: int = 256,
        depth: int = 2,
        dropout: float = 0.45,
        expansion: int = 2,
        use_linear_residual: bool = True,
    ):
        super().__init__()

        self.input_norm = nn.BatchNorm1d(num_features)
        self.feature_gate = nn.Parameter(torch.zeros(num_features))

        self.stem = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList([
            ResidualMLPBlock(dim=hidden_dim, dropout=dropout, expansion=expansion)
            for _ in range(depth)
        ])

        self.main_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.linear_head = nn.Linear(num_features, num_classes) if use_linear_residual else None
        self.linear_alpha = nn.Parameter(torch.tensor(-1.0))

        # Hierarchical auxiliary heads
        self.cn_vs_disease = nn.Linear(hidden_dim, 1)
        self.mci_vs_ad = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_norm(x)

        gate = torch.sigmoid(self.feature_gate)
        x_gated = x * gate

        h = self.stem(x_gated)
        for blk in self.blocks:
            h = blk(h)

        logits = self.main_head(h)

        if self.linear_head is not None:
            alpha = torch.sigmoid(self.linear_alpha)
            logits = logits + alpha * self.linear_head(x_gated)

        aux = {
            "feature_gate": gate,
            "cn_vs_disease_logits": self.cn_vs_disease(h).squeeze(-1),
            "mci_vs_ad_logits": self.mci_vs_ad(h).squeeze(-1),
        }
        return logits, aux


# ============================================================
# 2) Order-aware CNN baseline
# ============================================================

class MultiScaleCNNClassifier(nn.Module):
    def __init__(self, num_features: int, num_classes: int, width: int = 128, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(1, width)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(width, width, kernel_size=k, padding=k // 2, groups=1),
                nn.GELU(),
                nn.BatchNorm1d(width),
                nn.Conv1d(width, width, kernel_size=3, padding=1),
                nn.GELU(),
                nn.BatchNorm1d(width),
            )
            for k in [3, 9, 25]
        ])
        self.fuse = nn.Sequential(
            nn.Conv1d(width * 3, width, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(width),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, num_classes),
        )

    def forward(self, x):
        # x: [B, P]
        h = x.unsqueeze(-1)
        h = self.input_proj(h)      # [B, P, C]
        h = h.transpose(1, 2)       # [B, C, P]
        ys = [b(h) for b in self.branches]
        y = self.fuse(torch.cat(ys, dim=1))
        logits = self.head(y)
        return logits, {}


# ============================================================
# 3) Original-style multi-scale BiMamba
# ============================================================

class ScaleTokenizer(nn.Module):
    def __init__(self, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, d_model)

    def forward(self, x):
        B, P = x.shape
        pad = (self.patch_size - (P % self.patch_size)) % self.patch_size
        if pad > 0:
            x = F.pad(x, (0, pad), value=0.0)
        L = x.shape[1] // self.patch_size
        x = x.reshape(B, L, self.patch_size)
        return self.proj(x)


class MultiScaleEncoder(nn.Module):
    def __init__(self, patch_size: int, d_model: int, n_layers: int = 2, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, dropout: float = 0.2):
        super().__init__()
        self.tokenizer = ScaleTokenizer(patch_size, d_model)
        self.local_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
        )
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.pool = AttnPool(d_model)

    def forward(self, x):
        t = self.tokenizer(x)        # [B, L, D]
        t = self.local_conv(t.transpose(1, 2)).transpose(1, 2)
        for blk in self.layers:
            t = blk(t)
        pooled, weights = self.pool(t)
        return pooled, t, weights


class MultiScaleBiMambaClassifier(nn.Module):
    def __init__(self, num_features: int, num_classes: int, d_model: int = 128, patch_scales=(1, 5, 25),
                 n_layers: int = 2, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.2):
        super().__init__()
        self.patch_scales = list(patch_scales)
        self.scale_encoders = nn.ModuleList([
            MultiScaleEncoder(s, d_model, n_layers=n_layers, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
            for s in self.patch_scales
        ])
        self.scale_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(self, x):
        pooled_scales, token_maps, attn_maps = [], [], []
        for enc in self.scale_encoders:
            pooled, tok, attn = enc(x)
            pooled_scales.append(pooled)
            token_maps.append(tok)
            attn_maps.append(attn)
        scale_stack = torch.stack(pooled_scales, dim=1)
        gate = torch.softmax(self.scale_gate(scale_stack).squeeze(-1), dim=1)
        fused = torch.sum(scale_stack * gate.unsqueeze(-1), dim=1)
        logits = self.head(fused)
        aux = {"scale_weights": gate, "token_maps": token_maps, "attn_maps": attn_maps}
        return logits, aux


# ============================================================
# 4) Improved hybrid multi-scale Mamba for methylation
# ============================================================

class LocalStem1D(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.dw_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2, groups=d_model),
                nn.Conv1d(d_model, d_model, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm1d(d_model),
            )
            for k in [3, 7, 15]
        ])
        self.fuse = nn.Sequential(
            nn.Conv1d(d_model * 3, d_model, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, P]
        h = self.input_proj(x.unsqueeze(-1))   # [B, P, D]
        h = h.transpose(1, 2)                  # [B, D, P]
        ys = [b(h) for b in self.dw_branches]
        y = self.fuse(torch.cat(ys, dim=1)).transpose(1, 2)  # [B, P, D]
        return self.drop(y)


class HybridScaleEncoder(nn.Module):
    def __init__(self, patch_size: int, d_model: int = 128, n_layers: int = 2,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.2):
        super().__init__()
        self.patch_size = patch_size
        self.patch_proj = nn.Linear(d_model * patch_size, d_model)
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.pool = AttnPool(d_model)

    def forward(self, x_tok):
        # x_tok: [B, P, D]
        B, P, D = x_tok.shape
        pad = (self.patch_size - (P % self.patch_size)) % self.patch_size
        if pad > 0:
            x_tok = F.pad(x_tok.transpose(1, 2), (0, pad), value=0.0).transpose(1, 2)

        L = x_tok.shape[1] // self.patch_size
        t = x_tok.reshape(B, L, self.patch_size * D)
        # t = x_tok.contiguous().reshape(B, L, self.patch_size * D)

        t = self.patch_proj(t)
        for blk in self.layers:
            t = blk(t)
        pooled, weights = self.pool(t)
        return pooled, t, weights


class OrdinalHead(nn.Module):
    """
    CORAL-like ordinal head for ordered classes.
    For K classes, outputs K-1 ordered logits.
    For 3 classes, outputs 2 logits.
    """
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(in_dim, 1)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x):
        base = self.fc(x)  # [B,1]
        logits = base + self.bias.view(1, -1)
        return logits


class HybridMultiScaleMambaClassifier(nn.Module):
    """
    Improved architecture for methylation classification.
    Why this is more suitable:
    - local stem handles short-range feature interactions explicitly
    - multi-scale encoders model coarse groups without assuming only one good patch size
    - bidirectional Mamba keeps long-range dependency modeling
    - scale gate lets the model choose useful scales per sample
    - auxiliary ordinal head helps HC -> MCI -> AD ordering
    """
    def __init__(self, num_features: int, num_classes: int, d_model: int = 128,
                 patch_scales=(4, 16, 64), n_layers: int = 2, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, dropout: float = 0.25):
        super().__init__()
        self.local_stem = LocalStem1D(d_model=d_model, dropout=dropout)
        self.patch_scales = list(patch_scales)
        self.scale_encoders = nn.ModuleList([
            HybridScaleEncoder(
                patch_size=s,
                d_model=d_model,
                n_layers=n_layers,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for s in self.patch_scales
        ])
        self.scale_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        self.feature_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            GLUBlock(d_model, d_model, dropout=dropout),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.ordinal_head = OrdinalHead(d_model, num_classes)

    def forward(self, x):
        # x: [B, P]
        x_tok = self.local_stem(x)  # [B, P, D]
        pooled_scales, token_maps, attn_maps = [], [], []
        for enc in self.scale_encoders:
            pooled, tok, attn = enc(x_tok)
            pooled_scales.append(pooled)
            token_maps.append(tok)
            attn_maps.append(attn)
        scale_stack = torch.stack(pooled_scales, dim=1)   # [B, S, D]
        scale_w = torch.softmax(self.scale_gate(scale_stack).squeeze(-1), dim=1)
        fused = torch.sum(scale_stack * scale_w.unsqueeze(-1), dim=1)
        fused = fused * self.feature_gate(fused)
        logits = self.cls_head(fused)
        ordinal_logits = self.ordinal_head(fused)
        aux = {
            "scale_weights": scale_w,
            "token_maps": token_maps,
            "attn_maps": attn_maps,
            "ordinal_logits": ordinal_logits,
        }
        return logits, aux

def needs_scaling(model_name: str) -> bool:
    return model_name in ["elastic_net_lr", "svm_rbf"]




# ============================================================
# TabR-lite / ExcelFormer-lite self-contained baselines
# ============================================================

class NumericPatchTokenizer(nn.Module):
    """Numerical feature tokenizer for high-dimensional DNAm matrices.

    The original TabR / ExcelFormer implementations are not simple drop-in
    sklearn estimators for this training script. This tokenizer keeps the
    comparison practical by grouping thousands of CpGs into a manageable number
    of feature tokens before applying attention/retrieval blocks.
    """
    def __init__(self, num_features: int, d_model: int, max_tokens: int = 128, dropout: float = 0.1):
        super().__init__()
        self.num_features = int(num_features)
        self.max_tokens = max(1, int(max_tokens))
        self.patch_size = max(1, int((self.num_features + self.max_tokens - 1) // self.max_tokens))
        self.num_tokens = int((self.num_features + self.patch_size - 1) // self.patch_size)
        self.padded_features = self.num_tokens * self.patch_size
        self.proj = nn.Linear(self.patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pos = nn.Parameter(torch.zeros(1, self.num_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, f = x.shape
        if f != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {f}")
        pad = self.padded_features - f
        if pad > 0:
            x = F.pad(x, (0, pad), value=0.0)
        x = x.reshape(b, self.num_tokens, self.patch_size)
        x = self.proj(x) + self.pos
        return self.dropout(self.norm(x))


class TabRClassifier(nn.Module):
    """A compact retrieval-augmented tabular baseline compatible with this framework.

    It follows the key TabR idea—augmenting each representation with a
    k-nearest-neighbor-like context—while using the current mini-batch as the
    candidate set. This keeps the model self-contained and compatible with the
    existing PyTorch loop. For strict reproduction of the original TabR paper,
    use the official yandex-research/tabular-dl-tabr repository.
    """
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 128,
        depth: int = 3,
        num_heads: int = 4,
        k_neighbors: int = 8,
        dropout: float = 0.2,
        max_tokens: int = 128,
    ):
        super().__init__()
        self.k_neighbors = max(1, int(k_neighbors))
        self.tokenizer = NumericPatchTokenizer(num_features, d_model, max_tokens=max_tokens, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=max(1, int(num_heads)),
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=max(1, int(depth)))
        self.pool = AttnPool(d_model)
        self.query = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.key = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.value = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

    def _batch_retrieval(self, z: torch.Tensor) -> torch.Tensor:
        b, d = z.shape
        if b <= 1:
            return torch.zeros_like(z)
        q = F.normalize(self.query(z), dim=-1)
        k = F.normalize(self.key(z), dim=-1)
        v = self.value(z)
        sim = q @ k.T
        sim = sim.masked_fill(torch.eye(b, dtype=torch.bool, device=z.device), -1e4)
        kk = min(self.k_neighbors, b - 1)
        topv, topi = torch.topk(sim, k=kk, dim=1)
        w = torch.softmax(topv, dim=1)
        ctx = torch.sum(v[topi] * w.unsqueeze(-1), dim=1)
        return ctx

    def forward(self, x: torch.Tensor):
        tokens = self.tokenizer(x)
        h = self.encoder(tokens)
        z, attn = self.pool(h)
        ctx = self._batch_retrieval(z)
        fused = torch.cat([z, ctx, z - ctx, z * ctx], dim=-1)
        logits = self.head(fused)
        return logits, {"attn_maps": [attn]}


class ExcelFormerBlock(nn.Module):
    """ExcelFormer-inspired semi-permeable attention block for numerical tokens."""
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=max(1, int(num_heads)),
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.drop2 = nn.Dropout(dropout)

    @staticmethod
    def _semi_permeable_mask(length: int, device: torch.device) -> torch.Tensor:
        # Upper-triangular -inf mask: token i can receive information from itself
        # and earlier tokens, but later tokens are partially blocked.
        mask = torch.full((length, length), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        mask = self._semi_permeable_mask(h.shape[1], h.device)
        y, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop1(y)
        h = self.norm2(x)
        y = self.ffn(h) * self.gate(h)
        return x + self.drop2(y)


class ExcelFormerClassifier(nn.Module):
    """Self-contained ExcelFormer-inspired baseline for already-numerical DNAm tables.

    It uses patch-tokenized numerical columns, semi-permeable masked attention,
    attentive gated feed-forward layers, and attention pooling. The official
    torch-frame ExcelFormer requires TensorFrame metadata and column statistics;
    this class is a practical drop-in approximation for the current CSV/NumPy
    pipeline.
    """
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 128,
        depth: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        max_tokens: int = 128,
    ):
        super().__init__()
        self.tokenizer = NumericPatchTokenizer(num_features, d_model, max_tokens=max_tokens, dropout=dropout)
        self.blocks = nn.ModuleList([
            ExcelFormerBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(max(1, int(depth)))
        ])
        self.pool = AttnPool(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(self, x: torch.Tensor):
        h = self.tokenizer(x)
        for blk in self.blocks:
            h = blk(h)
        pooled, attn = self.pool(h)
        logits = self.head(pooled)
        return logits, {"attn_maps": [attn]}


def _none_if_zero(value):
    """Use 0/-1 as command-line friendly aliases for unlimited tree depth."""
    if value is None or int(value) <= 0:
        return None
    return int(value)


def _lgb_unlimited_depth(value):
    """LightGBM uses -1 rather than None for unlimited depth."""
    if value is None or int(value) <= 0:
        return -1
    return int(value)

# ============================================================
# Factory
# ============================================================


def build_model(
    model_name: str,
    num_features: int,
    num_classes: int,
    d_model: int = 128,
    dropout: float = 0.2,
    patch_scales=(1, 5, 25),
    n_mamba_layers: int = 2,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    hierarchical_class_indices=None,
    covariate_dim: int = 0,
    use_feature_gate: bool = False,
    use_domain_head: bool = False,
    domain_hidden_dim: int = 128,
    num_domains: int = 2,
    n_transformer_layers: int = 3,
    num_heads: int = 4,
    num_prototypes_per_class: int = 4,
    tabm_num_members: int = 4,
    grl_lambda: float = 1.0,
    use_linear_residual: bool = True,
    tabr_k_neighbors: int = 8,

    seed: int = 42,
     # elastic net logistic regression
    enet_C: float = 1.0,
    enet_l1_ratio: float = 0.5,
    enet_max_iter: int = 5000,

    # random forest / extra trees
    n_estimators: int = 500,
    max_depth: int = 0,   # 0 means None
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",

    # svm
    svm_C: float = 1.0,
    svm_gamma: str = "scale",

    # xgboost / lightgbm
    gb_learning_rate: float = 0.03,
    gb_max_depth: int = 6,
    gb_num_leaves: int = 31,
    gb_subsample: float = 0.9,
    gb_colsample_bytree: float = 0.8,
    gb_min_child_samples: int = 20,
    gb_reg_lambda: float = 1.0,
    args = None,
):
    name = model_name.lower()

    if name in {"tabpfn", "tabicl"}:
        raise ValueError(
            f"{model_name} is a sklearn-style in-context estimator and is handled by "
            "train_external_test_framework.py, not build_model()."
        )

    if name in {"tabr", "tabr_lite"}:
        return TabRClassifier(
            num_features=num_features,
            num_classes=num_classes,
            d_model=d_model,
            depth=max(1, n_transformer_layers),
            num_heads=num_heads,
            k_neighbors=tabr_k_neighbors,
            dropout=dropout,
            max_tokens=128,
        )

    if name in {"excelformer", "excel_former"}:
        return ExcelFormerClassifier(
            num_features=num_features,
            num_classes=num_classes,
            d_model=d_model,
            depth=max(1, n_transformer_layers),
            num_heads=num_heads,
            dropout=dropout,
            max_tokens=128,
        )

    if name in {"ft_transformer", "ft"}:
        if FTTransformerClassifier is None:
            raise ImportError("methylation_advanced_tabular_models.py is required for ft_transformer")
        idx = hierarchical_class_indices or {}
        return FTTransformerClassifier(
            num_features=num_features,
            num_classes=num_classes,
            d_model=d_model,
            num_layers=n_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            idx_ad=idx.get("idx_ad"),
            idx_mci=idx.get("idx_mci"),
            idx_hc=idx.get("idx_hc"),
            use_linear_residual=use_linear_residual,
        )

    if name in {"modernnca", "modern_nca", "nca"}:
        if ModernNCAClassifier is None:
            raise ImportError("methylation_advanced_tabular_models.py is required for modernnca")
        idx = hierarchical_class_indices or {}
        return ModernNCAClassifier(
            num_features=num_features,
            num_classes=num_classes,
            d_model=d_model,
            depth=max(2, n_transformer_layers),
            dropout=dropout,
            num_prototypes_per_class=num_prototypes_per_class,
            idx_ad=idx.get("idx_ad"),
            idx_mci=idx.get("idx_mci"),
            idx_hc=idx.get("idx_hc"),
            use_linear_residual=use_linear_residual,
        )

    if name in {"tabm", "tabular_mixup"}:
        if TabMClassifier is None:
            raise ImportError("methylation_advanced_tabular_models.py is required for tabm")
        idx = hierarchical_class_indices or {}
        return TabMClassifier(
            num_features=num_features,
            num_classes=num_classes,
            d_model=d_model,
            depth=max(2, n_transformer_layers),
            dropout=dropout,
            num_members=tabm_num_members,
            idx_ad=idx.get("idx_ad"),
            idx_mci=idx.get("idx_mci"),
            idx_hc=idx.get("idx_hc"),
            use_linear_residual=use_linear_residual,
        )

    if name in {"dann_ft", "dann_ft_transformer", "dann+ft"}:
        if DANNFTTransformerClassifier is None:
            raise ImportError("methylation_advanced_tabular_models.py is required for dann_ft")
        idx = hierarchical_class_indices or {}
        return DANNFTTransformerClassifier(
            num_features=num_features,
            num_classes=num_classes,
            d_model=d_model,
            num_layers=n_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_domains=num_domains,
            grl_lambda=grl_lambda,
            idx_ad=idx.get("idx_ad"),
            idx_mci=idx.get("idx_mci"),
            idx_hc=idx.get("idx_hc"),
            use_linear_residual=use_linear_residual,
        )

    if name in {"dann_ft_v1", "dann_ft_transformer_v1", "dann+ft_v1"}:
        if DANNFTTransformerClassifier_v1 is None:
            raise ImportError("methylation_advanced_tabular_models.py is required for dann_ft")
        idx = hierarchical_class_indices or {}
        return DANNFTTransformerClassifier_v1(
            num_features=num_features,
            num_classes=num_classes,
            d_model=d_model,
            num_layers=n_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_domains=num_domains,
            grl_lambda=grl_lambda,
            idx_ad=idx.get("idx_ad"),
            idx_mci=idx.get("idx_mci"),
            idx_hc=idx.get("idx_hc"),
            use_linear_residual=use_linear_residual,
        )

    if name in {'methy_ft_transformer', 'methylation_ft_transformer'}:
        if MethylationFTTransformerClassifier is None:
            raise ImportError("methylation_advanced_tabular_models.py is required for methy_ft_transformer")
        idx = hierarchical_class_indices or {}
        return MethylationFTTransformerClassifier(
            num_features=num_features,
            num_classes=num_classes,
            d_model=d_model,
            num_layers=n_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            idx_ad=idx.get("idx_ad"),
            idx_mci=idx.get("idx_mci"),
            idx_hc=idx.get("idx_hc"),
            use_linear_residual=use_linear_residual,
            cpg_annotation_ids = None,
            annotation_dropout = 0.0,
        )

    if name in {"improved_ft_transformer", "improv_ft_transformer"}:
        if ImprovedFTTransformerClassifier is None:
            raise ImportError("methylation_advanced_tabular_models.py is required for improved_ft_transformer")
        idx = hierarchical_class_indices or {}
        return ImprovedFTTransformerClassifier(
            num_features=num_features,
            num_classes=num_classes,
            d_model=d_model,
            num_layers=n_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            idx_ad=idx.get("idx_ad"),
            idx_mci=idx.get("idx_mci"),
            idx_hc=idx.get("idx_hc"),
            use_linear_residual=use_linear_residual,
        )


    if name == "mlp":
        return ResidualMLPClassifier(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=max(256, d_model * 4),
            depth=4,
            dropout=dropout,
        )

    if name in {"stable_mlp", "stable_methyl_res_mlp", "stable_methylation_res_mlp"}:
        return StableMethylationResMLP(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=max(256, d_model * 4),
            depth=4,
            dropout=dropout,
            expansion=expand,
            use_linear_residual=use_linear_residual,
        )

    if name == "cnn":
        return MultiScaleCNNClassifier(
            num_features=num_features,
            num_classes=num_classes,
            width=d_model,
            dropout=dropout,
        )

    if name == "ms_mamba":
        return MultiScaleBiMambaClassifier(
            num_features=num_features,
            num_classes=num_classes,
            d_model=d_model,
            patch_scales=tuple(int(x) for x in patch_scales),
            n_layers=n_mamba_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

    if name == "hybrid_ms_mamba":
        # better default scales for 5k+ methylation features
        return HybridMultiScaleMambaClassifier(
            num_features=num_features,
            num_classes=num_classes,
            d_model=d_model,
            patch_scales=tuple(int(x) for x in patch_scales),
            n_layers=n_mamba_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
    
    if name == "auc_boost_tabular":
        if AUCBoostTabularNet is None:
            raise ImportError("methylation_auc_boost_model.py is required for auc_boost_tabular")
        return AUCBoostTabularNet(
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=max(384, d_model * 4),
            token_dim=max(48, d_model // 2),
            num_tokens=32,
            num_token_layers=3,
            num_heads=4,
            dropout=dropout,
        )
    
    if name == "subspace_ensemble":
        if SubspaceEnsembleTabularNet is None:
            raise ImportError("methylation_subspace_ensemble_model.py is required for subspace_ensemble")
        return SubspaceEnsembleTabularNet(
            num_features=num_features,
            num_classes=num_classes,
            num_experts=8,
            subspace_ratio=0.35,
            hidden_dim=max(256, d_model * 2),
            emb_dim=d_model,
            expert_depth=2,
            router_hidden=max(128, d_model * 2),
            dropout=dropout,
            seed=42,
        )
    
    if name == "hfr_net":
        if hierarchical_class_indices is None:
            raise ValueError("hierarchical_class_indices must be provided for hfr_net")

        if HFRNet is None:
            raise ImportError("methylation_hfr_model.py is required for hfr_net")
        return HFRNet(
            num_features=num_features,
            num_classes=num_classes,
            idx_ad=hierarchical_class_indices["idx_ad"],
            idx_mci=hierarchical_class_indices["idx_mci"],
            idx_hc=hierarchical_class_indices["idx_hc"],
            hidden_dim=max(256, d_model * 4),
            depth=4,
            dropout=dropout,
        )
    
    if name == "rank_hfr_mlp":
        if hierarchical_class_indices is None:
            raise ValueError("hierarchical_class_indices must be provided for rank_hfr_mlp")

        if RankHFRMLP is None:
            raise ImportError("methylation_rank_hfr_mlp.py is required for rank_hfr_mlp")
        return RankHFRMLP(
            num_features=num_features,
            num_classes=num_classes,
            idx_ad=hierarchical_class_indices["idx_ad"],
            idx_mci=hierarchical_class_indices["idx_mci"],
            idx_hc=hierarchical_class_indices["idx_hc"],
            hidden_dim=max(256, d_model * 4),
            depth=4,
            dropout=dropout,
        )

    if name == "sparse_rank_hfr_mlp":
        if hierarchical_class_indices is None:
            raise ValueError("hierarchical_class_indices must be provided for sparse_rank_hfr_mlp")

        if SparseRankHFRMLP is None:
            raise ImportError("methylation_sparse_rank_hfr_mlp.py is required for sparse_rank_hfr_mlp")
        return SparseRankHFRMLP(
            num_features=num_features,
            num_classes=num_classes,
            idx_ad=hierarchical_class_indices["idx_ad"],
            idx_mci=hierarchical_class_indices["idx_mci"],
            idx_hc=hierarchical_class_indices["idx_hc"],
            hidden_dim=max(256, d_model * 4),
            depth=4,
            dropout=dropout,
            selector_bias=False,
        )

    if name == "linear_residual_rank_hfr_mlp":
        if hierarchical_class_indices is None:
            raise ValueError("hierarchical_class_indices must be provided for linear_residual_rank_hfr_mlp")

        if LinearResidualRankHFRMLP is None:
            raise ImportError("methylation_linear_residual_rank_hfr_mlp.py is required for linear_residual_rank_hfr_mlp")
        return LinearResidualRankHFRMLP(
            num_features=num_features,
            num_classes=num_classes,
            idx_ad=hierarchical_class_indices["idx_ad"],
            idx_mci=hierarchical_class_indices["idx_mci"],
            idx_hc=hierarchical_class_indices["idx_hc"],
            hidden_dim=max(256, d_model * 4),
            depth=4,
            dropout=dropout,
        )
    
    if name == "cross_cohort_linear_residual_rank_hfr_mlp":
        if hierarchical_class_indices is None:
            raise ValueError("hierarchical_class_indices must be provided for CrossCohortLinearResidualRankHFRMLP")

        if CrossCohortLinearResidualRankHFRMLP is None:
            raise ImportError("CrossCohortLinearResidualRankHFRMLP.py is required for cross_cohort_linear_residual_rank_hfr_mlp")
        return CrossCohortLinearResidualRankHFRMLP(
            num_features=num_features,
            num_classes=num_classes,
            idx_ad=hierarchical_class_indices["idx_ad"],
            idx_mci=hierarchical_class_indices["idx_mci"],
            idx_hc=hierarchical_class_indices["idx_hc"],
            hidden_dim=max(128, d_model * 2),   # 比原始 full MLP 更稳
            depth=3,
            dropout=dropout,
            covariate_dim=covariate_dim,
            use_feature_gate=use_feature_gate,
            use_domain_head=use_domain_head,
            domain_hidden_dim=domain_hidden_dim,
            num_domains=num_domains,
        )
    
    if name == "cross_cohort_lr_gate_hfr_mlp":
        if hierarchical_class_indices is None:
            raise ValueError("hierarchical_class_indices must be provided for CrossCohortLRGateHFRMLP")

        if CrossCohortLRGateHFRMLP is None:
            raise ImportError("methylation_cross_cohort_lr_gate_hfr_mlp.py is required for build_cross_cohort_lr_gate_hfr_mlp")
        return CrossCohortLRGateHFRMLP(
            num_features=num_features,
            num_classes=num_classes,
            idx_ad=hierarchical_class_indices["idx_ad"],
            idx_mci=hierarchical_class_indices["idx_mci"],
            idx_hc=hierarchical_class_indices["idx_hc"],
            dropout=dropout,
            covariate_dim=covariate_dim,
            use_feature_gate=args.use_feature_gate,
            use_domain_head=args.use_domain_head,
            domain_hidden_dim=args.domain_hidden_dim,
            num_domains=num_domains,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            gate_rank=args.gate_rank,
            gate_rho=args.gate_rho,
            gate_bias_scale=args.gate_bias_scale,
            use_token_pool=bool(args.use_token_pool),
            linear_alpha_init=args.linear_alpha_init,
        )
        
    if name == "cross_cohort_dg_linear_residual_rank_hfr_mlp":
        if hierarchical_class_indices is None:
            raise ValueError("hierarchical_class_indices must be provided")

        if CrossCohortDGLinearResidualRankHFRMLP is None:
            raise ImportError("methylation_cross_cohort_dg_lr_rank_hfr_mlp.py is required for cross_cohort_dg_linear_residual_rank_hfr_mlp")
        return CrossCohortDGLinearResidualRankHFRMLP(
            num_features=num_features,
            num_classes=num_classes,
            idx_ad=hierarchical_class_indices["idx_ad"],
            idx_mci=hierarchical_class_indices["idx_mci"],
            idx_hc=hierarchical_class_indices["idx_hc"],
            hidden_dim=max(128, d_model * 2),
            depth=3,
            dropout=dropout,
            covariate_dim=covariate_dim,
            use_feature_gate=False,
            use_domain_head=True,
            domain_hidden_dim=128,
            num_domains=num_domains,
            invariant_dim=128,
            private_dim=64,
            use_private_branch=True,
        )
        
    if name == "elastic_net_lr":
        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=enet_C,
            l1_ratio=enet_l1_ratio,
            max_iter=enet_max_iter,
            class_weight=None,
            random_state=seed,
            n_jobs=-1,
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=_none_if_zero(max_depth),
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=None,
            random_state=seed,
            n_jobs=-1,
        )

    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=_none_if_zero(max_depth),
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=None,
            random_state=seed,
            n_jobs=-1,
        )

    if name == "svm_rbf":
        clf = SVC(
            kernel="rbf",
            C=svm_C,
            gamma=svm_gamma,
            probability=True,
            class_weight=None,
            random_state=seed,
        )
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

    if name == "xgboost":
        xgb_mod = _require_xgboost()

        common_params = dict(
            n_estimators=n_estimators,
            learning_rate=gb_learning_rate,
            max_depth=gb_max_depth,
            subsample=gb_subsample,
            colsample_bytree=gb_colsample_bytree,
            reg_lambda=gb_reg_lambda,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        )
        if num_classes == 2:
            return xgb_mod.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                **common_params,
            )
        return xgb_mod.XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            **common_params,
        )

    if name == "lightgbm":
        lgb_mod = _require_lightgbm()

        common_params = dict(
            n_estimators=n_estimators,
            learning_rate=gb_learning_rate,
            max_depth=_lgb_unlimited_depth(gb_max_depth),
            num_leaves=gb_num_leaves,
            subsample=gb_subsample,
            colsample_bytree=gb_colsample_bytree,
            reg_lambda=gb_reg_lambda,
            min_child_samples=gb_min_child_samples,
            random_state=seed,
            n_jobs=-1,
        )
        if num_classes == 2:
            return lgb_mod.LGBMClassifier(
                objective="binary",
                **common_params,
            )
        return lgb_mod.LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            **common_params,
        )

    raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Use one of: mlp, cnn, ms_mamba, hybrid_ms_mamba, auc_boost_tabular, \
                subspace_ensemble, ft_transformer, modernnca, tabm, dann_ft, tabpfn, tabicl, tabr, excelformer, \
                hfr_net, rank_hfr_mlp, sparse_rank_hfr_mlp, \
                    linear_residual_rank_hfr_mlp, cross_cohort_linear_residual_rank_hfr_mlp, \
                        cross_cohort_dg_linear_residual_rank_hfr_mlp, cross_cohort_hodg_joint_lr_rank_hfr_mlp,\
                        elastic_net_lr, random_forest, extra_trees, svm_rbf, xgboost, lightgbm, cross_cohort_lr_gate_hfr_mlp"
        )
