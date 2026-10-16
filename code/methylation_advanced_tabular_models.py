#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced plug-and-play tabular models for DNA methylation classification.

This module follows the same interface as the existing project:
    logits, aux = model(x)

For canonical three-class HC/MCI/AD tasks, the models can expose the same
hierarchical auxiliary prediction logic used by LinearResidualRankHFRMLP:
    aux["hc_vs_disease_logits"]
    aux["ad_vs_mci_logits"]
    aux["ovr_logits"]
    aux["linear_l1"], aux["linear_l2"] when a linear residual branch is used

Models included
---------------
1) FTTransformerClassifier
2) ModernNCAClassifier
3) TabMClassifier with optional training-time mixup handled by the trainer
4) DANNFTTransformerClassifier

TabPFN is intentionally handled in the training script because it is a sklearn-like
non-PyTorch estimator rather than a torch.nn.Module.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Shared blocks
# ============================================================

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class OVRHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class LinearResidualHead(nn.Module):
    """Elastic-net-style linear residual branch on the standardized input."""
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(num_features, num_classes)
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.fc(x)
        alpha = F.softplus(self.alpha_raw)
        w = self.fc.weight
        linear_l1 = torch.mean(torch.abs(w))
        linear_l2 = torch.mean(w ** 2)
        return logits, alpha, linear_l1, linear_l2


class OrdinalHead(nn.Module):
    """CORAL-like ordinal head. For K classes, returns K-1 logits."""
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
        self.bias = nn.Parameter(torch.zeros(max(1, num_classes - 1)))
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_classes <= 1:
            raise ValueError("num_classes must be >=2")
        return self.fc(x) + self.bias.view(1, -1)


class HierarchicalPredictionHead(nn.Module):
    """
    Unified prediction head.

    For all tasks, it returns main logits. For canonical 3-class HC/MCI/AD tasks,
    it also returns HFR-style auxiliary logits so the existing CompositeCriterion
    can apply hierarchical losses without changing the call pattern.
    """
    def __init__(
        self,
        embed_dim: int,
        num_features: int,
        num_classes: int,
        dropout: float = 0.2,
        idx_ad: Optional[int] = None,
        idx_mci: Optional[int] = None,
        idx_hc: Optional[int] = None,
        use_linear_residual: bool = True,
        use_aux3: bool = True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.idx_ad = idx_ad
        self.idx_mci = idx_mci
        self.idx_hc = idx_hc
        self.is_hierarchical = (
            self.num_classes == 3 and idx_ad is not None and idx_mci is not None and idx_hc is not None
        )
        self.use_linear_residual = bool(use_linear_residual)
        self.use_aux3 = bool(use_aux3)

        hidden = max(embed_dim, 64)
        self.main_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self.ordinal_head = OrdinalHead(embed_dim, num_classes)

        if self.use_linear_residual:
            self.linear_head = LinearResidualHead(num_features=num_features, num_classes=num_classes)
        else:
            self.linear_head = None

        if self.is_hierarchical:
            self.hc_vs_disease_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            self.ad_vs_mci_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
            self.ovr_head = OVRHead(embed_dim, num_classes)
            self.aux3_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )

    def forward(self, z: torch.Tensor, x: torch.Tensor, aux: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        if aux is None:
            aux = {}
        main_logits = self.main_head(z)
        logits = main_logits
        aux["shared_embedding"] = z
        aux["mlp_logits"] = main_logits
        aux["ordinal_logits"] = self.ordinal_head(z)

        if self.linear_head is not None:
            linear_logits, alpha, linear_l1, linear_l2 = self.linear_head(x)
            logits = main_logits + alpha * linear_logits
            aux.update({
                "linear_logits": linear_logits,
                "linear_alpha": alpha,
                "linear_l1": linear_l1,
                "linear_l2": linear_l2,
            })

        if self.is_hierarchical:
            aux.update({
                "hc_vs_disease_logits": self.hc_vs_disease_head(z).squeeze(-1),
                "ad_vs_mci_logits": self.ad_vs_mci_head(z).squeeze(-1),
                "ovr_logits": self.ovr_head(z),
                "idx_ad": int(self.idx_ad),
                "idx_mci": int(self.idx_mci),
                "idx_hc": int(self.idx_hc),
            })
            if self.use_aux3:
                aux["aux3_logits"] = self.aux3_head(z)
        return logits, aux


# ============================================================
# FT-Transformer
# ============================================================

class FeatureTokenizer(nn.Module):
    """
    Numeric feature tokenizer used by FT-Transformer.

    For input x in [B, F], each feature f has a trainable vector W_f and bias b_f:
        token_f = x_f * W_f + b_f
    A learnable [CLS] token is prepended.
    """
    def __init__(self, num_features: int, token_dim: int, dropout: float = 0.0):
        super().__init__()
        self.num_features = int(num_features)
        self.token_dim = int(token_dim)
        self.weight = nn.Parameter(torch.empty(num_features, token_dim))
        self.bias = nn.Parameter(torch.empty(num_features, token_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        return self.dropout(torch.cat([cls, tokens], dim=1))


class MethylationFeatureTokenizer(nn.Module):
    """
    Methylation-aware feature tokenizer for CpG-level FT-Transformer.

    For each CpG feature f and sample methylation value x_f, the token is:
        token_f = x_f * W_f
                  + b_f
                  + E_cpg_id[f]
                  + E_chromosome[chr_f]
                  + E_genomic_region[region_f]
                  + E_gene_annotation[gene_f]
                  + E_cpg_island[island_f]

    Notes
    -----
    - cpg_id embedding represents feature identity after training-feature-order alignment.
    - Annotation IDs must be aligned to the exact order of the input matrix columns.
    - ID 0 is reserved for missing / unknown annotation values.
    """
    def __init__(
        self,
        num_features: int,
        token_dim: int,
        cpg_annotation_ids: Optional[Dict[str, torch.Tensor]] = None,
        dropout: float = 0.0,
        use_feature_bias: bool = True,
        annotation_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.token_dim = int(token_dim)
        self.use_feature_bias = bool(use_feature_bias)

        self.value_weight = nn.Parameter(torch.empty(self.num_features, self.token_dim))
        if self.use_feature_bias:
            self.feature_bias = nn.Parameter(torch.empty(self.num_features, self.token_dim))
        else:
            self.register_parameter("feature_bias", None)

        ids = self._prepare_annotation_ids(cpg_annotation_ids)
        for name, value in ids.items():
            self.register_buffer(name, value.long(), persistent=True)

        self.cpg_id_embedding = nn.Embedding(int(self.cpg_ids.max().item()) + 1, self.token_dim, padding_idx=0)
        self.chromosome_embedding = nn.Embedding(int(self.chromosome_ids.max().item()) + 1, self.token_dim, padding_idx=0)
        self.genomic_region_embedding = nn.Embedding(int(self.genomic_region_ids.max().item()) + 1, self.token_dim, padding_idx=0)
        self.gene_annotation_embedding = nn.Embedding(int(self.gene_annotation_ids.max().item()) + 1, self.token_dim, padding_idx=0)
        self.cpg_island_embedding = nn.Embedding(int(self.cpg_island_ids.max().item()) + 1, self.token_dim, padding_idx=0)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        self.annotation_scale = nn.Parameter(torch.tensor(1.0))
        self.token_norm = nn.LayerNorm(self.token_dim)
        self.annotation_dropout = nn.Dropout(annotation_dropout)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def _prepare_annotation_ids(self, cpg_annotation_ids: Optional[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if cpg_annotation_ids is None:
            cpg_annotation_ids = {}

        def _as_tensor(name: str, default):
            value = cpg_annotation_ids.get(name, default)
            value = torch.as_tensor(value, dtype=torch.long)
            if value.numel() != self.num_features:
                raise ValueError(
                    f"cpg_annotation_ids['{name}'] must have length {self.num_features}, got {value.numel()}"
                )
            value = value.view(self.num_features).clamp_min(0)
            return value

        return {
            "cpg_ids": _as_tensor("cpg_ids", torch.arange(1, self.num_features + 1, dtype=torch.long)),
            "chromosome_ids": _as_tensor("chromosome_ids", torch.zeros(self.num_features, dtype=torch.long)),
            "genomic_region_ids": _as_tensor("genomic_region_ids", torch.zeros(self.num_features, dtype=torch.long)),
            "gene_annotation_ids": _as_tensor("gene_annotation_ids", torch.zeros(self.num_features, dtype=torch.long)),
            "cpg_island_ids": _as_tensor("cpg_island_ids", torch.zeros(self.num_features, dtype=torch.long)),
        }

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.value_weight)
        if self.feature_bias is not None:
            nn.init.zeros_(self.feature_bias)
        nn.init.normal_(self.cls_token, std=0.02)
        for emb in [
            self.cpg_id_embedding,
            self.chromosome_embedding,
            self.genomic_region_embedding,
            self.gene_annotation_embedding,
            self.cpg_island_embedding,
        ]:
            nn.init.normal_(emb.weight, std=0.02)
            if emb.padding_idx is not None:
                with torch.no_grad():
                    emb.weight[emb.padding_idx].zero_()

    def annotation_tokens(self) -> torch.Tensor:
        ann = (
            self.cpg_id_embedding(self.cpg_ids)
            + self.chromosome_embedding(self.chromosome_ids)
            + self.genomic_region_embedding(self.genomic_region_ids)
            # + self.gene_annotation_embedding(self.gene_annotation_ids)
            # + self.cpg_island_embedding(self.cpg_island_ids)
        )
        return self.annotation_dropout(ann)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]
        value_tokens = x.unsqueeze(-1) * self.value_weight.unsqueeze(0)
        if self.feature_bias is not None:
            value_tokens = value_tokens + self.feature_bias.unsqueeze(0)
        tokens = value_tokens + self.annotation_scale * self.annotation_tokens().unsqueeze(0)
        tokens = self.token_norm(tokens)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        return self.dropout(torch.cat([cls, tokens], dim=1))


class MethylationFTTransformerEncoder(nn.Module):
    """FT-Transformer encoder using methylation-aware CpG annotation tokenizer."""
    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        cpg_annotation_ids: Optional[Dict[str, torch.Tensor]] = None,
        annotation_dropout: float = 0.0,
    ):
        super().__init__()
        self.tokenizer = MethylationFeatureTokenizer(
            num_features=num_features,
            token_dim=d_model,
            cpg_annotation_ids=cpg_annotation_ids,
            dropout=dropout,
            annotation_dropout=annotation_dropout,
        )
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        tokens = self.tokenizer(x)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)
        z = tokens[:, 0]
        aux = {
            "tokens": tokens,
            "cpg_tokens": tokens[:, 1:],
            "annotation_scale": self.tokenizer.annotation_scale.detach(),
        }
        return z, aux


class MethylationFTTransformerClassifier(nn.Module):
    """Disease classifier with methylation-aware CpG tokenizer."""
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        cpg_annotation_ids: Optional[Dict[str, torch.Tensor]] = None,
        annotation_dropout: float = 0.0,
        idx_ad: Optional[int] = None,
        idx_mci: Optional[int] = None,
        idx_hc: Optional[int] = None,
        use_linear_residual: bool = True,
    ):
        super().__init__()
        self.encoder = MethylationFTTransformerEncoder(
            num_features=num_features,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            cpg_annotation_ids=cpg_annotation_ids,
            annotation_dropout=annotation_dropout,
        )
        self.pred_head = HierarchicalPredictionHead(
            embed_dim=d_model,
            num_features=num_features,
            num_classes=num_classes,
            dropout=dropout,
            idx_ad=idx_ad,
            idx_mci=idx_mci,
            idx_hc=idx_hc,
            use_linear_residual=use_linear_residual,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        z, aux = self.encoder(x)
        return self.pred_head(z, x, aux)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.2, ffn_multiplier: int = 4):
        super().__init__()
        if d_model % num_heads != 0:
            # Use a safe one-head fallback rather than crashing for unusual d_model values.
            num_heads = 1
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_multiplier, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        y, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(y)
        x = x + self.ffn(self.norm2(x))
        return x


class FTTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features=num_features, token_dim=d_model, dropout=dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        tokens = self.tokenizer(x)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)
        z = tokens[:, 0]
        aux = {"tokens": tokens}
        return z, aux


def _init_weights(module: nn.Module):
    """统一权重初始化，与项目现有约定一致。"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _build_ffn(
    d_model: int,
    ffn_multiplier: int = 4,
    dropout: float = 0.2,
    activation: nn.Module = nn.GELU(),
) -> nn.Sequential:
    """标准膨胀-压缩 FFN，与 TransformerEncoderBlock 一致。"""
    return nn.Sequential(
        nn.Linear(d_model, d_model * ffn_multiplier),
        activation,
        nn.Dropout(dropout),
        nn.Linear(d_model * ffn_multiplier, d_model),
        nn.Dropout(dropout),
    )


# ============================================================================
# 改进 ①: Gated Feature Selector
# ============================================================================

class GatedFeatureSelector(nn.Module):
    """
    Token-wise soft feature gating applied BEFORE Transformer blocks.

    For each feature token t_f ∈ R^d:
        s_f = σ(W₂ · GELU(W₁ · t_f))      # scalar gate per feature token
        t_f' = s_f · t_f

    Input:
        tokens: [B, F+1, d], where position 0 is CLS.

    Output:
        gated tokens: [B, F+1, d]
    """

    def __init__(self, num_features: int, d_model: int, reduction: int = 16):
        super().__init__()
        self.num_features = int(num_features)
        self.d_model = int(d_model)

        inner = max(d_model // reduction, 8)

        # Apply Linear on the last dimension d_model for each feature token.
        self.squeeze = nn.Linear(d_model, inner)
        self.excite = nn.Linear(inner, 1)

        self.norm = nn.LayerNorm(d_model)
        self.apply(_init_weights)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, F+1, d]

        Returns:
            gated: [B, F+1, d]
        """
        if tokens.dim() != 3:
            raise ValueError(
                f"GatedFeatureSelector expects tokens with shape [B, F+1, d], "
                f"but got {tuple(tokens.shape)}"
            )

        cls = tokens[:, :1, :]        # [B, 1, d]
        feats = tokens[:, 1:, :]      # [B, F, d]

        if feats.size(1) != self.num_features:
            raise ValueError(
                f"Feature token number mismatch: expected {self.num_features}, "
                f"but got {feats.size(1)}. "
                f"Check whether num_features matches the input CpG dimension."
            )

        if feats.size(-1) != self.d_model:
            raise ValueError(
                f"Token dimension mismatch: expected d_model={self.d_model}, "
                f"but got {feats.size(-1)}."
            )

        h = F.gelu(self.squeeze(feats))          # [B, F, inner]
        gates = torch.sigmoid(self.excite(h))    # [B, F, 1]

        gated_feats = feats * gates              # [B, F, d]

        out = torch.cat([cls, gated_feats], dim=1)
        out = self.norm(out)

        return out

# ============================================================================
# 改进 ②: Genomic Distance-Biased Attention
# ============================================================================

class GenomicDistanceBias(nn.Module):
    """
    Learnable genomic-distance bias that modulates self-attention scores.

    For CpG i at position p_i and CpG j at position p_j:
        bias_{ij} = -log(1 + |p_i - p_j| / scale) · λ_h

    where λ_h > 0 is learned per attention head.  Closer CpGs get less
    negative bias → higher effective attention.  This injects the biological
    prior that physically proximal CpGs tend to be co-methylated.

    The CLS token (idx 0) receives distance 0 to all features so it
    attends freely without distance penalty.
    """

    def __init__(
        self,
        num_features: int,
        num_heads: int,
        genomic_positions: Optional[torch.Tensor] = None,
        max_distance_bp: float = 250_000_000.0,  # largest human chromosome ~250 Mb
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.num_heads = int(num_heads)

        # Per-head learnable decay strength (λ_h in [0, +∞))
        self.log_lambda = nn.Parameter(torch.zeros(num_heads))  # ln(1) = 0

        # Pre-compute distance matrix once if positions are available.
        if genomic_positions is not None:
            self._set_genomic_positions(genomic_positions, max_distance_bp)
        else:
            self.register_buffer("distances", None, persistent=True)
            self.max_distance_bp = float(max_distance_bp)

    def _set_genomic_positions(
        self,
        positions: torch.Tensor,
        max_distance_bp: float = 250_000_000.0,
    ):
        """Pre-compute a distance matrix that respects CLS placement."""
        positions = positions.view(-1).float()
        if positions.numel() != self.num_features:
            raise ValueError(
                f"genomic_positions length {positions.numel()} != num_features {self.num_features}"
            )

        # Feature-only pairwise distances (F × F), normalized to [0, 1]
        pos_diff = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()
        feat_dist = torch.log1p(pos_diff / max_distance_bp)  # log(1 + |Δ|/scale)

        # Pad for CLS token: CLS at row/col 0, features at 1..F.
        # CLS ↔ feature distance = 0 → no bias.
        # CLS ↔ CLS distance = 0.
        N = self.num_features + 1
        dist = torch.zeros(N, N, dtype=feat_dist.dtype, device=feat_dist.device)
        dist[1:, 1:] = feat_dist
        self.register_buffer("distances", dist, persistent=True)
        self.max_distance_bp = float(max_distance_bp)

    def update_positions(self, positions: torch.Tensor):
        """Recompute distance buffer with new positions (e.g. after feature alignment)."""
        self._set_genomic_positions(positions, self.max_distance_bp)

    def forward(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        Returns:
            bias: [num_heads, N, N] or None if no positions are registered.
                  Can be added to attention scores: scores += bias.
        """
        if self.distances is None:
            return None

        # λ_h = exp(log_lambda) ∈ (0, +∞)
        lam = torch.exp(self.log_lambda).view(self.num_heads, 1, 1)  # [H, 1, 1]
        dist = self.distances.to(device)                               # [N, N]
        bias = -lam * dist.unsqueeze(0)                                # [H, N, N]
        return bias.expand(self.num_heads, -1, -1) if batch_size > 0 else bias


class GenomicDistanceBiasedMHA(nn.Module):
    """
    Multi-head attention that optionally adds a genomic-distance bias.

    When genomic_positions are unavailable, this behaves exactly like
    standard nn.MultiheadAttention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_features: int,
        dropout: float = 0.1,
        genomic_positions: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            num_heads = 1
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.distance_bias = GenomicDistanceBias(
            num_features=num_features,
            num_heads=num_heads,
            genomic_positions=genomic_positions,
        )
        self.apply(_init_weights)

    def forward(
        self,
        x: torch.Tensor,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, N, d]
            need_weights: if True, also return attention weights.

        Returns:
            out: [B, N, d]
            attn_weights: [B, N, N] or None
        """
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, hd]
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, N, N]

        # --- Genomic distance bias (the key addition) ---
        bias = self.distance_bias(B, x.device)
        if bias is not None:
            attn_scores = attn_scores + bias.unsqueeze(0)  # [1, H, N, N] + [B, H, N, N]

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)                     # [B, H, N, hd]
        out = out.transpose(1, 2).contiguous().view(B, N, D)    # [B, N, D]
        out = self.out_proj(out)

        ret_weights = attn_weights.mean(dim=1) if need_weights else None  # average across heads
        return out, ret_weights


# ============================================================================
# 改进 ②+③: Hierarchical Attention Block
# ============================================================================

class HierarchicalAttentionBlock(nn.Module):
    """
    A Transformer block that performs two attention stages:

    Stage 1 — Intra-group Local Attention:
        Within each CpG group (e.g. a CpG island), a shared Transformer
        block with genomic-distance bias processes tokens independently.
        A learned [CLS_group] token is prepended and pooled.

    Stage 2 — Inter-group Global Attention:
        All group [CLS_group] tokens, together with the global [CLS],
        attend to each other via standard self-attention.

    If no groups are provided, falls back to a standard TransformerEncoderBlock
    (global self-attention only).

    This mirrors the biological hierarchy:
        CpG islands (local co-methylation) → cross-island regulatory interactions.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.2,
        ffn_multiplier: int = 4,
        cpg_groups: Optional[List[List[int]]] = None,
        genomic_positions: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)

        # Determine group structure
        if cpg_groups is not None and len(cpg_groups) > 1:
            self.groups = [list(g) for g in cpg_groups if len(g) > 0]
            self.use_hierarchical = len(self.groups) > 1
        else:
            self.groups = []
            self.use_hierarchical = False

        if self.use_hierarchical:
            # --- Stage 1: intra-group local attention ---
            self.local_norm1 = nn.LayerNorm(d_model)
            # Shared MHA for all groups
            self.local_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads,
                dropout=dropout, batch_first=True,
            )
            self.local_drop1 = nn.Dropout(dropout)
            self.local_norm2 = nn.LayerNorm(d_model)
            self.local_ffn = _build_ffn(d_model, ffn_multiplier, dropout)

            # Group summary tokens (one learnable CLS per group)
            self.num_groups = len(self.groups)
            self.group_cls_tokens = nn.Parameter(
                torch.zeros(1, self.num_groups, d_model)
            )
            nn.init.normal_(self.group_cls_tokens, std=0.02)

            # --- Stage 2: inter-group global attention ---
            self.global_block = _StandardTransformerBlock(
                d_model=d_model, num_heads=num_heads,
                dropout=dropout, ffn_multiplier=ffn_multiplier,
            )
        else:
            # --- Fallback: standard global self-attention ---
            self.global_block = _StandardTransformerBlock(
                d_model=d_model, num_heads=num_heads,
                dropout=dropout, ffn_multiplier=ffn_multiplier,
            )

        self.apply(_init_weights)

    def _intra_group_attention(
        self, feat_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process each group independently with local SA + FFN.

        Args:
            feat_tokens: [B, F, d]  (feature tokens only, without global CLS)

        Returns:
            group_summaries: [B, num_groups, d]
            updated_feats:   [B, F, d]
        """
        B, F, D = feat_tokens.shape
        device = feat_tokens.device
        group_summaries = []
        out_feats = feat_tokens.clone()

        for g_idx, group_indices in enumerate(self.groups):
            idx = torch.tensor(group_indices, device=device, dtype=torch.long)
            if idx.numel() <= 1:
                # Single-token group: just use its own token as summary
                summary = feat_tokens[:, idx, :].mean(dim=1, keepdim=True)  # [B, 1, d]
                group_summaries.append(summary)
                continue

            local = feat_tokens[:, idx, :]  # [B, n_g, d]
            g_cls = self.group_cls_tokens[:, g_idx:g_idx + 1, :].expand(B, -1, -1)

            # Prepend group CLS
            local_seq = torch.cat([g_cls, local], dim=1)  # [B, 1+n_g, d]

            # Local attention + FFN (Pre-Norm style)
            h = self.local_norm1(local_seq)
            y, _ = self.local_attn(h, h, h, need_weights=False)
            local_seq = local_seq + self.local_drop1(y)
            local_seq = local_seq + self.local_ffn(self.local_norm2(local_seq))

            # The group CLS at position 0 becomes the group summary
            group_summaries.append(local_seq[:, :1, :])   # [B, 1, d]
            out_feats[:, idx, :] = local_seq[:, 1:, :]    # updated feature tokens

        summaries = torch.cat(group_summaries, dim=1)      # [B, G, d]
        return summaries, out_feats

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, d] — CLS at position 0, then F feature tokens.

        Returns:
            tokens: [B, N, d] (same shape, updated representation).
        """
        if not self.use_hierarchical:
            return self.global_block(tokens)

        cls_token = tokens[:, :1, :]          # [B, 1, d]
        feat_tokens = tokens[:, 1:, :]        # [B, F, d]

        # Stage 1: intra-group local attention
        group_summaries, feat_tokens = self._intra_group_attention(feat_tokens)

        # Stage 2: inter-group global attention
        # Sequence = [global_CLS] + [group_CLS_0, group_CLS_1, ...] + [updated features]
        # Including the raw features ensures information flows both ways.
        global_seq = torch.cat(
            [cls_token, group_summaries, feat_tokens], dim=1
        )

        # Global attention over the combined sequence
        global_seq = self.global_block(global_seq)

        # Reassemble: CLS + updated group summaries + updated features
        # Keep only CLS and features; group summaries are intermediates.
        new_cls = global_seq[:, :1, :]                              # [B, 1, d]
        new_feats = global_seq[:, 1 + self.num_groups:, :]          # [B, F, d]
        return torch.cat([new_cls, new_feats], dim=1)


# ============================================================================
# 改进 ④: CLS Cross-Attention Aggregator
# ============================================================================

class CLSCrossAttentionAggregator(nn.Module):
    """
    Instead of taking CLS's final hidden state as the representation,
    let CLS actively QUERY all feature tokens via cross-attention.

    This is conceptually:
        z = CrossAttention(
            query   = CLS,            "what disease signal am I looking for?"
            key/val = feature tokens,  "here's all the CpG information"
        )

    Benefits:
      - Stronger aggregation: CLS extracts exactly what it needs
      - Interpretability: attention weights show which CpGs drive decisions
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(d_model)
        self.apply(_init_weights)

    def forward(
        self, tokens: torch.Tensor, need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            tokens: [B, N, d] — CLS at position 0, features at 1..F.
            need_weights: return [B, F] attention over features for interpretation.

        Returns:
            z:             [B, d] aggregated representation.
            attn_weights:  [B, F] or None.
        """
        cls = self.norm_q(tokens[:, :1, :])       # [B, 1, d]
        feats = self.norm_kv(tokens[:, 1:, :])    # [B, F, d]

        aggregated, attn = self.cross_attn(
            query=cls,
            key=feats,
            value=feats,
            need_weights=need_weights,
        )  # aggregated: [B, 1, d], attn: [B, 1, F] if need_weights else None

        z = cls + self.drop(aggregated)
        z = self.out_norm(z).squeeze(1)           # [B, d]

        weights = attn.squeeze(1) if attn is not None else None  # [B, F]
        return z, weights


# ============================================================================
# 辅助: 标准 Transformer Block（用作 hierarchical 的 fallback / global stage）
# ============================================================================

class _StandardTransformerBlock(nn.Module):
    """Pre-Norm Transformer block — identical in spirit to TransformerEncoderBlock."""

    def __init__(
        self, d_model: int, num_heads: int = 4,
        dropout: float = 0.2, ffn_multiplier: int = 4,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            num_heads = 1
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = _build_ffn(d_model, ffn_multiplier, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        y, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(y)
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================================
# 完整 Encoder
# ============================================================================

class ImprovedFTTransformerEncoder(nn.Module):
    """
    Improved FT-Transformer encoder with four orthogonal enhancements.

    Parameters
    ----------
    num_features : int
        Number of tabular features (e.g. CpG sites).
    d_model : int
        Token / hidden dimension.
    num_layers : int
        Total Transformer layers.  The first (num_layers - 1) are hierarchical
        blocks (when grouping is used); the last is always a global block.
    num_heads : int
        Attention heads per layer.
    dropout : float
        Dropout rate throughout.
    genomic_positions : Tensor of shape (num_features,), optional
        Genomic coordinates (e.g. bp position along chromosome) for each feature.
        When provided, genomic distance bias is injected into attention scores.
    cpg_groups : List[List[int]], optional
        Each inner list contains the 0-based feature indices belonging to one
        biological group (e.g. CpG island).  When provided with >1 groups,
        hierarchical intra/inter-group attention is used.
    use_gated_selection : bool
        Enable the GatedFeatureSelector (default: True).
    use_cls_cross_attention : bool
        Replace the final CLS extraction with cross-attention aggregation
        (default: True).
    ffn_multiplier : int
        Expansion factor for FFN layers.
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        genomic_positions: Optional[torch.Tensor] = None,
        cpg_groups: Optional[List[List[int]]] = None,
        use_gated_selection: bool = True,
        use_cls_cross_attention: bool = True,
        ffn_multiplier: int = 4,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.use_gated_selection = bool(use_gated_selection)
        self.use_cls_cross_attention = bool(use_cls_cross_attention)

        # ---- Tokenizer ----
        self.tokenizer = FeatureTokenizer(
            num_features=num_features, token_dim=d_model, dropout=dropout,
        )

        # ---- Gated Feature Selection ----
        if self.use_gated_selection:
            self.gate = GatedFeatureSelector(num_features, d_model)
        else:
            self.gate = None

        # ---- Build layer stack ----
        # Validate / normalize genomic positions
        if genomic_positions is not None:
            genomic_positions = torch.as_tensor(genomic_positions, dtype=torch.float32)
            if genomic_positions.numel() != num_features:
                raise ValueError(
                    f"genomic_positions length {genomic_positions.numel()} "
                    f"!= num_features {num_features}"
                )
        self.register_buffer("genomic_positions",
                             genomic_positions if genomic_positions is not None
                             else torch.empty(0), persistent=True)

        # Validate groups
        if cpg_groups is not None:
            flat = [idx for g in cpg_groups for idx in g]
            if len(flat) != len(set(flat)):
                raise ValueError("cpg_groups contain duplicate indices")
            if max(flat) >= num_features:
                raise ValueError(
                    f"cpg_groups contains index {max(flat)} >= num_features {num_features}"
                )
        self.cpg_groups = cpg_groups

        self.blocks = nn.ModuleList()
        for layer_idx in range(num_layers):
            # Last layer: always global block with genomic distance bias
            if layer_idx == num_layers - 1:
                block = GenomicDistanceBiasedTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    num_features=num_features,
                    dropout=dropout,
                    ffn_multiplier=ffn_multiplier,
                    genomic_positions=(
                        genomic_positions if genomic_positions is not None else None
                    ),
                )
            else:
                # Earlier layers: hierarchical (when groups provided) or standard
                block = HierarchicalAttentionBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    ffn_multiplier=ffn_multiplier,
                    cpg_groups=cpg_groups,
                    genomic_positions=(
                        genomic_positions if genomic_positions is not None else None
                    ),
                )
            self.blocks.append(block)

        # ---- CLS Cross-Attention Aggregator ----
        if self.use_cls_cross_attention:
            self.cls_aggregator = CLSCrossAttentionAggregator(
                d_model=d_model, num_heads=num_heads, dropout=dropout * 0.5,
            )
        else:
            self.cls_aggregator = None
            self.final_norm = nn.LayerNorm(d_model)

        self.apply(_init_weights)

    @property
    def has_genomic_positions(self) -> bool:
        return self.genomic_positions.numel() > 0

    @property
    def has_groups(self) -> bool:
        return self.cpg_groups is not None and len(self.cpg_groups) > 1

    def update_genomic_positions(self, positions: torch.Tensor):
        """Update genomic positions (e.g. after feature reordering)."""
        positions = torch.as_tensor(positions, dtype=torch.float32).view(-1)
        if positions.numel() != self.num_features:
            raise ValueError(
                f"positions length {positions.numel()} != num_features {self.num_features}"
            )
        self.register_buffer("genomic_positions", positions, persistent=True)
        # Propagate to the genomic-distance-biased block (last block)
        last_block = self.blocks[-1]
        if isinstance(last_block, GenomicDistanceBiasedTransformerBlock):
            last_block.distance_bias.update_positions(positions)

    def forward(
        self, x: torch.Tensor, need_attention: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [B, num_features]
            need_attention: if True, include CpG attention weights in aux
                            (for biomarker interpretation).

        Returns:
            z:   [B, d_model]    aggregated feature representation
            aux: dict with at least {"tokens": Tensor}
        """
        # Tokenize
        tokens = self.tokenizer(x)              # [B, F+1, d]

        # Gated feature selection
        if self.gate is not None:
            tokens = self.gate(tokens)

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        # CLS aggregation
        attn_weights = None
        if self.cls_aggregator is not None:
            z, attn_weights = self.cls_aggregator(tokens, need_weights=need_attention)
        else:
            tokens = self.final_norm(tokens)
            z = tokens[:, 0]                    # [B, d]

        aux = {
            "tokens": tokens,
            "cpg_tokens": tokens[:, 1:] if tokens.shape[1] > 1 else None,
        }
        if attn_weights is not None:
            aux["cpg_attention_weights"] = attn_weights  # [B, F] — for interpretation

        return z, aux


# ============================================================================
# Genomic-Distance-Biased Transformer Block
# ============================================================================

class GenomicDistanceBiasedTransformerBlock(nn.Module):
    """
    Pre-Norm Transformer block that replaces standard MHA with
    GenomicDistanceBiasedMHA.

    Used as the FINAL layer in the improved encoder to apply biological
    positional bias in a global context after hierarchical processing.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_features: int,
        dropout: float = 0.2,
        ffn_multiplier: int = 4,
        genomic_positions: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GenomicDistanceBiasedMHA(
            embed_dim=d_model,
            num_heads=num_heads,
            num_features=num_features,
            dropout=dropout,
            genomic_positions=genomic_positions,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = _build_ffn(d_model, ffn_multiplier, dropout)
        self.distance_bias = self.attn.distance_bias  # expose for position updates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        y, _ = self.attn(h, need_weights=False)
        x = x + self.drop1(y)
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================================
# 完整分类器
# ============================================================================

class ImprovedFTTransformerClassifier(nn.Module):
    """
    Complete classifier wrapping the ImprovedFTTransformerEncoder.

    Drop-in replacement for FTTransformerClassifier with the same interface:
        logits, aux = model(x)

    Additional aux outputs:
        - "cpg_attention_weights": [B, F] cross-attention over CpGs (interpretable)
        - "cpg_tokens":              [B, F, d] final per-CpG representations
        - "gate_values":             (internal, when gating is enabled)
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        genomic_positions: Optional[torch.Tensor] = None,
        cpg_groups: Optional[List[List[int]]] = None,
        use_gated_selection: bool = True,
        use_cls_cross_attention: bool = True,
        need_attention: bool = False,
        idx_ad: Optional[int] = None,
        idx_mci: Optional[int] = None,
        idx_hc: Optional[int] = None,
        use_linear_residual: bool = True,
    ):
        super().__init__()
        self.need_attention = bool(need_attention)
        self.num_features = int(num_features)
        self.num_classes = int(num_classes)

        self.encoder = ImprovedFTTransformerEncoder(
            num_features=num_features,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            genomic_positions=genomic_positions,
            cpg_groups=cpg_groups,
            use_gated_selection=use_gated_selection,
            use_cls_cross_attention=use_cls_cross_attention,
        )

        self.pred_head = HierarchicalPredictionHead(
            embed_dim=d_model,
            num_features=num_features,
            num_classes=num_classes,
            dropout=dropout,
            idx_ad=idx_ad,
            idx_mci=idx_mci,
            idx_hc=idx_hc,
            use_linear_residual=use_linear_residual,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        z, aux = self.encoder(x, need_attention=self.need_attention)
        logits, aux = self.pred_head(z, x, aux)
        return logits, aux


class FTTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        idx_ad: Optional[int] = None,
        idx_mci: Optional[int] = None,
        idx_hc: Optional[int] = None,
        use_linear_residual: bool = True,
    ):
        super().__init__()
        self.encoder = FTTransformerEncoder(
            num_features=num_features,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.pred_head = HierarchicalPredictionHead(
            embed_dim=d_model,
            num_features=num_features,
            num_classes=num_classes,
            dropout=dropout,
            idx_ad=idx_ad,
            idx_mci=idx_mci,
            idx_hc=idx_hc,
            use_linear_residual=use_linear_residual,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        z, aux = self.encoder(x)
        return self.pred_head(z, x, aux)


# ============================================================
# ModernNCA-style classifier
# ============================================================

class ModernNCAClassifier(nn.Module):
    """
    A differentiable ModernNCA-style classifier.

    The encoder maps tabular inputs into a metric embedding space. Each class owns
    multiple trainable prototypes. Logits are computed with temperature-scaled
    log-sum-exp over negative squared distances to class prototypes.
    """
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 128,
        depth: int = 4,
        dropout: float = 0.2,
        num_prototypes_per_class: int = 4,
        temperature: float = 0.7,
        idx_ad: Optional[int] = None,
        idx_mci: Optional[int] = None,
        idx_hc: Optional[int] = None,
        use_linear_residual: bool = True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_prototypes_per_class = int(num_prototypes_per_class)
        self.temperature_raw = nn.Parameter(torch.tensor(float(math.log(math.exp(temperature) - 1.0))))
        hidden = max(256, d_model * 2)
        self.stem = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.LayerNorm(d_model),
        )
        self.blocks = nn.ModuleList([ResidualMLPBlock(d_model, dropout=dropout) for _ in range(depth)])
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.prototypes = nn.Parameter(torch.randn(num_classes, num_prototypes_per_class, d_model) * 0.02)
        self.pred_head = HierarchicalPredictionHead(
            embed_dim=d_model,
            num_features=num_features,
            num_classes=num_classes,
            dropout=dropout,
            idx_ad=idx_ad,
            idx_mci=idx_mci,
            idx_hc=idx_hc,
            use_linear_residual=use_linear_residual,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        for block in self.blocks:
            z = block(z)
        z = self.proj(z)
        return F.normalize(z, dim=-1)

    def prototype_logits(self, z: torch.Tensor) -> torch.Tensor:
        p = F.normalize(self.prototypes, dim=-1)
        dist2 = torch.sum((z[:, None, None, :] - p[None, :, :, :]) ** 2, dim=-1)
        temperature = F.softplus(self.temperature_raw).clamp_min(1e-3)
        logits = torch.logsumexp(-dist2 / temperature, dim=-1)
        return logits

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        z = self.encode(x)
        nca_logits = self.prototype_logits(z)
        aux = {"nca_logits": nca_logits, "shared_embedding": z}
        head_logits, aux = self.pred_head(z, x, aux)
        # Blend metric logits with parametric head logits. This keeps NCA central while
        # retaining the same hierarchical auxiliary interface.
        logits = 0.5 * head_logits + 0.5 * nca_logits
        aux["mlp_logits"] = head_logits
        return logits, aux


# ============================================================
# TabM-style ensemble with training-time mixup support
# ============================================================

class TabMClassifier(nn.Module):
    """
    Lightweight TabM-style tabular ensemble.

    It uses multiple MLP members over a shared normalized input. The trainer may
    apply mixup through --mixup-alpha; this module itself stays deterministic so
    validation/external testing uses the same forward interface as all models.
    """
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 128,
        depth: int = 3,
        dropout: float = 0.2,
        num_members: int = 4,
        idx_ad: Optional[int] = None,
        idx_mci: Optional[int] = None,
        idx_hc: Optional[int] = None,
        use_linear_residual: bool = True,
    ):
        super().__init__()
        self.num_members = int(num_members)
        hidden = max(256, d_model * 2)
        self.input_norm = nn.LayerNorm(num_features)
        self.member_stems = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_features, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(self.num_members)
        ])
        self.member_blocks = nn.ModuleList([
            nn.ModuleList([ResidualMLPBlock(d_model, dropout=dropout) for _ in range(depth)])
            for _ in range(self.num_members)
        ])
        self.member_heads = nn.ModuleList([nn.Linear(d_model, num_classes) for _ in range(self.num_members)])
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_model * self.num_members),
            nn.Linear(d_model * self.num_members, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pred_head = HierarchicalPredictionHead(
            embed_dim=d_model,
            num_features=num_features,
            num_classes=num_classes,
            dropout=dropout,
            idx_ad=idx_ad,
            idx_mci=idx_mci,
            idx_hc=idx_hc,
            use_linear_residual=use_linear_residual,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        xn = self.input_norm(x)
        embeddings = []
        member_logits = []
        for stem, blocks, head in zip(self.member_stems, self.member_blocks, self.member_heads):
            z = stem(xn)
            for block in blocks:
                z = block(z)
            embeddings.append(z)
            member_logits.append(head(z))
        member_logits_t = torch.stack(member_logits, dim=1)  # [B, M, C]
        z_fused = self.fuse(torch.cat(embeddings, dim=-1))
        head_logits, aux = self.pred_head(z_fused, x, {"tabm_member_logits": member_logits_t})
        logits = 0.5 * head_logits + 0.5 * member_logits_t.mean(dim=1)
        return logits, aux


# ============================================================
# DANN + FT-Transformer
# ============================================================

class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReverseFunction.apply(x, lambd)


class DANNFTTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_domains: int = 2,
        grl_lambda: float = 1.0,
        idx_ad: Optional[int] = None,
        idx_mci: Optional[int] = None,
        idx_hc: Optional[int] = None,
        use_linear_residual: bool = True,
    ):
        super().__init__()
        self.grl_lambda = float(grl_lambda)
        self.encoder = FTTransformerEncoder(
            num_features=num_features,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.pred_head = HierarchicalPredictionHead(
            embed_dim=d_model,
            num_features=num_features,
            num_classes=num_classes,
            dropout=dropout,
            idx_ad=idx_ad,
            idx_mci=idx_mci,
            idx_hc=idx_hc,
            use_linear_residual=use_linear_residual,
        )
        self.domain_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(64, d_model)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(64, d_model), int(num_domains)),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        z, aux = self.encoder(x)
        logits, aux = self.pred_head(z, x, aux)
        aux["domain_logits"] = self.domain_head(grad_reverse(z, self.grl_lambda))
        return logits, aux


class DANNMethylationFTTransformerClassifier(nn.Module):
    """DANN variant of methylation-aware FT-Transformer."""
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_domains: int = 2,
        grl_lambda: float = 1.0,
        cpg_annotation_ids: Optional[Dict[str, torch.Tensor]] = None,
        annotation_dropout: float = 0.0,
        idx_ad: Optional[int] = None,
        idx_mci: Optional[int] = None,
        idx_hc: Optional[int] = None,
        use_linear_residual: bool = True,
    ):
        super().__init__()
        self.grl_lambda = float(grl_lambda)
        self.encoder = MethylationFTTransformerEncoder(
            num_features=num_features,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            cpg_annotation_ids=cpg_annotation_ids,
            annotation_dropout=annotation_dropout,
        )
        self.pred_head = HierarchicalPredictionHead(
            embed_dim=d_model,
            num_features=num_features,
            num_classes=num_classes,
            dropout=dropout,
            idx_ad=idx_ad,
            idx_mci=idx_mci,
            idx_hc=idx_hc,
            use_linear_residual=use_linear_residual,
        )
        self.domain_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(64, d_model)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(64, d_model), int(num_domains)),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        z, aux = self.encoder(x)
        logits, aux = self.pred_head(z, x, aux)
        aux["domain_logits"] = self.domain_head(grad_reverse(z, self.grl_lambda))
        return logits, aux


class ResidualMLPClassifier(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int = 512, depth: int = 4, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim, dropout=dropout) for _ in range(depth)])
        # self.head = nn.Sequential(
        #     nn.LayerNorm(hidden_dim),
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim // 2, num_classes),
        # )

    def forward(self, x):
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        # logits = self.head(h)
        return  h


class FTTransformerEncoder_v1(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features=num_features, token_dim=d_model, dropout=dropout)
        
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        tokens = self.tokenizer(x)
        for block in self.blocks:
            tokens = block(tokens)
        # tokens = self.dropout(torch.cat([tokens, mlp_features], dim=1))
        tokens = self.final_norm(tokens)
        z = tokens[:, 0] #+ mlp_features[:, 0]
        
        aux = {"tokens": tokens}
        return z, aux


class DANNFTTransformerClassifier_v1(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_domains: int = 2,
        grl_lambda: float = 1.0,
        idx_ad: Optional[int] = None,
        idx_mci: Optional[int] = None,
        idx_hc: Optional[int] = None,
        use_linear_residual: bool = True,
    ):
        super().__init__()
        self.grl_lambda = float(grl_lambda)
        self.encoder = FTTransformerEncoder_v1(
            num_features=num_features,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        # self.res_mlp = ResidualMLPClassifier(num_features=num_features, hidden_dim=d_model, dropout=dropout)
        self.pred_head = HierarchicalPredictionHead(
            embed_dim=d_model,
            num_features=num_features,
            num_classes=num_classes,
            dropout=dropout,
            idx_ad=idx_ad,
            idx_mci=idx_mci,
            idx_hc=idx_hc,
            use_linear_residual=use_linear_residual,
        )
        self.domain_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(64, d_model)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(64, d_model), int(num_domains)),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        z, aux = self.encoder(x)
        # r = self.res_mlp(x)
        logits, aux = self.pred_head(z, x)
        aux["domain_logits"] = self.domain_head(grad_reverse(z, self.grl_lambda))
        return logits, aux




# ============================================================
# Methylation-aware hierarchical epigenetic attention
# ============================================================

class CovariateTokenizer(nn.Module):
    """Tokenize sample-level covariates such as age, sex and blood cell proportions.

    Input covariates are expected to be train-only standardized in the training script.
    For each covariate c_i, token_i = c_i * W_i + b_i + E_covariate_id[i].
    """
    def __init__(self, covariate_dim: int, token_dim: int, dropout: float = 0.0):
        super().__init__()
        self.covariate_dim = int(covariate_dim)
        self.token_dim = int(token_dim)
        if self.covariate_dim > 0:
            self.weight = nn.Parameter(torch.empty(self.covariate_dim, self.token_dim))
            self.bias = nn.Parameter(torch.empty(self.covariate_dim, self.token_dim))
            self.id_embedding = nn.Embedding(self.covariate_dim + 1, self.token_dim, padding_idx=0)
            self.register_buffer("covariate_ids", torch.arange(1, self.covariate_dim + 1, dtype=torch.long), persistent=True)
            self.norm = nn.LayerNorm(self.token_dim)
            self.dropout = nn.Dropout(dropout)
            self.reset_parameters()
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
            self.id_embedding = None
            self.norm = None
            self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        if self.covariate_dim > 0:
            nn.init.xavier_uniform_(self.weight)
            nn.init.zeros_(self.bias)
            nn.init.normal_(self.id_embedding.weight, std=0.02)
            with torch.no_grad():
                self.id_embedding.weight[0].zero_()

    def forward(self, covariates: Optional[torch.Tensor], batch_size: int, device=None) -> torch.Tensor:
        if self.covariate_dim <= 0:
            if device is None:
                device = self.weight.device if self.weight is not None else torch.device("cpu")
            return torch.empty(batch_size, 0, self.token_dim, device=device)
        if covariates is None:
            raise ValueError("This model was created with covariate_dim>0, but forward(..., covariates=None) was received.")
        covariates = covariates.to(self.weight.device)
        if covariates.shape[-1] != self.covariate_dim:
            raise ValueError(f"Expected {self.covariate_dim} covariates, got {covariates.shape[-1]}")
        tokens = covariates.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        tokens = tokens + self.id_embedding(self.covariate_ids).unsqueeze(0)
        return self.dropout(self.norm(tokens))


class GroupLocalSelfAttention(nn.Module):
    """Apply a shared Transformer block independently inside each CpG group."""
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.block = TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)

    def forward(self, tokens: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        # tokens: [B, F, D], group_ids: [F]
        if tokens.shape[1] != group_ids.numel():
            raise ValueError("CpG token length and group_ids length mismatch")
        out = tokens.clone()
        unique_groups = torch.unique(group_ids.detach().cpu()).tolist()
        for g in unique_groups:
            idx = torch.nonzero(group_ids == int(g), as_tuple=False).view(-1)
            if idx.numel() <= 1:
                continue
            out[:, idx, :] = self.block(tokens[:, idx, :])
        return out


class GroupAttentionPooling(nn.Module):
    """First-level attention: CpG tokens -> gene/chromosome/pathway group tokens."""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, group_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # tokens: [B, F, D]
        unique_groups = torch.unique(group_ids.detach().cpu()).tolist()
        group_tokens = []
        attn_dense = tokens.new_zeros(tokens.shape[0], tokens.shape[1])
        for g in unique_groups:
            idx = torch.nonzero(group_ids == int(g), as_tuple=False).view(-1)
            local = tokens[:, idx, :]
            score = self.score(local).squeeze(-1)
            w = torch.softmax(score, dim=1)
            group_vec = torch.sum(local * w.unsqueeze(-1), dim=1)
            group_tokens.append(group_vec)
            attn_dense[:, idx] = w
        group_tokens = torch.stack(group_tokens, dim=1) if group_tokens else tokens.new_zeros(tokens.shape[0], 0, tokens.shape[-1])
        return self.out_norm(group_tokens), attn_dense


class EpigeneticHierarchicalEncoder(nn.Module):
    """Two-level methylation encoder.

    Level 1: CpG tokens are processed/pool within biological groups, usually genes.
    Level 2: group tokens plus sample-level covariate tokens are processed by global Transformer layers.

    The global sequence contains two special tokens:
      - disease CLS: used for AD/MCI/HC prediction;
      - confounder CLS: used to absorb cohort/age/sex/cell-composition information.
    """
    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        cpg_annotation_ids: Optional[Dict[str, torch.Tensor]] = None,
        annotation_dropout: float = 0.0,
        covariate_dim: int = 0,
        local_group_layers: int = 1,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.d_model = int(d_model)
        self.covariate_dim = int(covariate_dim)
        self.cpg_tokenizer = MethylationFeatureTokenizer(
            num_features=num_features,
            token_dim=d_model,
            cpg_annotation_ids=cpg_annotation_ids,
            dropout=dropout,
            annotation_dropout=annotation_dropout,
        )
        self.covariate_tokenizer = CovariateTokenizer(covariate_dim=covariate_dim, token_dim=d_model, dropout=dropout)

        ids = cpg_annotation_ids or {}
        group_ids = torch.as_tensor(ids.get("group_ids", ids.get("gene_annotation_ids", torch.ones(num_features, dtype=torch.long))), dtype=torch.long)
        if group_ids.numel() != num_features:
            raise ValueError(f"group_ids must have length {num_features}, got {group_ids.numel()}")
        group_ids = group_ids.view(num_features).clamp_min(0)
        # Remap to compact contiguous IDs. This keeps pooling order deterministic.
        uniq = sorted(set(int(x) for x in group_ids.tolist()))
        remap = {old: new for new, old in enumerate(uniq)}
        compact = torch.tensor([remap[int(x)] for x in group_ids.tolist()], dtype=torch.long)
        self.register_buffer("group_ids", compact, persistent=True)
        self.num_groups = int(compact.max().item()) + 1 if compact.numel() else 0

        self.local_layers = nn.ModuleList([
            GroupLocalSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(max(0, int(local_group_layers)))
        ])
        self.group_pool = GroupAttentionPooling(d_model=d_model, dropout=dropout)
        self.disease_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.confounder_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.group_type_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        self.covariate_type_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        self.global_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.disease_cls, std=0.02)
        nn.init.normal_(self.confounder_cls, std=0.02)
        nn.init.normal_(self.group_type_embedding, std=0.02)
        nn.init.normal_(self.covariate_type_embedding, std=0.02)

    def forward(self, x: torch.Tensor, covariates: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        B = x.shape[0]
        # cpg_tokenizer returns [CLS] + CpG tokens. We use only CpG tokens here.
        cpg_tokens = self.cpg_tokenizer(x)[:, 1:, :]
        for layer in self.local_layers:
            cpg_tokens = layer(cpg_tokens, self.group_ids)
        group_tokens, within_group_attn = self.group_pool(cpg_tokens, self.group_ids)
        group_tokens = group_tokens + self.group_type_embedding
        cov_tokens = self.covariate_tokenizer(covariates, batch_size=B, device=x.device)
        if cov_tokens.shape[1] > 0:
            cov_tokens = cov_tokens + self.covariate_type_embedding
        disease_cls = self.disease_cls.expand(B, -1, -1)
        conf_cls = self.confounder_cls.expand(B, -1, -1)
        seq = torch.cat([disease_cls, conf_cls, group_tokens, cov_tokens], dim=1)
        for block in self.global_blocks:
            seq = block(seq)
        seq = self.final_norm(seq)
        z_disease = seq[:, 0]
        z_confounder = seq[:, 1]
        aux = {
            "tokens": seq,
            "cpg_tokens": cpg_tokens,
            "group_tokens": group_tokens,
            "covariate_tokens": cov_tokens,
            "within_group_attention": within_group_attn,
            "group_ids": self.group_ids,
            "disease_embedding": z_disease,
            "confounder_embedding": z_confounder,
            "orthogonality_loss": torch.mean(F.cosine_similarity(z_disease, z_confounder, dim=-1).pow(2)),
        }
        return z_disease, aux


class HierarchicalEpigeneticAttentionClassifier(nn.Module):
    """Disease classifier with two-level epigenetic attention and a confounder branch."""
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_domains: int = 2,
        grl_lambda: float = 1.0,
        cpg_annotation_ids: Optional[Dict[str, torch.Tensor]] = None,
        annotation_dropout: float = 0.0,
        covariate_dim: int = 0,
        local_group_layers: int = 1,
        idx_ad: Optional[int] = None,
        idx_mci: Optional[int] = None,
        idx_hc: Optional[int] = None,
        use_linear_residual: bool = True,
    ):
        super().__init__()
        self.grl_lambda = float(grl_lambda)
        self.covariate_dim = int(covariate_dim)
        self.encoder = EpigeneticHierarchicalEncoder(
            num_features=num_features,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            cpg_annotation_ids=cpg_annotation_ids,
            annotation_dropout=annotation_dropout,
            covariate_dim=covariate_dim,
            local_group_layers=local_group_layers,
        )
        self.pred_head = HierarchicalPredictionHead(
            embed_dim=d_model,
            num_features=num_features,
            num_classes=num_classes,
            dropout=dropout,
            idx_ad=idx_ad,
            idx_mci=idx_mci,
            idx_hc=idx_hc,
            use_linear_residual=use_linear_residual,
        )
        hidden = max(64, d_model)
        # Adversarial branch: make disease embedding domain-invariant.
        self.domain_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, int(num_domains))
        )
        # Confounder branch: encourage confounder CLS to capture cohort/batch.
        self.confounder_domain_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, int(num_domains))
        )
        # Confounder branch: reconstruct standardized covariates from confounder CLS.
        self.covariate_reconstruction_head = None
        if self.covariate_dim > 0:
            self.covariate_reconstruction_head = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, self.covariate_dim)
            )

    def forward(self, x: torch.Tensor, covariates: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        z, aux = self.encoder(x, covariates=covariates)
        logits, aux = self.pred_head(z, x, aux)
        z_conf = aux["confounder_embedding"]
        aux["domain_logits"] = self.domain_head(grad_reverse(z, self.grl_lambda))
        aux["confounder_domain_logits"] = self.confounder_domain_head(z_conf)
        if self.covariate_reconstruction_head is not None:
            aux["covariate_reconstruction"] = self.covariate_reconstruction_head(z_conf)
        return logits, aux

# ============================================================
# FT-Transformer (Official) — paper-correct backbone + project head
# ============================================================


class FTTransformerOfficial(nn.Module):
    """FT-Transformer using the official rtdl_revisiting_models backbone.

    Key improvements over the simplified ``FTTransformerClassifier``:

    - Uses the paper-correct ``FTTransformer`` from ``rtdl_revisiting_models``,
      which implements:
      * ReGLU activation (paper default) instead of GELU
      * Skipping the first LayerNorm in the first Transformer layer (CRUCIAL)
      * CLS-only attention in the last Transformer layer (efficient)
    - Still attaches the project's ``HierarchicalPredictionHead`` so that
      HFR aux heads, ordinal head and linear residual are available.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 192,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.2,
        idx_ad: Optional[int] = None,
        idx_mci: Optional[int] = None,
        idx_hc: Optional[int] = None,
        use_linear_residual: bool = True,
    ):
        super().__init__()
        # Import lazily to keep the file importable even when rtdl is absent.
        try:
            from rtdl_revisiting_models import FTTransformer
        except ImportError:
            raise ImportError(
                "rtdl_revisiting_models.py is required for ft_transformer_official. "
                "Make sure it is in the same directory."
            )

        # Build the paper-official FT-Transformer backbone using the default config.
        self.backbone = FTTransformer(
            n_cont_features=num_features,
            cat_cardinalities=[],
            d_out=None,  # we only need the [CLS] embedding, not the final head
            n_blocks=num_layers,
            d_block=d_model,
            attention_n_heads=num_heads,
            attention_dropout=dropout * 0.75,  # paper default maps total→sub-rates
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4 / 3,
            ffn_dropout=dropout * 0.5,
            residual_dropout=0.0,
            _is_default=False,
        )

        self.pred_head = HierarchicalPredictionHead(
            embed_dim=d_model,
            num_features=num_features,
            num_classes=num_classes,
            dropout=dropout,
            idx_ad=idx_ad,
            idx_mci=idx_mci,
            idx_hc=idx_hc,
            use_linear_residual=use_linear_residual,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        z = self.backbone(x, None)       # x_cont=x, x_cat=None → [CLS] embedding [B, d]
        logits, aux = self.pred_head(z, x, {})
        return logits, aux


# ============================================================
# 本地 ResidualMLPBlock — 避免与 methylation_model_zoo 的循环导入
# ============================================================

class _ResidualMLPBlock(nn.Module):
    """PreNorm residual FFN block, same structure as ResidualMLPBlock in model zoo."""
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


# ============================================================
# Model A: MLPTransformerFusion — 门控双路径融合
# ============================================================

class _GatedFusion(nn.Module):
    """自适应门控融合：让模型学习每个样本更信任哪条路径。"""
    def __init__(self, dim_mlp: int, dim_ft: int, dim_fusion: int, dropout: float = 0.1):
        super().__init__()
        self.proj_mlp = nn.Linear(dim_mlp, dim_fusion)
        self.proj_ft = nn.Linear(dim_ft, dim_fusion)
        self.gate = nn.Sequential(
            nn.Linear(dim_mlp + dim_ft, dim_fusion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_fusion, dim_fusion),
            nn.Sigmoid(),
        )

    def forward(self, z_mlp: torch.Tensor, z_ft: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_mlp_p = self.proj_mlp(z_mlp)
        z_ft_p = self.proj_ft(z_ft)
        gate = self.gate(torch.cat([z_mlp, z_ft], dim=-1))
        z_fused = gate * z_mlp_p + (1.0 - gate) * z_ft_p
        return z_fused, gate


class MLPTransformerFusion(nn.Module):
    """ResidualMLP + FT-Transformer 门控双路径融合模型。

    两条路径共享同一个输入 x：
    - 路径 A (ResidualMLP): stem → 多个 ResidualMLPBlock → z_mlp [B, mlp_hidden]
    - 路径 B (FT-Transformer): FeatureTokenizer → Transformer blocks → z_ft [B, d_model]
    然后通过 _GatedFusion 自适应融合，送入 HierarchicalPredictionHead。
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        mlp_hidden_dim: int = 256,
        mlp_depth: int = 3,
        dim_fusion: int = 128,
        idx_ad: Optional[int] = None,
        idx_mci: Optional[int] = None,
        idx_hc: Optional[int] = None,
        use_linear_residual: bool = True,
    ):
        super().__init__()

        # ---- FT-Transformer 编码器 ----
        self.ft_encoder = FTTransformerEncoder(
            num_features=num_features,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ---- ResidualMLP 编码器 ----
        self.mlp_stem = nn.Sequential(
            nn.Linear(num_features, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mlp_blocks = nn.ModuleList([
            _ResidualMLPBlock(mlp_hidden_dim, dropout=dropout)
            for _ in range(mlp_depth)
        ])

        # ---- 门控融合 ----
        self.fusion = _GatedFusion(
            dim_mlp=mlp_hidden_dim,
            dim_ft=d_model,
            dim_fusion=dim_fusion,
            dropout=dropout,
        )

        # ---- 共享预测头 ----
        self.pred_head = HierarchicalPredictionHead(
            embed_dim=dim_fusion,
            num_features=num_features,
            num_classes=num_classes,
            dropout=dropout,
            idx_ad=idx_ad,
            idx_mci=idx_mci,
            idx_hc=idx_hc,
            use_linear_residual=use_linear_residual,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # FT-Transformer 路径
        z_ft, aux_ft = self.ft_encoder(x)

        # ResidualMLP 路径
        h_mlp = self.mlp_stem(x)
        for blk in self.mlp_blocks:
            h_mlp = blk(h_mlp)

        # 门控融合
        z_fused, gate_values = self.fusion(h_mlp, z_ft)

        # 共享预测头
        logits, aux = self.pred_head(z_fused, x, {})
        aux["gate_values"] = gate_values.detach()
        aux["features"] = z_fused
        return logits, aux


# ============================================================
# Model B: FTTransformerDeepFFN — Transformer + ResidualMLP 增强 FFN
# ============================================================

class _TransformerBlockDeepFFN(nn.Module):
    """Transformer block 的 FFN 被多层 ResidualMLPBlock 替换。

    保持 Self-Attention 不变（保留特征交互能力），
    将原来单薄的 2 层 FFN 替换为 mlp_ffn_depth 层残差 MLP。
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.2,
                 ffn_multiplier: int = 4, mlp_ffn_depth: int = 2):
        super().__init__()
        if d_model % num_heads != 0:
            num_heads = 1
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.ModuleList([
            _ResidualMLPBlock(d_model, dropout=dropout)
            for _ in range(mlp_ffn_depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        y, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(y)
        h = self.norm2(x)
        for blk in self.ffn:
            h = blk(h)
        x = x + h
        return x


class FTTransformerDeepFFN(nn.Module):
    """FT-Transformer 变体：Transformer 块中的 FFN 替换为深层 ResidualMLP。

    特点：
    - 保留 FeatureTokenizer（特征嵌入）
    - 保留 Multi-Head Self-Attention（特征交互）
    - 将原始 2 层 GELU FFN 替换为 ResidualMLPBlock 堆叠
    - 更深的位置级别处理能力，同时不丢失跨特征注意力
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        mlp_ffn_depth: int = 2,
        idx_ad: Optional[int] = None,
        idx_mci: Optional[int] = None,
        idx_hc: Optional[int] = None,
        use_linear_residual: bool = True,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(
            num_features=num_features, token_dim=d_model, dropout=dropout,
        )
        self.blocks = nn.ModuleList([
            _TransformerBlockDeepFFN(
                d_model=d_model, num_heads=num_heads, dropout=dropout,
                mlp_ffn_depth=mlp_ffn_depth,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.pred_head = HierarchicalPredictionHead(
            embed_dim=d_model,
            num_features=num_features,
            num_classes=num_classes,
            dropout=dropout,
            idx_ad=idx_ad,
            idx_mci=idx_mci,
            idx_hc=idx_hc,
            use_linear_residual=use_linear_residual,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        tokens = self.tokenizer(x)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.final_norm(tokens)
        z = tokens[:, 0]                     # CLS token
        logits, aux = self.pred_head(z, x, {})
        aux["features"] = z
        return logits, aux
