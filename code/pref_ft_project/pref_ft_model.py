#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PREF-FT: Positive-Aware Residual Evidence Fusion Transformer.

CERM -> CCEM -> REPM -> SMEM -> PREF Head.
The four logit branches represent contextual, linear CpG, positive and prototype
evidence. Existing encoder/head state_dict keys are preserved.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CpGEvidenceRecalibrationModule(nn.Module):
    """Learnable CpG-wise reliability gate, initialized close to identity."""

    def __init__(self, num_features: int, init_logit: float = 4.0, gate_dropout: float = 0.05):
        super().__init__()
        self.gate_logits = nn.Parameter(torch.full((int(num_features),), float(init_logit)))
        self.gate_dropout = float(gate_dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        gate = torch.sigmoid(self.gate_logits).view(1, -1)
        if self.training and self.gate_dropout > 0:
            gate = F.dropout(gate, p=self.gate_dropout, training=True)
        x_gated = x * gate
        return x_gated, {
            "feature_gate": gate.squeeze(0).detach(),
            "feature_gate_mean": gate.mean().detach(),
        }


class FeatureTokenizer(nn.Module):
    """
    Numeric feature tokenizer used by FT-Transformer.

    For input x in [B, F], each CpG feature f has a trainable vector W_f and
    bias b_f:
        token_f = x_f * W_f + b_f
    A learnable [CLS] token is prepended.
    """

    def __init__(self, num_features: int, token_dim: int, dropout: float = 0.2):
        super().__init__()
        self.num_features = int(num_features)
        self.token_dim = int(token_dim)
        self.weight = nn.Parameter(torch.empty(self.num_features, self.token_dim))
        self.bias = nn.Parameter(torch.empty(self.num_features, self.token_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.token_dim))
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


class TransformerEncoderBlock(nn.Module):
    """PreNorm transformer encoder block for CpG tokens."""

    def __init__(self, d_model: int = 128, num_heads: int = 4, dropout: float = 0.2, ffn_multiplier: int = 4):
        super().__init__()
        if int(d_model) % int(num_heads) != 0:
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


class ResidualEvidencePoolingModule(nn.Module):
    """CLS token plus residual attention pooling over CpG tokens."""

    def __init__(self, d_model: int = 128, dropout: float = 0.2, init_alpha: float = -2.5):
        super().__init__()
        hidden = max(32, int(d_model) // 2)
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.alpha_raw = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cls = tokens[:, 0]
        cpg_tokens = tokens[:, 1:]
        attn_logits = self.score(cpg_tokens).squeeze(-1)
        attn = torch.softmax(attn_logits, dim=1)
        pooled = torch.sum(cpg_tokens * attn.unsqueeze(-1), dim=1)
        alpha = torch.sigmoid(self.alpha_raw)
        z = cls + alpha * self.proj(pooled)
        return z, {
            "token_pool_alpha": alpha.detach(),
            "token_pool_attention": attn.detach(),
            "token_pool_embedding": pooled.detach(),
        }


class ContextualCpGEncodingModule(nn.Module):
    """CCEM, paper equations (3)-(6): CpG tokens, CLS, Transformer, normalization.

    The encoder subclasses CCEM so legacy encoder.tokenizer/blocks/final_norm
    state_dict paths remain unchanged. The linear residual still receives x.
    """

    def __init__(self, num_features: int, d_model: int = 128,
                 num_layers: int = 3, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features=num_features, token_dim=d_model, dropout=dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(int(num_layers))
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x_gated: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x_gated)
        for block in self.blocks:
            tokens = block(tokens)
        return self.final_norm(tokens)


class StableMethylationEmbeddingModule(nn.Sequential):
    """SMEM: LayerNorm -> Linear -> GELU -> Dropout -> LayerNorm.

    Sequential numeric parameter paths preserve existing checkpoints.
    """

    def __init__(self, d_model: int = 128, dropout: float = 0.2):
        super().__init__(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.LayerNorm(d_model),
        )


class MethylationEvidenceEncoder(ContextualCpGEncodingModule):
    """CERM -> CCEM -> REPM -> SMEM; preserves legacy checkpoint keys."""

    def __init__(self, num_features: int, d_model: int = 128,
                 num_layers: int = 3, num_heads: int = 4,
                 dropout: float = 0.2, gate_dropout: float = 0.05):
        # CERM initialization uses constants and consumes no random numbers.
        super().__init__(num_features, d_model, num_layers, num_heads, dropout)
        self.feature_gate = CpGEvidenceRecalibrationModule(
            num_features=num_features, init_logit=4.0, gate_dropout=gate_dropout)
        self.pool = ResidualEvidencePoolingModule(d_model=d_model, dropout=dropout, init_alpha=-2.5)
        self.stable_bottleneck = StableMethylationEmbeddingModule(d_model, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_gated, gate_aux = self.feature_gate(x)                 # CERM (1)-(2)
        tokens = super().forward(x_gated)                      # CCEM (3)-(6)
        z_raw, pool_aux = self.pool(tokens)                     # REPM (7)-(9)
        z = self.stable_bottleneck(z_raw)                       # SMEM
        aux: Dict[str, torch.Tensor] = {
            "features": z, "shared_embedding": z,
            "x_gated": x_gated, "z_raw": z_raw.detach(),
        }
        aux.update(gate_aux)
        aux.update(pool_aux)
        return z, aux


class EnsembleContextualEvidenceBranch(nn.Module):
    """TabM-inspired multi-member head; logits are averaged across members."""

    def __init__(self, embed_dim: int, num_classes: int = 2, num_members: int = 4, dropout: float = 0.2):
        super().__init__()
        self.num_members = int(max(1, num_members))
        hidden = max(64, int(embed_dim))
        self.members = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, int(num_classes)),
            )
            for _ in range(self.num_members)
        ])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        member_logits = torch.stack([head(z) for head in self.members], dim=1)
        logits = member_logits.mean(dim=1)
        return logits, {
            "member_logits": member_logits.detach(),
            "member_logit_std": (member_logits.std(dim=1).mean().detach()
                               if self.num_members > 1 else z.new_zeros(())),
        }


class LinearCpGEvidenceBranch(nn.Module):
    """Linear CpG evidence residual (equation 12); no separate elastic-net penalty."""

    def __init__(self, num_features: int, num_classes: int = 2, dropout: float = 0.1, init_alpha: float = -2.0):
        super().__init__()
        self.input_dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(int(num_features), int(num_classes), bias=True)
        self.alpha_raw = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        linear_logits = self.linear(self.input_dropout(x))
        alpha = torch.sigmoid(self.alpha_raw)
        return alpha * linear_logits, {
            "elastic_linear_logits": linear_logits.detach(),
            "elastic_linear_alpha": alpha.detach(),
        }


class PositiveEvidenceAmplificationBranch(nn.Module):
    """Positive-class one-vs-rest residual for class_order='unaffected, affected'."""

    def __init__(self, embed_dim: int, dropout: float = 0.2, init_alpha: float = -3.0, positive_bias_init: float = 0.0):
        super().__init__()
        hidden = max(32, int(embed_dim) // 2)
        self.ovr = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.alpha_raw = nn.Parameter(torch.tensor(float(init_alpha)))
        self.positive_bias = nn.Parameter(torch.tensor(float(positive_bias_init)))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        s = self.ovr(z).view(-1) + self.positive_bias
        residual = torch.stack([-s, s], dim=1)
        alpha = torch.sigmoid(self.alpha_raw)
        return alpha * residual, {
            "positive_ovr_score": s.detach(),
            "positive_ovr_alpha": alpha.detach(),
        }


class PrototypeAnchoredEvidenceBranch(nn.Module):
    """Small class-anchor residual logits in representation space."""

    def __init__(self, embed_dim: int, num_classes: int = 2, init_alpha: float = -4.0):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(int(num_classes), int(embed_dim)) * 0.02)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.alpha_raw = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z_norm = F.normalize(z, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        proto_logits = torch.matmul(z_norm, p_norm.t()) * F.softplus(self.logit_scale)
        alpha = torch.sigmoid(self.alpha_raw)
        return alpha * proto_logits, {
            "prototype_logits": proto_logits.detach(),
            "prototype_alpha": alpha.detach(),
        }


class PositiveAwareResidualEvidenceFusionHead(nn.Module):
    """Final CE-trained head combining nonlinear, linear, positive-aware and prototype evidence."""

    def __init__(
        self,
        embed_dim: int,
        num_features: int,
        num_classes: int = 2,
        tabm_num_members: int = 4,
        dropout: float = 0.2,
        linear_dropout: float = 0.1,
        use_linear_residual: bool = True,
        use_positive_ovr: bool = True,
        use_prototype_residual: bool = True,
    ):
        super().__init__()
        if int(num_classes) != 2:
            raise ValueError("PositiveAwareResidualEvidenceFusionHead is intended for num_classes=2.")
        self.member_head = EnsembleContextualEvidenceBranch(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_members=tabm_num_members,
            dropout=dropout,
        )
        self.linear_residual = LinearCpGEvidenceBranch(
            num_features=num_features,
            num_classes=num_classes,
            dropout=linear_dropout,
            init_alpha=-2.0,
        ) if use_linear_residual else None
        self.positive_ovr = PositiveEvidenceAmplificationBranch(
            embed_dim=embed_dim,
            dropout=dropout,
            init_alpha=-3.0,
            positive_bias_init=0.0,
        ) if use_positive_ovr else None
        self.prototype = PrototypeAnchoredEvidenceBranch(
            embed_dim=embed_dim,
            num_classes=num_classes,
            init_alpha=-4.0,
        ) if use_prototype_residual else None

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits, aux = self.member_head(z)
        if self.linear_residual is not None:
            res, res_aux = self.linear_residual(x)
            logits = logits + res
            aux.update(res_aux)
        if self.positive_ovr is not None:
            res, res_aux = self.positive_ovr(z)
            logits = logits + res
            aux.update(res_aux)
        if self.prototype is not None:
            res, res_aux = self.prototype(z)
            logits = logits + res
            aux.update(res_aux)
        return logits, aux


class PREF_FT(nn.Module):
    """Positive-aware CE-only binary FT model."""

    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        d_model: int = 128,
        n_transformer_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        tabm_num_members: int = 4,
        use_linear_residual: bool = True,
        gate_dropout: float = 0.05,
        linear_dropout: float = 0.1,
        use_positive_ovr: bool = True,
        use_prototype_residual: bool = True,
    ):
        super().__init__()
        if int(num_classes) != 2:
            raise ValueError(f"PREF_FT only supports num_classes=2, got {num_classes}.")
        self.encoder = MethylationEvidenceEncoder(
            num_features=int(num_features),
            d_model=int(d_model),
            num_layers=int(n_transformer_layers),
            num_heads=int(num_heads),
            dropout=float(dropout),
            gate_dropout=float(gate_dropout),
        )
        self.head = PositiveAwareResidualEvidenceFusionHead(
            embed_dim=int(d_model),
            num_features=int(num_features),
            num_classes=2,
            tabm_num_members=int(tabm_num_members),
            dropout=float(dropout),
            linear_dropout=float(linear_dropout),
            use_linear_residual=bool(use_linear_residual),
            use_positive_ovr=bool(use_positive_ovr),
            use_prototype_residual=bool(use_prototype_residual),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z, aux = self.encoder(x)
        logits, head_aux = self.head(z, x)
        aux.update(head_aux)
        return logits, aux



# Paper abbreviations and legacy imports; aliases do not register extra parameters.
CERM = CpGEvidenceRecalibrationModule
CCEM = ContextualCpGEncodingModule
REPM = ResidualEvidencePoolingModule
SMEM = StableMethylationEmbeddingModule
PREFHead = PositiveAwareResidualEvidenceFusionHead
CpGStabilityGate = CpGEvidenceRecalibrationModule
CLSPlusAttentionPooling = ResidualEvidencePoolingModule
StableFTEncoder = MethylationEvidenceEncoder
MultiMemberHead = EnsembleContextualEvidenceBranch
ElasticLinearResidual = LinearCpGEvidenceBranch
PositiveOVRResidual = PositiveEvidenceAmplificationBranch
PrototypeResidualHead = PrototypeAnchoredEvidenceBranch
PositiveAwareBinaryHead = PositiveAwareResidualEvidenceFusionHead
BinaryPositiveAwareFT = PREF_FT
model = PREF_FT
