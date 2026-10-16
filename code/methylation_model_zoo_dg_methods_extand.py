# ============================================================
# Enhanced DANN + FT-Transformer wrapper for DNA methylation
# ============================================================
# This file is generated as a drop-in model-zoo extension.
# It delegates all original model names to the original build_model(),
# and only overrides the new enhanced FT/DANN-FT names.

from typing import Dict, Tuple, Optional
from pref_ft_project.pref_ft_model import PREF_FT
import torch
import torch.nn as nn
import torch.nn.functional as F

from methylation_advanced_tabular_models import (
    FeatureTokenizer,
    TransformerEncoderBlock,
    HierarchicalPredictionHead,
    grad_reverse
    
)
from methylation_model_zoo_dg_methods import build_model as _original_build_model

_ENHANCED_NAMES = {
    "enhanced_dann_ft_transformer",
    "dann_ft_transformer_plus",
    "dann_ft_plus",
    "stable_dann_ft_transformer",
    "stable_dann_ft",
    "enhanced_ft_transformer",
    "ft_transformer_plus",
    "stable_ft_transformer",
}


class CpGStabilityGate(nn.Module):
    """
    Learnable CpG-wise stability gate.

    Rationale
    ---------
    Cross-cohort DNAm classification is sensitive to unstable CpGs.
    This module learns a soft feature reliability gate but is initialized
    close to identity, so the enhanced model starts very close to the
    original FT-Transformer baseline.
    """
    def __init__(self, num_features: int, init_logit: float = 4.0, gate_dropout: float = 0.0):
        super().__init__()
        self.gate_logits = nn.Parameter(torch.full((int(num_features),), float(init_logit)))
        self.gate_dropout = float(gate_dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        gate = torch.sigmoid(self.gate_logits).view(1, -1)
        if self.training and self.gate_dropout > 0:
            # Drop feature gates rather than raw features; expectation is kept by scaling.
            gate = F.dropout(gate, p=self.gate_dropout, training=True)
        x_gated = x * gate
        aux = {
            "feature_gate": gate.squeeze(0),
            "feature_gate_mean": gate.mean().detach(),
            "feature_gate_l1": torch.mean(torch.abs(gate.squeeze(0))),
        }
        return x_gated, aux


class CLSPlusAttentionPooling(nn.Module):
    """
    Add an attention-pooled CpG-token summary to the [CLS] representation.

    The original FT-Transformer usually uses only the [CLS] token. In small-n,
    high-p methylation data, a weak residual global summary often stabilizes
    class ranking. The residual coefficient is initialized near zero to preserve
    the current baseline at the start of training.
    """
    def __init__(self, d_model: int, dropout: float = 0.2, init_alpha: float = -3.0):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(32, d_model // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, d_model // 2), 1),
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
        aux = {
            "token_pool_alpha": alpha.detach(),
            "token_pool_attention": attn.detach(),
            "token_pool_embedding": pooled.detach(),
        }
        return z, aux


class EnhancedFTTransformerEncoder(nn.Module):
    """
    FT-Transformer encoder with two conservative additions:
    1) CpGStabilityGate before tokenization;
    2) CLSPlusAttentionPooling after Transformer blocks.
    """
    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        gate_dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_gate = CpGStabilityGate(num_features=num_features, init_logit=4.0, gate_dropout=gate_dropout)
        self.tokenizer = FeatureTokenizer(num_features=num_features, token_dim=d_model, dropout=dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.pool = CLSPlusAttentionPooling(d_model=d_model, dropout=dropout, init_alpha=-3.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        x_gated, gate_aux = self.feature_gate(x)
        tokens = self.tokenizer(x_gated)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)
        z, pool_aux = self.pool(tokens)
        aux = {
            "tokens": tokens,
            "cpg_tokens": tokens[:, 1:],
            "x_gated": x_gated,
        }
        aux.update(gate_aux)
        aux.update(pool_aux)
        return z, aux


class PrototypeResidualHead(nn.Module):
    """
    Learnable disease prototypes in representation space.

    It creates a small residual logit from cosine similarity to class prototypes.
    The residual weight is near zero initially, making this a safe enhancement.
    It is particularly useful when the minority class, e.g. AD, needs a stable
    class anchor across cohorts.
    """
    def __init__(self, embed_dim: int, num_classes: int, init_alpha: float = -4.0):
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
            "prototype_logits": proto_logits,
            "prototype_alpha": alpha.detach(),
        }


class HierarchicalLogitRefiner(nn.Module):
    """
    Convert HC-vs-disease and AD-vs-MCI auxiliary predictions into a conservative
    residual 3-class logit. It only activates when the hierarchical auxiliary
    heads are available. This makes the auxiliary heads useful at inference time
    without replacing the original flat classifier.
    """
    def __init__(
        self,
        num_classes: int,
        idx_hc: Optional[int] = None,
        idx_mci: Optional[int] = None,
        idx_ad: Optional[int] = None,
        init_alpha: float = -4.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.idx_hc = idx_hc
        self.idx_mci = idx_mci
        self.idx_ad = idx_ad
        self.alpha_raw = nn.Parameter(torch.tensor(float(init_alpha)))
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, aux: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if (
            self.num_classes != 3
            or self.idx_hc is None
            or self.idx_mci is None
            or self.idx_ad is None
            or "hc_vs_disease_logits" not in aux
            or "ad_vs_mci_logits" not in aux
        ):
            return logits, {"hier_refine_alpha": torch.tensor(0.0, device=logits.device)}

        p_dis = torch.sigmoid(aux["hc_vs_disease_logits"].view(-1))
        p_ad = torch.sigmoid(aux["ad_vs_mci_logits"].view(-1))
        probs = torch.zeros_like(logits)
        probs[:, self.idx_hc] = 1.0 - p_dis
        probs[:, self.idx_mci] = p_dis * (1.0 - p_ad)
        probs[:, self.idx_ad] = p_dis * p_ad
        hier_logits = torch.log(probs.clamp_min(self.eps))
        alpha = torch.sigmoid(self.alpha_raw)
        refined = logits + alpha * hier_logits
        return refined, {
            "hier_refined_logits": hier_logits,
            "hier_refine_alpha": alpha.detach(),
        }


class ConditionalDomainAdversary(nn.Module):
    """
    CDAN-lite domain adversary.

    Standard DANN aligns only marginal z distributions. For AD/CN/MCI data,
    disease prevalence often differs across cohorts, so class-conditional
    domain alignment is safer. This head conditions the domain classifier on
    both representation z and predicted class probabilities.
    """
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_domains: int,
        dropout: float = 0.2,
        detach_prob: bool = True,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.detach_prob = bool(detach_prob)
        in_dim = embed_dim * (self.num_classes + 1)
        hidden = max(64, embed_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, int(num_domains)),
        )

    def forward(self, z: torch.Tensor, probs: torch.Tensor, grl_lambda: float) -> torch.Tensor:
        p = probs.detach() if self.detach_prob else probs
        # Outer product p(y|x) ⊗ z, flattened. Shape: [B, K*D]
        outer = torch.bmm(p.unsqueeze(2), z.unsqueeze(1)).flatten(1)
        cond = torch.cat([z, outer], dim=1)
        return self.net(grad_reverse(cond, grl_lambda))


class EnhancedDANNFTTransformerClassifier(nn.Module):
    """
    Enhanced DANN-FT model.

    Compared with the original DANNFTTransformerClassifier, this model adds:
    - CpG stability gate;
    - residual token attention pooling;
    - prototype residual logits;
    - hierarchical logit refinement;
    - class-conditional domain adversary.

    All residual coefficients are initialized close to zero, so this is a
    conservative upgrade rather than a disruptive architecture replacement.
    """
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_domains: int = 2,
        grl_lambda: float = 0.1,
        idx_ad: Optional[int] = None,
        idx_mci: Optional[int] = None,
        idx_hc: Optional[int] = None,
        use_linear_residual: bool = True,
        gate_dropout: float = 0.0,
        use_conditional_domain: bool = True,
    ):
        super().__init__()
        self.grl_lambda = float(grl_lambda)
        self.num_classes = int(num_classes)
        self.encoder = EnhancedFTTransformerEncoder(
            num_features=num_features,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            gate_dropout=gate_dropout,
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
        self.prototype_head = PrototypeResidualHead(embed_dim=d_model, num_classes=num_classes, init_alpha=-4.0)
        self.hier_refiner = HierarchicalLogitRefiner(
            num_classes=num_classes,
            idx_hc=idx_hc,
            idx_mci=idx_mci,
            idx_ad=idx_ad,
            init_alpha=-4.0,
        )
        self.use_conditional_domain = bool(use_conditional_domain)
        if self.use_conditional_domain:
            self.domain_head = ConditionalDomainAdversary(
                embed_dim=d_model,
                num_classes=num_classes,
                num_domains=num_domains,
                dropout=dropout,
                detach_prob=True,
            )
        else:
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

        proto_residual, proto_aux = self.prototype_head(z)
        logits = logits + proto_residual
        aux.update(proto_aux)

        logits, hier_aux = self.hier_refiner(logits, aux)
        aux.update(hier_aux)

        probs = torch.softmax(logits, dim=1)
        if self.use_conditional_domain:
            aux["domain_logits"] = self.domain_head(z, probs, self.grl_lambda)
        else:
            aux["domain_logits"] = self.domain_head(grad_reverse(z, self.grl_lambda))
        aux["shared_embedding"] = z
        return logits, aux

class EnhancedDANNFTTransformerClassifier_ce(nn.Module):
    """
    Enhanced DANN-FT model.

    Compared with the original DANNFTTransformerClassifier, this model adds:
    - CpG stability gate;
    - residual token attention pooling;
    - prototype residual logits;
    - hierarchical logit refinement;
    - class-conditional domain adversary.

    All residual coefficients are initialized close to zero, so this is a
    conservative upgrade rather than a disruptive architecture replacement.
    """
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_domains: int = 2,
        grl_lambda: float = 0.1,
        idx_ad: Optional[int] = None,
        idx_mci: Optional[int] = None,
        idx_hc: Optional[int] = None,
        use_linear_residual: bool = True,
        gate_dropout: float = 0.0,
        use_conditional_domain: bool = True,
    ):
        super().__init__()
        self.grl_lambda = float(grl_lambda)
        self.num_classes = int(num_classes)
        self.encoder = EnhancedFTTransformerEncoder(
            num_features=num_features,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            gate_dropout=gate_dropout,
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
        self.prototype_head = PrototypeResidualHead(embed_dim=d_model, num_classes=num_classes, init_alpha=-4.0)
        self.hier_refiner = HierarchicalLogitRefiner(
            num_classes=num_classes,
            idx_hc=idx_hc,
            idx_mci=idx_mci,
            idx_ad=idx_ad,
            init_alpha=-4.0,
        )
        self.use_conditional_domain = bool(use_conditional_domain)
        if self.use_conditional_domain:
            self.domain_head = ConditionalDomainAdversary(
                embed_dim=d_model,
                num_classes=num_classes,
                num_domains=num_domains,
                dropout=dropout,
                detach_prob=True,
            )
        else:
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

        proto_residual, proto_aux = self.prototype_head(z)
        logits = logits + proto_residual
        aux.update(proto_aux)

        logits, hier_aux = self.hier_refiner(logits, aux)
        aux.update(hier_aux)

        probs = torch.softmax(logits, dim=1)
        if self.use_conditional_domain:
            aux["domain_logits"] = self.domain_head(z, probs, self.grl_lambda)
        else:
            aux["domain_logits"] = self.domain_head(grad_reverse(z, self.grl_lambda))
        aux["shared_embedding"] = z
        return logits, aux


class CpGStabilityGate(nn.Module):
    """Elastic-net-inspired soft CpG reliability gate, initialized close to 1."""

    def __init__(self, num_features: int, init_logit: float = 4.0, gate_dropout: float = 0.0):
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


class CLSPlusAttentionPooling(nn.Module):
    """CLS token plus residual attention pooling over CpG tokens."""

    def __init__(self, d_model: int, dropout: float = 0.2, init_alpha: float = -2.5):
        super().__init__()
        hidden = max(32, d_model // 2)
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


class StableFTEncoder(nn.Module):
    """DANN-FT-inspired stable shared representation, without requiring domain loss."""

    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        gate_dropout: float = 0.05,
    ):
        super().__init__()
        self.feature_gate = CpGStabilityGate(num_features, init_logit=4.0, gate_dropout=gate_dropout)
        self.tokenizer = FeatureTokenizer(num_features=num_features, token_dim=d_model, dropout=dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(int(num_layers))
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.pool = CLSPlusAttentionPooling(d_model=d_model, dropout=dropout, init_alpha=-2.5)
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
            "features": z,
            "shared_embedding": z,
            "x_gated": x_gated,
            "z_raw": z_raw.detach(),
        }
        aux.update(gate_aux)
        aux.update(pool_aux)
        return z, aux


class ElasticLinearResidual(nn.Module):
    """Direct CpG-to-logit path, analogous to elastic-net logistic evidence."""

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


class MultiMemberHead(nn.Module):
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
        member_logits = torch.stack([head(z) for head in self.members], dim=1)  # [B, M, K]
        logits = member_logits.mean(dim=1)
        return logits, {
            "member_logits": member_logits.detach(),
            "member_logit_std": member_logits.std(dim=1).mean().detach(),
        }


class PositiveOVRResidual(nn.Module):
    """Positive-class one-vs-rest residual. For class-order 'unaffected, affected', positive id is 1."""

    def __init__(self, embed_dim: int, dropout: float = 0.2, init_alpha: float = -3.0, positive_bias_init: float = 0.0):
        super().__init__()
        self.ovr = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, max(32, embed_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, embed_dim // 2), 1),
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


class PrototypeResidualHead(nn.Module):
    """Class-anchor residual logits in representation space."""

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


class PositiveAwareBinaryHead(nn.Module):
    """Final CE-trained head combining nonlinear, linear, ensemble, and positive-aware evidence."""

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
            raise ValueError("PositiveAwareBinaryHead is intended for num_classes=2.")
        self.member_head = MultiMemberHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_members=tabm_num_members,
            dropout=dropout,
        )
        self.linear_residual = ElasticLinearResidual(
            num_features=num_features,
            num_classes=num_classes,
            dropout=linear_dropout,
            init_alpha=-2.0,
        ) if use_linear_residual else None
        self.positive_ovr = PositiveOVRResidual(
            embed_dim=embed_dim,
            dropout=dropout,
            init_alpha=-3.0,
            positive_bias_init=0.0,
        ) if use_positive_ovr else None
        self.prototype = PrototypeResidualHead(
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


EnhancedDANNFTTransformerClassifier_positive_aware = PREF_FT

EnhancedDANNFTTransformerClassifier_positive_aware_2 = PREF_FT



class EnhancedFTTransformerClassifier(nn.Module):
    """Non-adversarial version using the same conservative encoder/head upgrades."""
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
        gate_dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = EnhancedFTTransformerEncoder(
            num_features=num_features,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            gate_dropout=gate_dropout,
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
        self.prototype_head = PrototypeResidualHead(embed_dim=d_model, num_classes=num_classes, init_alpha=-4.0)
        self.hier_refiner = HierarchicalLogitRefiner(
            num_classes=num_classes,
            idx_hc=idx_hc,
            idx_mci=idx_mci,
            idx_ad=idx_ad,
            init_alpha=-4.0,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        z, aux = self.encoder(x)
        logits, aux = self.pred_head(z, x, aux)
        proto_residual, proto_aux = self.prototype_head(z)
        logits = logits + proto_residual
        aux.update(proto_aux)
        logits, hier_aux = self.hier_refiner(logits, aux)
        aux.update(hier_aux)
        aux["shared_embedding"] = z
        return logits, aux


def build_model(
    model_name: str,
    num_features: int,
    num_classes: int,
    d_model: int = 128,
    dropout: float = 0.2,
    patch_scales=None,
    n_mamba_layers: int = 2,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    hierarchical_class_indices: Optional[Dict[str, int]] = None,
    num_domains: int = 2,
    n_transformer_layers: int = 3,
    num_heads: int = 4,
    num_prototypes_per_class: int = 4,
    tabm_num_members: int = 4,
    grl_lambda: float = 0.1,
    use_linear_residual: bool = True,
    tabr_k_neighbors: int = 8,
    **kwargs,
):
    name = str(model_name).strip().lower()
    if name in {"pref_ft", "pref-ft", "binary_positive_aware_ft", "binary_positive_aware_ft_2"}:
        return PREF_FT(
            num_features=num_features, num_classes=num_classes, d_model=d_model,
            n_transformer_layers=n_transformer_layers, num_heads=num_heads,
            dropout=dropout, tabm_num_members=tabm_num_members,
            use_linear_residual=use_linear_residual,
            gate_dropout=float(kwargs.get("gate_dropout", 0.05)),
            linear_dropout=float(kwargs.get("linear_dropout", 0.1)),
            use_positive_ovr=bool(kwargs.get("use_positive_ovr", True)),
            use_prototype_residual=bool(kwargs.get("use_prototype_residual", True)),
        )
    idx_ad = idx_mci = idx_hc = None
    if hierarchical_class_indices is not None:
        idx_ad = hierarchical_class_indices.get("idx_ad")
        idx_mci = hierarchical_class_indices.get("idx_mci")
        idx_hc = hierarchical_class_indices.get("idx_hc")

    if name in {"enhanced_dann_ft_transformer", "dann_ft_transformer_plus", "dann_ft_plus", "stable_dann_ft_transformer", "stable_dann_ft"}:
        return EnhancedDANNFTTransformerClassifier(
            num_features=num_features,
            num_classes=num_classes,
            d_model=d_model,
            num_layers=n_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_domains=num_domains,
            grl_lambda=grl_lambda,
            idx_ad=idx_ad,
            idx_mci=idx_mci,
            idx_hc=idx_hc,
            use_linear_residual=use_linear_residual,
            gate_dropout=float(kwargs.get("gate_dropout", 0.0)),
            use_conditional_domain=bool(kwargs.get("use_conditional_domain", True)),
        )

    if name in {"enhanced_ft_transformer", "ft_transformer_plus", "stable_ft_transformer"}:
        return EnhancedFTTransformerClassifier(
            num_features=num_features,
            num_classes=num_classes,
            d_model=d_model,
            num_layers=n_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            idx_ad=idx_ad,
            idx_mci=idx_mci,
            idx_hc=idx_hc,
            use_linear_residual=use_linear_residual,
            gate_dropout=float(kwargs.get("gate_dropout", 0.0)),
        )



    # Delegate all original models to the original model zoo.
    return _original_build_model(
        model_name=model_name,
        num_features=num_features,
        num_classes=num_classes,
        d_model=d_model,
        dropout=dropout,
        patch_scales=patch_scales,
        n_mamba_layers=n_mamba_layers,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        hierarchical_class_indices=hierarchical_class_indices,
        num_domains=num_domains,
        n_transformer_layers=n_transformer_layers,
        num_heads=num_heads,
        num_prototypes_per_class=num_prototypes_per_class,
        tabm_num_members=tabm_num_members,
        grl_lambda=grl_lambda,
        use_linear_residual=use_linear_residual,
        tabr_k_neighbors=tabr_k_neighbors,
        **kwargs,
    )
