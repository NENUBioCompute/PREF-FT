"""Shared PREF-FT model exports for the explanation workflow."""
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from pref_ft_project.pref_ft_model import (
    PREF_FT,
    CpGEvidenceRecalibrationModule,
    ContextualCpGEncodingModule,
    ResidualEvidencePoolingModule,
    StableMethylationEmbeddingModule,
    PositiveAwareResidualEvidenceFusionHead,
    EnsembleContextualEvidenceBranch,
    LinearCpGEvidenceBranch,
    PositiveEvidenceAmplificationBranch,
    PrototypeAnchoredEvidenceBranch,
    CERM,
    CCEM,
    REPM,
    SMEM,
    PREFHead,
    BinaryPositiveAwareFT,
    model
)

__all__ = ['PREF_FT', 'CpGEvidenceRecalibrationModule', 'ContextualCpGEncodingModule', 'ResidualEvidencePoolingModule', 'StableMethylationEmbeddingModule', 'PositiveAwareResidualEvidenceFusionHead', 'EnsembleContextualEvidenceBranch', 'LinearCpGEvidenceBranch', 'PositiveEvidenceAmplificationBranch', 'PrototypeAnchoredEvidenceBranch', 'CERM', 'CCEM', 'REPM', 'SMEM', 'PREFHead', 'BinaryPositiveAwareFT', 'model']
