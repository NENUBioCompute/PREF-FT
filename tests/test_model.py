import pytest
import torch

from pref_ft_project import PREF_FT, CERM, CCEM, REPM, SMEM
from pref_ft_shap_project.pref_ft_model import PREF_FT as ExplanationModel


def make_model(**options):
    return PREF_FT(num_features=10, d_model=16, n_transformer_layers=1,
                   num_heads=4, **options)


def test_training_and_explanation_share_one_model():
    assert PREF_FT is ExplanationModel


def test_paper_modules_and_four_evidence_sum():
    model = make_model().eval()
    assert isinstance(model.encoder.feature_gate, CERM)
    assert isinstance(model.encoder, CCEM)
    assert isinstance(model.encoder.pool, REPM)
    assert isinstance(model.encoder.stable_bottleneck, SMEM)
    components, linear_inputs = [], []
    hooks = []
    for branch in [model.head.member_head, model.head.linear_residual,
                   model.head.positive_ovr, model.head.prototype]:
        hooks.append(branch.register_forward_hook(lambda module, inputs, result: components.append(result[0])))
    hooks.append(model.head.linear_residual.register_forward_pre_hook(
        lambda module, inputs: linear_inputs.append(inputs[0])))
    x = torch.randn(4, 10)
    logits, auxiliary = model(x)
    for hook in hooks:
        hook.remove()
    assert logits.shape == (4, 2)
    torch.testing.assert_close(logits, sum(components), rtol=0, atol=0)
    assert linear_inputs[0] is x
    torch.testing.assert_close(auxiliary["token_pool_attention"].sum(1), torch.ones(4))


def test_final_cross_entropy_reaches_all_parameters():
    model = make_model()
    logits, _ = model(torch.randn(4, 10))
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1])).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_single_member_diagnostic_is_finite():
    _, auxiliary = make_model(tabm_num_members=1)(torch.randn(2, 10))
    assert auxiliary["member_logit_std"].item() == 0


def test_binary_task_and_checkpoint_contract():
    with pytest.raises(ValueError):
        PREF_FT(num_features=10, num_classes=3)
    model = make_model()
    restored = make_model()
    restored.load_state_dict(model.state_dict(), strict=True)
    keys = model.state_dict()
    assert "encoder.tokenizer.weight" in keys
    assert "encoder.stable_bottleneck.1.weight" in keys
    assert "head.linear_residual.linear.weight" in keys
