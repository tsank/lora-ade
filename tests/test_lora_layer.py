import pytest
import torch
import torch.nn as nn

from lora_ade.lora_layer import LoRALinear


@pytest.fixture
def setup():
    torch.manual_seed(0)
    base = nn.Linear(64, 64)
    x = torch.randn(4, 64)
    expected = base(x).detach()
    return LoRALinear(base, r=8, alpha=16), x, expected


def test_same_output_at_start(setup):
    layer, x, expected = setup
    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_only_a_and_b_are_trainable(setup):
    layer, _, _ = setup
    names = [n for n, p in layer.named_parameters() if p.requires_grad]
    assert names == ["A", "B"]


def test_trainable_parameter_count(setup):
    layer, _, _ = setup
    trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    assert trainable == 8 * 64 * 2


def test_gradients_reach_adapter_only(setup):
    layer, x, _ = setup
    layer(x).sum().backward()
    assert layer.A.grad is not None
    assert layer.B.grad is not None
    assert layer.base.weight.grad is None


def test_merge_matches_lora_output(setup):
    layer, x, _ = setup
    with torch.no_grad():
        layer.B.normal_(0, 0.02)
    out_lora = layer(x).detach()
    merged = layer.merge()
    assert torch.allclose(merged(x), out_lora, atol=1e-4)
