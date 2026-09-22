import torch

from NeuralOperators import LocalHighPassBlock2d


def test_identity_disables_only_activation_and_keeps_pointwise_gradient():
    torch.manual_seed(17)
    block = LocalHighPassBlock2d(4, activation="identity")
    x = torch.randn(2, 4, 12, 10, requires_grad=True)
    output = block(x)
    output.square().mean().backward()
    assert output.shape == x.shape
    assert block.pointwise.weight.grad is not None
    assert torch.count_nonzero(block.pointwise.weight.grad).item() > 0


def test_activation_arms_have_identical_parameters_but_different_forward_maps():
    torch.manual_seed(23)
    identity = LocalHighPassBlock2d(3, activation="identity")
    gelu = LocalHighPassBlock2d(3, activation="gelu")
    gelu.load_state_dict(identity.state_dict())
    x = torch.randn(2, 3, 9, 11)
    assert sum(parameter.numel() for parameter in identity.parameters()) == sum(
        parameter.numel() for parameter in gelu.parameters()
    )
    assert not torch.allclose(identity(x), gelu(x))
