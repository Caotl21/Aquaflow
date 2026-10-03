import torch

from hofa_policy.model import HofaPolicyNet
from hofa_policy.train import overfit_indices


def test_model_outputs_bounded_forces_and_finite_wrench():
    torch.manual_seed(3)
    model = HofaPolicyNet()
    state = torch.randn(4, 3)
    reference = torch.randn(4, 20, 10)
    previous = torch.randn(4, 3)
    wrench, force, normalized = model(state, reference, previous)
    assert wrench.shape == (4, 3)
    assert force.shape == (4, 4)
    assert normalized.shape == (4, 4)
    assert torch.isfinite(wrench).all()
    assert torch.all(normalized <= 1.0)
    assert torch.all(normalized >= -1.0)
    assert torch.all(force <= model.force_mid + model.force_half + 1e-6)
    assert torch.all(force >= model.force_mid - model.force_half - 1e-6)


def test_model_backward_reaches_reference_and_state_encoders():
    model = HofaPolicyNet()
    state = torch.randn(2, 3, requires_grad=True)
    reference = torch.randn(2, 20, 10, requires_grad=True)
    previous = torch.randn(2, 3, requires_grad=True)
    wrench, _, _ = model(state, reference, previous)
    wrench.square().mean().backward()
    assert state.grad is not None and state.grad.abs().sum() > 0
    assert reference.grad is not None and reference.grad.abs().sum() > 0
    assert previous.grad is not None and previous.grad.abs().sum() > 0


def test_overfit_subset_is_deterministic_and_spans_run():
    indices = overfit_indices(576, 128)
    assert len(indices) == 128
    assert indices[0] == 0
    assert indices[-1] == 575
    assert indices == sorted(set(indices))
