import numpy as np

from hofa_policy.shadow_core import ShadowPolicy


def test_shadow_checkpoint_predicts_finite_bounded_outputs():
    checkpoint = (
        "results/hofa_policy/train_v1_epoch500/best.pt")
    policy = ShadowPolicy(checkpoint, device="cpu", torch_threads=1)
    state = np.zeros(3, dtype=np.float32)
    reference = np.zeros((20, 10), dtype=np.float32)
    reference[:, 3] = 1.0
    wrench, force, normalized = policy.predict(
        state, reference, np.zeros(3, dtype=np.float32))
    assert wrench.shape == (3,)
    assert force.shape == (4,)
    assert normalized.shape == (4,)
    assert np.all(np.isfinite(wrench))
    assert np.all(np.abs(normalized) <= 1.0)

