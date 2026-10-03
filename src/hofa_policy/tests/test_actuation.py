import numpy as np

from hofa_policy.actuation import ThrusterGeometry


def test_bricsbot_geometry_is_planar_full_rank():
    geometry = ThrusterGeometry.bricsbot()
    assert geometry.allocation_matrix.shape == (3, 4)
    assert np.linalg.matrix_rank(geometry.allocation_matrix) == 3


def test_normalized_action_respects_force_limits():
    geometry = ThrusterGeometry.bricsbot()
    actions = np.asarray([[-2.0, -1.0, 0.0, 2.0],
                          [1.0, 0.5, -0.5, -1.0]])
    force = geometry.force_from_normalized(actions)
    assert np.all(force >= geometry.force_min)
    assert np.all(force <= geometry.force_max)


def test_synthesized_wrench_matches_allocation_matrix():
    geometry = ThrusterGeometry.bricsbot()
    action = np.asarray([0.2, -0.4, 0.6, -0.8])
    force = geometry.force_from_normalized(action)
    np.testing.assert_allclose(
        geometry.wrench_from_normalized(action),
        geometry.allocation_matrix @ force)


def test_midpoint_has_zero_horizontal_wrench_for_symmetric_layout():
    geometry = ThrusterGeometry.bricsbot()
    np.testing.assert_allclose(geometry.wrench_center, np.zeros(3), atol=1e-10)
