import math

import numpy as np

from hofa_policy.features import reference_features, state_features


def make_reference(**updates):
    row = dict(x_ned=0.0, y_ned=0.0, yaw_ned=0.0,
               dx_ned=0.0, dy_ned=0.0, dpsi_ned=0.0,
               ddx_ned=0.0, ddy_ned=0.0, ddpsi_ned=0.0)
    row.update(updates)
    return row


def test_reference_ahead_maps_to_body_x():
    odom = dict(x=1.0, y=2.0, yaw=0.0, u=0.2, v=-0.1, r=0.03)
    rows = [make_reference(x_ned=2.0, y_ned=2.0) for _ in range(20)]
    features = reference_features(odom, rows)
    np.testing.assert_allclose(features[:, 0], 1.0)
    np.testing.assert_allclose(features[:, 1], 0.0)
    np.testing.assert_allclose(state_features(odom), [0.2, -0.1, 0.03])


def test_reference_rotation_uses_current_body_frame():
    odom = dict(x=0.0, y=0.0, yaw=math.pi / 2, u=0.0, v=0.0, r=0.0)
    rows = [make_reference(x_ned=0.0, y_ned=1.0, yaw_ned=math.pi / 2)
            for _ in range(20)]
    features = reference_features(odom, rows)
    np.testing.assert_allclose(features[:, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(features[:, 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(features[:, 2], 0.0, atol=1e-6)
    np.testing.assert_allclose(features[:, 3], 1.0, atol=1e-6)
