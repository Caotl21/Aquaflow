# Bearing-only 2-D numerical environment

This package is the first, ROS-free stage of the BricsBot bearing-only policy.
It is a planar closed-loop surrogate, not an identified Stonefish model.  A
policy submits a local `Trajectory` in the current body frame.  The environment
executes it through a first-order, acceleration-limited tracker, updates the
robot pose, and then generates the next bearing/angle-width observation from
that executed pose.

The policy observation contains only:

- fixed-size obstacle tokens `[sin(beta), cos(beta), beta_left, beta_right,
  half_width, confidence, fov_truncated]`;
- detection masks, ideal simulator association IDs, capture age and causal
  history masks;
- relative self-motion/velocity and goal direction.

Obstacle centres, radii, distances, collision clearance and the true state are
kept in the `info` dictionary for labels and evaluation.  They are not part of
the policy observation.

The circular footprint is inflated into the collision test.  The default robot
radius is 0.365 m, matching the existing AquaFlow teacher envelope; this is a
conservative geometric default, not a Stonefish identification result.  The
default camera FOV is 80 degrees and the camera is 0.20 m ahead of the body
origin, matching the current BricsBot scenario.

Install for local development:

```bash
pip install -e src/bearing_sim
pytest -q src/bearing_sim/tests
```

The environment supports an `ideal_tracking` configuration for an upper-bound
ablation.  Formal training/evaluation should use the non-ideal tracker and
randomise delays, dropout, bearing noise, command saturation and current.
