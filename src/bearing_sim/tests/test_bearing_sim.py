import math
import numpy as np
import pytest

from bearing_sim import BearingEnv, Circle, Config, State
from bearing_sim.geometry import camera_tokens, clearance, first_contact, swept_clearance
from bearing_sim.trajectory import Trajectory


def test_circle_is_visible_with_angular_extent_and_camera_offset():
    cfg = Config(bounds=(-20,20,-20,20), rays=401, dropout=0, bearing_noise=0)
    tokens, mask, ids = camera_tokens(np.array([0.,0.,0.]), [Circle(2,1,.2)],
                                      cfg.bounds, cfg, np.random.default_rng(1))
    assert mask[0]
    assert ids[0] == 0
    assert tokens[0, 0] > 0  # centre bearing is to the left
    assert tokens[0, 4] > 0  # non-zero half angular width


def test_footprint_and_swept_collision():
    circle = Circle(2, 0, .2)
    assert clearance((1,0), [circle], .365, (-5,5,-5,5)) == pytest.approx(.435)
    assert clearance((1.5,0), [circle], .365, (-5,5,-5,5)) < 0
    assert swept_clearance((0,2),(3,2),[circle],.365,(-5,5,-5,5)) > 0
    assert first_contact((0,0),(3,0),[circle],.365,(-5,5,-5,5)) < 1


def test_trajectory_has_local_origin_and_time_derivative():
    trajectory = Trajectory.constant_velocity([.2, .1, .05], dt=.1, count=4)
    pose, velocity = trajectory.sample(.15, np.array([1.,2.,.3]))
    assert np.allclose(velocity, [.2*math.cos(.3)-.1*math.sin(.3),
                                  .2*math.sin(.3)+.1*math.cos(.3), .05])
    assert pose[0] > 1 and pose[1] > 2


def test_environment_tracks_after_command_delay_and_returns_causal_history():
    cfg = Config(bounds=(-20,20,-20,20), sensor_delay=.1, command_delay=.1,
                 dropout=0, bearing_noise=0, history=4, rays=101)
    env = BearingEnv([], goal=(5,0), config=cfg)
    initial, _ = env.reset()
    assert not initial["frame_mask"].all()
    action = Trajectory.constant_velocity([.2,0,0], dt=.1, count=20)
    obs, _, terminated, truncated, info = env.step(action)
    assert not terminated and not truncated
    assert info["state"][0] == 0
    assert obs["tokens"].shape == (4,8,7)
    assert np.all(obs["age"][obs["frame_mask"]] >= cfg.sensor_delay-1e-6)
    _, _, _, _, info = env.step(None)
    assert 0 < info["state"][3] < .2


def test_environment_rejects_unsafe_reference_and_ends_on_footprint_contact():
    cfg = Config(bounds=(-5,5,-5,5), sensor_delay=0, command_delay=0,
                 dropout=0, bearing_noise=0, rays=101, max_time=8)
    env = BearingEnv([Circle(1.0,0,.2)], goal=(4,0), config=cfg)
    with pytest.raises(ValueError):
        env.step(Trajectory.constant_velocity([1.0,0,0], dt=.1, count=20))
    # A safe-speed plan still collides because the circular footprint is inflated.
    plan = Trajectory.constant_velocity([.2,0,0], dt=.1, count=20)
    terminated = False
    for _ in range(80):
        _, _, terminated, _, info = env.step(plan)
        if terminated:
            break
    assert terminated
    assert info["collision"]
    assert info["state"][0] == pytest.approx(.435, abs=1e-8)


def test_hidden_circle_not_reported_and_fov_edges_clipped():
    cfg=Config(bounds=(-20,20,-20,20),rays=1001,dropout=0,bearing_noise=0)
    _,mask,ids=camera_tokens(np.zeros(3),[Circle(2,0,.5),Circle(4,0,.2)],
                           cfg.bounds,cfg,np.random.default_rng(0))
    assert set(ids[mask]) == {0}
    tok,mask,ids=camera_tokens(np.zeros(3),[Circle(2,1.4,.4)],
                             cfg.bounds,cfg,np.random.default_rng(0))
    assert mask[0] and tok[0,6] == 1
    assert tok[0,3] == pytest.approx(cfg.fov/2)


def test_tunnelling_and_wall_contact():
    circles=[Circle(2,0,.2)]
    assert clearance((0,0),circles,.3,(-5,5,-5,5)) > 0
    assert clearance((4,0),circles,.3,(-5,5,-5,5)) > 0
    assert swept_clearance((0,0),(4,0),circles,.3,(-5,5,-5,5)) < 0
    assert first_contact((0,0),(4,0),circles,.3,(-5,5,-5,5)) == pytest.approx(.375)
    assert first_contact((0,0),(6,0),[],.3,(-5,5,-5,5)) == pytest.approx(4.7/6)


def test_deterministic_reset_and_truth_separation():
    env=BearingEnv([Circle(2,0,.2)],seed=11)
    first,_=env.reset(seed=11)
    plan=Trajectory.constant_velocity([.1,.05,0])
    sequence=[env.step(plan)[0] for _ in range(10)]
    again,_=env.reset(seed=11)
    for key in first:
        np.testing.assert_array_equal(first[key],again[key])
    for original in sequence:
        repeated=env.step(plan)[0]
        for key in original:
            np.testing.assert_array_equal(original[key],repeated[key])
    assert set(first) == {"tokens","track_mask","track_ids","frame_mask",
                          "age","ego_history","velocity","goal"}


def test_safety_margin_is_not_physical_collision_and_timeout_is_truncation():
    env=BearingEnv([Circle(.64,0,.2)],goal=(4,0),
                   config=Config(max_time=.1,dropout=0,bearing_noise=0))
    _,_,terminated,truncated,info=env.step(Trajectory.constant_velocity([0,0,0]))
    assert info["margin_violation"] and not info["collision"]
    assert not terminated and truncated
    with pytest.raises(RuntimeError):
        env.step(None)


def test_camera_angular_width_matches_circle_tangents():
    cfg=Config(bounds=(-20,20,-20,20),dropout=0,bearing_noise=0,rays=2001)
    tok,mask,_=camera_tokens(np.zeros(3),[Circle(2,0,.2)],cfg.bounds,cfg,np.random.default_rng(0))
    assert mask[0]
    assert tok[0,4] == pytest.approx(math.asin(.2/1.8),abs=cfg.fov/(cfg.rays-1))
