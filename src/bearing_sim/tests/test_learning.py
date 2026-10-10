import copy
import numpy as np
import pytest
import torch
from bearing_sim import BearingEnv, Config
from bearing_sim.model import BearingPolicy, predict, decode_actions, policy_loss
from bearing_sim.records import FIELDS
from bearing_sim.scenes import make_suite
from bearing_sim.evaluate import run_episode, aggregate

torch.set_num_threads(2)


def batch(obs):
    return {k: torch.as_tensor(obs[k]).unsqueeze(0) for k in FIELDS}


def test_padding_and_object_order_do_not_change_policy():
    env = BearingEnv(config=Config(sensor_delay=0))
    obs, _ = env.reset()
    torch.manual_seed(4)
    model = BearingPolicy().eval()
    original = model(**batch(obs))
    changed = copy.deepcopy(obs)
    changed['tokens'][~changed['track_mask']] = np.nan
    changed['ego_history'][~changed['frame_mask']] = np.nan
    changed['age'][~changed['frame_mask']] = np.nan
    changed['tokens'] = changed['tokens'][:, ::-1].copy()
    changed['track_mask'] = changed['track_mask'][:, ::-1].copy()
    altered = model(**batch(changed))
    for key in original:
        torch.testing.assert_close(original[key], altered[key])


def test_prediction_ignores_privileged_fields_and_respects_limits(tmp_path):
    env = BearingEnv()
    obs, _ = env.reset()
    model = BearingPolicy().eval()
    first, _ = predict(model, obs)
    obs.update(track_ids=np.full((12,8),999),state=np.ones(6)*999,obstacles='not input')
    second, _ = predict(model, obs)
    np.testing.assert_array_equal(first.points, second.points)
    extremes = np.tile([10.,-10.,10.], (19,1))
    plan = decode_actions(extremes,np.zeros(3),model.config)
    speeds = np.diff(plan.points[:,1:],axis=0)/.1
    accel = np.diff(np.vstack([np.zeros(3),speeds]),axis=0)/.1
    assert np.linalg.norm(speeds[:,:2],axis=1).max() <= .2+1e-8
    assert np.linalg.norm(accel[:,:2],axis=1).max() <= .12+1e-8
    assert abs(accel[:,2]).max() <= .25+1e-8
    env.step(first)
    torch.save(dict(model=model.state_dict(),config=model.config),tmp_path/'model.pt')
    stored = torch.load(tmp_path/'model.pt',weights_only=True)
    restored = BearingPolicy(**stored['config']).eval()
    restored.load_state_dict(stored['model'])
    np.testing.assert_array_equal(first.points,predict(restored,obs)[0].points)


def test_training_has_finite_gradients_and_rejects_wrong_horizon():
    obs, _ = BearingEnv().reset()
    model = BearingPolicy()
    out = model(**batch(obs))
    target = torch.zeros(1,19,3)
    loss = policy_loss(out,target,torch.tensor([1]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError,match='horizons'):
        policy_loss(out,torch.zeros(1,10,3),torch.tensor([1]))


def test_suite_disjoint_and_timeout_is_counted():
    suite = make_suite(dict(train=2,val=2,test=2),seed=99,config=Config(max_time=.1))
    scenes = [s for split in suite['splits'].values() for s in split]
    assert len({s['scene_id'] for s in scenes}) == 6
    result = run_episode(BearingPolicy().eval(),suite['splits']['test'][0])
    assert result['truncated'] and not result['reached'] and not result['collision']
    assert aggregate([result])['timeout_rate'] == 1
