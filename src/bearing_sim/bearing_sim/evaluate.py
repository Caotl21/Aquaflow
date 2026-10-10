"""Complete held-out rollouts; student execution has no teacher intervention."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from .model import BearingPolicy, predict
from .expert import RouteTeacher
from .trajectory import Trajectory
from .scenes import environment
from .records import file_hash, source_hashes, write_json, scene_id
from .demo import svg_replay


def run_episode(model, scene, controller='student', output=None):
    env = environment(scene)
    obs, info = env.reset()
    teacher = RouteTeacher(env) if controller == 'teacher' else None
    rows, decisions = [], []
    terminated, truncated = bool(env.reached), False
    while not env.done:
        if controller == 'student':
            plan, diag = predict(model, obs)
            decisions.append(dict(time=env.time, **diag))
        elif controller == 'teacher':
            plan, _, _ = teacher.action(env.state, dt=env.config.dt)
        elif controller == 'goal_only':
            goal = obs['goal'].astype(float)
            velocity = goal/max(np.linalg.norm(goal),1e-12)*min(env.config.max_speed,np.linalg.norm(goal))
            plan = Trajectory.constant_velocity([*velocity,0.],dt=env.config.dt)
        else:
            raise ValueError('unknown controller')
        obs, _, terminated, truncated, info = env.step(plan)
        rows.append(dict(time=info['time'], **dict(zip(('x','y','yaw','u','v','r'),info['state'])),
                         clearance=info['interval_clearance']))
    if output is not None:
        output = Path(output)
        if rows:
            with output.with_suffix('.csv').open('w',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            svg = svg_replay(rows,scene['obstacles'],env.config.robot_radius)
            svg = svg.replace('Numerical environment check (not learned policy)',controller+' closed-loop rollout')
            output.with_suffix('.svg').write_text(svg)
        if controller == 'student':
            write_json(output.with_suffix('.candidates.json'),decisions)
    return dict(info, scene_id=scene['scene_id'],name=scene['name'],steps=len(rows),
                terminated=terminated,truncated=truncated,controller=controller)


def aggregate(results):
    if not results:
        raise ValueError('empty evaluation')
    if any(int(r['reached'])+int(r['collision'])+int(r['truncated']) != 1 for r in results):
        raise RuntimeError('episode outcomes must partition into success/collision/timeout')
    return dict(episodes=len(results),success_rate=float(np.mean([r['reached'] for r in results])),
                collision_rate=float(np.mean([r['collision'] for r in results])),
                timeout_rate=float(np.mean([r['truncated'] for r in results])),
                margin_violation_rate=float(np.mean([r['margin_violation'] for r in results])),
                mean_path_length=float(np.mean([r['path_length'] for r in results])),
                worst_clearance=float(min(r['min_clearance'] for r in results)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=2)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    payload = torch.load(a.checkpoint,map_location=a.device,weights_only=True)
    model = BearingPolicy(**payload['config']).to(a.device)
    model.load_state_dict(payload['model'])
    model.eval()
    suite = json.loads(a.suite.read_text())
    scenes = suite['splits']['test']
    if not scenes or any(s['split']!='test' or s['scene_id']!=scene_id(s) for s in scenes):
        raise ValueError('invalid test suite')
    ids = [s['scene_id'] for s in scenes]
    used = payload.get('provenance',{}).get('used_scene_ids')
    if used is None:
        raise ValueError('checkpoint lacks training/validation provenance')
    if len(set(ids)) != len(ids) or set(ids) & set(used):
        raise ValueError('test scene duplication or training overlap')
    for scene in scenes:
        for k in ('dt','max_speed','max_yaw_rate','max_accel','max_yaw_accel'):
            if scene['config'][k] != model.config[k]:
                raise ValueError('evaluation action contract mismatch: '+k)
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'suite.json',suite)
    results, summary = {}, {}
    for controller in ('student','teacher','goal_only'):
        results[controller] = []
        folder = a.output/controller
        folder.mkdir()
        for i,scene in enumerate(scenes):
            result = run_episode(model,scene,controller,folder/str(i))
            results[controller].append(result)
            print(json.dumps(dict(controller=controller,episode=i,reached=result['reached'],
                                  collision=result['collision'],timeout=result['truncated'])),flush=True)
        summary[controller] = aggregate(results[controller])
    write_json(a.output/'summary.json',dict(schema='bearing_closed_loop_v2',metrics=summary,results=results,
        checkpoint_sha256=file_hash(a.checkpoint),suite_sha256=file_hash(a.suite),source_sha256=source_hashes()))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
