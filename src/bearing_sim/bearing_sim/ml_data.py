"""Collect complete route-teacher episodes with scene-level splits."""
import argparse
from pathlib import Path
import numpy as np
from .expert import RouteTeacher
from .records import FIELDS, HORIZON, SCHEMA, file_hash, source_hashes, write_json
from .scenes import environment, make_suite

def collect(output, scenes, split, stride=3):
    output=Path(output)
    if output.exists() or output.with_suffix('.json').exists(): raise FileExistsError(output)
    if stride<1 or not scenes or any(s['split']!=split for s in scenes): raise ValueError('invalid split')
    output.parent.mkdir(parents=True,exist_ok=True); samples=[]; outcomes=[]
    for episode,scene in enumerate(scenes):
        env=environment(scene); teacher=RouteTeacher(env); obs,_=env.reset(); steps=0
        while not env.done:
            traj,actions,mode=teacher.action(env.state,HORIZON,env.config.dt)
            if steps%stride==0:
                row={k:np.asarray(obs[k]).copy() for k in FIELDS}
                row.update(expert_action=actions,expert_mode=np.int64(mode),scene_episode=np.int64(episode)); samples.append(row)
            obs,_,_,_,info=env.step(traj); steps+=1
        outcomes.append(dict(scene_id=scene['scene_id'],steps=steps,**info))
        if not info['reached'] or info['collision'] or info['margin_violation']: raise RuntimeError('teacher failed safety checks')
    if not samples: raise ValueError('no samples')
    packed={k:np.stack([row[k] for row in samples]) for k in samples[0]}; np.savez_compressed(output,**packed)
    write_json(output.with_suffix('.json'),dict(schema=SCHEMA,split=split,horizon=HORIZON,samples=len(samples),stride=stride,scenes=scenes,outcomes=outcomes,data_sha256=file_hash(output),source_sha256=source_hashes()))
    return len(samples)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--train-scenes',type=int,default=32); p.add_argument('--val-scenes',type=int,default=8); p.add_argument('--test-scenes',type=int,default=8); p.add_argument('--stride',type=int,default=3); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    counts=dict(train=a.train_scenes,val=a.val_scenes,test=a.test_scenes)
    if min(counts.values())<1: p.error('each split needs one scene')
    suite=make_suite(counts,a.seed); a.output.mkdir(parents=True,exist_ok=False); write_json(a.output/'suite.json',suite)
    for split in ('train','val'): print(split,collect(a.output/(split+'.npz'),suite['splits'][split],split,a.stride),flush=True)

if __name__=='__main__': main()
