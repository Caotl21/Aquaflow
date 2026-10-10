"""Reproducible environment checks, not learned-policy performance evaluation."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from .env import BearingEnv, Config
from .geometry import Circle, rotation
from .trajectory import Trajectory


def run_case(name, config, seed):
    obstacles=[] if name=="empty" else [Circle(2.5,0,.45)]
    env=BearingEnv(obstacles,goal=(5,0),config=config,seed=seed)
    observation,info=env.reset()
    rows=[dict(time=0.,x=0.,y=0.,yaw=0.,u=0.,v=0.,r=0.,
               min_clearance=info["min_clearance"],visible_tracks=0,frame_age=0.)]
    # Privileged fixed waypoints check execution only; never a bearing-policy baseline.
    waypoints=[(1.0,1.5),(3.7,1.5),(5.,0.)] if name=="oracle_bypass" else [(5.,0.)]
    waypoint=0
    min_observation_age=float("inf")
    while not env.done:
        goal=observation["goal"].astype(float)
        if name=="oracle_bypass":
            offset=np.asarray(waypoints[waypoint])-env.state[:2]
            if np.linalg.norm(offset)<.18 and waypoint<len(waypoints)-1:
                waypoint+=1
                offset=np.asarray(waypoints[waypoint])-env.state[:2]
            goal=rotation(env.state[2]).T@offset
        norm=np.linalg.norm(goal)
        velocity=goal/max(norm,1e-9)*min(config.max_speed,norm)
        plan=Trajectory.constant_velocity([*velocity,0.])
        observation,reward,terminated,truncated,info=env.step(plan)
        valid=observation["frame_mask"]
        frame_age=float(observation["age"][valid][-1]) if valid.any() else 0.
        if valid.any():
            min_observation_age=min(min_observation_age,frame_age)
        rows.append(dict(zip(["time","x","y","yaw","u","v","r"],
                             [info["time"],*info["state"]]),
                         min_clearance=info["min_clearance"],
                         visible_tracks=int(observation["track_mask"][-1].sum()),frame_age=frame_age))
    info={key:(bool(value) if isinstance(value,(np.bool_,bool)) else value)
          for key,value in info.items()}
    return rows,dict(**info,terminated=bool(terminated),truncated=bool(truncated),seed=seed,
                     min_observation_age=min_observation_age,
                     controller="privileged_fixed_waypoints" if name=="oracle_bypass" else "goal_only",
                     obstacles=[asdict(o) for o in obstacles])


def svg_replay(rows, obstacles, radius):
    def point(x,y):
        return 90+100*x,270-100*y
    points=" ".join("%.2f,%.2f"%point(row["x"],row["y"]) for row in rows)
    elements=['<svg xmlns="http://www.w3.org/2000/svg" width="720" height="500" viewBox="0 0 720 500">',
              '<rect width="720" height="500" fill="white"/>',
              '<text x="20" y="30" font-size="17">Numerical environment check (not learned policy)</text>']
    for o in obstacles:
        x,y=point(o["x"],o["y"])
        elements.append(f'<circle cx="{x}" cy="{y}" r="{100*(o["radius"]+radius)}" fill="#ffe3e3"/>')
        elements.append(f'<circle cx="{x}" cy="{y}" r="{100*o["radius"]}" fill="#555"/>')
    elements.append(f'<polyline points="{points}" fill="none" stroke="#1477bb" stroke-width="3"/>')
    for row in (rows[0],rows[-1]):
        x,y=point(row["x"],row["y"])
        elements.append(f'<circle cx="{x}" cy="{y}" r="{100*radius}" fill="none" stroke="#1477bb"/>')
    elements.append('<text x="20" y="460" font-size="14">Blue: actual motion / footprint. Grey: obstacle. Pink: inflated obstacle.</text></svg>')
    return "\n".join(elements)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    config=Config(**json.loads(args.config.read_text())) if args.config else Config()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"config.json").write_text(json.dumps(asdict(config),indent=2))
    sources={str(p.relative_to(Path(__file__).parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(Path(__file__).parent.glob("*.py"))}
    summary=dict(schema="bearing_sim_demo_v1",seed=args.seed,source_sha256=sources,
                 note="Engineering execution checks only; no network trained.",cases={})
    for name in ("empty","straight_collision","oracle_bypass"):
        rows,result=run_case(name,config,args.seed)
        with (args.output/(name+".csv")).open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (args.output/(name+".svg")).write_text(svg_replay(rows,result["obstacles"],config.robot_radius))
        summary["cases"][name]=result
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
