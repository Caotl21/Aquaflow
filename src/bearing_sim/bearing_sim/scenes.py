"""Reproducible scene suites with disjoint train/validation/test geometry."""
from dataclasses import asdict
import numpy as np
from .env import BearingEnv, Config, State
from .geometry import Circle
from .records import scene_id

def make_suite(counts, seed=42, config=None):
    config = config or Config()
    rng = np.random.default_rng(seed)
    used, splits = set(), {}
    for split, count in counts.items():
        scenes = []
        for index in range(int(count)):
            obstacle = Circle(float(rng.uniform(2.4,3.6)), float(rng.uniform(-1.1,1.1)),
                              float(rng.uniform(.25,.6)))
            obstacles = [] if index % 8 == 7 else [asdict(obstacle)]
            scene = dict(name=f"{split}_{index:04d}", split=split,
                         obstacles=obstacles,
                         goal=[float(rng.uniform(4.8,6.0)),float(rng.uniform(-.8,.8))],
                         state=asdict(State(y=float(rng.uniform(-.15,.15)),
                                            yaw=float(rng.uniform(-.08,.08)))),
                         config=asdict(config),
                         observation_seed=int(rng.integers(0,2**31-1)))
            scene["scene_id"] = scene_id(scene)
            if scene["scene_id"] in used:
                raise RuntimeError("duplicate scene id")
            used.add(scene["scene_id"])
            scenes.append(scene)
        splits[split] = scenes
    return dict(schema="bearing_scenes_v1", seed=seed, splits=splits)

def environment(scene):
    return BearingEnv([Circle(**item) for item in scene["obstacles"]],
                      goal=scene["goal"], state=State(**scene["state"]),
                      config=Config(**scene["config"]), seed=scene["observation_seed"])
