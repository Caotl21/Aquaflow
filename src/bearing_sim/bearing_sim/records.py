"""Shared serialization and provenance helpers for numerical experiments."""
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "bearing_bc_v2"
FIELDS = ("tokens", "track_mask", "frame_mask", "age", "ego_history", "velocity", "goal")
HORIZON = 19


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def write_json(path, value):
    def encode(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=encode)+"\n")


def scene_id(scene):
    geometry = {k: scene[k] for k in ("obstacles", "goal", "state")}
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()
