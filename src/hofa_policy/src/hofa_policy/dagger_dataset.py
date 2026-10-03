"""Sequential DAgger windows for multi-step behavior cloning."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .contract import CONTRACT


class DaggerSequenceDataset(Dataset):
    def __init__(self, directories, rollout_steps=3):
        if not directories:
            raise ValueError("at least one DAgger dataset is required")
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        self.steps = int(rollout_steps)
        self.sequences = []
        self.sources = []
        for directory in directories:
            directory = Path(directory).resolve()
            metadata = json.loads((directory / "metadata.json").read_text())
            if metadata.get("schema") != "hofa_policy_dagger_v1":
                raise ValueError("DAgger schema mismatch: %s" % directory)
            data = np.load(directory / "samples.npz", allow_pickle=False)
            count = len(data["state"])
            if count < self.steps:
                continue
            for name, shape in (
                    ("state", (count, CONTRACT.state_features)),
                    ("reference", (
                        count, CONTRACT.horizon,
                        CONTRACT.reference_features)),
                    ("previous_wrench", (count, 3)),
                    ("target_wrench", (count, 3)),
                    ("stamp", (count,))):
                if data[name].shape != shape or not np.all(
                        np.isfinite(data[name])):
                    raise ValueError("invalid %s in %s" % (name, directory))
            state = np.asarray(data["state"], dtype=np.float32)
            reference = np.asarray(data["reference"], dtype=np.float32)
            previous = np.asarray(data["previous_wrench"], dtype=np.float32)
            target = np.asarray(data["target_wrench"], dtype=np.float32)
            stamps = np.asarray(data["stamp"], dtype=np.float64)
            for start in range(count - self.steps + 1):
                if np.all(np.diff(stamps[start:start + self.steps]) > 0.0):
                    self.sequences.append((
                        state[start:start + self.steps],
                        reference[start:start + self.steps],
                        previous[start],
                        target[start:start + self.steps]))
            self.sources.append(metadata)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        state, reference, previous, target = self.sequences[index]
        return tuple(torch.from_numpy(value) for value in (
            state, reference, previous, target))
