"""PyTorch datasets for exported HOFA policy runs."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .contract import CONTRACT, SCHEMA


class PolicyDataset(Dataset):
    def __init__(self, directories):
        if not directories:
            raise ValueError("at least one dataset directory is required")
        arrays = {name: [] for name in (
            "state", "reference", "previous_wrench",
            "target_wrench", "stamp")}
        self.sources = []
        for directory in directories:
            directory = Path(directory).resolve()
            metadata = json.loads((directory / "metadata.json").read_text())
            if metadata["contract"]["schema"] != SCHEMA:
                raise ValueError("dataset contract mismatch: %s" % directory)
            data = np.load(directory / "samples.npz", allow_pickle=False)
            self._validate(data, directory)
            for name in arrays:
                arrays[name].append(np.asarray(data[name]))
            self.sources.append(metadata)
        self.data = {
            name: np.concatenate(parts, axis=0)
            for name, parts in arrays.items()
        }

    @staticmethod
    def _validate(data, directory):
        count = len(data["state"])
        expected = {
            "state": (count, CONTRACT.state_features),
            "reference": (count, CONTRACT.horizon,
                            CONTRACT.reference_features),
            "previous_wrench": (count, 3),
            "target_wrench": (count, 3),
            "stamp": (count,),
        }
        for name, shape in expected.items():
            if name not in data or data[name].shape != shape:
                raise ValueError(
                    "%s has invalid %s shape: %s expected %s" %
                    (directory, name, getattr(data.get(name), "shape", None),
                     shape))
            if not np.all(np.isfinite(data[name])):
                raise ValueError("%s contains nonfinite %s" %
                                 (directory, name))

    def __len__(self):
        return len(self.data["state"])

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.data["state"][index]),
            torch.from_numpy(self.data["reference"][index]),
            torch.from_numpy(self.data["previous_wrench"][index]),
            torch.from_numpy(self.data["target_wrench"][index]),
        )
