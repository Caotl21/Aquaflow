"""Checkpoint loading and deterministic inference for Shadow deployment."""
import numpy as np
import torch

from .contract import CONTRACT
from .model import HofaPolicyNet, ModelConfig


class ShadowPolicy:
    def __init__(self, checkpoint_path, device="cpu", torch_threads=1):
        self.device = torch.device(device)
        if torch_threads > 0:
            torch.set_num_threads(torch_threads)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True)
        if checkpoint.get("contract") != CONTRACT.to_dict():
            raise ValueError("checkpoint input/action contract mismatch")
        config = ModelConfig(**checkpoint["model_config"])
        self.model = HofaPolicyNet(config=config).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.epoch = int(checkpoint.get("epoch", -1))
        self.metrics = checkpoint.get("metrics", {})

    def predict(self, state, reference, previous_wrench):
        state = np.asarray(state, dtype=np.float32)
        reference = np.asarray(reference, dtype=np.float32)
        previous_wrench = np.asarray(previous_wrench, dtype=np.float32)
        if state.shape != (CONTRACT.state_features,):
            raise ValueError("state does not match policy contract")
        if reference.shape != (CONTRACT.horizon,
                               CONTRACT.reference_features):
            raise ValueError("reference does not match policy contract")
        if previous_wrench.shape != (3,):
            raise ValueError("previous_wrench must have shape (3,)")
        tensors = [
            torch.from_numpy(state).unsqueeze(0).to(self.device),
            torch.from_numpy(reference).unsqueeze(0).to(self.device),
            torch.from_numpy(previous_wrench).unsqueeze(0).to(self.device),
        ]
        with torch.inference_mode():
            wrench, force, normalized = self.model(*tensors)
        return (wrench[0].cpu().numpy(), force[0].cpu().numpy(),
                normalized[0].cpu().numpy())

