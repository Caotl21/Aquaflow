"""State/reference policy with a bounded latent-thruster output head."""
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .actuation import ThrusterGeometry
from .contract import CONTRACT


@dataclass(frozen=True)
class ModelConfig:
    point_hidden: int = 64
    reference_hidden: int = 96
    context_hidden: int = 64
    fusion_hidden: int = 128

    def to_dict(self):
        return asdict(self)


class HofaPolicyNet(nn.Module):
    def __init__(self, geometry=None, config=None):
        super().__init__()
        geometry = geometry or ThrusterGeometry.bricsbot()
        config = config or ModelConfig()
        self.config = config

        self.point_encoder = nn.Sequential(
            nn.Linear(CONTRACT.reference_features, config.point_hidden),
            nn.SiLU(),
            nn.Linear(config.point_hidden, config.reference_hidden),
            nn.SiLU(),
        )
        self.reference_conv = nn.Sequential(
            nn.Conv1d(config.reference_hidden, config.reference_hidden, 3,
                      padding=1),
            nn.SiLU(),
            nn.Conv1d(config.reference_hidden, config.reference_hidden, 3,
                      padding=1),
            nn.SiLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(CONTRACT.state_features + 3, config.context_hidden),
            nn.SiLU(),
            nn.Linear(config.context_hidden, config.context_hidden),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * config.reference_hidden + config.context_hidden,
                      config.fusion_hidden),
            nn.SiLU(),
            nn.Linear(config.fusion_hidden, config.fusion_hidden),
            nn.SiLU(),
            nn.Linear(config.fusion_hidden, CONTRACT.action_features),
        )

        self.register_buffer(
            "state_scale", torch.tensor([0.5, 0.5, 1.2],
                                           dtype=torch.float32))
        self.register_buffer(
            "reference_scale",
            torch.tensor([2.0, 2.0, 1.0, 1.0, 0.5, 0.5, 1.0,
                          0.5, 0.5, 1.0], dtype=torch.float32))
        self.register_buffer(
            "allocation_matrix",
            torch.tensor(geometry.allocation_matrix, dtype=torch.float32))
        self.register_buffer(
            "force_mid", torch.tensor(geometry.force_mid,
                                        dtype=torch.float32))
        self.register_buffer(
            "force_half", torch.tensor(geometry.force_half,
                                         dtype=torch.float32))
        self.register_buffer(
            "wrench_scale", torch.tensor(geometry.wrench_scale,
                                           dtype=torch.float32))

    def forward(self, state, reference, previous_wrench):
        if reference.ndim != 3 or reference.shape[1:] != (
                CONTRACT.horizon, CONTRACT.reference_features):
            raise ValueError("reference tensor does not match policy contract")
        point = self.point_encoder(reference / self.reference_scale)
        sequence = self.reference_conv(point.transpose(1, 2))
        reference_mean = sequence.mean(dim=2)
        reference_max = sequence.amax(dim=2)
        context = torch.cat((state / self.state_scale,
                             previous_wrench / self.wrench_scale), dim=-1)
        context = self.context_encoder(context)
        raw_action = self.fusion(torch.cat(
            (reference_mean, reference_max, context), dim=-1))
        normalized_force = torch.tanh(raw_action)
        force = self.force_mid + self.force_half * normalized_force
        wrench = force @ self.allocation_matrix.transpose(0, 1)
        return wrench, force, normalized_force
