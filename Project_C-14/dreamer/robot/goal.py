from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn

from dreamer.robot.types import GoalSpec


def _safe_get(obj, key, default):
    if obj is None:
        return default
    try:
        return obj[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(obj, key, default)


class GoalEncoder(nn.Module):
    """Embedding lookup for predefined natural language robot commands."""

    def __init__(self, commands: Sequence[str], goal_size: int):
        super().__init__()
        if not commands:
            raise ValueError("commands must contain at least one command")
        self.commands = list(commands)
        self.command_to_id = {command: index for index, command in enumerate(self.commands)}
        self.goal_size = int(goal_size)
        self.embedding = nn.Embedding(len(self.commands), self.goal_size)

    @staticmethod
    def default_commands() -> List[str]:
        return [
            "go to the orange ball",
            "go to the blue cube",
            "go to the green cone",
            "return to start",
        ]

    @classmethod
    def from_config(cls, config):
        robot_cfg = _safe_get(config, "robot", None)
        commands = _safe_get(robot_cfg, "commands", cls.default_commands())
        parameters = _safe_get(config, "parameters", None)
        dreamer_cfg = _safe_get(parameters, "dreamer", None)
        goal_size = _safe_get(dreamer_cfg, "goal_size", 0)
        if goal_size <= 0:
            raise ValueError("goal_size must be greater than 0 to use GoalEncoder")
        return cls(commands, goal_size)

    def resolve_id(self, command_text: str) -> int:
        if command_text in self.command_to_id:
            return self.command_to_id[command_text]
        return 0

    def encode_tensor(self, command_text: str, batch_size: int = 1, device=None, dtype=None):
        command_id = self.resolve_id(command_text)
        command_ids = torch.full(
            (batch_size,), command_id, dtype=torch.long, device=device
        )
        goal = self.embedding(command_ids)
        if dtype is not None:
            goal = goal.to(dtype=dtype)
        return goal

    @torch.no_grad()
    def encode(self, command_text: str, device=None, dtype=None) -> GoalSpec:
        command_id = self.resolve_id(command_text)
        goal = self.encode_tensor(command_text, batch_size=1, device=device, dtype=dtype)
        return GoalSpec(
            command_id=command_id,
            command_text=self.commands[command_id],
            goal_vec=goal.squeeze(0).detach().cpu().numpy().astype(np.float32),
        )
