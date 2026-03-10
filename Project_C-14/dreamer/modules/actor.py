import torch
import torch.nn as nn
from torch.distributions import TanhTransform

from dreamer.utils.utils import create_normal_dist, build_network


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    try:
        return obj[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(obj, key, default)


class Actor(nn.Module):
    def __init__(self, discrete_action_bool, action_size, config):
        super().__init__()
        self.config = config.parameters.dreamer.agent.actor
        self.discrete_action_bool = discrete_action_bool
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size
        self.goal_size = _cfg_get(config.parameters.dreamer, "goal_size", 0)

        action_size = action_size if discrete_action_bool else 2 * action_size

        self.network = build_network(
            self.stochastic_size + self.deterministic_size + self.goal_size,
            self.config.hidden_size,
            self.config.num_layers,
            self.config.activation,
            action_size,
        )

    def _resolve_goal(self, goal, batch_size, device, dtype):
        if self.goal_size <= 0:
            return None
        if goal is None:
            return torch.zeros(batch_size, self.goal_size, device=device, dtype=dtype)
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)
        if goal.shape[0] != batch_size:
            goal = goal.expand(batch_size, -1)
        return goal.to(device=device, dtype=dtype)

    def forward(self, posterior, deterministic, goal=None):
        x = torch.cat((posterior, deterministic), -1)
        goal_tensor = self._resolve_goal(goal, x.shape[0], x.device, x.dtype)
        if goal_tensor is not None:
            x = torch.cat((x, goal_tensor), -1)

        x = self.network(x)
        if self.discrete_action_bool:
            dist = torch.distributions.OneHotCategorical(logits=x)
            action = dist.sample() + dist.probs - dist.probs.detach()
        else:
            dist = create_normal_dist(
                x,
                mean_scale=self.config.mean_scale,
                init_std=self.config.init_std,
                min_std=self.config.min_std,
                activation=torch.tanh,
            )
            dist = torch.distributions.TransformedDistribution(dist, TanhTransform())
            action = torch.distributions.Independent(dist, 1).rsample()
        return action
