import numpy as np
import torch.nn as nn

from dreamer.utils.utils import (
    initialize_weights,
    horizontal_forward,
    create_normal_dist,
    build_network,
)


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    try:
        return obj[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(obj, key, default)


class Decoder(nn.Module):
    """
    Lightweight decoder used during training only.
    Can be disabled for deployment-only configs.
    """

    def __init__(self, observation_shape, config):
        super().__init__()
        self.config = config.parameters.dreamer.decoder
        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        self.observation_shape = observation_shape
        self.flat_observation_size = int(np.prod(observation_shape))
        self.enabled = bool(_cfg_get(self.config, "enabled", True))

        if self.enabled:
            hidden_size = int(
                _cfg_get(
                    self.config,
                    "hidden_size",
                    max(512, self.stochastic_size + self.deterministic_size),
                )
            )
            num_layers = int(_cfg_get(self.config, "num_layers", 3))
            activation = _cfg_get(self.config, "activation", "ELU")
            self.network = build_network(
                self.deterministic_size + self.stochastic_size,
                hidden_size,
                num_layers,
                activation,
                self.flat_observation_size,
            )
            self.network.apply(initialize_weights)
        else:
            self.network = None

    def forward(self, posterior, deterministic):
        if not self.enabled or self.network is None:
            return None

        x = horizontal_forward(
            self.network,
            posterior,
            deterministic,
            output_shape=(self.flat_observation_size,),
        )
        x = x.reshape(*x.shape[:-1], *self.observation_shape)
        dist = create_normal_dist(x, std=1, event_shape=len(self.observation_shape))
        return dist
