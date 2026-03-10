import unittest
from types import SimpleNamespace

import torch

from dreamer.algorithms.dreamer import Dreamer
from dreamer.modules.model import RSSM


def _cfg():
    return SimpleNamespace(
        operation=SimpleNamespace(device="cpu"),
        parameters=SimpleNamespace(
            dreamer=SimpleNamespace(
                stochastic_size=30,
                deterministic_size=200,
                embedded_state_size=128,
                goal_size=32,
                rssm=SimpleNamespace(
                    recurrent_model=SimpleNamespace(hidden_size=200, activation="ELU"),
                    transition_model=SimpleNamespace(
                        hidden_size=200,
                        num_layers=2,
                        activation="ELU",
                        min_std=0.1,
                    ),
                    representation_model=SimpleNamespace(
                        hidden_size=200,
                        num_layers=2,
                        activation="ELU",
                        min_std=0.1,
                    ),
                ),
                reward=SimpleNamespace(hidden_size=400, num_layers=2, activation="ELU"),
                continue_=SimpleNamespace(hidden_size=400, num_layers=3, activation="ELU"),
                agent=SimpleNamespace(
                    actor=SimpleNamespace(
                        hidden_size=400,
                        min_std=0.0001,
                        init_std=5.0,
                        mean_scale=5,
                        activation="ELU",
                        num_layers=2,
                    ),
                    critic=SimpleNamespace(hidden_size=400, activation="ELU", num_layers=3),
                ),
            )
        ),
    )


class RSSMAndUncertaintyTests(unittest.TestCase):
    def test_rssm_step_shapes_with_goal(self):
        config = _cfg()
        rssm = RSSM(action_size=2, config=config)

        prior, deterministic = rssm.recurrent_model_input_init(batch_size=4)
        action = torch.zeros(4, 2)
        goal = torch.zeros(4, config.parameters.dreamer.goal_size)

        deterministic = rssm.recurrent_model(prior, action, deterministic, goal=goal)
        prior_dist, prior = rssm.transition_model(deterministic)

        self.assertEqual(deterministic.shape, (4, config.parameters.dreamer.deterministic_size))
        self.assertEqual(prior.shape, (4, config.parameters.dreamer.stochastic_size))
        self.assertEqual(prior_dist.mean.shape, (4, config.parameters.dreamer.stochastic_size))

    def test_latent_mse(self):
        predicted = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        observed = torch.tensor([[1.0, 1.0], [3.0, 5.0]])

        mse = Dreamer.compute_latent_mse(predicted, observed)
        self.assertAlmostEqual(mse, 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
