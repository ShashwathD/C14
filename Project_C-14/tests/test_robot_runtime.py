import unittest

from dreamer.algorithms.dreamer import Dreamer
from dreamer.envs.envs import make_robot_env, get_env_infos
from dreamer.robot.fallback import calibrate_uncertainty_threshold
from dreamer.utils.utils import load_config


class _DummyWriter:
    def add_scalar(self, *_args, **_kwargs):
        return None

    def close(self):
        return None


class RobotRuntimeTests(unittest.TestCase):
    def test_mock_runtime_loop_and_fallback_trace(self):
        config = load_config("robot-c14")
        env = make_robot_env(config)
        try:
            obs_shape, discrete_action_bool, action_size = get_env_infos(env)
            agent = Dreamer(
                obs_shape,
                discrete_action_bool,
                action_size,
                _DummyWriter(),
                config.operation.device,
                config,
            )

            nominal_mse = agent.collect_nominal_mse(
                env,
                command_text="go to the orange ball",
                num_steps=10,
                fp16=False,
            )
            self.assertGreater(len(nominal_mse), 0)

            threshold = calibrate_uncertainty_threshold(nominal_mse)
            result = agent.run_robot_runtime(
                env,
                command_text="go to the orange ball",
                num_steps=12,
                threshold=threshold,
                trigger_frames=3,
                release_frames=5,
                fp16=False,
            )

            self.assertGreater(result["steps"], 0)
            self.assertEqual(result["steps"], len(result["trace"]))
            self.assertIn("goal", result)
            self.assertTrue(all("mse" in item for item in result["trace"]))
            self.assertTrue(all("use_fallback" in item for item in result["trace"]))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
