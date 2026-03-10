import argparse
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from dreamer.algorithms.dreamer import Dreamer
from dreamer.envs.envs import make_robot_env, get_env_infos
from dreamer.robot.export import export_runtime_artifacts
from dreamer.robot.fallback import calibrate_uncertainty_threshold
from dreamer.utils.utils import load_config, get_base_directory


def create_agent(config, obs_shape, discrete_action_bool, action_size, writer):
    if config.algorithm != "dreamer-v1":
        raise ValueError("Only algorithm='dreamer-v1' is supported in this pruned C14 repo")

    device = config.operation.device
    return Dreamer(obs_shape, discrete_action_bool, action_size, writer, device, config)


def maybe_load_checkpoint(agent, checkpoint_path):
    if not checkpoint_path:
        return

    checkpoint = torch.load(checkpoint_path, map_location=agent.device)
    module_map = {
        "encoder": agent.encoder,
        "rssm": agent.rssm,
        "actor": agent.actor,
        "critic": agent.critic,
        "reward_predictor": agent.reward_predictor,
        "decoder": agent.decoder,
    }
    if hasattr(agent, "continue_predictor"):
        module_map["continue_predictor"] = agent.continue_predictor
    if getattr(agent, "goal_encoder", None) is not None:
        module_map["goal_encoder"] = agent.goal_encoder

    for key, module in module_map.items():
        state = checkpoint.get(key)
        if state is not None:
            module.load_state_dict(state)


def run(config_file, args):
    config = load_config(config_file)
    if config.environment.benchmark != "robot":
        raise RuntimeError("This pruned repo supports only benchmark='robot'")

    env = make_robot_env(config)
    obs_shape, discrete_action_bool, action_size = get_env_infos(env)

    log_dir = (
        get_base_directory()
        + "/runs/"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "_"
        + config.operation.log_dir
    )
    writer = SummaryWriter(log_dir)

    agent = create_agent(config, obs_shape, discrete_action_bool, action_size, writer)
    maybe_load_checkpoint(agent, args.checkpoint)

    try:
        if args.mode == "calibrate_threshold":
            mse_values = agent.collect_nominal_mse(
                env,
                command_text=args.command,
                num_steps=args.calibration_steps,
                fp16=args.fp16,
                rollout_horizon=args.rollout_horizon,
                rollout_candidates=args.rollout_candidates,
                rollout_noise=args.rollout_noise,
                verbose=args.console_trace,
                trace_interval=args.trace_interval,
                latent_preview_dims=args.latent_preview_dims,
            )
            threshold = calibrate_uncertainty_threshold(mse_values)
            print(f"calibrated_threshold={threshold:.6f}")
            print(f"num_samples={len(mse_values)}")

        elif args.mode == "robot_runtime":
            result = agent.run_robot_runtime(
                env,
                command_text=args.command,
                num_steps=args.runtime_steps,
                threshold=args.threshold,
                trigger_frames=args.trigger_frames,
                release_frames=args.release_frames,
                fp16=args.fp16,
                rollout_horizon=args.rollout_horizon,
                rollout_candidates=args.rollout_candidates,
                rollout_noise=args.rollout_noise,
                verbose=args.console_trace,
                trace_interval=args.trace_interval,
                latent_preview_dims=args.latent_preview_dims,
            )
            print(
                f"runtime_complete steps={result['steps']} "
                f"fallback_steps={result['fallback_steps']} "
                f"threshold={result['threshold']:.6f}"
            )

        elif args.mode == "export":
            try:
                export_result = export_runtime_artifacts(
                    agent,
                    observation_shape=obs_shape,
                    export_dir=args.export_dir,
                    fp16=args.fp16,
                )
                print("export_complete")
                for key, value in export_result.items():
                    print(f"{key}={value}")
            except RuntimeError as exc:
                print(f"export_failed: {exc}")

        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
    finally:
        if hasattr(env, "close"):
            env.close()
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="robot-c14.yml",
        help="config file to run(default: robot-c14.yml)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="robot_runtime",
        choices=["robot_runtime", "calibrate_threshold", "export"],
    )
    parser.add_argument(
        "--command",
        type=str,
        default="go to the orange ball",
        help="Goal command used for robot runtime modes",
    )
    parser.add_argument(
        "--runtime_steps",
        type=int,
        default=300,
        help="Number of runtime control steps",
    )
    parser.add_argument(
        "--calibration_steps",
        type=int,
        default=200,
        help="Number of nominal steps for threshold calibration",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional fallback threshold. If omitted, threshold is calibrated online.",
    )
    parser.add_argument("--trigger_frames", type=int, default=3)
    parser.add_argument("--release_frames", type=int, default=5)
    parser.add_argument(
        "--rollout_horizon",
        type=int,
        default=5,
        help="Imagined rollout horizon used to score candidate runtime actions",
    )
    parser.add_argument(
        "--rollout_candidates",
        type=int,
        default=5,
        help="Number of candidate first-actions evaluated per runtime step",
    )
    parser.add_argument(
        "--rollout_noise",
        type=float,
        default=0.20,
        help="Gaussian perturbation scale for non-base rollout candidates",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional checkpoint path for runtime/export",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default="runtime_exports",
        help="Directory for ONNX/TensorRT exports",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Run runtime/export in FP16",
    )
    parser.add_argument(
        "--console_trace",
        action="store_true",
        help="Print per-step camera stats + latent previews in console",
    )
    parser.add_argument(
        "--trace_interval",
        type=int,
        default=1,
        help="Print every N steps when --console_trace is enabled",
    )
    parser.add_argument(
        "--latent_preview_dims",
        type=int,
        default=8,
        help="How many latent dimensions to print in console trace",
    )

    args = parser.parse_args()
    run(args.config, args)
