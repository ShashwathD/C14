from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from dreamer.modules.model import RSSM, RewardModel, ContinueModel
from dreamer.modules.encoder import Encoder
from dreamer.modules.decoder import Decoder
from dreamer.modules.actor import Actor
from dreamer.modules.critic import Critic

from dreamer.robot.fallback import (
    UncertaintyAwareFallback,
    calibrate_uncertainty_threshold,
)
from dreamer.robot.goal import GoalEncoder
from dreamer.robot.types import GoalSpec, RobotObservation
from dreamer.utils.utils import (
    compute_lambda_values,
    create_normal_dist,
    DynamicInfos,
)
from dreamer.utils.buffer import ReplayBuffer


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    try:
        return obj[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(obj, key, default)


class Dreamer:
    def __init__(
        self,
        observation_shape,
        discrete_action_bool,
        action_size,
        writer,
        device,
        config,
    ):
        self.device = device
        self.action_size = action_size
        self.discrete_action_bool = discrete_action_bool

        self.encoder = Encoder(observation_shape, config).to(self.device)
        self.decoder = Decoder(observation_shape, config).to(self.device)
        self.rssm = RSSM(action_size, config).to(self.device)
        self.reward_predictor = RewardModel(config).to(self.device)
        if config.parameters.dreamer.use_continue_flag:
            self.continue_predictor = ContinueModel(config).to(self.device)
        self.actor = Actor(discrete_action_bool, action_size, config).to(self.device)
        self.critic = Critic(config).to(self.device)

        self.buffer = ReplayBuffer(observation_shape, action_size, self.device, config)

        self.config = config.parameters.dreamer
        self.goal_size = int(_cfg_get(self.config, "goal_size", 0))
        self.goal_encoder = self._build_goal_encoder(config)

        # optimizer
        self.model_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.rssm.parameters())
            + list(self.reward_predictor.parameters())
        )
        if self.goal_encoder is not None:
            self.model_params += list(self.goal_encoder.parameters())
        if self.config.use_continue_flag:
            self.model_params += list(self.continue_predictor.parameters())

        self.model_optimizer = torch.optim.Adam(
            self.model_params, lr=self.config.model_learning_rate
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_learning_rate
        )

        self.continue_criterion = nn.BCELoss()

        self.dynamic_learning_infos = DynamicInfos(self.device)
        self.behavior_learning_infos = DynamicInfos(self.device)

        self.writer = writer
        self.num_total_episode = 0

    def _build_goal_encoder(self, config):
        if self.goal_size <= 0:
            return None
        try:
            return GoalEncoder.from_config(config).to(self.device)
        except Exception:
            return GoalEncoder(GoalEncoder.default_commands(), self.goal_size).to(
                self.device
            )

    def _zero_goal(self, batch_size: int, dtype=torch.float32):
        if self.goal_size <= 0:
            return None
        return torch.zeros(batch_size, self.goal_size, device=self.device, dtype=dtype)

    def _resolve_goal(
        self, command_text: str, batch_size: int = 1, dtype=torch.float32
    ) -> Tuple[Optional[GoalSpec], Optional[torch.Tensor]]:
        if self.goal_size <= 0 or self.goal_encoder is None:
            return None, None
        goal_spec = self.goal_encoder.encode(command_text, device=self.device, dtype=dtype)
        goal_tensor = self.goal_encoder.encode_tensor(
            command_text,
            batch_size=batch_size,
            device=self.device,
            dtype=dtype,
        )
        return goal_spec, goal_tensor

    def train(self, env):
        if len(self.buffer) < 1:
            self.environment_interaction(env, self.config.seed_episodes)

        for _iteration in range(self.config.train_iterations):
            for _collect_interval in range(self.config.collect_interval):
                data = self.buffer.sample(
                    self.config.batch_size, self.config.batch_length
                )
                posteriors, deterministics = self.dynamic_learning(data)
                self.behavior_learning(posteriors, deterministics)

            self.environment_interaction(env, self.config.num_interaction_episodes)
            self.evaluate(env)

    def evaluate(self, env):
        self.environment_interaction(env, self.config.num_evaluate, train=False)

    def dynamic_learning(self, data):
        prior, deterministic = self.rssm.recurrent_model_input_init(len(data.action))
        goal_context = self._zero_goal(len(data.action), dtype=prior.dtype)

        data.embedded_observation = self.encoder(data.observation)

        for t in range(1, self.config.batch_length):
            deterministic = self.rssm.recurrent_model(
                prior, data.action[:, t - 1], deterministic, goal=goal_context
            )
            prior_dist, prior = self.rssm.transition_model(deterministic)
            posterior_dist, posterior = self.rssm.representation_model(
                data.embedded_observation[:, t], deterministic
            )

            self.dynamic_learning_infos.append(
                priors=prior,
                prior_dist_means=prior_dist.mean,
                prior_dist_stds=prior_dist.scale,
                posteriors=posterior,
                posterior_dist_means=posterior_dist.mean,
                posterior_dist_stds=posterior_dist.scale,
                deterministics=deterministic,
            )

            prior = posterior

        infos = self.dynamic_learning_infos.get_stacked()
        self._model_update(data, infos)
        return infos.posteriors.detach(), infos.deterministics.detach()

    def _model_update(self, data, posterior_info):
        reconstruction_loss = None
        reconstructed_observation_dist = self.decoder(
            posterior_info.posteriors, posterior_info.deterministics
        )
        if reconstructed_observation_dist is not None:
            reconstruction_loss = reconstructed_observation_dist.log_prob(
                data.observation[:, 1:]
            )

        if self.config.use_continue_flag:
            continue_dist = self.continue_predictor(
                posterior_info.posteriors, posterior_info.deterministics
            )
            continue_loss = self.continue_criterion(
                continue_dist.probs, 1 - data.done[:, 1:]
            )

        reward_dist = self.reward_predictor(
            posterior_info.posteriors, posterior_info.deterministics
        )
        reward_loss = reward_dist.log_prob(data.reward[:, 1:])

        prior_dist = create_normal_dist(
            posterior_info.prior_dist_means,
            posterior_info.prior_dist_stds,
            event_shape=1,
        )
        posterior_dist = create_normal_dist(
            posterior_info.posterior_dist_means,
            posterior_info.posterior_dist_stds,
            event_shape=1,
        )
        kl_divergence_loss = torch.mean(
            torch.distributions.kl.kl_divergence(posterior_dist, prior_dist)
        )
        kl_divergence_loss = torch.max(
            torch.tensor(self.config.free_nats).to(self.device), kl_divergence_loss
        )

        model_loss = self.config.kl_divergence_scale * kl_divergence_loss - reward_loss.mean()
        if reconstruction_loss is not None:
            model_loss -= reconstruction_loss.mean()
        if self.config.use_continue_flag:
            model_loss += continue_loss.mean()

        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            self.model_params,
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.model_optimizer.step()

    def behavior_learning(self, states, deterministics):
        """
        posterior shape : (batch, timestep, stochastic)
        """
        state = states.reshape(-1, self.config.stochastic_size)
        deterministic = deterministics.reshape(-1, self.config.deterministic_size)
        goal_context = self._zero_goal(state.shape[0], dtype=state.dtype)

        for _t in range(self.config.horizon_length):
            action = self.actor(state, deterministic, goal=goal_context)
            deterministic = self.rssm.recurrent_model(
                state, action, deterministic, goal=goal_context
            )
            _, state = self.rssm.transition_model(deterministic)
            self.behavior_learning_infos.append(
                priors=state, deterministics=deterministic
            )

        self._agent_update(self.behavior_learning_infos.get_stacked())

    def _agent_update(self, behavior_learning_infos):
        predicted_rewards = self.reward_predictor(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean
        values = self.critic(
            behavior_learning_infos.priors, behavior_learning_infos.deterministics
        ).mean

        if self.config.use_continue_flag:
            continues = self.continue_predictor(
                behavior_learning_infos.priors, behavior_learning_infos.deterministics
            ).mean
        else:
            continues = self.config.discount * torch.ones_like(values)

        lambda_values = compute_lambda_values(
            predicted_rewards,
            values,
            continues,
            self.config.horizon_length,
            self.device,
            self.config.lambda_,
        )

        actor_loss = -torch.mean(lambda_values)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.actor_optimizer.step()

        value_dist = self.critic(
            behavior_learning_infos.priors.detach()[:, :-1],
            behavior_learning_infos.deterministics.detach()[:, :-1],
        )
        value_loss = -torch.mean(value_dist.log_prob(lambda_values.detach()))

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.critic_optimizer.step()

    @staticmethod
    def compute_latent_mse(predicted_latent: torch.Tensor, observed_latent: torch.Tensor) -> float:
        return float(torch.mean((predicted_latent - observed_latent) ** 2).detach().cpu().item())

    @staticmethod
    def _tensor_head(tensor: torch.Tensor, dims: int = 8) -> str:
        vector = tensor.detach().reshape(-1).float().cpu().numpy()
        dims = int(max(1, min(dims, vector.size)))
        return np.array2string(vector[:dims], precision=4, separator=", ")

    @staticmethod
    def _observation_stats(observation):
        if not isinstance(observation, RobotObservation):
            return None
        rgb = observation.rgb.astype(np.float32)
        depth = observation.depth.astype(np.float32)
        return {
            "rgb_mean": float(rgb.mean()),
            "rgb_std": float(rgb.std()),
            "depth_min": float(np.nanmin(depth)),
            "depth_mean": float(np.nanmean(depth)),
            "depth_max": float(np.nanmax(depth)),
            "odometry": observation.odometry,
            "timestamp": observation.timestamp,
        }

    @staticmethod
    def _to_model_observation(env, observation):
        if isinstance(observation, RobotObservation):
            return env.preprocess_observation(observation)
        return observation

    def _encode_robot_observation(self, env, observation, dtype=torch.float32):
        model_observation = self._to_model_observation(env, observation)

        observation_tensor = (
            torch.from_numpy(model_observation)
            .to(self.device)
            .to(dtype=dtype)
            .reshape(1, *model_observation.shape)
        )
        embedded = self.encoder(observation_tensor)
        return embedded.reshape(1, -1)

    def _configure_runtime_precision(self, fp16: bool):
        if fp16:
            self.encoder.half()
            self.rssm.half()
            self.actor.half()
            self.reward_predictor.half()
        else:
            self.encoder.float()
            self.rssm.float()
            self.actor.float()
            self.reward_predictor.float()

        self.encoder.eval()
        self.rssm.eval()
        self.actor.eval()
        self.reward_predictor.eval()

    @torch.no_grad()
    def _sample_rollout_candidates(
        self,
        posterior: torch.Tensor,
        deterministic: torch.Tensor,
        goal: Optional[torch.Tensor],
        num_candidates: int,
        rollout_noise: float,
    ) -> torch.Tensor:
        base_action = self.actor(posterior, deterministic, goal=goal).detach()
        num_candidates = max(1, int(num_candidates))
        if num_candidates == 1:
            return base_action

        candidates = base_action.expand(num_candidates, -1).clone()
        noise = float(max(0.0, rollout_noise))
        if noise > 0.0:
            perturbation = noise * torch.randn(
                num_candidates - 1,
                self.action_size,
                device=self.device,
                dtype=base_action.dtype,
            )
            candidates[1:] = torch.clamp(candidates[1:] + perturbation, -1.0, 1.0)
        return candidates

    @torch.no_grad()
    def _rollout_candidate_returns(
        self,
        posterior: torch.Tensor,
        deterministic: torch.Tensor,
        goal: Optional[torch.Tensor],
        first_actions: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        horizon = max(1, int(horizon))
        num_candidates = first_actions.shape[0]

        state = posterior.expand(num_candidates, -1)
        deterministic_state = deterministic.expand(num_candidates, -1)

        if goal is not None:
            if goal.dim() == 1:
                goal = goal.unsqueeze(0)
            if goal.shape[0] != num_candidates:
                goal_batch = goal.expand(num_candidates, -1)
            else:
                goal_batch = goal
            goal_batch = goal_batch.to(device=self.device, dtype=first_actions.dtype)
        else:
            goal_batch = None

        action = first_actions
        discounted_returns = torch.zeros(
            num_candidates, device=self.device, dtype=first_actions.dtype
        )
        discount_factor = 1.0

        for _ in range(horizon):
            deterministic_state = self.rssm.recurrent_model(
                state, action, deterministic_state, goal=goal_batch
            )
            prior_dist, _prior = self.rssm.transition_model(deterministic_state)
            # Use distribution mean for stable planning scores.
            state = prior_dist.mean

            reward_dist = self.reward_predictor(state, deterministic_state)
            rewards = reward_dist.mean.reshape(-1)
            discounted_returns = discounted_returns + discount_factor * rewards

            action = self.actor(state, deterministic_state, goal=goal_batch).detach()
            discount_factor *= float(self.config.discount)

        return discounted_returns

    @torch.no_grad()
    def _select_runtime_action(
        self,
        posterior: torch.Tensor,
        deterministic: torch.Tensor,
        goal: Optional[torch.Tensor],
        rollout_horizon: int,
        rollout_candidates: int,
        rollout_noise: float,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        candidates = self._sample_rollout_candidates(
            posterior=posterior,
            deterministic=deterministic,
            goal=goal,
            num_candidates=rollout_candidates,
            rollout_noise=rollout_noise,
        )

        if candidates.shape[0] == 1 or int(rollout_horizon) <= 1:
            return candidates[:1], None

        candidate_returns = self._rollout_candidate_returns(
            posterior=posterior,
            deterministic=deterministic,
            goal=goal,
            first_actions=candidates,
            horizon=rollout_horizon,
        )
        best_index = int(torch.argmax(candidate_returns).item())
        planning_info = {
            "best_return": float(candidate_returns[best_index].detach().cpu().item()),
            "mean_return": float(candidate_returns.mean().detach().cpu().item()),
            "best_index": float(best_index),
        }
        return candidates[best_index : best_index + 1], planning_info

    @torch.no_grad()
    def collect_nominal_mse(
        self,
        env,
        command_text: str,
        num_steps: int = 200,
        fp16: bool = False,
        rollout_horizon: int = 5,
        rollout_candidates: int = 5,
        rollout_noise: float = 0.20,
        verbose: bool = False,
        trace_interval: int = 1,
        latent_preview_dims: int = 8,
    ) -> List[float]:
        self._configure_runtime_precision(fp16)
        runtime_dtype = torch.float16 if fp16 else torch.float32

        goal_spec, goal = self._resolve_goal(
            command_text,
            batch_size=1,
            dtype=runtime_dtype,
        )
        if goal is None:
            goal = self._zero_goal(1, dtype=runtime_dtype)

        posterior, deterministic = self.rssm.recurrent_model_input_init(1)
        posterior = posterior.to(dtype=runtime_dtype)
        deterministic = deterministic.to(dtype=runtime_dtype)
        action = torch.zeros(1, self.action_size, device=self.device, dtype=runtime_dtype)

        observation = env.reset()
        embedded_observation = self._encode_robot_observation(
            env, observation, dtype=runtime_dtype
        )

        mse_values = []
        done = False
        step = 0
        while not done and step < num_steps:
            deterministic = self.rssm.recurrent_model(
                posterior, action, deterministic, goal=goal
            )
            _, prior = self.rssm.transition_model(deterministic)
            _, posterior = self.rssm.representation_model(
                embedded_observation, deterministic
            )
            mse = self.compute_latent_mse(prior, posterior)
            mse_values.append(mse)

            action, planning_info = self._select_runtime_action(
                posterior=posterior,
                deterministic=deterministic,
                goal=goal,
                rollout_horizon=rollout_horizon,
                rollout_candidates=rollout_candidates,
                rollout_noise=rollout_noise,
            )
            control_action = env.tensor_to_action(action)
            if verbose and step % max(trace_interval, 1) == 0:
                obs_stats = self._observation_stats(observation)
                rgb_mean = obs_stats["rgb_mean"] if obs_stats else float("nan")
                depth_mean = obs_stats["depth_mean"] if obs_stats else float("nan")
                planning_text = ""
                if planning_info is not None:
                    planning_text = (
                        f" rolloutH={int(rollout_horizon)} cand={int(rollout_candidates)} "
                        f"ret(best/mean)={planning_info['best_return']:.4f}/"
                        f"{planning_info['mean_return']:.4f}"
                    )
                print(
                    f"[CALIB step={step}] mse={mse:.6f} "
                    f"action=({control_action.linear_vel:.3f}, {control_action.angular_vel:.3f}) "
                    f"rgb_mean={rgb_mean:.2f} depth_mean={depth_mean:.3f} "
                    f"embed_head={self._tensor_head(embedded_observation, latent_preview_dims)} "
                    f"prior_head={self._tensor_head(prior, latent_preview_dims)} "
                    f"posterior_head={self._tensor_head(posterior, latent_preview_dims)}"
                    f"{planning_text}"
                )
            next_observation, _reward, done, _info = env.step(control_action)

            action = env.action_to_tensor(control_action, self.device).to(runtime_dtype)
            embedded_observation = self._encode_robot_observation(
                env, next_observation, dtype=runtime_dtype
            )
            observation = next_observation
            step += 1

        return mse_values

    @torch.no_grad()
    def run_robot_runtime(
        self,
        env,
        command_text: str,
        num_steps: int = 300,
        threshold: Optional[float] = None,
        trigger_frames: int = 3,
        release_frames: int = 5,
        fp16: bool = False,
        rollout_horizon: int = 5,
        rollout_candidates: int = 5,
        rollout_noise: float = 0.20,
        verbose: bool = False,
        trace_interval: int = 1,
        latent_preview_dims: int = 8,
    ) -> Dict:
        runtime_dtype = torch.float16 if fp16 else torch.float32
        self._configure_runtime_precision(fp16)

        goal_spec, goal = self._resolve_goal(
            command_text,
            batch_size=1,
            dtype=runtime_dtype,
        )
        if goal is None:
            goal = self._zero_goal(1, dtype=runtime_dtype)

        if threshold is None:
            calibration_mse = self.collect_nominal_mse(
                env,
                command_text=command_text,
                num_steps=min(50, num_steps),
                fp16=fp16,
                rollout_horizon=rollout_horizon,
                rollout_candidates=rollout_candidates,
                rollout_noise=rollout_noise,
            )
            threshold = calibrate_uncertainty_threshold(calibration_mse)

        fallback = UncertaintyAwareFallback(
            threshold=threshold,
            trigger_frames=trigger_frames,
            release_frames=release_frames,
        )

        posterior, deterministic = self.rssm.recurrent_model_input_init(1)
        posterior = posterior.to(dtype=runtime_dtype)
        deterministic = deterministic.to(dtype=runtime_dtype)
        action = torch.zeros(1, self.action_size, device=self.device, dtype=runtime_dtype)

        observation = env.reset()
        embedded_observation = self._encode_robot_observation(
            env, observation, dtype=runtime_dtype
        )

        trace = []
        done = False
        step = 0

        while not done and step < num_steps:
            deterministic = self.rssm.recurrent_model(
                posterior, action, deterministic, goal=goal
            )
            _, prior = self.rssm.transition_model(deterministic)
            _, posterior = self.rssm.representation_model(
                embedded_observation, deterministic
            )

            learned_action_tensor, planning_info = self._select_runtime_action(
                posterior=posterior,
                deterministic=deterministic,
                goal=goal,
                rollout_horizon=rollout_horizon,
                rollout_candidates=rollout_candidates,
                rollout_noise=rollout_noise,
            )
            learned_action = env.tensor_to_action(learned_action_tensor)

            mse = self.compute_latent_mse(prior, posterior)
            decision = fallback.update(mse)
            selected_action = (
                env.fallback_action(observation)
                if decision.use_fallback
                else learned_action
            )
            if verbose and step % max(trace_interval, 1) == 0:
                obs_stats = self._observation_stats(observation)
                rgb_mean = obs_stats["rgb_mean"] if obs_stats else float("nan")
                depth_min = obs_stats["depth_min"] if obs_stats else float("nan")
                depth_mean = obs_stats["depth_mean"] if obs_stats else float("nan")
                depth_max = obs_stats["depth_max"] if obs_stats else float("nan")
                odometry = obs_stats["odometry"] if obs_stats else ("?", "?", "?")
                planning_text = ""
                if planning_info is not None:
                    planning_text = (
                        f" rolloutH={int(rollout_horizon)} cand={int(rollout_candidates)} "
                        f"ret(best/mean)={planning_info['best_return']:.4f}/"
                        f"{planning_info['mean_return']:.4f}"
                    )
                print(
                    f"[RUN step={step}] mse={mse:.6f} threshold={decision.threshold:.6f} "
                    f"fallback={decision.use_fallback} hc={decision.high_count} lc={decision.low_count} "
                    f"action=({selected_action.linear_vel:.3f}, {selected_action.angular_vel:.3f}) "
                    f"rgb_mean={rgb_mean:.2f} depth(min/mean/max)="
                    f"{depth_min:.3f}/{depth_mean:.3f}/{depth_max:.3f} "
                    f"odom={odometry} "
                    f"embed_head={self._tensor_head(embedded_observation, latent_preview_dims)} "
                    f"prior_head={self._tensor_head(prior, latent_preview_dims)} "
                    f"posterior_head={self._tensor_head(posterior, latent_preview_dims)}"
                    f"{planning_text}"
                )

            next_observation, _reward, done, info = env.step(selected_action)

            action = env.action_to_tensor(selected_action, self.device).to(runtime_dtype)
            embedded_observation = self._encode_robot_observation(
                env, next_observation, dtype=runtime_dtype
            )
            observation = next_observation

            trace.append(
                {
                    "step": step,
                    "mse": mse,
                    "threshold": decision.threshold,
                    "use_fallback": decision.use_fallback,
                    "high_count": decision.high_count,
                    "low_count": decision.low_count,
                    "linear_vel": selected_action.linear_vel,
                    "angular_vel": selected_action.angular_vel,
                    "timestamp": info.get("timestamp"),
                }
            )
            step += 1

        fallback_steps = sum(1 for item in trace if item["use_fallback"])
        return {
            "goal": None
            if goal_spec is None
            else {
                "command_id": goal_spec.command_id,
                "command_text": goal_spec.command_text,
                "goal_vec": goal_spec.goal_vec,
            },
            "threshold": threshold,
            "steps": step,
            "fallback_steps": fallback_steps,
            "trace": trace,
        }

    @torch.no_grad()
    def environment_interaction(self, env, num_interaction_episodes, train=True):
        score_lst = np.array([])
        for _epi in range(num_interaction_episodes):
            posterior, deterministic = self.rssm.recurrent_model_input_init(1)
            action = torch.zeros(1, self.action_size).to(self.device)
            goal_context = self._zero_goal(1, dtype=action.dtype)

            observation = env.reset()
            model_observation = self._to_model_observation(env, observation)
            embedded_observation = self.encoder(
                torch.from_numpy(model_observation).float().to(self.device)
            )

            score = 0
            done = False

            while not done:
                deterministic = self.rssm.recurrent_model(
                    posterior, action, deterministic, goal=goal_context
                )
                embedded_observation = embedded_observation.reshape(1, -1)
                _, posterior = self.rssm.representation_model(
                    embedded_observation, deterministic
                )
                action = self.actor(posterior, deterministic, goal=goal_context).detach()

                if self.discrete_action_bool:
                    buffer_action = action.cpu().numpy()
                    env_action = buffer_action.argmax()

                else:
                    buffer_action = action.cpu().numpy()[0]
                    env_action = buffer_action

                next_observation, reward, done, _info = env.step(env_action)
                next_model_observation = self._to_model_observation(env, next_observation)
                if train:
                    self.buffer.add(
                        model_observation,
                        buffer_action,
                        reward,
                        next_model_observation,
                        done,
                    )
                score += reward
                embedded_observation = self.encoder(
                    torch.from_numpy(next_model_observation).float().to(self.device)
                )
                model_observation = next_model_observation
                observation = next_observation
                if done:
                    if train:
                        self.num_total_episode += 1
                        self.writer.add_scalar(
                            "training score", score, self.num_total_episode
                        )
                    else:
                        score_lst = np.append(score_lst, score)
                    break
        if not train:
            evaluate_score = score_lst.mean()
            print("evaluate score : ", evaluate_score)
            self.writer.add_scalar("test score", evaluate_score, self.num_total_episode)
