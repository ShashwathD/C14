import os
import shutil
import subprocess
import importlib.util
from typing import Dict

import torch
import torch.nn as nn


def _deterministic_actor_forward(actor, posterior, deterministic, goal=None):
    x = torch.cat((posterior, deterministic), -1)
    if actor.goal_size > 0:
        if goal is None:
            goal = torch.zeros(
                x.shape[0], actor.goal_size, device=x.device, dtype=x.dtype
            )
        elif goal.dim() == 1:
            goal = goal.unsqueeze(0)
        if goal.shape[0] != x.shape[0]:
            goal = goal.expand(x.shape[0], -1)
        x = torch.cat((x, goal.to(device=x.device, dtype=x.dtype)), -1)

    logits = actor.network(x)
    if actor.discrete_action_bool:
        return torch.softmax(logits, dim=-1)

    mean, _std = torch.chunk(logits, 2, -1)
    mean = mean / actor.config.mean_scale
    mean = torch.tanh(mean)
    mean = actor.config.mean_scale * mean
    return torch.tanh(mean)


class RSSMStepExporter(nn.Module):
    def __init__(self, rssm):
        super().__init__()
        self.rssm = rssm
        self.goal_size = rssm.recurrent_model.goal_size

    def forward(self, state, action, deterministic, goal=None):
        deterministic = self.rssm.recurrent_model(
            state, action, deterministic, goal=goal
        )
        _prior_dist, prior = self.rssm.transition_model(deterministic)
        return deterministic, prior


class ActorExporter(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor
        self.goal_size = actor.goal_size

    def forward(self, posterior, deterministic, goal=None):
        return _deterministic_actor_forward(self.actor, posterior, deterministic, goal)


def _try_build_tensorrt_engine(onnx_path: str, fp16: bool) -> str:
    trtexec_path = shutil.which("trtexec")
    if trtexec_path is None:
        return ""

    engine_path = os.path.splitext(onnx_path)[0] + ".engine"
    command = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
    ]
    if fp16:
        command.append("--fp16")

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return engine_path
    except Exception:
        return ""


def export_runtime_artifacts(
    agent,
    observation_shape,
    export_dir: str,
    fp16: bool = False,
    opset: int = 17,
) -> Dict[str, str]:
    if importlib.util.find_spec("onnx") is None:
        raise RuntimeError(
            "ONNX export requires the 'onnx' package. Install it with 'pip install onnx==1.14.1'."
        )

    os.makedirs(export_dir, exist_ok=True)

    dtype = torch.float16 if fp16 else torch.float32
    device = torch.device(agent.device)

    agent.encoder.eval()
    agent.rssm.eval()
    agent.actor.eval()

    agent.encoder.to(device=device, dtype=dtype)
    agent.rssm.to(device=device, dtype=dtype)
    agent.actor.to(device=device, dtype=dtype)

    encoder_path = os.path.join(export_dir, "encoder.onnx")
    rssm_path = os.path.join(export_dir, "rssm_step.onnx")
    actor_path = os.path.join(export_dir, "actor.onnx")

    dummy_obs = torch.zeros(1, *observation_shape, device=device, dtype=dtype)
    torch.onnx.export(
        agent.encoder,
        (dummy_obs,),
        encoder_path,
        opset_version=opset,
        input_names=["observation"],
        output_names=["embedded"],
        dynamic_axes={"observation": {0: "batch"}, "embedded": {0: "batch"}},
    )

    rssm_exporter = RSSMStepExporter(agent.rssm).to(device=device, dtype=dtype).eval()
    dummy_state = torch.zeros(
        1, agent.config.stochastic_size, device=device, dtype=dtype
    )
    dummy_action = torch.zeros(1, agent.action_size, device=device, dtype=dtype)
    dummy_deterministic = torch.zeros(
        1, agent.config.deterministic_size, device=device, dtype=dtype
    )

    if agent.goal_size > 0:
        dummy_goal = torch.zeros(1, agent.goal_size, device=device, dtype=dtype)
        torch.onnx.export(
            rssm_exporter,
            (dummy_state, dummy_action, dummy_deterministic, dummy_goal),
            rssm_path,
            opset_version=opset,
            input_names=["state", "action", "deterministic", "goal"],
            output_names=["next_deterministic", "prior"],
            dynamic_axes={
                "state": {0: "batch"},
                "action": {0: "batch"},
                "deterministic": {0: "batch"},
                "goal": {0: "batch"},
                "next_deterministic": {0: "batch"},
                "prior": {0: "batch"},
            },
        )
    else:
        class _RSSMStepNoGoal(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, state, action, deterministic):
                return self.model(state, action, deterministic, None)

        rssm_no_goal = _RSSMStepNoGoal(rssm_exporter).to(device=device, dtype=dtype).eval()
        torch.onnx.export(
            rssm_no_goal,
            (dummy_state, dummy_action, dummy_deterministic),
            rssm_path,
            opset_version=opset,
            input_names=["state", "action", "deterministic"],
            output_names=["next_deterministic", "prior"],
            dynamic_axes={
                "state": {0: "batch"},
                "action": {0: "batch"},
                "deterministic": {0: "batch"},
                "next_deterministic": {0: "batch"},
                "prior": {0: "batch"},
            },
        )

    actor_exporter = ActorExporter(agent.actor).to(device=device, dtype=dtype).eval()
    dummy_posterior = torch.zeros(
        1, agent.config.stochastic_size, device=device, dtype=dtype
    )

    if agent.goal_size > 0:
        dummy_goal = torch.zeros(1, agent.goal_size, device=device, dtype=dtype)
        torch.onnx.export(
            actor_exporter,
            (dummy_posterior, dummy_deterministic, dummy_goal),
            actor_path,
            opset_version=opset,
            input_names=["posterior", "deterministic", "goal"],
            output_names=["action"],
            dynamic_axes={
                "posterior": {0: "batch"},
                "deterministic": {0: "batch"},
                "goal": {0: "batch"},
                "action": {0: "batch"},
            },
        )
    else:
        class _ActorNoGoal(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, posterior, deterministic):
                return self.model(posterior, deterministic, None)

        actor_no_goal = _ActorNoGoal(actor_exporter).to(device=device, dtype=dtype).eval()
        torch.onnx.export(
            actor_no_goal,
            (dummy_posterior, dummy_deterministic),
            actor_path,
            opset_version=opset,
            input_names=["posterior", "deterministic"],
            output_names=["action"],
            dynamic_axes={
                "posterior": {0: "batch"},
                "deterministic": {0: "batch"},
                "action": {0: "batch"},
            },
        )

    engines = {}
    for key, path in {
        "encoder_engine": encoder_path,
        "rssm_engine": rssm_path,
        "actor_engine": actor_path,
    }.items():
        engine_path = _try_build_tensorrt_engine(path, fp16=fp16)
        engines[key] = engine_path

    result = {
        "encoder_onnx": encoder_path,
        "rssm_onnx": rssm_path,
        "actor_onnx": actor_path,
    }
    result.update(engines)
    return result
