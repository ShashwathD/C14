# C14 Robot Runtime Baseline (Pruned)

This repo is a pruned C14-focused baseline for an edge-deployable world-model robot stack.

## What is included
- Goal-conditioned Dreamer/RSSM core (`dreamer/modules`, `dreamer/algorithms/dreamer.py`)
- Robot runtime environment abstraction (`dreamer/envs/envs.py`)
- RGB-D preprocessing (`dreamer/envs/wrappers.py`)
- Command-to-goal encoder (`dreamer/robot/goal.py`)
- Uncertainty-aware fallback (`dreamer/robot/fallback.py`)
- ONNX/TensorRT export helper (`dreamer/robot/export.py`)
- Robot config preset (`dreamer/configs/robot-c14.yml`)

## Install
```bash
pip install -r requirements.txt
```

## Run
Calibrate fallback threshold:
```bash
python main.py --config robot-c14 --mode calibrate_threshold --command "go to the orange ball"
```

Run robot runtime loop:
```bash
python main.py --config robot-c14 --mode robot_runtime --command "go to the orange ball" --fp16
```

Verbose console tracing (camera stats + latent previews):
```bash
python main.py --config robot-c14 --mode robot_runtime --command "go to the orange ball" --console_trace --trace_interval 1 --latent_preview_dims 8
```

Export ONNX/TensorRT-ready artifacts:
```bash
python main.py --config robot-c14 --mode export --export_dir runtime_exports --fp16
```

## Notes
- `camera_source: mock` (default) runs without hardware.
- Set `camera_source: oakd` and `serial_port` in `robot-c14.yml` for hardware integration.
- TensorRT engine build requires `trtexec` on path.
