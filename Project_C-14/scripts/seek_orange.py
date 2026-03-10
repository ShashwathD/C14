#!/usr/bin/env python3
import argparse
import time
from typing import Optional, Tuple

import numpy as np

from dreamer.envs.envs import make_robot_env
from dreamer.robot.types import ControlAction
from dreamer.utils.utils import load_config


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    try:
        return obj[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(obj, key, default)


def _flip_hw(rgb: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return rgb[::-1, ::-1], depth[::-1, ::-1]


def _orange_mask(rgb: np.ndarray, r_min: int, g_min: int, g_max: int, b_max: int, rg_delta: int) -> np.ndarray:
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    return (
        (r >= r_min)
        & (g >= g_min)
        & (g <= g_max)
        & (b <= b_max)
        & (r >= g + rg_delta)
    )


def _mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple orange-seeking script (no training).")
    parser.add_argument("--config", type=str, default="dreamer/configs/robot-c14.yml")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--forward-speed", type=float, default=0.12)
    parser.add_argument("--turn-kp", type=float, default=0.9)
    parser.add_argument("--search-omega", type=float, default=0.6)
    parser.add_argument("--min-area", type=float, default=0.01, help="Fraction of image that must be orange.")
    parser.add_argument("--stop-depth", type=float, default=0.35, help="Stop when median depth < this (m).")
    parser.add_argument("--flip", action="store_true", help="Rotate frames 180 degrees.")
    parser.add_argument("--no-depth", action="store_true", help="Ignore depth for stopping.")
    parser.add_argument("--r-min", type=int, default=120)
    parser.add_argument("--g-min", type=int, default=50)
    parser.add_argument("--g-max", type=int, default=200)
    parser.add_argument("--b-max", type=int, default=80)
    parser.add_argument("--rg-delta", type=int, default=30)
    args = parser.parse_args()

    config = load_config(args.config)
    env = make_robot_env(config)

    fps = int(_cfg_get(_cfg_get(config, "environment", None), "camera_fps", 15))
    sleep_s = 1.0 / max(1, fps)

    # Default to encoder flip setting if present and no explicit flag given.
    if not args.flip:
        enc_cfg = _cfg_get(_cfg_get(_cfg_get(config, "parameters", None), "dreamer", None), "encoder", None)
        args.flip = bool(_cfg_get(enc_cfg, "flip_hw", False))

    try:
        for step in range(int(args.max_steps)):
            observation = env.observe()
            rgb = observation.rgb
            depth = observation.depth
            if args.flip:
                rgb, depth = _flip_hw(rgb, depth)

            mask = _orange_mask(rgb, args.r_min, args.g_min, args.g_max, args.b_max, args.rg_delta)
            area = float(mask.mean())

            if area >= args.min_area:
                centroid = _mask_centroid(mask)
                if centroid is None:
                    action = ControlAction(0.0, args.search_omega)
                else:
                    cx, _cy = centroid
                    w = rgb.shape[1]
                    offset = (cx - (w - 1) / 2.0) / max(1.0, w / 2.0)
                    angular = float(-args.turn_kp * offset)
                    linear = float(args.forward_speed * max(0.2, 1.0 - abs(offset) * 1.2))

                    if not args.no_depth:
                        depth_vals = depth[mask]
                        if depth_vals.size > 0:
                            depth_median = float(np.nanmedian(depth_vals))
                            if np.isfinite(depth_median) and depth_median <= args.stop_depth:
                                env.motor_controller.send(ControlAction(0.0, 0.0))
                                break
                    action = ControlAction(linear, angular)
            else:
                action = ControlAction(0.0, args.search_omega)

            env.motor_controller.send(action)
            time.sleep(sleep_s)
    finally:
        env.motor_controller.stop()
        env.close()


if __name__ == "__main__":
    main()
