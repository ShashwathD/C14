from typing import Iterable

import numpy as np

from dreamer.robot.types import ControlAction, FallbackDecision, RobotObservation


def calibrate_uncertainty_threshold(mse_values: Iterable[float], sigma: float = 3.0) -> float:
    values = np.asarray(list(mse_values), dtype=np.float32)
    if values.size == 0:
        raise ValueError("mse_values must contain at least one element")
    mean = float(values.mean())
    std = float(values.std())
    return mean + sigma * std


class UncertaintyAwareFallback:
    def __init__(
        self,
        threshold: float,
        trigger_frames: int = 3,
        release_frames: int = 5,
    ):
        self.threshold = float(threshold)
        self.trigger_frames = int(trigger_frames)
        self.release_frames = int(release_frames)
        self.high_count = 0
        self.low_count = 0
        self.use_fallback = False

    def reset(self):
        self.high_count = 0
        self.low_count = 0
        self.use_fallback = False

    def update(self, mse: float) -> FallbackDecision:
        mse = float(mse)
        if mse > self.threshold:
            self.high_count += 1
            self.low_count = 0
        else:
            self.low_count += 1
            self.high_count = 0

        if not self.use_fallback and self.high_count >= self.trigger_frames:
            self.use_fallback = True
        elif self.use_fallback and self.low_count >= self.release_frames:
            self.use_fallback = False

        return FallbackDecision(
            use_fallback=self.use_fallback,
            mse=mse,
            threshold=self.threshold,
            high_count=self.high_count,
            low_count=self.low_count,
        )


class DeterministicFallbackPolicy:
    """
    Picks a safe direction from depth by selecting the clearest of left/center/right.
    """

    def __init__(
        self,
        forward_speed: float = 0.10,
        turn_speed: float = 0.60,
        min_depth_m: float = 0.35,
    ):
        self.forward_speed = float(forward_speed)
        self.turn_speed = float(turn_speed)
        self.min_depth_m = float(min_depth_m)

    def select_action(self, observation: RobotObservation) -> ControlAction:
        depth = observation.depth
        if depth is None or depth.size == 0:
            return ControlAction(linear_vel=0.0, angular_vel=self.turn_speed)

        depth = np.asarray(depth, dtype=np.float32)
        finite = np.isfinite(depth)
        if not np.any(finite):
            return ControlAction(linear_vel=0.0, angular_vel=self.turn_speed)

        clean = np.where(finite, depth, 0.0)
        height, width = clean.shape[:2]
        one_third = max(width // 3, 1)

        left = clean[:, :one_third].mean()
        center = clean[:, one_third : 2 * one_third].mean()
        right = clean[:, 2 * one_third :].mean()

        sectors = np.asarray([left, center, right], dtype=np.float32)
        best = int(np.argmax(sectors))

        if sectors[best] < self.min_depth_m:
            # Surrounded; rotate in place to seek free space.
            return ControlAction(linear_vel=0.0, angular_vel=self.turn_speed)

        if best == 1:
            return ControlAction(linear_vel=self.forward_speed, angular_vel=0.0)
        if best == 0:
            return ControlAction(
                linear_vel=0.5 * self.forward_speed, angular_vel=self.turn_speed
            )
        return ControlAction(
            linear_vel=0.5 * self.forward_speed, angular_vel=-self.turn_speed
        )
