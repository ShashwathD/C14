import unittest

from dreamer.robot.fallback import (
    UncertaintyAwareFallback,
    calibrate_uncertainty_threshold,
)


class FallbackTests(unittest.TestCase):
    def test_calibration_mean_plus_three_std(self):
        threshold = calibrate_uncertainty_threshold([1.0, 2.0, 3.0])
        self.assertAlmostEqual(threshold, 2.0 + 3.0 * 0.81649658, places=5)

    def test_fallback_trigger_and_release(self):
        fallback = UncertaintyAwareFallback(threshold=1.0, trigger_frames=3, release_frames=5)

        # Trigger after 3 consecutive high-MSE frames.
        self.assertFalse(fallback.update(1.2).use_fallback)
        self.assertFalse(fallback.update(1.1).use_fallback)
        self.assertTrue(fallback.update(1.3).use_fallback)

        # Stay in fallback until 5 consecutive low-MSE frames.
        for _ in range(4):
            self.assertTrue(fallback.update(0.4).use_fallback)
        self.assertFalse(fallback.update(0.3).use_fallback)


if __name__ == "__main__":
    unittest.main()
