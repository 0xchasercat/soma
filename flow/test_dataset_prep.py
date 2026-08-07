"""Regression tests for preserving genuine human movement tails."""

import unittest

import numpy as np
import torch
from nflows.transforms.splines.rational_quadratic import (
    unconstrained_rational_quadratic_spline,
)

from audit_eval import (
    add_default_tremor,
    pin_dispatch_terminal_stop,
    resample_for_dispatch,
    sigma_lognormal_fit,
)
from dataset_prep import DROPPED_DELTA_INDEX, MODEL_DIM, N_FREE, N_STEPS, process_movement
from onnx_spline import branch_free_rational_quadratic
from train_flow import split_indices


def movement(points: list[tuple[float, float, float]]) -> dict:
    return {
        "sessionId": "session-a",
        "trajectory": [
            {"x": x, "y": y, "tMs": t} for x, y, t in points
        ],
        "clickTarget": {
            "boundingBox": {"x": 0, "y": 0, "width": 12, "height": 8}
        },
    }


class DatasetPreservationTest(unittest.TestCase):
    def test_branch_free_spline_matches_nflows_inside_and_outside_tail(self) -> None:
        generator = torch.Generator().manual_seed(42)
        inputs = torch.randn(4, 7, generator=generator) * 4
        inputs[0, 0], inputs[0, 1] = -8, 8
        widths = torch.randn(4, 7, 8, generator=generator)
        heights = torch.randn(4, 7, 8, generator=generator)
        derivatives = torch.randn(4, 7, 7, generator=generator)
        for inverse in (False, True):
            expected = unconstrained_rational_quadratic_spline(
                inputs,
                widths,
                heights,
                derivatives,
                inverse=inverse,
                tails="linear",
                tail_bound=6.0,
            )
            actual = branch_free_rational_quadratic(
                inputs,
                widths,
                heights,
                derivatives,
                inverse=inverse,
                tail_bound=6.0,
                min_bin_width=1e-3,
                min_bin_height=1e-3,
                min_derivative=1e-3,
            )
            np.testing.assert_allclose(
                actual[0].numpy(), expected[0].numpy(), rtol=1e-5, atol=1e-5
            )
            np.testing.assert_allclose(
                actual[1].numpy(), expected[1].numpy(), rtol=1e-5, atol=1e-5
            )

    def test_sigma_lognormal_metric_is_not_a_dead_binary_feature(self) -> None:
        time = np.linspace(0.01, 1.0, 60)
        bell = np.exp(-((np.log(time) + 0.8) ** 2) / (2 * 0.35**2)) / time
        self.assertGreater(sigma_lognormal_fit(bell), 0.7)
        self.assertEqual(sigma_lognormal_fit(np.full(60, 10.0)), 0.0)

    def test_runtime_audit_caps_frames_and_pins_tremor_endpoints(self) -> None:
        u = np.array([0.0, 10.0])
        v = np.array([0.0, 0.0])
        t = np.array([0.0, 3_600_000.0])
        resampled = resample_for_dispatch(u, v, t)
        self.assertEqual(len(resampled[0]), 4096)
        stopped = pin_dispatch_terminal_stop(resampled, True)
        self.assertEqual(float(stopped[0][-2]), float(stopped[0][-1]))
        tremored = add_default_tremor(
            *resampled, amplitude=0.3, rng=np.random.default_rng(42)
        )
        self.assertEqual(float(tremored[0][0]), 0.0)
        self.assertEqual(float(tremored[0][-1]), 10.0)
        self.assertEqual(float(tremored[2][-1]), 3_600_000.0)

    def assert_preserved(self, points: list[tuple[float, float, float]]) -> None:
        result, reason = process_movement(movement(points))
        self.assertIsNone(reason)
        self.assertIsNotNone(result)
        assert result is not None
        x, context, meta = result
        self.assertEqual(x.shape, (MODEL_DIM,))
        self.assertEqual(context.shape, (3,))
        self.assertEqual(meta.shape, (4,))

    def test_preserves_short_fast_reach(self) -> None:
        self.assert_preserved([(0, 0, 0), (2, 1, 8)])

    def test_preserves_long_hesitant_reach(self) -> None:
        self.assert_preserved([(0, 0, 0), (30, 40, 12_000), (4, 2, 25_000)])

    def test_preserves_high_curvature_and_correction(self) -> None:
        self.assert_preserved(
            [(0, 0, 0), (100, 0, 20), (-80, 50, 40), (3, 2, 60)]
        )

    def test_touchdown_delta_is_modeled_and_endpoint_is_recoverable(self) -> None:
        result, reason = process_movement(
            movement([(0, 0, 0), (1, 0, 99), (10, 0, 100)])
        )
        self.assertIsNone(reason)
        assert result is not None
        x, _, _ = result
        du_free = x[:N_FREE]
        dv_free = x[N_FREE : 2 * N_FREE]
        du = np.insert(du_free, DROPPED_DELTA_INDEX, 1.0 - du_free.sum())
        dv = np.insert(dv_free, DROPPED_DELTA_INDEX, -dv_free.sum())
        self.assertEqual(len(du), N_STEPS)
        self.assertAlmostEqual(float(du.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(dv.sum()), 0.0, places=6)
        # The abrupt measured arrival remains the final modeled delta instead
        # of being manufactured from the endpoint constraint.
        self.assertGreater(float(du[-1]), float(du[-2]))

    def test_reports_only_noncanonical_zero_displacement(self) -> None:
        result, reason = process_movement(
            movement([(0, 0, 0), (20, 20, 10), (0, 0, 20)])
        )
        self.assertIsNone(result)
        self.assertEqual(reason, "zero endpoint displacement")

    def test_session_groups_do_not_cross_validation_boundary(self) -> None:
        groups = ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"]
        train, validation, kind, _, _ = split_indices(len(groups), groups, seed=42)
        self.assertEqual(kind, "session-grouped")
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertTrue(
            {groups[index] for index in train}.isdisjoint(
                {groups[index] for index in validation}
            )
        )
        repeated = split_indices(len(groups), groups, seed=42)
        self.assertEqual((train, validation), repeated[:2])


if __name__ == "__main__":
    unittest.main()
