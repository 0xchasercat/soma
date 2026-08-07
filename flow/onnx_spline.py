"""ONNX-safe rational-quadratic coupling transform.

`nflows` implements linear tails by boolean-indexing only values inside the
spline bound. Legacy ONNX tracing freezes the number of selected values from
the dummy input, so a later sample with a different mask crashes at runtime.
This module evaluates the spline on a clamped full tensor and selects tails with
`torch.where`, preserving identical mathematics without data-dependent shapes.
"""

import math

import torch
from torch.nn import functional as F
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform


def branch_free_rational_quadratic(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    *,
    inverse: bool,
    tail_bound: float,
    min_bin_width: float,
    min_bin_height: float,
    min_derivative: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_bins = unnormalized_widths.shape[-1]
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    # Outside values are restored exactly by torch.where below. Clamping only
    # gives the vectorized spline a valid bin for those otherwise-unused rows.
    spline_inputs = torch.clamp(inputs, -tail_bound, tail_bound)

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    widths_cumulative = -tail_bound + 2 * tail_bound * torch.cumsum(widths, dim=-1)
    left = torch.full_like(widths_cumulative[..., :1], -tail_bound)
    right = torch.full_like(widths_cumulative[..., :1], tail_bound)
    cumwidths = torch.cat([left, widths_cumulative[..., :-1], right], dim=-1)
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    heights_cumulative = -tail_bound + 2 * tail_bound * torch.cumsum(heights, dim=-1)
    bottom = torch.full_like(heights_cumulative[..., :1], -tail_bound)
    top = torch.full_like(heights_cumulative[..., :1], tail_bound)
    cumheights = torch.cat([bottom, heights_cumulative[..., :-1], top], dim=-1)
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    boundary_constant = math.log(math.exp(1 - min_derivative) - 1)
    boundary = torch.full_like(unnormalized_derivatives[..., :1], boundary_constant)
    derivatives = min_derivative + F.softplus(
        torch.cat([boundary, unnormalized_derivatives, boundary], dim=-1)
    )

    bin_locations = cumheights if inverse else cumwidths
    # Count lower edges. Excluding the final upper edge naturally maps an input
    # exactly at +tail_bound to the last bin without an in-place epsilon edit.
    bin_index = torch.sum(
        spline_inputs.unsqueeze(-1) >= bin_locations[..., :-1], dim=-1
    ) - 1
    bin_index = torch.clamp(bin_index, 0, num_bins - 1).unsqueeze(-1)

    input_cumwidths = cumwidths.gather(-1, bin_index)[..., 0]
    input_bin_widths = widths.gather(-1, bin_index)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_index)[..., 0]
    input_heights = heights.gather(-1, bin_index)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_index)[..., 0]
    input_derivatives = derivatives.gather(-1, bin_index)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_index)[..., 0]

    if inverse:
        a = (spline_inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (
            spline_inputs - input_cumheights
        ) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
        c = -input_delta * (spline_inputs - input_cumheights)
        discriminant = torch.clamp(b.square() - 4 * a * c, min=0)
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        spline_outputs = root * input_bin_widths + input_cumwidths
        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) * theta_one_minus_theta
        derivative_numerator = input_delta.square() * (
            input_derivatives_plus_one * root.square()
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).square()
        )
        spline_logabsdet = -(
            torch.log(derivative_numerator) - 2 * torch.log(denominator)
        )
    else:
        theta = (spline_inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)
        numerator = input_heights * (
            input_delta * theta.square()
            + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) * theta_one_minus_theta
        spline_outputs = input_cumheights + numerator / denominator
        derivative_numerator = input_delta.square() * (
            input_derivatives_plus_one * theta.square()
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).square()
        )
        spline_logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    return (
        torch.where(inside, spline_outputs, inputs),
        torch.where(inside, spline_logabsdet, torch.zeros_like(inputs)),
    )


class OnnxSafeRationalQuadraticCoupling(
    PiecewiseRationalQuadraticCouplingTransform
):
    """State-dict-compatible coupling using branch-free linear tails."""

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        widths = transform_params[..., : self.num_bins]
        heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        derivatives = transform_params[..., 2 * self.num_bins :]
        return branch_free_rational_quadratic(
            inputs,
            widths,
            heights,
            derivatives,
            inverse=inverse,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )
