"""
Validates the correctness of the implementations by comparing the number of
parameters to the expected number of parameters.
"""
from typing import Literal

import jax
import pytest

import pdequinox as pdeqx


@pytest.mark.parametrize(
    "num_spatial_dims,in_channels,out_channels,kernel_size,stride,dilation,use_bias,boundary_mode,zero_bias_init",
    [
        (
            num_spatial_dims,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            use_bias,
            boundary_mode,
            zero_bias_init,
        )
        for num_spatial_dims in [1, 2, 3]
        for in_channels in [1, 2, 5]
        for out_channels in [1, 2, 5]
        for kernel_size in [2, 3, 4, 5]
        for stride in [1, 2]
        for dilation in [1, 2, 3]
        for use_bias in [True, False]
        for boundary_mode in ["periodic", "dirichlet", "neumann"]
        for zero_bias_init in [True, False]
    ],
)
def test_physics_conv(
    num_spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    dilation: int,
    use_bias: bool,
    boundary_mode: Literal["periodic", "dirichlet", "neumann"],
    zero_bias_init: bool,
):
    net = pdeqx.conv.PhysicsConv(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        use_bias=use_bias,
        key=jax.random.PRNGKey(0),
        boundary_mode=boundary_mode,
        zero_bias_init=zero_bias_init,
    )

    num_parameters = pdeqx.count_parameters(net)

    # Compute the expected number of parameters
    num_parameters_expected = (
        in_channels * out_channels * kernel_size**num_spatial_dims
    ) + (out_channels if use_bias else 0)

    assert num_parameters == num_parameters_expected


@pytest.mark.parametrize(
    "num_spatial_dims,in_channels,out_channels,num_modes",
    [
        (num_spatial_dims, in_channels, out_channels, num_modes)
        for num_spatial_dims in [1, 2, 3]
        for in_channels in [1, 2, 5]
        for out_channels in [1, 2, 5]
        for num_modes in [1, 2, 5, 12]
    ],
)
def test_spectral_conv(
    num_spatial_dims: int,
    in_channels: int,
    out_channels: int,
    num_modes: int,
):
    net = pdeqx.conv.SpectralConv(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        num_modes=num_modes,
        key=jax.random.PRNGKey(0),
    )

    num_parameters = pdeqx.count_parameters(net)

    # Compute the expected number of parameters; factor of 2 for real and
    # imaginary parts; raising to 2**(D-1) because we use the real-valued FFT
    # (only have to save parameters for half of the relevant modes)
    num_parameters_expected = (
        in_channels * out_channels * num_modes**num_spatial_dims * 2
    ) * 2 ** (num_spatial_dims - 1)

    assert num_parameters == num_parameters_expected