"""
Validates the correctness of the implementations by comparing the number of
parameters to the expected number of parameters.
"""
from typing import Callable, Literal

import jax
import pytest

import pdequinox as pdeqx

# Basic Convolutional Layers


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


@pytest.mark.parametrize(
    "num_spatial_dims,in_channels,out_channels,use_bias,zero_bias_init",
    [
        (num_spatial_dims, in_channels, out_channels, use_bias, zero_bias_init)
        for num_spatial_dims in [1, 2, 3]
        for in_channels in [1, 2, 5]
        for out_channels in [1, 2, 5]
        for use_bias in [True, False]
        for zero_bias_init in [True, False]
    ],
)
def test_pointwise_linear_conv(
    num_spatial_dims: int,
    in_channels: int,
    out_channels: int,
    use_bias: bool,
    zero_bias_init: bool,
):
    net = pdeqx.conv.PointwiseLinearConv(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        use_bias=use_bias,
        key=jax.random.PRNGKey(0),
        zero_bias_init=zero_bias_init,
    )

    num_parameters = pdeqx.count_parameters(net)

    # Compute the expected number of parameters
    num_parameters_expected = in_channels * out_channels

    if use_bias:
        num_parameters_expected += out_channels

    assert num_parameters == num_parameters_expected


# Blocks


@pytest.mark.parametrize(
    "num_spatial_dims,in_channels,out_channels,boundary_mode,activation,kernel_size,use_norm,use_bias,zero_bias_init",
    [
        (
            num_spatial_dims,
            in_channels,
            out_channels,
            boundary_mode,
            activation,
            kernel_size,
            use_norm,
            use_bias,
            zero_bias_init,
        )
        for num_spatial_dims in [1, 2, 3]
        for in_channels in [1, 2, 5]
        for out_channels in [1, 2, 5]
        for boundary_mode in ["periodic", "dirichlet", "neumann"]
        for activation in [jax.nn.relu, jax.nn.sigmoid]
        for kernel_size in [2, 3, 4, 5]
        for use_norm in [True, False]
        for use_bias in [True, False]
        for zero_bias_init in [True, False]
    ],
)
def test_ClassicDoubleConvBlock(
    num_spatial_dims: int,
    in_channels: int,
    out_channels: int,
    boundary_mode: Literal["periodic", "dirichlet", "neumann"],
    activation: Callable,
    kernel_size: int,
    use_norm: bool,
    use_bias: bool,
    zero_bias_init: bool,
):
    block = pdeqx.blocks.ClassicDoubleConvBlock(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        boundary_mode=boundary_mode,
        activation=activation,
        key=jax.random.PRNGKey(0),
        kernel_size=kernel_size,
        use_norm=use_norm,
        use_bias=use_bias,
        zero_bias_init=zero_bias_init,
    )

    num_parameters = pdeqx.count_parameters(block)

    # Compute the expected number of parameters
    first_conv_num_parameters = (
        in_channels * out_channels * kernel_size**num_spatial_dims
    ) + (out_channels if use_bias else 0)
    second_conv_num_parameters = (
        out_channels * out_channels * kernel_size**num_spatial_dims
    ) + (out_channels if use_bias else 0)
    first_norm_num_parameters = 2 * out_channels if use_norm else 0
    second_norm_num_parameters = 2 * out_channels if use_norm else 0

    num_parameters_expected = (
        first_conv_num_parameters
        + second_conv_num_parameters
        + first_norm_num_parameters
        + second_norm_num_parameters
    )

    assert num_parameters == num_parameters_expected

    block_factory = pdeqx.blocks.ClassicDoubleConvBlockFactory(
        kernel_size=kernel_size,
        use_norm=use_norm,
        use_bias=use_bias,
        zero_bias_init=zero_bias_init,
    )
    block_from_factory = block_factory(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        activation=activation,
        boundary_mode=boundary_mode,
        key=jax.random.PRNGKey(0),
    )

    num_parameters_from_factory = pdeqx.count_parameters(block_from_factory)

    assert num_parameters_from_factory == num_parameters_expected


@pytest.mark.parametrize(
    "num_spatial_dims,in_channels,out_channels,boundary_mode,activation,kernel_size,use_norm,use_bias,zero_bias_init",
    [
        (
            num_spatial_dims,
            in_channels,
            out_channels,
            boundary_mode,
            activation,
            kernel_size,
            use_norm,
            use_bias,
            zero_bias_init,
        )
        for num_spatial_dims in [1, 2, 3]
        for in_channels in [1, 2, 5]
        for out_channels in [1, 2, 5]
        for boundary_mode in ["periodic", "dirichlet", "neumann"]
        for activation in [jax.nn.relu, jax.nn.sigmoid]
        for kernel_size in [2, 3, 4, 5]
        for use_norm in [True, False]
        for use_bias in [True, False]
        for zero_bias_init in [True, False]
    ],
)
def test_ClassicResBlock(
    num_spatial_dims: int,
    in_channels: int,
    out_channels: int,
    boundary_mode: Literal["periodic", "dirichlet", "neumann"],
    activation: Callable,
    kernel_size: int,
    use_norm: bool,
    use_bias: bool,
    zero_bias_init: bool,
):
    block = pdeqx.blocks.ClassicResBlock(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        boundary_mode=boundary_mode,
        activation=activation,
        key=jax.random.PRNGKey(0),
        kernel_size=kernel_size,
        use_norm=use_norm,
        use_bias=use_bias,
        zero_bias_init=zero_bias_init,
    )

    num_parameters = pdeqx.count_parameters(block)

    # Compute the expected number of parameters
    num_parameters_expected = 0

    # First conv kernel
    num_parameters_expected += (
        in_channels * out_channels * kernel_size**num_spatial_dims
    )
    # First conv bias
    if use_bias:
        num_parameters_expected += out_channels
    # First norm
    if use_norm:
        num_parameters_expected += 2 * out_channels

    # Second conv kernel
    num_parameters_expected += (
        out_channels * out_channels * kernel_size**num_spatial_dims
    )
    # Second conv bias
    if use_bias:
        num_parameters_expected += out_channels
    # Second norm
    if use_norm:
        num_parameters_expected += 2 * out_channels

    # Bypass conv kernel
    if in_channels != out_channels:
        # Bypass is a 1x1 convolution
        num_parameters_expected += in_channels * out_channels
        # The bias in the bypass conv is always deactivated!
    # Bypass norm
    if in_channels != out_channels and use_norm:
        num_parameters_expected += 2 * out_channels

    assert num_parameters == num_parameters_expected

    block_factory = pdeqx.blocks.ClassicResBlockFactory(
        kernel_size=kernel_size,
        use_norm=use_norm,
        use_bias=use_bias,
        zero_bias_init=zero_bias_init,
    )
    block_from_factory = block_factory(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        activation=activation,
        boundary_mode=boundary_mode,
        key=jax.random.PRNGKey(0),
    )

    num_parameters_from_factory = pdeqx.count_parameters(block_from_factory)

    assert num_parameters_from_factory == num_parameters_expected


@pytest.mark.parametrize(
    "num_spatial_dims,in_channels,out_channels,activation,num_modes,use_bias,zero_bias_init",
    [
        (
            num_spatial_dims,
            in_channels,
            out_channels,
            activation,
            num_modes,
            use_bias,
            zero_bias_init,
        )
        for num_spatial_dims in [1, 2, 3]
        for in_channels in [1, 2, 5]
        for out_channels in [1, 2, 5]
        for activation in [jax.nn.relu, jax.nn.sigmoid]
        for num_modes in [1, 2, 5, 12]
        for use_bias in [True, False]
        for zero_bias_init in [True, False]
    ],
)
def test_ClassicSpectralBlock(
    num_spatial_dims: int,
    in_channels: int,
    out_channels: int,
    activation: Callable,
    num_modes: int,
    use_bias: bool,
    zero_bias_init: bool,
):
    block = pdeqx.blocks.ClassicSpectralBlock(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        activation=activation,
        key=jax.random.PRNGKey(0),
        num_modes=num_modes,
        use_bias=use_bias,
        zero_bias_init=zero_bias_init,
    )

    num_parameters = pdeqx.count_parameters(block)

    # Compute the expected number of parameters
    num_parameters_expected = 0

    # Spectral conv kernel  (factor of two to separately save real and imaginary
    # parts) (factor of 2**(D-1) because we use the real-valued FFT
    # (only have to save parameters for half of the relevant modes))
    num_parameters_expected += (
        in_channels * out_channels * num_modes**num_spatial_dims * 2
    ) * 2 ** (num_spatial_dims - 1)

    # Spectral conv bypass 1x1 conv
    num_parameters_expected += in_channels * out_channels

    # Spectral conv bypass bias
    if use_bias:
        num_parameters_expected += out_channels

    assert num_parameters == num_parameters_expected

    block_factory = pdeqx.blocks.ClassicSpectralBlockFactory(
        num_modes=num_modes,
        use_bias=use_bias,
        zero_bias_init=zero_bias_init,
    )

    block_from_factory = block_factory(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        activation=activation,
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
    )

    num_parameters_from_factory = pdeqx.count_parameters(block_from_factory)

    assert num_parameters_from_factory == num_parameters_expected


@pytest.mark.parametrize(
    "num_spatial_dims,in_channels,out_channels,boundary_mode,activation,kernel_size,dilation_rates,use_norm,use_bias,zero_bias_init",
    [
        (
            num_spatial_dims,
            in_channels,
            out_channels,
            boundary_mode,
            activation,
            kernel_size,
            dilation_rates,
            use_norm,
            use_bias,
            zero_bias_init,
        )
        for num_spatial_dims in [1, 2, 3]
        for in_channels in [
            1,
            5,
        ]
        for out_channels in [
            1,
            3,
        ]
        for boundary_mode in ["periodic", "dirichlet", "neumann"]
        for activation in [jax.nn.relu, jax.nn.sigmoid]
        for kernel_size in [2, 3, 4, 5]
        for dilation_rates in [(1,), (1, 2, 1), (1, 2, 4, 2, 1), (1, 3, 8)]
        for use_norm in [True, False]
        for use_bias in [True, False]
        for zero_bias_init in [True, False]
    ],
)
def test_DilatedResBlock(
    num_spatial_dims: int,
    in_channels: int,
    out_channels: int,
    boundary_mode: Literal["periodic", "dirichlet", "neumann"],
    activation: Callable,
    kernel_size: int,
    dilation_rates: tuple[int],
    use_norm: bool,
    use_bias: bool,
    zero_bias_init: bool,
):
    block = pdeqx.blocks.DilatedResBlock(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        boundary_mode=boundary_mode,
        activation=activation,
        key=jax.random.PRNGKey(0),
        kernel_size=kernel_size,
        dilation_rates=dilation_rates,
        use_norm=use_norm,
        use_bias=use_bias,
        zero_bias_init=zero_bias_init,
    )

    num_parameters = pdeqx.count_parameters(block)

    # Compute the expected number of parameters
    num_parameters_expected = 0

    # First norm layer
    if use_norm:
        num_parameters_expected += 2 * in_channels

    # First conv kernel that performs the adjustment in the number of channels
    num_parameters_expected += (
        in_channels * out_channels * kernel_size**num_spatial_dims
    )
    # First conv bias
    if use_bias:
        num_parameters_expected += out_channels

    # Iterate over all the other convolutions
    for dilation_rate in dilation_rates[1:]:
        if use_norm:
            num_parameters_expected += 2 * out_channels
        num_parameters_expected += (
            out_channels * out_channels * kernel_size**num_spatial_dims
        )
        if use_bias:
            num_parameters_expected += out_channels

    # Bypass norm
    if in_channels != out_channels and use_norm:
        num_parameters_expected += 2 * in_channels
    # Bypass conv kernel
    if in_channels != out_channels:
        num_parameters_expected += in_channels * out_channels
        # The bias in the bypass conv is toggled
        if use_bias:
            num_parameters_expected += out_channels

    assert num_parameters == num_parameters_expected

    block_factory = pdeqx.blocks.DilatedResBlockFactory(
        kernel_size=kernel_size,
        dilation_rates=dilation_rates,
        use_norm=use_norm,
        use_bias=use_bias,
        zero_bias_init=zero_bias_init,
    )

    block_from_factory = block_factory(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        activation=activation,
        boundary_mode=boundary_mode,
        key=jax.random.PRNGKey(0),
    )

    num_parameters_from_factory = pdeqx.count_parameters(block_from_factory)

    assert num_parameters_from_factory == num_parameters_expected


@pytest.mark.parametrize(
    "num_spatial_dims,in_channels,out_channels,use_bias,zero_bias_init",
    [
        (
            num_spatial_dims,
            in_channels,
            out_channels,
            use_bias,
            zero_bias_init,
        )
        for num_spatial_dims in [1, 2, 3]
        for in_channels in [
            1,
            5,
        ]
        for out_channels in [1, 3]
        for use_bias in [True, False]
        for zero_bias_init in [True, False]
    ],
)
def test_LinearChannelAdjustBlock(
    num_spatial_dims: int,
    in_channels: int,
    out_channels: int,
    use_bias: bool,
    zero_bias_init: bool,
):
    block = pdeqx.blocks.LinearChannelAdjustBlock(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        key=jax.random.PRNGKey(0),
        use_bias=use_bias,
        zero_bias_init=zero_bias_init,
    )

    num_parameters = pdeqx.count_parameters(block)

    # Compute the expected number of parameters
    num_parameters_expected = in_channels * out_channels

    if use_bias:
        num_parameters_expected += out_channels

    assert num_parameters == num_parameters_expected

    block_factory = pdeqx.blocks.LinearChannelAdjustBlockFactory(
        use_bias=use_bias,
        zero_bias_init=zero_bias_init,
    )

    block_from_factory = block_factory(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        activation=jax.nn.relu,
        boundary_mode="periodic",
        key=jax.random.PRNGKey(0),
    )

    num_parameters_from_factory = pdeqx.count_parameters(block_from_factory)

    assert num_parameters_from_factory == num_parameters_expected


@pytest.mark.parametrize(
    "num_spatial_dims,in_channels,out_channels,kernel_size,use_bias,boundary_mode",
    [
        (
            num_spatial_dims,
            in_channels,
            out_channels,
            kernel_size,
            use_bias,
            boundary_mode,
        )
        for num_spatial_dims in [1, 2, 3]
        for in_channels in [
            1,
            5,
        ]
        for out_channels in [
            1,
            3,
        ]
        for kernel_size in [2, 3, 4, 5]
        for use_bias in [True, False]
        for boundary_mode in ["periodic", "dirichlet", "neumann"]
    ],
)
def test_LinearConvBlock(
    num_spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    use_bias: bool,
    boundary_mode: Literal["periodic", "dirichlet", "neumann"],
):
    block = pdeqx.blocks.LinearConvBlock(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        key=jax.random.PRNGKey(0),
        use_bias=use_bias,
        boundary_mode=boundary_mode,
    )

    num_parameters = pdeqx.count_parameters(block)

    # Compute the expected number of parameters
    num_parameters_expected = (
        in_channels * out_channels * kernel_size**num_spatial_dims
    ) + (out_channels if use_bias else 0)

    assert num_parameters == num_parameters_expected

    block_factory = pdeqx.blocks.LinearConvBlockFactory(
        kernel_size=kernel_size,
        use_bias=use_bias,
    )

    block_from_factory = block_factory(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        activation=jax.nn.relu,
        boundary_mode=boundary_mode,
        key=jax.random.PRNGKey(0),
    )

    num_parameters_from_factory = pdeqx.count_parameters(block_from_factory)

    assert num_parameters_from_factory == num_parameters_expected


@pytest.mark.parametrize(
    "num_spatial_dims,in_channels,out_channels,kernel_size,factor,use_bias,boundary_mode",
    [
        (
            num_spatial_dims,
            in_channels,
            out_channels,
            kernel_size,
            factor,
            use_bias,
            boundary_mode,
        )
        for num_spatial_dims in [1, 2, 3]
        for in_channels in [
            1,
            5,
        ]
        for out_channels in [
            1,
            3,
        ]
        for kernel_size in [2, 3, 4, 5]
        for factor in [2, 3, 4]
        for use_bias in [True, False]
        for boundary_mode in ["periodic", "dirichlet", "neumann"]
    ],
)
def test_LinearConvDownBlock(
    num_spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    factor: int,
    use_bias: bool,
    boundary_mode: Literal["periodic", "dirichlet", "neumann"],
):
    block = pdeqx.blocks.LinearConvDownBlock(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        factor=factor,
        key=jax.random.PRNGKey(0),
        use_bias=use_bias,
        boundary_mode=boundary_mode,
    )

    num_parameters = pdeqx.count_parameters(block)

    # Compute the expected number of parameters
    num_parameters_expected = (
        in_channels * out_channels * kernel_size**num_spatial_dims
    ) + (out_channels if use_bias else 0)

    assert num_parameters == num_parameters_expected

    block_factory = pdeqx.blocks.LinearConvDownBlockFactory(
        kernel_size=kernel_size,
        factor=factor,
        use_bias=use_bias,
    )

    block_from_factory = block_factory(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        activation=jax.nn.relu,
        boundary_mode=boundary_mode,
        key=jax.random.PRNGKey(0),
    )

    num_parameters_from_factory = pdeqx.count_parameters(block_from_factory)

    assert num_parameters_from_factory == num_parameters_expected


@pytest.mark.parametrize(
    "num_spatial_dims,in_channels,out_channels,kernel_size,factor,output_padding,use_bias,boundary_mode",
    [
        (
            num_spatial_dims,
            in_channels,
            out_channels,
            kernel_size,
            factor,
            output_padding,
            use_bias,
            boundary_mode,
        )
        for num_spatial_dims in [1, 2, 3]
        for in_channels in [
            1,
            5,
        ]
        for out_channels in [
            1,
            3,
        ]
        for kernel_size in [2, 3, 4, 5]
        for factor in [2, 3, 4]
        for output_padding in [
            0,
            1,
        ]
        for use_bias in [True, False]
        for boundary_mode in ["periodic", "dirichlet", "neumann"]
    ],
)
def test_LinearConvUpBlock(
    num_spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    factor: int,
    output_padding: int,
    use_bias: bool,
    boundary_mode: Literal["periodic", "dirichlet", "neumann"],
):
    block = pdeqx.blocks.LinearConvUpBlock(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        factor=factor,
        output_padding=output_padding,
        key=jax.random.PRNGKey(0),
        use_bias=use_bias,
        boundary_mode=boundary_mode,
    )

    num_parameters = pdeqx.count_parameters(block)

    # Compute the expected number of parameters
    num_parameters_expected = (
        in_channels * out_channels * kernel_size**num_spatial_dims
    ) + (out_channels if use_bias else 0)

    assert num_parameters == num_parameters_expected

    block_factory = pdeqx.blocks.LinearConvUpBlockFactory(
        kernel_size=kernel_size,
        factor=factor,
        use_bias=use_bias,
        output_padding=output_padding,
    )

    block_from_factory = block_factory(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        activation=jax.nn.relu,
        boundary_mode=boundary_mode,
        key=jax.random.PRNGKey(0),
    )

    num_parameters_from_factory = pdeqx.count_parameters(block_from_factory)

    assert num_parameters_from_factory == num_parameters_expected


@pytest.mark.parametrize(
    "num_spatial_dims,in_channels,out_channels,activation,kernel_size,use_norm,use_bias,zero_bias_init,boundary_mode",
    [
        (
            num_spatial_dims,
            in_channels,
            out_channels,
            activation,
            kernel_size,
            use_norm,
            use_bias,
            zero_bias_init,
            boundary_mode,
        )
        for num_spatial_dims in [1, 2, 3]
        for in_channels in [1, 2, 5]
        for out_channels in [1, 2, 5]
        for activation in [jax.nn.relu, jax.nn.sigmoid]
        for kernel_size in [2, 3, 4, 5]
        for use_norm in [True, False]
        for use_bias in [True, False]
        for zero_bias_init in [True, False]
        for boundary_mode in ["periodic", "dirichlet", "neumann"]
    ],
)
def test_ModernResBlock(
    num_spatial_dims: int,
    in_channels: int,
    out_channels: int,
    activation: Callable,
    kernel_size: int,
    use_norm: bool,
    use_bias: bool,
    zero_bias_init: bool,
    boundary_mode: Literal["periodic", "dirichlet", "neumann"],
):
    block = pdeqx.blocks.ModernResBlock(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        activation=activation,
        key=jax.random.PRNGKey(0),
        kernel_size=kernel_size,
        use_norm=use_norm,
        use_bias=use_bias,
        zero_bias_init=zero_bias_init,
        boundary_mode=boundary_mode,
    )

    num_parameters = pdeqx.count_parameters(block)

    # Compute the expected number of parameters
    num_parameters_expected = 0

    # First norm layer
    if use_norm:
        num_parameters_expected += 2 * in_channels

    # First conv kernel that performs the adjustment in the number of channels
    num_parameters_expected += (
        in_channels * out_channels * kernel_size**num_spatial_dims
    )
    # First conv bias
    if use_bias:
        num_parameters_expected += out_channels

    # Second norm layer
    if use_norm:
        num_parameters_expected += 2 * out_channels

    # Second conv kernel
    num_parameters_expected += (
        out_channels * out_channels * kernel_size**num_spatial_dims
    )
    # Second conv bias
    if use_bias:
        num_parameters_expected += out_channels

    if in_channels != out_channels:
        if use_norm:
            # Bypass norm
            num_parameters_expected += 2 * in_channels

        # Bypass is a 1x1 convolution
        num_parameters_expected += in_channels * out_channels
        # The bias in the bypass conv is always deactivated!

    assert num_parameters == num_parameters_expected

    block_factory = pdeqx.blocks.ModernResBlockFactory(
        kernel_size=kernel_size,
        use_norm=use_norm,
        use_bias=use_bias,
        zero_bias_init=zero_bias_init,
    )

    block_from_factory = block_factory(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        activation=activation,
        boundary_mode=boundary_mode,
        key=jax.random.PRNGKey(0),
    )

    num_parameters_from_factory = pdeqx.count_parameters(block_from_factory)

    assert num_parameters_from_factory == num_parameters_expected
