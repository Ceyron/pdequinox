from collections.abc import Sequence
from typing import Union

import jax.numpy as jnp
from equinox import field
from jaxtyping import PRNGKeyArray

from ._conv import MorePaddingConv, MorePaddingConvTranspose, _ntuple


def compute_same_padding(
    num_spatial_dims: int,
    kernel_size: Sequence[int],
    dilation: Sequence[int],
) -> Sequence[Sequence[int]]:
    parse = _ntuple(num_spatial_dims)
    kernel_size = parse(kernel_size)
    dilation = parse(dilation)
    same_padding = tuple(
        (
            ((k - 1) // 2) * d,
            (((k - 1) // 2) + ((k - 1) % 2)) * d,
        )
        for k, d in zip(kernel_size, dilation)
    )
    return same_padding


class PhysicsConv(MorePaddingConv):
    boundary_mode: str = field(static=True)
    boundary_kwargs: dict = field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        # no padding because it always chosen to retain spatial size
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
        boundary_mode: str,
        zero_bias_init: bool = False,
        **boundary_kwargs,
    ):
        """
        General n-dimensional convolution with "same" padding to operate on
        fields. Allows to choose a `boundary_mode` affecting the type of padding
        used. No option to set the padding. Some boundary modes may require
        additional `boundary_kwargs`.

        This is a thin wrapper around `equinox.nn.Conv`.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example
            traditional, convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `kernel_size`: The size of the convolutional kernel.
        - `stride`: The stride of the convolution.
        - `dilation`: The dilation of the convolution.
        - `groups`: The number of input channel groups. At `groups=1`,
            all input channels contribute to all output channels. Values higher
            than `1` are equivalent to running `groups` independent `Conv`
            operations side-by-side, each having access only to `in_channels` //
            `groups` input channels, and concatenating the results along the
            output channel dimension. `in_channels` must be divisible by
            `groups`.
        - `use_bias`: Whether to add on a bias after the convolution.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        - `boundary_mode`: The type of boundary padding to use. Use one of
            ["periodic", "dirichlet", "neumann"]. Currently only "periodic" is
            supported. (Keyword only argument.)
        - `zero_bias_init`: Whether to initialise the bias to zero. (Keyword
            only argument.)
        - `boundary_kwargs`: Additional keyword arguments to pass to the
          boundary
            padding function. (Keyword only argument.)

        !!! info

            All of `kernel_size`, `stride`, `dilation` can be either an integer
            or a sequence of integers. If they are a sequence then the sequence
            should be of length equal to `num_spatial_dims`, and specify the
            value of each property down each spatial dimension in turn.

            If they are an integer then the same kernel size / stride / dilation
            will be used along every spatial dimension.
        """
        if boundary_mode.lower() != "periodic":
            raise ValueError(
                f"Only 'periodic' boundary mode is supported, got {boundary_mode}"
            )
        self.boundary_mode = boundary_mode.lower()
        self.boundary_kwargs = boundary_kwargs

        if boundary_mode == "periodic":
            padding_mode = "circular"

        super().__init__(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=compute_same_padding(num_spatial_dims, kernel_size, dilation),
            padding_mode=padding_mode,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            key=key,
        )

        if use_bias and zero_bias_init:
            self.bias = jnp.zeros_like(self.bias)


class PhysicsConvTranspose(MorePaddingConvTranspose):
    boundary_mode: str = field(static=True)
    boundary_kwargs: dict = field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        # no padding because it always chosen to retain spatial size
        output_padding: Union[int, Sequence[int]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
        boundary_mode: str,
        zero_bias_init: bool = False,
        **boundary_kwargs,
    ):
        """
        General n-dimensional transposed convolution with "same" padding to
        operate on fields. Allows to choose a `boundary_mode` affecting the type
        of padding used. No option to set the padding. Some boundary modes may
        require additional `boundary_kwargs`.

        This is a thin wrapper around `equinox.nn.ConvTranspose`.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example
            traditional, convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `kernel_size`: The size of the convolutional kernel.
        - `stride`: The stride of the convolution.
        - `output_padding`: Additional padding for the output shape.
        - `dilation`: The dilation of the convolution.
        - `groups`: The number of input channel groups. At `groups=1`,
            all input channels contribute to all output channels. Values higher
            than `1` are equivalent to running `groups` independent `Conv`
            operations side-by-side, each having access only to `in_channels` //
            `groups` input channels, and concatenating the results along the
            output channel dimension. `in_channels` must be divisible by
            `groups`.
        - `use_bias`: Whether to add on a bias after the convolution.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        - `boundary_mode`: The type of boundary padding to use. Use one of
            ["periodic", "dirichlet", "neumann"]. Currently only "periodic" is
            supported. (Keyword only argument.)
        - `zero_bias_init`: Whether to initialise the bias to zero. (Keyword
            only argument.)
        - `boundary_kwargs`: Additional keyword arguments to pass to the
          boundary
            padding function. (Keyword only argument.)

        !!! info

            All of `kernel_size`, `stride`, `dilation` can be either an integer
            or a sequence of integers. If they are a sequence then the sequence
            should be of length equal to `num_spatial_dims`, and specify the
            value of each property down each spatial dimension in turn.

            If they are an integer then the same kernel size / stride / dilation
            will be used along every spatial dimension.

        !!! tip

            Transposed convolutions are often used to go in the "opposite
            direction" to a normal convolution. That is, from something with the
            shape of the output of a convolution to something with the shape of
            the input to a convolution. Moreover, to do so with the same
            "connectivity", i.e. which inputs can affect which outputs.

            Relative to an [`PhysicsConv`][] layer, this can be accomplished by
            switching the values of `in_channels` and `out_channels`, whilst
            keeping `kernel_size`, `stride`, `dilation`, and `groups` the same.

            When `stride > 1` then [`PhysicsConv`][] maps multiple input shapes
            to the same output shape. `output_padding` is provided to resolve
            this ambiguity, by adding a little extra padding to just the
            bottom/right edges of the input.

            See [these
            animations](https://github.com/vdumoulin/conv_arithmetic/blob/af6f818b0bb396c26da79899554682a8a499101d/README.md#transposed-convolution-animations)
            and [this report](https://arxiv.org/abs/1603.07285) for a nice
            reference.
        """
        if boundary_mode.lower() != "periodic":
            raise ValueError(
                f"Only 'periodic' boundary mode is supported, got {boundary_mode}"
            )
        self.boundary_mode = boundary_mode.lower()
        self.boundary_kwargs = boundary_kwargs

        if boundary_mode == "periodic":
            padding_mode = "circular"

        super().__init__(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=compute_same_padding(num_spatial_dims, kernel_size, dilation),
            padding_mode=padding_mode,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            key=key,
        )

        if use_bias and zero_bias_init:
            self.bias = jnp.zeros_like(self.bias)
