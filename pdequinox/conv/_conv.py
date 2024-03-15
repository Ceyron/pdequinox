"""
Patch more padding modes into Equinox
"""

import itertools as it
import math
from collections.abc import Callable, Sequence
from typing import Any, Optional, TypeVar, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from equinox import Module, field
from jaxtyping import Array, PRNGKeyArray

_T = TypeVar("_T")


def _ntuple(n: int) -> Callable[[Union[_T, Sequence[_T]]], tuple[_T, ...]]:
    def parse(x: Union[_T, Sequence[_T]]) -> tuple[_T, ...]:
        if isinstance(x, Sequence):
            if len(x) == n:
                return tuple(x)
            else:
                raise ValueError(
                    f"Length of {x} (length = {len(x)}) is not equal to {n}"
                )
        else:
            return tuple(it.repeat(x, n))

    return parse


def all_sequences(x: Union[Sequence[Any], Sequence[_T]]) -> bool:
    return all(isinstance(xi, Sequence) for xi in x)


class MorePaddingConv(Module):
    """General N-dimensional convolution."""

    num_spatial_dims: int = field(static=True)
    weight: Array
    bias: Optional[Array]
    in_channels: int = field(static=True)
    out_channels: int = field(static=True)
    kernel_size: tuple[int, ...] = field(static=True)
    stride: tuple[int, ...] = field(static=True)
    padding: tuple[tuple[int, int], ...] = field(static=True)
    padding_mode: str = field(static=True)
    dilation: tuple[int, ...] = field(static=True)
    groups: int = field(static=True)
    use_bias: bool = field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int], Sequence[tuple[int, int]]] = 0,
        padding_mode: str = "zeros",
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example traditional
            convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `kernel_size`: The size of the convolutional kernel.
        - `stride`: The stride of the convolution.
        - `padding`: The amount of padding to apply before and after each spatial
            dimension.
        - `padding_mode`: (NEW!) The mode of padding to apply. Can be one of
            `["zeros", "reflect", "replicate", "circular"]`.
        - `dilation`: The dilation of the convolution.
        - `groups`: The number of input channel groups. At `groups=1`,
            all input channels contribute to all output channels. Values
            higher than `1` are equivalent to running `groups` independent
            `Conv` operations side-by-side, each having access only to
            `in_channels` // `groups` input channels, and
            concatenating the results along the output channel dimension.
            `in_channels` must be divisible by `groups`.
        - `use_bias`: Whether to add on a bias after the convolution.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        !!! info

            All of `kernel_size`, `stride`, `padding`, `dilation` can be either an
            integer or a sequence of integers. If they are a sequence then the sequence
            should be of length equal to `num_spatial_dims`, and specify the value of
            each property down each spatial dimension in turn.

            If they are an integer then the same kernel size / stride / padding /
            dilation will be used along every spatial dimension.

            `padding` can alternatively also be a sequence of 2-element tuples,
            each representing the padding to apply before and after each spatial
            dimension.

        """
        if not (padding_mode in ["zeros", "reflect", "replicate", "circular"]):
            raise ValueError(
                f"`padding_mode` must be one of ['zeros', 'reflect', 'replicate', 'circular'],"
                f" but got {padding_mode}."
            )

        wkey, bkey = jrandom.split(key, 2)

        parse = _ntuple(num_spatial_dims)
        kernel_size = parse(kernel_size)
        stride = parse(stride)
        dilation = parse(dilation)

        if in_channels % groups != 0:
            raise ValueError(
                f"`in_channels` (={in_channels}) must be divisible "
                f"by `groups` (={groups})."
            )

        grouped_in_channels = in_channels // groups
        lim = 1 / np.sqrt(grouped_in_channels * math.prod(kernel_size))
        self.weight = jrandom.uniform(
            wkey,
            (out_channels, grouped_in_channels) + kernel_size,
            minval=-lim,
            maxval=lim,
        )
        if use_bias:
            self.bias = jrandom.uniform(
                bkey,
                (out_channels,) + (1,) * num_spatial_dims,
                minval=-lim,
                maxval=lim,
            )
        else:
            self.bias = None

        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(padding, int):
            self.padding = tuple((padding, padding) for _ in range(num_spatial_dims))
        elif isinstance(padding, Sequence) and len(padding) == num_spatial_dims:
            if all_sequences(padding):
                self.padding = tuple(padding)
            else:
                self.padding = tuple((p, p) for p in padding)
        else:
            raise ValueError(
                "`padding` must either be an int or tuple of length "
                f"{num_spatial_dims} containing ints or tuples of length 2."
            )
        self.padding_mode = padding_mode
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias

    @jax.named_scope("eqx.nn.Conv")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape
            `(in_channels, dim_1, ..., dim_N)`, where `N = num_spatial_dims`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(out_channels, new_dim_1, ..., new_dim_N)`.
        """

        unbatched_rank = self.num_spatial_dims + 1
        if x.ndim != unbatched_rank:
            raise ValueError(
                f"Input to `Conv` needs to have rank {unbatched_rank},",
                f" but input has shape {x.shape}.",
            )
        if self.padding_mode == "circular":
            x = jnp.pad(x, ((0, 0),) + self.padding, mode="wrap")
            padding_lax = ((0, 0),) * self.num_spatial_dims
        elif self.padding_mode == "zeros":
            x = x
            padding_lax = self.padding
        elif self.padding_mode == "reflect":
            x = jnp.pad(x, ((0, 0),) + self.padding, mode="reflect")
            padding_lax = ((0, 0),) * self.num_spatial_dims
        elif self.padding_mode == "replicate":
            x = jnp.pad(x, ((0, 0),) + self.padding, mode="edge")
            padding_lax = ((0, 0),) * self.num_spatial_dims
        else:
            raise ValueError(
                f"`padding_mode` must be one of ['zeros', 'reflect', 'replicate', 'circular'],"
                f" but got {self.padding_mode}."
            )

        x = jnp.expand_dims(x, axis=0)
        x = lax.conv_general_dilated(
            lhs=x,
            rhs=self.weight,
            window_strides=self.stride,
            padding=padding_lax,
            rhs_dilation=self.dilation,
            feature_group_count=self.groups,
        )
        x = jnp.squeeze(x, axis=0)
        if self.use_bias:
            x = x + self.bias
        return x

    @property
    def receptive_field(self) -> tuple[tuple[float, float], ...]:
        """The receptive field of the convolutional kernel."""
        return tuple(
            (
                ((k - 1) // 2) * d,
                (k // 2) * d,
            )
            for k, d in zip(self.kernel_size, self.dilation)
        )


class MorePaddingConvTranspose(Module):
    """General N-dimensional transposed convolution."""

    num_spatial_dims: int = field(static=True)
    weight: Array
    bias: Optional[Array]
    in_channels: int = field(static=True)
    out_channels: int = field(static=True)
    kernel_size: tuple[int, ...] = field(static=True)
    stride: tuple[int, ...] = field(static=True)
    padding: tuple[tuple[int, int], ...] = field(static=True)
    padding_mode: str = field(static=True)
    output_padding: tuple[int, ...] = field(static=True)
    dilation: tuple[int, ...] = field(static=True)
    groups: int = field(static=True)
    use_bias: bool = field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        padding: Union[int, Sequence[int], Sequence[tuple[int, int]]] = 0,
        padding_mode: str = "zeros",
        output_padding: Union[int, Sequence[int]] = 0,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example traditional
            convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `kernel_size`: The size of the transposed convolutional kernel.
        - `stride`: The stride used on the equivalent [`equinox.nn.Conv`][].
        - `padding`: The amount of padding used on the equivalent [`equinox.nn.Conv`][].
        - `padding_mode`: (NEW!) The mode of padding to apply. Can be one of
            `["zeros", "reflect", "replicate", "circular"]`.
        - `output_padding`: Additional padding for the output shape.
        - `dilation`: The spacing between kernel points.
        - `groups`: The number of input channel groups. At `groups=1`,
            all input channels contribute to all output channels. Values
            higher than 1 are equivalent to running `groups` independent
            `ConvTranspose` operations side-by-side, each having access only to
            `in_channels` // `groups` input channels, and
            concatenating the results along the output channel dimension.
            `in_channels` must be divisible by `groups`.
        - `use_bias`: Whether to add on a bias after the transposed convolution.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        !!! info

            All of `kernel_size`, `stride`, `padding`, `output_padding`, `dilation` can
            be either an integer or a sequence of integers. If they are a sequence then
            the sequence should be of length equal to `num_spatial_dims`, and specify
            the value of each property down each spatial dimension in turn.

            If they are an integer then the same kernel size / stride / padding /
            dilation will be used along every spatial dimension.

            `padding` can alternatively also be a sequence of 2-element tuples,
            each representing the padding to apply before and after each spatial
            dimension.

        !!! tip

            Transposed convolutions are often used to go in the "opposite direction" to
            a normal convolution. That is, from something with the shape of the output
            of a convolution to something with the shape of the input to a convolution.
            Moreover, to do so with the same "connectivity", i.e. which inputs can
            affect which outputs.

            Relative to an [`equinox.nn.Conv`][] layer, this can be accomplished by
            switching the values of `in_channels` and `out_channels`, whilst keeping
            `kernel_size`, `stride`, `padding`, `dilation`, and `groups` the same.

            When `stride > 1` then [`equinox.nn.Conv`][] maps multiple input shapes to the
            same output shape. `output_padding` is provided to resolve this ambiguity,
            by adding a little extra padding to just the bottom/right edges of the
            input.

            See [these animations](https://github.com/vdumoulin/conv_arithmetic/blob/af6f818b0bb396c26da79899554682a8a499101d/README.md#transposed-convolution-animations)
            and [this report](https://arxiv.org/abs/1603.07285) for a nice reference.
        """  # noqa: E501

        wkey, bkey = jrandom.split(key, 2)

        parse = _ntuple(num_spatial_dims)
        kernel_size = parse(kernel_size)
        stride = parse(stride)
        output_padding = parse(output_padding)
        dilation = parse(dilation)

        for s, o in zip(stride, output_padding):
            if output_padding >= stride:
                raise ValueError("Must have `output_padding < stride` (elementwise).")

        grouped_in_channels = in_channels // groups
        lim = 1 / np.sqrt(grouped_in_channels * math.prod(kernel_size))
        self.weight = jrandom.uniform(
            wkey,
            (out_channels, grouped_in_channels) + kernel_size,
            minval=-lim,
            maxval=lim,
        )
        if use_bias:
            self.bias = jrandom.uniform(
                bkey,
                (out_channels,) + (1,) * num_spatial_dims,
                minval=-lim,
                maxval=lim,
            )
        else:
            self.bias = None

        self.num_spatial_dims = num_spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(padding, int):
            self.padding = tuple((padding, padding) for _ in range(num_spatial_dims))
        elif isinstance(padding, Sequence) and len(padding) == num_spatial_dims:
            if all_sequences(padding):
                self.padding = tuple(padding)
            else:
                self.padding = tuple((p, p) for p in padding)
        else:
            raise ValueError(
                "`padding` must either be an int or tuple of length "
                f"{num_spatial_dims} containing ints or tuples of length 2."
            )
        self.padding_mode = padding_mode
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias

    @jax.named_scope("eqx.nn.ConvTranspose")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape
            `(in_channels, dim_1, ..., dim_N)`, where `N = num_spatial_dims`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array of shape `(out_channels, new_dim_1, ..., new_dim_N)`.
        """

        unbatched_rank = self.num_spatial_dims + 1
        if x.ndim != unbatched_rank:
            raise ValueError(
                f"Input to `ConvTranspose` needs to have rank {unbatched_rank},",
                f" but input has shape {x.shape}.",
            )
        # Given by Relationship 14 of https://arxiv.org/abs/1603.07285
        transpose_padding = tuple(
            (d * (k - 1) - p0, d * (k - 1) - p1 + o)
            for k, (p0, p1), o, d in zip(
                self.kernel_size, self.padding, self.output_padding, self.dilation
            )
        )
        # Decide how much has to pre-paded (for everything non "zeros" padding
        # mode)
        if self.padding_mode != "zeros":
            pre_dilation_padding = tuple(
                (
                    (p_l + (s - 1)) // s,
                    (p_r + (s - 1)) // s,
                )
                for (p_l, p_r), s in zip(transpose_padding, self.stride)
            )
            # Can also be negative
            post_dilation_padding = tuple(
                (
                    p_l - pd_l * s,
                    p_r - pd_r * s + o,
                )
                for (p_l, p_r), (pd_l, pd_r), s, o in zip(
                    self.padding, pre_dilation_padding, self.stride, self.output_padding
                )
            )
            if self.padding_mode == "circular":
                x = jnp.pad(x, ((0, 0),) + pre_dilation_padding, mode="wrap")
            elif self.padding_mode == "reflect":
                x = jnp.pad(x, ((0, 0),) + pre_dilation_padding, mode="reflect")
            elif self.padding_mode == "replicate":
                x = jnp.pad(x, ((0, 0),) + pre_dilation_padding, mode="edge")
            else:
                raise ValueError(
                    f"`padding_mode` must be one of ['zeros', 'reflect', 'replicate', 'circular'],"
                    f" but got {self.padding_mode}."
                )
        else:
            post_dilation_padding = self.padding
            x = x

        x = jnp.expand_dims(x, axis=0)
        x = lax.conv_general_dilated(
            lhs=x,
            rhs=self.weight,
            window_strides=(1,) * self.num_spatial_dims,
            padding=post_dilation_padding,
            lhs_dilation=self.stride,
            rhs_dilation=self.dilation,
            feature_group_count=self.groups,
        )
        x = jnp.squeeze(x, axis=0)
        if self.use_bias:
            x = x + self.bias
        return x

    @property
    def receptive_field(self) -> tuple[tuple[float, float], ...]:
        """The receptive field of the transposed convolutional kernel."""
        return tuple(
            (
                float(((k - 1) // 2) * d),
                float((k // 2) * d),
            )
            for k, d in zip(self.kernel_size, self.dilation)
        )
