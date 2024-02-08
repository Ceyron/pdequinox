"""
Instead of monkey patching the equinox conv operations (and not succeeding with
periodic/circular padding on transpose convolutions), I will just implement them
myselves, but following equinox as close as possible
"""

import itertools as it
import math
from collections.abc import Callable, Sequence
from typing import Optional, TypeVar, Union, Any

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

def periodic_padding(
    x: Array, padding: tuple[tuple[int, int], ...]
) -> Array:
    """Periodic padding for convolution.

    Args:
        x: Input array with shape (C, *N).
        padding: Padding to apply before and after each spatial dimension.

    Returns:
        Array with shape (C, *(N + 2 * padding)).
    """
    return np.pad(x, ((0, 0),) + padding, mode="wrap")


class PhysicsConv(Module, strict=True):
    """General N-dimensional convolution."""

    num_spatial_dims: int = field(static=True)
    weight: Array
    bias: Optional[Array]
    in_channels: int = field(static=True)
    out_channels: int = field(static=True)
    kernel_size: tuple[int, ...] = field(static=True)
    stride: tuple[int, ...] = field(static=True)
    padding: tuple[tuple[int, int], ...] = field(static=True)
    dilation: tuple[int, ...] = field(static=True)
    groups: int = field(static=True)
    use_bias: bool = field(static=True)
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
        **boundary_kwargs,
    ):
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

        padding = tuple(
            ((k - 1) * d // 2, (k - 1) * d // 2)
            for k, d in zip(kernel_size, dilation)
        )

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
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        self.boundary_mode = boundary_mode
        self.boundary_kwargs = boundary_kwargs

    @jax.named_scope("PhysicsConv")
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
        if self.boundary_mode == "periodic":
            x = periodic_padding(x, self.padding)
        else:
            raise ValueError(f"boundary_mode={self.boundary_mode} not implemented")

        x = jnp.expand_dims(x, axis=0)
        padding_lax = ((0, 0),) * self.num_spatial_dims
        x = lax.conv_general_dilated(
            lhs=x,
            rhs=self.weight,
            window_strides=self.stride,
            padding=padding_lax,  # no padding is applied here
            rhs_dilation=self.dilation,
            feature_group_count=self.groups,
        )
        x = jnp.squeeze(x, axis=0)
        if self.use_bias:
            x = x + self.bias
        return x

class PhysicsConvTranspose(Module, strict=True):
    """General N-dimensional transposed convolution."""

    num_spatial_dims: int = field(static=True)
    weight: Array
    bias: Optional[Array]
    in_channels: int = field(static=True)
    out_channels: int = field(static=True)
    kernel_size: tuple[int, ...] = field(static=True)
    stride: tuple[int, ...] = field(static=True)
    padding: tuple[tuple[int, int], ...] = field(static=True)
    output_padding: tuple[int, ...] = field(static=True)
    dilation: tuple[int, ...] = field(static=True)
    groups: int = field(static=True)
    use_bias: bool = field(static=True)
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
        **boundary_kwargs,
    ):
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

        padding = tuple(
            (
                int(np.floor((k - 1) * d / 2.0)),
                int(np.ceil ((k - 1) * d / 2.0)),
            )
            for k, d in zip(kernel_size, dilation)
        )

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
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        self.boundary_mode = boundary_mode
        self.boundary_kwargs = boundary_kwargs

    @jax.named_scope("PhysicsConvTranspose")
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
        padding = tuple(
            (
                int(np.floor((d * (k - 1) - p0) / s)),
                int( np.ceil((d * (k - 1) - p1) / s)),
            )
            for k, s, (p0, p1), o, d in zip(
                self.kernel_size, self.stride, self.padding, self.output_padding, self.dilation
            )
        )
        if self.boundary_mode == "periodic":
            x = periodic_padding(x, padding)
        else:
            raise ValueError(f"boundary_mode={self.boundary_mode} not implemented")

        x = jnp.expand_dims(x, axis=0)
        # ToDo: is this correct?
        padding_lax = ((0, o) for o in self.output_padding)
        # padding_lax = ((0, 0),) * self.num_spatial_dims
        x = lax.conv_general_dilated(
            lhs=x,
            rhs=self.weight,
            window_strides=(1,) * self.num_spatial_dims,
            padding=padding_lax,  # no padding is applied here, only output padding to correct shapes
            lhs_dilation=self.stride,
            rhs_dilation=self.dilation,
            feature_group_count=self.groups,
        )
        x = jnp.squeeze(x, axis=0)
        if self.use_bias:
            x = x + self.bias
        return x
