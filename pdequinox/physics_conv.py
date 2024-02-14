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
    return jnp.pad(x, ((0, 0),) + padding, mode="wrap")


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
        zero_bias_init: bool = False,
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
            if zero_bias_init:
                self.bias = jnp.zeros((out_channels,) + (1,) * num_spatial_dims)
            else:
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

        self.padding = tuple(
            ((k - 1) * d // 2, (k - 1) * d // 2)
            for k, d in zip(kernel_size, dilation)
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
    """General N-dimensional transposed convolution.
    
    inspired by circular padding in jax flax:
    https://github.com/google/flax/blob/c25f546ff4c72c6c85e5225fe8467331720034be/flax/linen/linear.py#L864

    and jax-cfd:
    https://github.com/google/jax-cfd/blob/d215f13282bd63045fb3455f8fac061653428040/jax_cfd/ml/layers.py#L218

    ToDo: Does not yet support dilation!
    """

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
        zero_bias_init: bool = False,
        **boundary_kwargs,
    ):
        if dilation != 1:
            raise NotImplementedError("Dilation not yet supported in `ConvTranspose`.")

        wkey, bkey = jrandom.split(key, 2)

        parse = _ntuple(num_spatial_dims)
        kernel_size = parse(kernel_size)
        stride = parse(stride)
        output_padding = parse(output_padding)
        dilation = parse(dilation)

        for s, o in zip(stride, output_padding):
            if o >= s:
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
            if zero_bias_init:
                self.bias = jnp.zeros((out_channels,) + (1,) * num_spatial_dims)
            else:
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

        # Note that those are effectively multiplied by stride because we use
        # the lhs dilation
        self.padding = tuple((k//2, k//2) for k in kernel_size)

        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        self.boundary_mode = boundary_mode
        self.boundary_kwargs = boundary_kwargs

    @jax.named_scope("PhysicsConvTranspose")
    def __call__(
        self,
        x: Array,
        *,
        key: Optional[PRNGKeyArray] = None,
        output_padding: Optional[Union[int, Sequence[int]]] = None,
    ) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape
            `(in_channels, dim_1, ..., dim_N)`, where `N = num_spatial_dims`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        - `output_padding`: Overwrite the `output_padding` attribute specified in the
            constructor for this call only. If `None`, the attribute is used. (Keyword
            only argument.)

        **Returns:**

        A JAX array of shape `(out_channels, new_dim_1, ..., new_dim_N)`.
        """
        if output_padding is not None:
            parse = _ntuple(self.num_spatial_dims)
            this_output_padding = parse(output_padding)
        else:
            this_output_padding = self.output_padding

        unbatched_rank = self.num_spatial_dims + 1
        if x.ndim != unbatched_rank:
            raise ValueError(
                f"Input to `ConvTranspose` needs to have rank {unbatched_rank},",
                f" but input has shape {x.shape}.",
            )
        
        # needed for slicing out later
        spatial_shape = x.shape[1:]
        
        if self.boundary_mode == "periodic":
            x = periodic_padding(x, self.padding)
        else:
            raise ValueError(f"boundary_mode={self.boundary_mode} not implemented")

        x = jnp.expand_dims(x, axis=0)
        padding_lax = ((0, 0),) * self.num_spatial_dims
        x = lax.conv_general_dilated(
            lhs=x,
            rhs=self.weight,
            window_strides=(1,) * self.num_spatial_dims,
            padding=padding_lax, # no padding is applied here
            lhs_dilation=self.stride,
            rhs_dilation=self.dilation,
            feature_group_count=self.groups,
        )
        x = jnp.squeeze(x, axis=0)
        
        output_slices = ((None, None),) + tuple(
            (
                (k // s) * (s - 1),
                (k // s) * (s - 1) + s * dims - 1 + o,
            )
            for s, k, dims, o in zip(self.stride, self.kernel_size, spatial_shape, this_output_padding)
        )
        output_slices = tuple(slice(*entry) for entry in output_slices)

        # Following JAX-CFD
        x = x[output_slices]

        if self.use_bias:
            x = x + self.bias
        return x
