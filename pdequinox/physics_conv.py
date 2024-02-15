from equinox import field

from .conv import MorePaddingConv, MorePaddingConvTranspose, _ntuple

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
        if boundary_mode.lower() != "periodic":
            raise ValueError(f"Only 'periodic' boundary mode is supported, got {boundary_mode}")
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
        if boundary_mode.lower() != "periodic":
            raise ValueError(f"Only 'periodic' boundary mode is supported, got {boundary_mode}")
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