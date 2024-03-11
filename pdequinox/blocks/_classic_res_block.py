from typing import Callable

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from .._physics_conv import PhysicsConv


class ClassicResBlock(eqx.Module):
    conv_1: eqx.Module
    conv_2: eqx.Module
    activation: Callable

    def __init__(
        self,
        num_spatial_dims: int,
        channels: int,
        activation: Callable,
        kernel_size: int = 3,
        *,
        boundary_mode: str,
        key,
        use_bias: bool = True,
        zero_bias_init: bool = False,
        **boundary_kwargs,
    ):
        k_1, k_2 = jax.random.split(key)
        self.conv_1 = PhysicsConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=1,
            boundary_mode=boundary_mode,
            use_bias=use_bias,
            zero_bias_init=zero_bias_init,
            key=k_1,
            **boundary_kwargs,
        )
        self.conv_2 = PhysicsConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=1,
            boundary_mode=boundary_mode,
            use_bias=use_bias,
            zero_bias_init=zero_bias_init,
            key=k_2,
            **boundary_kwargs,
        )
        self.activation = activation

    def __call__(self, x):
        x_skip = x
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = x + x_skip
        x = self.activation(x)
        return x


class ClassicResBlockFactory(eqx.Module):
    kernel_size: int
    use_bias: bool
    zero_bias_init: bool

    def __init__(
        self,
        kernel_size: int = 3,
        *,
        use_bias: bool = True,
        zero_bias_init: bool = False,
    ):
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.zero_bias_init = zero_bias_init

    def __call__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        activation: Callable,
        *,
        boundary_mode: str,
        key: PRNGKeyArray,
        **boundary_kwargs,
    ):
        if in_channels != out_channels:
            raise ValueError(
                "ClassicResBlock only supports in_channels == out_channels"
            )
        else:
            channels = in_channels
        return ClassicResBlock(
            num_spatial_dims,
            channels,
            activation,
            self.kernel_size,
            boundary_mode=boundary_mode,
            key=key,
            use_bias=self.use_bias,
            zero_bias_init=self.zero_bias_init,
            **boundary_kwargs,
        )
