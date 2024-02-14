import jax
import equinox as eqx

from ..physics_conv import PhysicsConv, PhysicsConvTranspose
from ..spectral_conv import SpectralConv
from ..pointwise_linear_conv import PointwiseLinearConv
from typing import Any, Callable
from jaxtyping import PRNGKeyArray

from .base_block import Block, BlockFactory

class ClassicDoubleConvBlock(Block):
    """Also does channel adjustment"""
    conv_1: PhysicsConv
    norm_1: eqx.nn.GroupNorm
    conv_2: PhysicsConv
    norm_2: eqx.nn.GroupNorm
    activation: Callable

    def __init__(
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
        k_1, k_2 = jax.random.split(key)
        self.conv_1 = PhysicsConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            use_bias=False,
            key=k_1,
            boundary_mode=boundary_mode,
            **boundary_kwargs,
        )
        self.norm_1 = eqx.nn.GroupNorm(out_channels, out_channels)
        self.conv_2 = PhysicsConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            use_bias=False,
            key=k_2,
            boundary_mode=boundary_mode,
            **boundary_kwargs,
        )
        self.norm_2 = eqx.nn.GroupNorm(out_channels, out_channels)

        self.activation = activation

    def __call__(self, x):
        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = self.norm_2(x)
        x = self.activation(x)
        return x
    
class ClassicDoubleConvBlockFactory(BlockFactory):
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
        return ClassicDoubleConvBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            boundary_mode=boundary_mode,
            key=key,
            **boundary_kwargs,
        )