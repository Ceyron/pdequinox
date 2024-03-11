from typing import Callable

import jax
from jaxtyping import PRNGKeyArray

from ..pointwise_linear_conv import PointwiseLinearConv
from ..spectral_conv import SpectralConv
from .base_block import Block, BlockFactory


class ClassicSpectralBlock(Block):
    spectral_conv: SpectralConv
    by_pass_conv: PointwiseLinearConv
    activation: Callable

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_modes: int,
        activation: Callable,
        *,
        use_bias: bool = True,
        zero_bias_init: bool = False,
        key: PRNGKeyArray,
    ):
        k_1, k_2 = jax.random.split(key)
        self.spectral_conv = SpectralConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_modes=num_modes,
            key=k_1,
        )
        self.by_pass_conv = PointwiseLinearConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=use_bias,
            zero_bias_init=zero_bias_init,
            key=k_2,
        )
        self.activation = activation

    def __call__(self, x):
        x = self.spectral_conv(x) + self.by_pass_conv(x)
        x = self.activation(x)
        return x


class ClassicSpectralBlockFactory(BlockFactory):
    num_modes: int or tuple[int, ...]
    use_bias: bool = True
    zero_bias_init: bool = False

    def __call__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        activation: Callable,
        *,
        boundary_mode: str,  # unused
        key: PRNGKeyArray,
        **boundary_kwargs,  # unused
    ):
        return ClassicSpectralBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_modes=self.num_modes,
            activation=activation,
            key=key,
            use_bias=self.use_bias,
            zero_bias_init=self.zero_bias_init,
        )
