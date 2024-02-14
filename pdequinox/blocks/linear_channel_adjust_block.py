import jax
import equinox as eqx

from ..physics_conv import PhysicsConv, PhysicsConvTranspose
from ..spectral_conv import SpectralConv
from ..pointwise_linear_conv import PointwiseLinearConv
from typing import Any, Callable
from jaxtyping import PRNGKeyArray

from .base_block import Block, BlockFactory


LinearChannelAdjustBlock = PointwiseLinearConv

class LinearChannelAdjustBlockFactory(BlockFactory):
    use_bias: bool = True
    zero_bias_init: bool = False

    def __call__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        activation: Callable,  # unused
        *,
        boundary_mode: str,  # unused
        key: PRNGKeyArray,
        **boundary_kwargs,  # unused
    ):
        return LinearChannelAdjustBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=self.use_bias,
            zero_bias_init=self.zero_bias_init,
            key=key,
        )