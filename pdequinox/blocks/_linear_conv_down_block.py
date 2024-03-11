from typing import Callable

from jaxtyping import PRNGKeyArray

from .._physics_conv import PhysicsConv
from ._base_block import BlockFactory

LinearConvDownBlock = PhysicsConv


class LinearConvDownBlockFactory(BlockFactory):
    kernel_size: int = 3
    factor: int = 2
    use_bias: bool = True

    def __call__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        activation: Callable,  # unused
        *,
        boundary_mode: str,
        key: PRNGKeyArray,
        **boundary_kwargs,
    ):
        return LinearConvDownBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.factor,
            boundary_mode=boundary_mode,
            use_bias=self.use_bias,
            key=key,
            **boundary_kwargs,
        )
