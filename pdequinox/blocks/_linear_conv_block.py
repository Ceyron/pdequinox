from typing import Callable

from jaxtyping import PRNGKeyArray

from ..conv import PhysicsConv
from ._base_block import BlockFactory

LinearConvBlock = PhysicsConv


class LinearConvBlockFactory(BlockFactory):
    kernel_size: int = 3
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
        return LinearConvBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            dilation=1,
            boundary_mode=boundary_mode,
            use_bias=self.use_bias,
            key=key,
            **boundary_kwargs,
        )
