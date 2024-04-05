from typing import Callable

from jaxtyping import PRNGKeyArray

from ..conv import PhysicsConv
from ._base_block import BlockFactory

LinearConvBlock = PhysicsConv


class LinearConvBlockFactory(BlockFactory):
    kernel_size: int
    use_bias: bool

    def __init__(
        self,
        *,
        kernel_size: int = 3,
        use_bias: bool = True,
    ):
        """
        Factory for creating `LinearConvBlock` instances.

        **Arguments:**

        - `kernel_size`: The size of the convolutional kernel. Default is `3`.
        - `use_bias`: Whether to use bias in the convolutional layers. Default is
            `True`.
        """
        self.kernel_size = kernel_size
        self.use_bias = use_bias

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
