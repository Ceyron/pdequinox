from typing import Callable, Literal

from jaxtyping import PRNGKeyArray

from ..conv import PhysicsConv
from ._base_block import BlockFactory

LinearConvDownBlock = PhysicsConv


class LinearConvDownBlockFactory(BlockFactory):
    kernel_size: int
    factor: int
    use_bias: bool

    def __init__(
        self,
        *,
        kernel_size: int = 3,
        factor: int = 2,
        use_bias: bool = True,
    ):
        """
        Factory for creating `LinearConvDownBlock` instances.

        **Arguments:**

        - `kernel_size`: The size of the convolutional kernel. Default is `3`.
        - `factor`: The downsampling factor. Default is `2`. This will become
            the stride of the convolution.
        - `use_bias`: Whether to use bias after the convolution. Default
            is `True`.
        """
        self.kernel_size = kernel_size
        self.factor = factor
        self.use_bias = use_bias

    def __call__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        activation: Callable,  # unused
        boundary_mode: Literal["periodic", "dirichlet", "neumann"],
        key: PRNGKeyArray,
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
        )
