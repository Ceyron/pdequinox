from typing import Callable

from jaxtyping import PRNGKeyArray

from ..conv import PhysicsConvTranspose
from ._base_block import BlockFactory

LinearConvUpBlock = PhysicsConvTranspose


class LinearConvUpBlockFactory(BlockFactory):
    kernel_size: int
    factor: int
    use_bias: bool
    output_padding: int

    def __init__(
        self,
        *,
        kernel_size: int = 3,
        factor: int = 2,
        use_bias: bool = True,
        output_padding: int = 1,
    ):
        """
        Factory for creating `LinearConvUpBlock` instances.

        **Arguments:**

        - `kernel_size`: The size of the convolutional kernel. Default is `3`.
        - `factor`: The upsampling factor. Default is `2`. This will become
            the stride of the transposed convolution. Set this to the same value
            as in a corresponding `LinearConvDownBlockFactory` instance.
        - `use_bias`: Whether to use bias after the convolution. Default
            is `True`.
        - `output_padding`: The amount of additional padding used by the
            transposed convolution. Use this to resolve the ambiguity that the
            result of an integer division with `factor` is not bijective. If you
            have `factor=2` and work with spatial dimensions divisible by `2`,
            set this to `1`. Default is `1`.
        """
        self.kernel_size = kernel_size
        self.factor = factor
        self.use_bias = use_bias
        self.output_padding = output_padding

    def __call__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        activation: Callable,  # unused
        *,
        boundary_mode: str,
        key: PRNGKeyArray,
    ):
        return LinearConvUpBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.factor,
            output_padding=self.output_padding,
            boundary_mode=boundary_mode,
            use_bias=self.use_bias,
            key=key,
        )
