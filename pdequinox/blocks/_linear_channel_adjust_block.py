from typing import Callable

from jaxtyping import PRNGKeyArray

from ..conv import PointwiseLinearConv
from ._base_block import BlockFactory

LinearChannelAdjustBlock = PointwiseLinearConv


class LinearChannelAdjustBlockFactory(BlockFactory):
    use_bias: bool
    zero_bias_init: bool

    def __init__(
        self,
        *,
        use_bias: bool = True,
        zero_bias_init: bool = False,
    ):
        """
        Factory for creating `LinearChannelAdjustBlock` instances.

        **Arguments:**

        - `use_bias`: Whether to use a bias in the convolution.
        - `zero_bias_init`: Whether to initialise the bias to zero.
        """
        self.use_bias = use_bias
        self.zero_bias_init = zero_bias_init

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
