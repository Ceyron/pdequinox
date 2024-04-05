from typing import Callable

import jax
from jaxtyping import PRNGKeyArray

from .._sequential import Sequential
from ..blocks import ClassicResBlockFactory, LinearChannelAdjustBlockFactory


class ClassicResNet(Sequential):
    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: int = 32,
        num_blocks: int = 6,
        use_norm: bool = False,
        activation: Callable = jax.nn.relu,
        boundary_mode: str = "periodic",
        key: PRNGKeyArray,
        **boundary_kwargs,
    ):
        """
        Vanilla ResNet architecture very close the original He et al. (2016)
        paper.

        Performs a sequence of blocks consisting of two convolutions and a
        bypass. The structure of the blocks are "post-activation" (original
        ResNet paper). For the modern "pre-activation" ResNet, see
        `ModernResNet`. By default, no group normalization is used. The original
        paper used batch normalization.

        The total number of convolutions is `2 * num_blocks` (3x3 convolutions)
        and two 1x1 convolutions for the lifting and projection.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example
            traditional convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `hidden_channels`: The number of channels in the hidden layers.
            Default is `32`.
        - `num_blocks`: The number of blocks to use. Must be an integer greater
            or equal than `1`. Default is `6`.
        - `use_norm`: Whether to use group normalization. Default is `False`.
        - `activation`: The activation function to use in the blocks. Default is
            `jax.nn.relu`. Lifting and projection are **not** activated.
        - `boundary_mode`: The boundary mode to use for the convolution. Default
            is `"periodic"`.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            activation=activation,
            key=key,
            boundary_mode=boundary_mode,
            lifting_factory=LinearChannelAdjustBlockFactory(
                use_bias=True,
                zero_bias_init=False,
            ),
            block_factory=ClassicResBlockFactory(
                use_norm=use_norm,
                use_bias=True,
                zero_bias_init=False,
            ),
            projection_factory=LinearChannelAdjustBlockFactory(
                use_bias=True,
                zero_bias_init=False,
            ),
            **boundary_kwargs,
        )
