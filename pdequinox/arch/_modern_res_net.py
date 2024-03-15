from typing import Callable

import jax
from jaxtyping import PRNGKeyArray

from .._block_net import BaseBlockNet
from ..blocks import LinearChannelAdjustBlockFactory, ModernResBlockFactory


class ModernResNet(BaseBlockNet):
    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: int = 32,
        num_blocks: int = 6,
        use_norm: bool = True,
        activation: Callable = jax.nn.relu,
        boundary_mode: str = "periodic",
        key: PRNGKeyArray,
        **boundary_kwargs,
    ):
        """
        Modern ResNet
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
            block_factory=ModernResBlockFactory(
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
