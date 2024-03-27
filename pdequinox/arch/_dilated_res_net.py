from typing import Callable

import jax
from jaxtyping import PRNGKeyArray

from .._sequential import Sequential
from ..blocks import DilatedResBlockFactory, LinearChannelAdjustBlockFactory


class DilatedResNet(Sequential):
    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: int = 32,
        num_blocks: int = 2,
        dilation_rates: tuple[int, ...] = (1, 2, 4, 8, 4, 2, 1),
        use_norm: bool = True,
        activation: Callable = jax.nn.relu,
        boundary_mode: str = "periodic",
        key: PRNGKeyArray,
        **boundary_kwargs,
    ):
        """
        Classic ResNet
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
            block_factory=DilatedResBlockFactory(
                dilation_rates=dilation_rates,
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
