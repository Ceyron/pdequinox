from typing import Callable, Literal, Optional

import jax
from jaxtyping import PRNGKeyArray

from .._sequential import Sequential
from ..blocks import AxialAttentionBlockFactory, LinearChannelAdjustBlockFactory


class AxialTransformer(Sequential):
    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: int = 64,
        num_heads: int = 8,
        num_blocks: int = 6,
        feed_forward_multiplier: int = 4,
        activation: Callable = jax.nn.gelu,
        boundary_mode: Optional[
            Literal["periodic", "dirichlet", "neumann"]
        ] = None,  # unused
        key: PRNGKeyArray,
    ):
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            activation=activation,
            key=key,
            boundary_mode="periodic",  # Does not matter
            lifting_factory=LinearChannelAdjustBlockFactory(),
            block_factory=AxialAttentionBlockFactory(
                num_heads=num_heads,
                feed_forward_multiplier=feed_forward_multiplier,
            ),
            projection_factory=LinearChannelAdjustBlockFactory(),
        )
