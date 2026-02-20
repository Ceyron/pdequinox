from typing import Callable, Literal

import jax
from jaxtyping import PRNGKeyArray

from .._transformer_utils import SpatialAttention, to_channel_first, to_channel_last
from ._base_block import Block, BlockFactory


class DenseAttentionBlock(Block):
    num_spatial_dims: int
    spatial_attention: SpatialAttention

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        num_heads: int,
        *,
        feed_forward_multiplier: int = 4,
        activation: Callable = jax.nn.gelu,
        key: PRNGKeyArray,
    ):
        self.num_spatial_dims = num_spatial_dims
        self.spatial_attention = SpatialAttention(
            num_spatial_dims=num_spatial_dims,
            embedding_size=in_channels,
            num_heads=num_heads,
            feed_forward_multiplier=feed_forward_multiplier,
            activation=activation,
            key=key,
        )

    def __call__(self, x):
        x = to_channel_last(x)
        x = self.spatial_attention(x)
        x = to_channel_first(x)
        return x

    @property
    def receptive_field(self) -> tuple[tuple[float, float], ...]:
        return ((float("inf"), float("inf")),) * self.num_spatial_dims


class DenseAttentionBlockFactory(BlockFactory):
    num_heads: int
    feed_forward_multiplier: int

    def __init__(
        self,
        *,
        num_heads: int = 8,
        feed_forward_multiplier: int = 4,
    ):
        self.num_heads = num_heads
        self.feed_forward_multiplier = feed_forward_multiplier

    def __call__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        activation: Callable,
        boundary_mode: Literal["periodic", "dirichlet", "neumann"],  # unused
        key: PRNGKeyArray,
    ):
        if in_channels != out_channels:
            raise ValueError(
                f"DenseAttentionBlock requires in_channels == out_channels, "
                f"got {in_channels} != {out_channels}"
            )
        return DenseAttentionBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            num_heads=self.num_heads,
            feed_forward_multiplier=self.feed_forward_multiplier,
            activation=activation,
            key=key,
        )
