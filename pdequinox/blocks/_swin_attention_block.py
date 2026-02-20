from typing import Callable, Literal

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from .._transformer_utils import (
    SpatialAttention,
    boundary_shift,
    boundary_unshift,
    patchify,
    to_channel_first,
    to_channel_last,
    unpatchify,
)
from ._base_block import Block, BlockFactory


class SwinAttentionBlock(Block):
    num_spatial_dims: int = eqx.field(static=True)
    window_size: int = eqx.field(static=True)
    shift: int = eqx.field(static=True)
    boundary_mode: str = eqx.field(static=True)
    spatial_attention: SpatialAttention

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        num_heads: int,
        window_size: int,
        shift: int,
        boundary_mode: str,
        *,
        feed_forward_multiplier: int = 4,
        activation: Callable = jax.nn.gelu,
        key: PRNGKeyArray,
    ):
        self.num_spatial_dims = num_spatial_dims
        self.window_size = window_size
        self.shift = shift
        self.boundary_mode = boundary_mode
        self.spatial_attention = SpatialAttention(
            num_spatial_dims=num_spatial_dims,
            embedding_size=in_channels,
            num_heads=num_heads,
            feed_forward_multiplier=feed_forward_multiplier,
            activation=activation,
            key=key,
        )

    def __call__(self, x):
        x = to_channel_last(x)  # (*spatial, C)

        # Boundary-aware shift
        for dim in range(self.num_spatial_dims):
            x = boundary_shift(x, self.shift, dim, self.boundary_mode)

        # Patchify each channel: vmap over channel dim (last axis)
        x_patched = jax.vmap(
            lambda _x: patchify(_x, patch_size=self.window_size),
            in_axes=-1,
            out_axes=-1,
        )(x)
        # x_patched: (*num_patches, *patch_sizes, C)

        # Apply attention within each window: vmap over num_patches dims
        attn = self.spatial_attention
        for _ in range(self.num_spatial_dims):
            attn = jax.vmap(attn)
        x_patched = attn(x_patched)

        # Unpatchify
        x = jax.vmap(unpatchify, in_axes=-1, out_axes=-1)(x_patched)

        # Boundary-aware unshift
        for dim in range(self.num_spatial_dims):
            x = boundary_unshift(x, self.shift, dim, self.boundary_mode)

        x = to_channel_first(x)  # (C, *spatial)
        return x

    @property
    def receptive_field(self) -> tuple[tuple[float, float], ...]:
        w = float(self.window_size - 1)
        return ((w, w),) * self.num_spatial_dims


class SwinAttentionBlockFactory(BlockFactory):
    num_heads: int
    window_size: int
    feed_forward_multiplier: int
    _call_count: list

    def __init__(
        self,
        *,
        num_heads: int = 8,
        window_size: int = 8,
        feed_forward_multiplier: int = 4,
    ):
        self.num_heads = num_heads
        self.window_size = window_size
        self.feed_forward_multiplier = feed_forward_multiplier
        self._call_count = [0]

    def __call__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        activation: Callable,
        boundary_mode: Literal["periodic", "dirichlet", "neumann"],
        key: PRNGKeyArray,
    ):
        if in_channels != out_channels:
            raise ValueError(
                f"SwinAttentionBlock requires in_channels == out_channels, "
                f"got {in_channels} != {out_channels}"
            )
        shift = (self.window_size // 2) if (self._call_count[0] % 2 == 1) else 0
        self._call_count[0] += 1
        return SwinAttentionBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            num_heads=self.num_heads,
            window_size=self.window_size,
            shift=shift,
            boundary_mode=boundary_mode,
            feed_forward_multiplier=self.feed_forward_multiplier,
            activation=activation,
            key=key,
        )
