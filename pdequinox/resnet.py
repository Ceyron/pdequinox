from typing import Callable, List

import equinox as eqx
import jax.random as jr
from jaxtyping import PRNGKeyArray

from .blocks import (
    Block,
    BlockFactory,
    ClassicResBlockFactory,
    LinearChannelAdjustBlockFactory,
)


class ResNet(eqx.Module):
    lifting: Block
    blocks: List[Block]
    projection: Block

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_blocks: int,
        activation: Callable,
        *,
        key: PRNGKeyArray,
        boundary_mode: str,
        lifting_factory: BlockFactory = LinearChannelAdjustBlockFactory(),
        block_factory: BlockFactory = ClassicResBlockFactory(),
        projection_factory: BlockFactory = LinearChannelAdjustBlockFactory(),
        **boundary_kwargs,
    ):
        subkey, key = jr.split(key)
        self.lifting = lifting_factory(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=hidden_channels,
            activation=activation,
            boundary_mode=boundary_mode,
            key=subkey,
            **boundary_kwargs,
        )
        self.blocks = []
        for _ in range(num_blocks):
            subkey, key = jr.split(key)
            self.blocks.append(
                block_factory(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    activation=activation,
                    boundary_mode=boundary_mode,
                    key=subkey,
                    **boundary_kwargs,
                )
            )
        self.projection = projection_factory(
            num_spatial_dims=num_spatial_dims,
            in_channels=hidden_channels,
            out_channels=out_channels,
            activation=activation,
            boundary_mode=boundary_mode,
            key=key,
            **boundary_kwargs,
        )

    def __call__(self, x):
        x = self.lifting(x)
        for block in self.blocks:
            x = block(x)
        x = self.projection(x)
        return x
