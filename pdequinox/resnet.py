import jax.random as jr
import equinox as eqx

from typing import Any, Callable, List
from jaxtyping import PRNGKeyArray

from channel_adjustment import ChannelAdjuster, ChannelAdjusterFactory, LinearAdjusterFactory
from blocks import Block, BlockFactory, ClassicResBlockFactory

class ResNet(eqx.Module):
    lifting: ChannelAdjuster
    blocks: List[Block]
    projection: ChannelAdjuster

    def __init__(
        self,
        num_spacial_dims: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_blocks: int,
        activation: Callable,
        *,
        key: PRNGKeyArray,
        lifting_factory: ChannelAdjusterFactory = LinearAdjusterFactory(),
        block_factory: BlockFactory = ClassicResBlockFactory(),
        projection_factory: ChannelAdjusterFactory = LinearAdjusterFactory(),
    ):
        subkey, key = jr.split(key)
        self.lifting = lifting_factory(
            num_spacial_dims,
            in_channels,
            hidden_channels,
            key=subkey,
        )
        self.blocks = []
        for _ in range(num_blocks):
            subkey, key = jr.split(key)
            self.blocks.append(block_factory(
                num_spacial_dims,
                hidden_channels,
                activation=activation,
                key=subkey,
            ))
        self.projection = projection_factory(
            num_spacial_dims,
            hidden_channels,
            out_channels,
            key=key,
        )

    def __call__(self, x):
        x = self.lifting(x)
        for block in self.blocks:
            x = block(x)
        x = self.projection(x)
        return x

