from typing import Callable, List

import equinox as eqx
import jax.random as jr
from jaxtyping import PRNGKeyArray

from ._utils import sum_receptive_fields
from .blocks import (
    Block,
    BlockFactory,
    ClassicResBlockFactory,
    LinearChannelAdjustBlockFactory,
)


class Sequential(eqx.Module):
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
        """
        Generic constructor for block-based architectures like ResNets.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example
            traditional convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `hidden_channels`: The number of channels in the hidden layers.
        - `num_blocks`: The number of blocks to use. TODO add clarification
        - `activation`: The activation function to use in the blocks.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        - `boundary_mode`: The boundary mode to use for the convolution.
            (Keyword only argument)
        - `lifting_factory`: The factory to use for the lifting block.
            Default is `LinearChannelAdjustBlockFactory` which is simply a
            linear 1x1 convolution for channel adjustment.
        - `block_factory`: The factory to use for the blocks. Default is
            `ClassicResBlockFactory` which is a classic ResNet block (with
            postactivation)
        - `projection_factory`: The factory to use for the projection block.
            Default is `LinearChannelAdjustBlockFactory` which is simply a
            linear 1x1 convolution for channel adjustment.
        """
        subkey, key = jr.split(key)
        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")

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

    @property
    def receptive_field(self) -> tuple[tuple[float, float], ...]:
        lifting_receptive_field = self.lifting.receptive_field
        block_receptive_fields = tuple(block.receptive_field for block in self.blocks)
        projection_receptive_field = self.projection.receptive_field
        receptive_fields = (
            (lifting_receptive_field,)
            + block_receptive_fields
            + (projection_receptive_field,)
        )
        return sum_receptive_fields(receptive_fields)
