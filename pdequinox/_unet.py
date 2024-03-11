from typing import Any, Callable, List

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

from .blocks import (
    Block,
    BlockFactory,
    ClassicDoubleConvBlockFactory,
    LinearChannelAdjustBlockFactory,
    LinearConvDownBlockFactory,
    LinearConvUpBlockFactory,
)
from .conv import PhysicsConv


class UNet(eqx.Module):
    """
    Uses convolution for downsampling instead of max pooling
    """

    lifting: Block
    down_sampling_blocks: List[Block]
    left_arch_blocks: List[Block]  # Includes the bottleneck
    up_sampling_blocks: List[Block]
    right_arch_blocks: List[Block]
    projection: PhysicsConv
    reduction_factor: int
    levels: int

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_levels: int,
        activation: Callable,
        *,
        key: PRNGKeyArray,
        reduction_factor: int = 2,
        boundary_mode: str,
        lifting_factory: BlockFactory = ClassicDoubleConvBlockFactory(),
        down_sampling_factory: BlockFactory = LinearConvDownBlockFactory(),
        left_arch_factory: BlockFactory = ClassicDoubleConvBlockFactory(),
        up_sampling_factory: BlockFactory = LinearConvUpBlockFactory(),
        right_arch_factory: BlockFactory = ClassicDoubleConvBlockFactory(),
        projection_factory: BlockFactory = LinearChannelAdjustBlockFactory(),
        **boundary_kwargs,
    ):
        self.down_sampling_blocks = []
        self.left_arch_blocks = []
        self.up_sampling_blocks = []
        self.right_arch_blocks = []
        self.reduction_factor = reduction_factor
        self.levels = num_levels

        key, lifting_key, projection_key = jr.split(key, 3)

        self.lifting = lifting_factory(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=hidden_channels,
            activation=activation,
            boundary_mode=boundary_mode,
            key=lifting_key,
            **boundary_kwargs,
        )
        self.projection = projection_factory(
            num_spatial_dims=num_spatial_dims,
            in_channels=hidden_channels,
            out_channels=out_channels,
            activation=activation,
            boundary_mode=boundary_mode,
            key=projection_key,
            **boundary_kwargs,
        )

        channel_list = [
            hidden_channels * self.reduction_factor**i for i in range(num_levels)
        ]

        for (
            fan_in,
            fan_out,
        ) in zip(
            channel_list[:-1],
            channel_list[1:],
        ):
            key, down_key, left_key, up_key, right_key = jr.split(key, 5)
            self.down_sampling_blocks.append(
                down_sampling_factory(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=fan_in,
                    out_channels=fan_in,
                    activation=activation,
                    boundary_mode=boundary_mode,
                    key=down_key,
                    **boundary_kwargs,
                )
            )
            self.left_arch_blocks.append(
                left_arch_factory(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=fan_in,
                    out_channels=fan_out,
                    activation=activation,
                    boundary_mode=boundary_mode,
                    key=left_key,
                    **boundary_kwargs,
                )
            )
            self.up_sampling_blocks.append(
                up_sampling_factory(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=fan_out,
                    out_channels=fan_out // self.reduction_factor,
                    activation=activation,
                    boundary_mode=boundary_mode,
                    key=up_key,
                    **boundary_kwargs,
                )
            )
            self.right_arch_blocks.append(
                right_arch_factory(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=self.reduction_factor * fan_in,
                    out_channels=fan_in,
                    activation=activation,
                    boundary_mode=boundary_mode,
                    key=right_key,
                    **boundary_kwargs,
                )
            )

    def __call__(self, x: Any) -> Any:
        spatial_shape = x.shape[1:]
        for dims in spatial_shape:
            if dims % self.reduction_factor**self.levels != 0:
                raise ValueError("Spatial dim issue")

        x = self.lifting(x)

        skips = []
        for down, left in zip(self.down_sampling_blocks, self.left_arch_blocks):
            skips.append(x)
            x = down(x)
            x = left(x)  # The last in the loop is the bottleneck block

        for up, right in zip(
            reversed(self.up_sampling_blocks),
            reversed(self.right_arch_blocks),
        ):
            skip = skips.pop()
            x = up(x)
            # Equinox models are by default single batch, hence the channels are
            # at axis=0
            x = jnp.concatenate((skip, x), axis=0)
            x = right(x)

        x = self.projection(x)

        return x
