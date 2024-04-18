from typing import Any, Callable, List, Literal, Optional

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

from ._utils import sum_receptive_fields
from .blocks import (
    Block,
    BlockFactory,
    ClassicDoubleConvBlockFactory,
    LinearChannelAdjustBlockFactory,
    LinearConvDownBlockFactory,
    LinearConvUpBlockFactory,
)
from .conv import PhysicsConv


class Hierarchical(eqx.Module):
    lifting: Block
    down_sampling_blocks: List[Block]
    left_arc_blocks: List[List[Block]]  # Includes the bottleneck
    up_sampling_blocks: List[Block]
    right_arc_blocks: List[List[Block]]
    projection: PhysicsConv
    reduction_factor: int
    num_levels: int
    num_blocks: int
    channel_multipliers: tuple[int, ...]

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: int,
        num_levels: int,
        num_blocks: int,
        activation: Callable,
        key: PRNGKeyArray,
        reduction_factor: int = 2,
        boundary_mode: Literal["periodic", "dirichlet", "neumann"],
        channel_multipliers: Optional[tuple[int, ...]] = None,
        lifting_factory: BlockFactory = ClassicDoubleConvBlockFactory(),
        down_sampling_factory: BlockFactory = LinearConvDownBlockFactory(),
        left_arc_factory: BlockFactory = ClassicDoubleConvBlockFactory(),
        up_sampling_factory: BlockFactory = LinearConvUpBlockFactory(),
        right_arc_factory: BlockFactory = ClassicDoubleConvBlockFactory(),
        projection_factory: BlockFactory = LinearChannelAdjustBlockFactory(),
    ):
        """
        Generic constructor for hierarchical block-based architectures like
        UNets. (For the classic UNet, use `pdequinox.arch.ClassicUNet` instead.)

        Hierarchical architectures us a number of different spatial resolutions.
        The lower the resolution, the wider the receptive field of convolutions.

        Allows to increase the number of blocks per level via the `num_blocks`
        argument. This will be identical for the left arc (=encoder) and the
        right arc (=decoder). **No multi-skip** as in PDEArena.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example
            traditional convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `hidden_channels`: The number of channels in the hidden layers. This
            refers to the highest resolution. Right after the input, the input
            channels will be lifted to this feature dimension without changing
            the spatial resolution.
        - `num_levels`: The number of levels in the hierarchy. This is the
            number of down and up sampling blocks. If set to 0, this will just
            be a classical conv net. If set to 1, this will be a single down and
            up sampling block etc. The total number of resolutions are
            `num_levels + 1`.
        - `num_blocks`: The number of blocks to use at each level. (Also affects
            the number of blocks in the bottleneck.)
        - `activation`: The activation function to use in the blocks.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        - `reduction_factor`: The factor by which the spatial resolution is
            reduced at each level. This has to be an integer. In order to avoid
            ambiguities in shapes, it is best if the input spatial resolution is
            a multiple of `reduction_factor ** num_levels`. Default is `2`.
        - `boundary_mode`: The boundary mode to use for the convolution.
            (Keyword only argument)
        - `channel_multipliers`: The factor by which the number of channels is
            multiplied at each level. If set to `None`, the channels will grow
            by a factor of `reduction_factor` at each level. This is similar to
            the classical UNet which trades spatial resolution for feature
            dimension. Note however, that the parameters of convolutions scale
            with the mapped channels, hence the majority of numbers will then be
            in the coarsest representation. Supply a tuple of integers that
            represent the desired number of channels at each resolution
            different than the original one. The length of the tuple must match
            `num_levels`. For example, to not change the number of channels at
            any level, set this to `(1,) * num_levels`. Default is `None`.
        - `lifting_factory`: The factory to use for the lifting block.
            Default is `ClassicDoubleConvBlockFactory` which is a classic double
            convolution block.
        - `down_sampling_factory`: The factory to use for the down sampling
            blocks. This must be a block that is able to change the spatial
            resolution. Default is `LinearConvDownBlockFactory` which is a
            simple linear strided convolution block.
        - `left_arch_factory`: The factory to use for the left architecture
            blocks. Default is `ClassicDoubleConvBlockFactory` which is a
            classic double convolution block.
        - `up_sampling_factory`: The factory to use for the up sampling blocks.
            This must be a block that is able to change the spatial resolution.
            It should work in conjunction with the `down_sampling_factory`.
            Default is `LinearConvUpBlockFactory` which is a simple linear
            strided transposed convolution block.
        - `right_arch_factory`: The factory to use for the right architecture
            blocks. Default is `ClassicDoubleConvBlockFactory` which is a
            classic double convolution block.
        - `projection_factory`: The factory to use for the projection block.
            Default is `LinearChannelAdjustBlockFactory` which is simply a
            linear 1x1 convolution for channel adjustment.
        """
        self.down_sampling_blocks = []
        self.left_arc_blocks = []
        self.up_sampling_blocks = []
        self.right_arc_blocks = []
        self.reduction_factor = reduction_factor
        self.num_levels = num_levels

        key, lifting_key, projection_key = jr.split(key, 3)

        self.lifting = lifting_factory(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=hidden_channels,
            activation=activation,
            boundary_mode=boundary_mode,
            key=lifting_key,
        )
        self.projection = projection_factory(
            num_spatial_dims=num_spatial_dims,
            in_channels=hidden_channels,
            out_channels=out_channels,
            activation=activation,
            boundary_mode=boundary_mode,
            key=projection_key,
        )

        if channel_multipliers is not None:
            if len(channel_multipliers) != num_levels:
                raise ValueError("len(channel_multipliers) must match num_levels")
        else:
            channel_multipliers = tuple(
                self.reduction_factor**i for i in range(1, num_levels + 1)
            )

        self.channel_multipliers = channel_multipliers

        channel_list = [
            hidden_channels,
        ] + [hidden_channels * m for m in channel_multipliers]

        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1")
        self.num_blocks = num_blocks

        for (
            fan_in,
            fan_out,
        ) in zip(
            channel_list[:-1],
            channel_list[1:],
        ):
            # If num_levels is 0, the loop will not run
            key, down_key, left_key, up_key, right_key = jr.split(key, 5)
            self.down_sampling_blocks.append(
                down_sampling_factory(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=fan_in,
                    out_channels=fan_in,
                    activation=activation,
                    boundary_mode=boundary_mode,
                    key=down_key,
                )
            )

            this_level_left_arc_blocks = []
            # The first block changes the number of channels
            this_level_left_arc_blocks.append(
                left_arc_factory(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=fan_in,
                    out_channels=fan_out,
                    activation=activation,
                    boundary_mode=boundary_mode,
                    key=left_key,
                )
            )
            for _ in range(num_blocks - 1):
                this_level_left_arc_blocks.append(
                    left_arc_factory(
                        num_spatial_dims=num_spatial_dims,
                        # All subsequent blocks have the same number of channels
                        in_channels=fan_out,
                        out_channels=fan_out,
                        activation=activation,
                        boundary_mode=boundary_mode,
                        key=left_key,
                    )
                )
            self.left_arc_blocks.append(this_level_left_arc_blocks)

            self.up_sampling_blocks.append(
                up_sampling_factory(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=fan_out,
                    out_channels=fan_out // self.reduction_factor,
                    activation=activation,
                    boundary_mode=boundary_mode,
                    key=up_key,
                )
            )

            this_level_right_arc_blocks = []
            # The first block changes the number of channels, and operates
            # together with incoming skip connection
            this_level_right_arc_blocks.append(
                right_arc_factory(
                    num_spatial_dims=num_spatial_dims,
                    in_channels=fan_out // self.reduction_factor + fan_in,
                    out_channels=fan_in,
                    activation=activation,
                    boundary_mode=boundary_mode,
                    key=right_key,
                )
            )
            for _ in range(num_blocks - 1):
                # All subsequent blocks have the same number of channels
                this_level_right_arc_blocks.append(
                    right_arc_factory(
                        num_spatial_dims=num_spatial_dims,
                        in_channels=fan_in,
                        out_channels=fan_in,
                        activation=activation,
                        boundary_mode=boundary_mode,
                        key=right_key,
                    )
                )
            self.right_arc_blocks.append(this_level_right_arc_blocks)

    def __call__(self, x: Any) -> Any:
        spatial_shape = x.shape[1:]
        for dims in spatial_shape:
            if dims % self.reduction_factor**self.num_levels != 0:
                raise ValueError("Spatial dim issue")

        x = self.lifting(x)

        skips = []
        for down, left_list in zip(self.down_sampling_blocks, self.left_arc_blocks):
            # If num_levels is 0, the loop will not run
            skips.append(x)
            x = down(x)

            # The last in the loop is the bottleneck block
            for left in left_list:
                x = left(x)

        for up, right_list in zip(
            reversed(self.up_sampling_blocks),
            reversed(self.right_arc_blocks),
        ):
            # If num_levels is 0, the loop will not run
            skip = skips.pop()
            x = up(x)
            # Equinox models are by default single batch, hence the channels are
            # at axis=0
            x = jnp.concatenate((skip, x), axis=0)
            for right in right_list:
                x = right(x)

        x = self.projection(x)

        return x

    @property
    def receptive_field(self) -> tuple[tuple[float, float], ...]:
        lifting_receptive_field = self.lifting.receptive_field
        projection_receptive_fields = self.projection.receptive_field

        down_receptive_fields = tuple(
            block.receptive_field for block in self.down_sampling_blocks
        )
        left_receptive_fields = tuple(
            tuple(block.receptive_field for block in block_list)
            for block_list in self.left_arc_blocks
        )
        up_receptive_fields = tuple(
            block.receptive_field for block in self.up_sampling_blocks
        )
        right_receptive_fields = tuple(
            tuple(block.receptive_field for block in block_list)
            for block_list in self.right_arc_blocks
        )

        spatial_reduction = tuple(
            self.reduction_factor**level for level in range(0, self.num_levels + 1)
        )

        # Down block acts on the fan_in spatial resolution
        scaled_down_receptive_field = tuple(
            tuple(
                (c_i_backward * r, c_i_forward * r)
                for (c_i_backward, c_i_forward) in conv_receptive_field
            )
            for conv_receptive_field, r in zip(
                down_receptive_fields, spatial_reduction[:-1]
            )
        )

        # Left block acts on the fan_out spatial resolution
        scaled_left_receptive_field = tuple(
            tuple(
                (c_i_backward * r, c_i_forward * r)
                for (c_i_backward, c_i_forward) in conv_receptive_field
            )
            for conv_receptive_field_list, r in zip(
                left_receptive_fields, spatial_reduction[1:]
            )
            for conv_receptive_field in conv_receptive_field_list
        )

        # Up block acts on the fan_out spatial resolution
        scaled_up_receptive_field = tuple(
            tuple(
                (c_i_backward * r, c_i_forward * r)
                for (c_i_backward, c_i_forward) in conv_receptive_field
            )
            for conv_receptive_field, r in zip(
                up_receptive_fields, spatial_reduction[1:]
            )
        )

        # Right block acts on the fan_in spatial resolution
        scaled_right_receptive_field = tuple(
            tuple(
                (c_i_backward * r, c_i_forward * r)
                for (c_i_backward, c_i_forward) in conv_receptive_field
            )
            for conv_receptive_field_list, r in zip(
                right_receptive_fields, spatial_reduction[:-1]
            )
            for conv_receptive_field in conv_receptive_field_list
        )

        collection_of_receptive_fields = (
            lifting_receptive_field,
            *scaled_down_receptive_field,
            *scaled_left_receptive_field,
            *scaled_up_receptive_field,
            *scaled_right_receptive_field,
            projection_receptive_fields,
        )

        return sum_receptive_fields(collection_of_receptive_fields)
