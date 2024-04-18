from typing import Callable, Optional

import jax
from jaxtyping import PRNGKeyArray

from .._hierarchical import Hierarchical
from ..blocks import (
    LinearChannelAdjustBlockFactory,
    LinearConvDownBlockFactory,
    LinearConvUpBlockFactory,
    ModernResBlockFactory,
)


class ModernUNet(Hierarchical):
    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: int = 16,
        num_levels: int = 4,
        num_blocks: int = 2,
        channel_multipliers: Optional[tuple[int, ...]] = None,
        use_norm: bool = True,
        activation: Callable = jax.nn.relu,
        key: PRNGKeyArray,
        boundary_mode: str = "periodic",
        **boundary_kwargs,
    ):
        """
        A modern UNet version close to the ones used by Gupta & Brandstetter
        (2023) in PDEArena.

        Uses ResNet blocks for the left and right arc of the UNet.

        In comparison to the version in PDEArena, the `num_block` in the left
        and right arc of the UNet are identical (PDEArena uses one additional in
        the right arc). Here, we also do not do multi-skips, only the last state
        in the processing of one hierarchy level is skip-connected to the
        decoder.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example
            traditional convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `hidden_channels`: The number of channels in the hidden layers.
            Default is `16`. This is the number of channels in finest (input)
            spatial resolution.
        - `num_levels`: The number of levels in the hierarchy. Default is `4`.
            Each level halves the spatial resolution. By default, it also
            doubles the number of channels. This can be changed by setting
            `channel_multipliers`.
        - `num_blocks`: The number of blocks in the left and right arc of the
            UNet, for each level. One block is a single modern ResNet block
            (using pre-activation) consisting of two convolutions. The default
            value of `num_blocks` is `2` meaning that for each level in both the
            encoder, bottleneck and decoder, two blocks are used. Hence, there
            are a total of four convolutions contributing receptive field per
            level.
        - `channel_multipliers`: A tuple of integers that specify the channel
            multipliers for each level. If `None`, the default is to double the
            number of channels at each level (for `num_levels=4` this would mean
            `(2, 4, 8, 16)`). The length of the tuple should be equal to
            `num_levels`.
        - `use_norm`: If `True`, uses group norm as part of the ResNet blocks.
            Default is `True`.
        - `activation`: The activation function to use in the blocks. Default is
            `jax.nn.relu`.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        - `boundary_mode`: The boundary mode to use. Default is `periodic`.
        """
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            num_blocks=num_blocks,
            channel_multipliers=channel_multipliers,
            activation=activation,
            key=key,
            boundary_mode=boundary_mode,
            lifting_factory=ModernResBlockFactory(
                use_norm=use_norm,
            ),
            down_sampling_factory=LinearConvDownBlockFactory(),
            left_arc_factory=ModernResBlockFactory(
                use_norm=use_norm,
            ),
            up_sampling_factory=LinearConvUpBlockFactory(),
            right_arc_factory=ModernResBlockFactory(
                use_norm=use_norm,
            ),
            projection_factory=LinearChannelAdjustBlockFactory(),
            **boundary_kwargs,
        )
