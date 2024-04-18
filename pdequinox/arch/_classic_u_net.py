from typing import Callable, Literal

import jax
from jaxtyping import PRNGKeyArray

from .._hierarchical import Hierarchical
from ..blocks import (
    ClassicDoubleConvBlockFactory,
    LinearChannelAdjustBlockFactory,
    LinearConvDownBlockFactory,
    LinearConvUpBlockFactory,
)


class ClassicUNet(Hierarchical):
    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: int = 16,
        num_levels: int = 4,
        use_norm: bool = True,
        activation: Callable = jax.nn.relu,
        key: PRNGKeyArray,
        boundary_mode: Literal["periodic", "dirichlet", "neumann"] = "periodic",
    ):
        """
        The vanilla UNet archiecture very close to the original Ronneberger et
        al. (2015) paper.

        Uses a hierarchy of spatial resolutions to obtain a wide receptive
        field.

        This version does **not** use maxpool for downsampling but instead uses
        a strided convolution. Up- and downsampling use 3x3 operations (instead
        of 2x2 operations). If active, uses group norm instead of batch norm.


        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example
            traditional convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `hidden_channels`: The number of channels in the hidden layers.
            Default is `16`. This is the number of channels in finest (input)
            spatial resolution.
        - `num_levels`: The number of levels in the hierarchy. Default is `4`.
            Each level halves the spatial resolution while doubling the number
            of channels.
        - `use_norm`: If `True`, uses group norm as part of double convolutions.
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
            num_blocks=1,
            activation=activation,
            key=key,
            boundary_mode=boundary_mode,
            lifting_factory=ClassicDoubleConvBlockFactory(
                use_norm=use_norm,
            ),
            down_sampling_factory=LinearConvDownBlockFactory(),
            left_arc_factory=ClassicDoubleConvBlockFactory(
                use_norm=use_norm,
            ),
            up_sampling_factory=LinearConvUpBlockFactory(),
            right_arc_factory=ClassicDoubleConvBlockFactory(
                use_norm=use_norm,
            ),
            projection_factory=LinearChannelAdjustBlockFactory(),
        )
