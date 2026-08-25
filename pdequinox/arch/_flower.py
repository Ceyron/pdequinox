from typing import Callable, Literal, Optional

import jax
from jaxtyping import PRNGKeyArray

from .._hierarchical import Hierarchical
from ..blocks import (
    CoordinateAugmentationBlockFactory,
    FlowerBlockFactory,
    LinearChannelAdjustBlockFactory,
    LinearConvDownBlockFactory,
    LinearConvUpBlockFactory,
    PointwiseMLPBlockFactory,
)


class Flower(Hierarchical):
    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: int = 16,
        num_levels: int = 4,
        num_blocks: int = 1,
        num_heads: int = 8,
        channel_multipliers: Optional[tuple[int, ...]] = None,
        use_norm: bool = True,
        num_groups: Optional[int] = None,
        coordinate_encoding: Literal["linear", "fourier", "auto", "none"] = "linear",
        displacement_scale: float = 1.0,
        zero_init_displacement: bool = False,
        activation: Callable = jax.nn.gelu,
        mlp_activation: Callable = jax.nn.relu,
        key: PRNGKeyArray,
        boundary_mode: Literal["periodic", "dirichlet", "neumann"] = "periodic",
    ):
        """
        The Flower architecture of Muser et al. (2026),
        [arXiv:2603.04430](https://arxiv.org/abs/2603.04430).

        A UNet-like multiscale network whose only mixing primitive is a
        multi-head **warp**: at each position the network predicts one
        displacement per head, purely from the local state, and reads its
        features at the displaced coordinates. Every learned map is pointwise;
        all nonlocality comes from that sampling. See
        [`pdequinox.conv.SelfWarp`][] for the primitive and
        [`pdequinox.blocks.FlowerBlock`][] for the residual block.

        The multiscale scaffold is a UNet: a coordinate-augmented pointwise
        lifting, strided convolutions for the resolution changes, one Flower
        block per level in each arc, and a pointwise MLP as the projection head.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example
            traditional convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `hidden_channels`: The number of channels at the finest resolution.
            Default is `16`. Must be divisible by `num_heads`.
        - `num_levels`: The number of down- and up-sampling steps. Default is
            `4`. See the note on conventions below.
        - `num_blocks`: The number of Flower blocks per level in each arc.
            Default is `1`, matching the reference implementation.
        - `num_heads`: The number of warping heads, held fixed across all
            levels (as in the reference), so that the number of channels per
            head grows with depth. Default is `8`.
        - `channel_multipliers`: The channel multiplier per level. If `None`
            (the default), the channels double at every level.
        - `use_norm`: Whether the Flower blocks use group normalization.
            Default is `True`.
        - `num_groups`: The number of groups for that normalization. If `None`
            (the default), `num_heads` is used, as in the reference.
        - `coordinate_encoding`: How the lifting encodes position; see
            [`pdequinox.blocks.CoordinateAugmentationBlockFactory`][]. Default
            is `"linear"`, reproducing the reference. Use `"none"` to keep the
            architecture translation-equivariant.
        - `displacement_scale`: A constant multiplier on the predicted
            displacements. Default is `1.0`.
        - `zero_init_displacement`: Whether to initialise all warps to the
            identity. Default is `False`.
        - `activation`: The block nonlinearity. Default is `jax.nn.gelu`,
            matching the reference. Note this does **not** reach the
            displacement MLPs; see `mlp_activation`.
        - `mlp_activation`: The activation inside the displacement MLPs.
            Default is `jax.nn.relu`, matching the reference.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        - `boundary_mode`: The boundary mode, used both for the convolutions and
            for what a warp reads when it samples outside the domain. Default is
            `"periodic"`.

        !!! tip "Reproducing the configuration from the paper"

            The defaults above are sized in line with the other architectures in
            this package (roughly 0.5M parameters in 2d), not with the paper,
            whose width was chosen to hit the 15-20M parameter budget of The
            Well benchmark. For that configuration use

            ```python
            Flower(..., hidden_channels=160, num_heads=40, num_levels=3)
            ```

        !!! info "Differences to the reference implementation"

            This is built on [`pdequinox.Hierarchical`][], whose UNet scaffold
            differs from the reference in a few places. These are structural
            choices, not oversights, but they mean the two are not
            parameter-identical:

            1. **Block placement is mirrored.** The reference applies a Flower
                block at a resolution and *then* downsamples; `Hierarchical`
                downsamples first and applies its blocks at the coarser
                resolution. Both skip-connect the same tensor (the level input).
            2. **The decoder is arranged differently.** The reference does
                block, then up-convolution, then concatenation of the skip;
                `Hierarchical` does up-convolution, concatenation, then block,
                with correspondingly different channel counts.
            3. **`num_levels` counts down- and up-sampling steps**, following
                the convention of this package, so there are `num_levels + 1`
                resolutions. The paper counts resolutions instead, so its
                "`n_levels=4`" corresponds to `num_levels=3` here.
            4. **Expect around 0.76x the parameters** of the reference at
                matched settings: at `hidden_channels=160, num_heads=40,
                num_levels=3` this yields roughly 13.2M against the reference's
                17.3M. That difference comes from points 1-3 and is expected.
            5. There is no FiLM conditioning on simulation parameters. For
                scalar metadata see
                [`pdequinox.ConstantEmbeddingMetadataNetwork`][].
            6. `boundary_mode` applies to all axes. The reference supports a
                different boundary condition per axis, which is the common case
                in The Well (periodic along one axis, walls along another).
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
            lifting_factory=CoordinateAugmentationBlockFactory(
                LinearChannelAdjustBlockFactory(),
                coordinate_encoding=coordinate_encoding,
            ),
            down_sampling_factory=LinearConvDownBlockFactory(kernel_size=2),
            left_arc_factory=FlowerBlockFactory(
                num_heads=num_heads,
                mlp_activation=mlp_activation,
                use_norm=use_norm,
                num_groups=num_groups,
                displacement_scale=displacement_scale,
                zero_init_displacement=zero_init_displacement,
            ),
            up_sampling_factory=LinearConvUpBlockFactory(kernel_size=2),
            right_arc_factory=FlowerBlockFactory(
                num_heads=num_heads,
                mlp_activation=mlp_activation,
                use_norm=use_norm,
                num_groups=num_groups,
                displacement_scale=displacement_scale,
                zero_init_displacement=zero_init_displacement,
            ),
            projection_factory=PointwiseMLPBlockFactory(),
        )
