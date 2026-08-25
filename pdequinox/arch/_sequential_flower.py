from typing import Callable, Literal, Optional

import jax
from jaxtyping import PRNGKeyArray

from .._sequential import Sequential
from ..blocks import (
    CoordinateAugmentationBlockFactory,
    FlowerBlockFactory,
    LinearChannelAdjustBlockFactory,
    PointwiseMLPBlockFactory,
)


class SequentialFlower(Sequential):
    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: int = 16,
        num_blocks: int = 4,
        num_heads: int = 8,
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
        A single-scale variant of the Flower architecture of Muser et al.
        (2026), [arXiv:2603.04430](https://arxiv.org/abs/2603.04430).

        Stacks [`pdequinox.blocks.FlowerBlock`][] at a single resolution,
        without the UNet scaffold of [`pdequinox.arch.Flower`][]. The blocks
        themselves are identical.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example
            traditional convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `hidden_channels`: The number of channels in the hidden layers.
            Default is `16`. Must be divisible by `num_heads`.
        - `num_blocks`: The number of Flower blocks. Default is `4`.
        - `num_heads`: The number of warping heads per block. Default is `8`.
        - `use_norm`: Whether the blocks use group normalization. Default is
            `True`.
        - `num_groups`: The number of groups for that normalization. If `None`
            (the default), `num_heads` is used.
        - `coordinate_encoding`: How the lifting encodes position; see
            [`pdequinox.blocks.CoordinateAugmentationBlockFactory`][]. Default
            is `"linear"`. Use `"none"` to keep the architecture
            translation-equivariant.
        - `displacement_scale`: A constant multiplier on the predicted
            displacements. Default is `1.0`.
        - `zero_init_displacement`: Whether to initialise all warps to the
            identity. Default is `False`.
        - `activation`: The block nonlinearity. Default is `jax.nn.gelu`. Note
            this does **not** reach the displacement MLPs; see
            `mlp_activation`.
        - `mlp_activation`: The activation inside the displacement MLPs.
            Default is `jax.nn.relu`.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        - `boundary_mode`: The boundary mode, used for what a warp reads when it
            samples outside the domain. Default is `"periodic"`.

        !!! info

            Unlike a convolutional ResNet, this has an **unbounded receptive
            field already at a depth of one block**: the displacements are not
            constrained, so a single head can in principle sample anywhere in
            the domain. The usual argument that a hierarchy is needed in order
            to grow the receptive field therefore does not apply here; the
            scaffold in [`pdequinox.arch.Flower`][] buys multiscale feature
            modelling, not reach.

        !!! info "Relation to the paper"

            The paper does report a single-scale variant (its Table 9), but that
            one embeds the input into 16x16 patches and operates at the reduced
            resolution, and it rearranges the block to a pre-normalization form.
            This implementation does neither: it runs at full resolution with
            the same post-normalization block as
            [`pdequinox.arch.Flower`][], so it contains no convolutional mixing
            and no patching at all. That configuration is not one the paper
            evaluated. The authors describe their single-scale results as a
            proof of concept they did not iterate on.
        """
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            activation=activation,
            key=key,
            boundary_mode=boundary_mode,
            lifting_factory=CoordinateAugmentationBlockFactory(
                LinearChannelAdjustBlockFactory(),
                coordinate_encoding=coordinate_encoding,
            ),
            block_factory=FlowerBlockFactory(
                num_heads=num_heads,
                mlp_activation=mlp_activation,
                use_norm=use_norm,
                num_groups=num_groups,
                displacement_scale=displacement_scale,
                zero_init_displacement=zero_init_displacement,
            ),
            projection_factory=PointwiseMLPBlockFactory(),
        )
