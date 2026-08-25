"""
The residual pullback block of the Flower architecture (Muser et al., ICML 2026):
https://arxiv.org/abs/2603.04430

Wraps the multi-head warp primitive in a residual block. Following the paper
(Sec. 2.1 and App. C.3), the block is

    Block = phi . Norm . (SelfWarp + IdProj)

i.e. post-normalization and post-activation. This differs from the
pre-activation ordering of `ModernResBlock`, and is deliberate: it is the
ordering the paper uses for the multiscale (U-Net) Flower.
"""

from typing import Callable, Literal, Optional

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from ..conv import PointwiseLinearConv, SelfWarp
from ._base_block import Block, BlockFactory


class FlowerBlock(Block):
    self_warp: SelfWarp
    id_proj: PointwiseLinearConv
    norm: eqx.Module
    activation: Callable

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        num_heads: int = 8,
        activation: Callable = jax.nn.gelu,
        mlp_activation: Callable = jax.nn.relu,
        use_norm: bool = True,
        num_groups: Optional[int] = None,
        displacement_scale: float = 1.0,
        zero_init_displacement: bool = False,
        use_bias: bool = True,
        boundary_mode: Literal["periodic", "dirichlet", "neumann"],
        key: PRNGKeyArray,
    ):
        """
        Residual pullback block: the core building block of the Flower
        architecture.

        Computes

            x -> activation(Norm(SelfWarp(x) + IdProj(x)))

        where [`pdequinox.conv.SelfWarp`][] is the multi-head warp, `IdProj` is a
        pointwise (1x1) convolution providing the residual path, and `Norm` is
        group normalization.

        Since `IdProj` already maps `in_channels` to `out_channels`, this block
        handles a change in the number of channels without any special case (in
        contrast to `ModernResBlock`, which needs a conditional bypass).

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example
            traditional convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels. Must be divisible by
            `num_heads`.
        - `num_heads`: The number of warping heads. Default is `8`. (Keyword
            only argument.)
        - `activation`: The nonlinearity applied at the **end of the block**.
            Default is `jax.nn.gelu`. See the note below; this does *not* affect
            the displacement MLP.
        - `mlp_activation`: The activation used **inside the displacement MLP**
            of the warp. Default is `jax.nn.relu`. See the note below.
        - `use_norm`: Whether to use group normalization. Default is `True`.
            (Keyword only argument.)
        - `num_groups`: The number of groups for the group normalization. If
            `None` (the default), this is set to `num_heads`, which is what the
            reference implementation does and which always divides
            `out_channels`. (Keyword only argument.)
        - `displacement_scale`: A constant multiplier on the predicted
            displacements. Default is `1.0`. (Keyword only argument.)
        - `zero_init_displacement`: Whether to initialise the warp to the
            identity. Default is `False`. (Keyword only argument.)
        - `use_bias`: Whether to add a bias in the pointwise convolutions.
            Default is `True`. (Keyword only argument.)
        - `boundary_mode`: The boundary mode used when a head samples outside of
            the domain. (Keyword only argument.)
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        !!! warning "Two separate activations"

            This block uses **two different activation functions**, and
            `activation` only controls one of them:

            - `activation` is the block nonlinearity, applied once at the very
                end (after the normalization). The reference uses GELU here.
            - `mlp_activation` is used *inside* the displacement MLP of the
                warp, between its two pointwise convolutions. The reference uses
                ReLU here.

            The reference implementation hardcodes both. Since the block factory
            protocol of this package passes down only a single `activation`,
            that one is routed to the block nonlinearity, and the displacement
            MLP keeps its own argument. So passing `activation=...` will **not**
            change the activation inside the displacement MLP; use
            `mlp_activation` for that.

        !!! info

            `out_channels` must be divisible by `num_heads`. When this block is
            used inside [`pdequinox.Hierarchical`][], the number of channels a
            block is constructed with is always an integer multiple of
            `hidden_channels`, so it is sufficient that `num_heads` divides
            `hidden_channels`.
        """
        warp_key, id_proj_key = jax.random.split(key)

        self.self_warp = SelfWarp(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            activation=mlp_activation,
            displacement_scale=displacement_scale,
            zero_init_displacement=zero_init_displacement,
            use_bias=use_bias,
            boundary_mode=boundary_mode,
            key=warp_key,
        )
        self.id_proj = PointwiseLinearConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=use_bias,
            key=id_proj_key,
        )

        if use_norm:
            groups = num_heads if num_groups is None else num_groups
            self.norm = eqx.nn.GroupNorm(groups=groups, channels=out_channels)
        else:
            self.norm = eqx.nn.Identity()

        self.activation = activation

    def __call__(self, x):
        return self.activation(self.norm(self.self_warp(x) + self.id_proj(x)))

    @property
    def receptive_field(self) -> tuple[tuple[float, float], ...]:
        # The pointwise residual contributes nothing spatially, so the block
        # inherits the (unbounded) reach of the warp.
        return self.self_warp.receptive_field


class FlowerBlockFactory(BlockFactory):
    num_heads: int
    mlp_activation: Callable
    use_norm: bool
    num_groups: Optional[int]
    displacement_scale: float
    zero_init_displacement: bool
    use_bias: bool

    def __init__(
        self,
        *,
        num_heads: int = 8,
        mlp_activation: Callable = jax.nn.relu,
        use_norm: bool = True,
        num_groups: Optional[int] = None,
        displacement_scale: float = 1.0,
        zero_init_displacement: bool = False,
        use_bias: bool = True,
    ):
        """
        Factory for creating `FlowerBlock` instances.

        **Arguments:**

        - `num_heads`: The number of warping heads. Default is `8`. This is kept
            fixed across all levels of a hierarchical architecture (as in the
            reference implementation), so the number of channels per head grows
            with depth.
        - `mlp_activation`: The activation used inside the displacement MLP of
            the warp. Default is `jax.nn.relu`. Note that this is *separate*
            from the `activation` handed to the block by the architecture, which
            only controls the block nonlinearity; see
            [`pdequinox.blocks.FlowerBlock`][].
        - `use_norm`: Whether to use group normalization. Default is `True`.
        - `num_groups`: The number of groups for the group normalization. If
            `None` (the default), this is set to `num_heads`.
        - `displacement_scale`: A constant multiplier on the predicted
            displacements. Default is `1.0`.
        - `zero_init_displacement`: Whether to initialise the warps to the
            identity. Default is `False`.
        - `use_bias`: Whether to add a bias in the pointwise convolutions.
            Default is `True`.

        !!! info

            The number of output channels of every block must be divisible by
            `num_heads`. Inside [`pdequinox.Hierarchical`][] the channel counts
            are integer multiples of `hidden_channels`, so it suffices that
            `num_heads` divides `hidden_channels`.
        """
        self.num_heads = num_heads
        self.mlp_activation = mlp_activation
        self.use_norm = use_norm
        self.num_groups = num_groups
        self.displacement_scale = displacement_scale
        self.zero_init_displacement = zero_init_displacement
        self.use_bias = use_bias

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
        return FlowerBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_heads=self.num_heads,
            # Only the block nonlinearity; the displacement MLP keeps its own
            activation=activation,
            mlp_activation=self.mlp_activation,
            use_norm=self.use_norm,
            num_groups=self.num_groups,
            displacement_scale=self.displacement_scale,
            zero_init_displacement=self.zero_init_displacement,
            use_bias=self.use_bias,
            boundary_mode=boundary_mode,
            key=key,
        )
