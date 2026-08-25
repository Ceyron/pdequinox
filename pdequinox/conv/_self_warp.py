"""
The core primitive of the Flower architecture (Muser et al., ICML 2026):
https://arxiv.org/abs/2603.04430

A multi-head warp (or "pullback") layer. All learned maps are pointwise (1x1
convolutions); nonlocality enters *only* through evaluating the features at
displaced coordinates.
"""

from typing import Callable, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from jaxtyping import Array, Float, PRNGKeyArray

from ._pointwise_linear_conv import PointwiseLinearConv

# The sampler extends the field beyond the domain in exactly the same way that
# `PhysicsConv` pads it, hence the boundary mode carries over one-to-one:
#
#     boundary mode | PhysicsConv padding | sampler mode
#     --------------|---------------------|-------------
#     periodic      | circular            | wrap
#     dirichlet     | zeros               | constant (cval=0)
#     neumann       | reflect             | mirror
#
# ("mirror" is the mode that matches `jnp.pad(..., mode='reflect')`, i.e. it
# mirrors without repeating the edge sample.)
BOUNDARY_MODE_TO_SAMPLER_MODE = {
    "periodic": "wrap",
    "dirichlet": "constant",
    "neumann": "mirror",
}


def _pullback(
    u: Float[Array, "C ..."],
    coordinates: Float[Array, "D ..."],
    *,
    boundary_mode: Literal["periodic", "dirichlet", "neumann"],
) -> Float[Array, "C ..."]:
    """
    Evaluate a field at (generally non-integer) coordinates, i.e. compute the
    pullback `(τ^* u)(x) = u(τ(x))`.

    Uses multi-linear interpolation (linear in 1d, bilinear in 2d, trilinear in
    3d) and is differentiable both with respect to `u` **and** with respect to
    `coordinates`. The latter is essential: it is what allows a displacement
    field to be learned.

    **Arguments:**

    - `u`: The field to read from, shape `(C, *spatial_in)`.
    - `coordinates`: Where to read, shape `(D, *spatial_out)`, given in **index
        units** (not normalized to `[-1, 1]`). `D` must match the number of
        spatial dimensions of `u`.
    - `boundary_mode`: How the field is extended beyond the domain. (Keyword
        only argument.)

    **Returns:**

    - The sampled field, shape `(C, *spatial_out)`.

    !!! info

        The output spatial shape is taken from `coordinates`, **not** from `u`.
        The two are deliberately decoupled: sampling onto a coarser or finer
        grid than the input is therefore expressible with this same routine.
    """
    if boundary_mode not in BOUNDARY_MODE_TO_SAMPLER_MODE:
        raise ValueError(
            "Only 'periodic', 'dirichlet', 'neumann' boundary modes are "
            f"supported, got {boundary_mode}"
        )
    sampler_mode = BOUNDARY_MODE_TO_SAMPLER_MODE[boundary_mode]

    if coordinates.shape[0] != u.ndim - 1:
        raise ValueError(
            f"coordinates must have {u.ndim - 1} leading entries to match the "
            f"spatial dimensions of the field, got {coordinates.shape[0]}"
        )

    # `map_coordinates` wants one array per spatial dimension
    coordinate_list = list(coordinates)

    return jax.vmap(
        lambda u_channel: map_coordinates(
            u_channel,
            coordinate_list,
            order=1,
            mode=sampler_mode,
            cval=0.0,
        )
    )(u)


class SelfWarp(eqx.Module):
    """
    Multi-head warp (pullback), the core primitive of the Flower architecture.
    """

    value_conv: PointwiseLinearConv
    flow_conv_1: PointwiseLinearConv
    flow_conv_2: PointwiseLinearConv
    activation: Callable
    num_spatial_dims: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    displacement_scale: float = eqx.field(static=True)
    boundary_mode: Literal["periodic", "dirichlet", "neumann"] = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_heads: int,
        *,
        activation: Callable = jax.nn.relu,
        displacement_scale: float = 1.0,
        zero_init_displacement: bool = False,
        use_bias: bool = True,
        boundary_mode: Literal["periodic", "dirichlet", "neumann"],
        key: PRNGKeyArray,
    ):
        """
        General n-dimensional multi-head warp, `u ↦ (Id + ϱ)^*(V u)`.

        Given an input field `u`, this layer

        1. computes a *value* field `v = V u` by a 1x1 convolution and splits it
            into `num_heads` groups of `out_channels // num_heads` channels
            each,
        2. computes one *displacement* field `ϱ⁽ʰ⁾(x) ∈ ℝ^d` per head via a
            small pointwise MLP (two 1x1 convolutions with an activation in
            between), and
        3. lets each head read its own values at the displaced coordinates,
            `w⁽ʰ⁾(x) = v⁽ʰ⁾(x + ϱ⁽ʰ⁾(x))`, before concatenating all heads.

        Every learned map is pointwise; the displacement at `x` depends only on
        `u(x)`, not on `u(x')` for `x' ≠ x`. All nonlocality of this layer comes
        from step 3, i.e. from *one* sparse sample per head. The cost is
        therefore linear in the number of grid points.

        Fixed (non-predicted) displacements would reduce this operation to a
        kernel-integral operator, and to an ordinary convolution if additionally
        the weights were shared across space. In that sense the warp generalizes
        convolution by making the sampling locations state-dependent.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example
            traditional convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels. Must be divisible by
            `num_heads`.
        - `num_heads`: The number of warping heads. Each head predicts its own
            displacement field and warps its own slice of the value field. One
            head is enough to represent a single transport direction; several
            heads allow concurrently active transport modes (and empirically
            matter most for the stability of long autoregressive rollouts).
        - `activation`: The activation function used inside the displacement
            MLP. Default is `jax.nn.relu`.
        - `displacement_scale`: A constant multiplier on the predicted
            displacements. Default is `1.0`, in which case a predicted value of
            `1.0` corresponds to a shift by half the domain extent. (Keyword
            only argument.)
        - `zero_init_displacement`: Whether to initialise the last layer of the
            displacement MLP to zero. The layer then starts out as a pure
            pointwise map (an identity warp) and grows its nonlocality during
            training. Default is `False`. (Keyword only argument.)
        - `use_bias`: Whether to add a bias in the pointwise convolutions.
            Default is `True`. (Keyword only argument.)
        - `boundary_mode`: The boundary mode used when a head samples outside
            of the domain. (Keyword only argument.)
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        !!! info

            Displacements are predicted in **normalized domain units**: they are
            multiplied by `(N_i - 1) / 2` per axis before being used, so a
            predicted displacement of `0.1` means a shift by 5% of the distance
            between the first and the last grid point, irrespective of the
            resolution `N_i`.

            This scaling is inherited from the reference implementation (which
            uses `align_corners=True`). Note that on a periodic axis the full
            period spans `N_i` cells rather than `N_i - 1`, so there the
            normalized unit is off by a factor of `(N_i - 1) / N_i` from a
            literal "fraction of the domain". This is a convention, not an
            error - it merely rescales what the displacement MLP has to learn -
            but it is worth knowing when interpreting displacements
            quantitatively.

            The coordinate grid is derived from the shape of the input inside
            `__call__` rather than being fixed at construction time. This layer
            (and every architecture built from it) is therefore agnostic to the
            spatial resolution: the same trained weights can be evaluated on a
            grid of any size. Under `jax.jit` the grid is a compile-time
            constant, so inferring it costs nothing.

        !!! tip

            This layer maps onto a grid of the same shape as its input, but the
            underlying pullback routine decouples the output grid
            from the input grid. A learned *adaptive resampling* layer - warping
            and changing the resolution in one step - is therefore expressible
            with the same primitive. The Flower paper does not do this; it uses
            ordinary strided convolutions for all resolution changes.
        """
        if out_channels % num_heads != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        boundary_mode = boundary_mode.lower()
        if boundary_mode not in BOUNDARY_MODE_TO_SAMPLER_MODE:
            raise ValueError(
                "Only 'periodic', 'dirichlet', 'neumann' boundary modes are "
                f"supported, got {boundary_mode}"
            )

        self.num_spatial_dims = num_spatial_dims
        self.num_heads = num_heads
        self.displacement_scale = displacement_scale
        self.boundary_mode = boundary_mode
        self.activation = activation

        value_key, flow_1_key, flow_2_key = jax.random.split(key, 3)

        self.value_conv = PointwiseLinearConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=use_bias,
            key=value_key,
        )

        # The displacement map is a two-layer pointwise MLP; its hidden width
        # matches the value map's output width.
        self.flow_conv_1 = PointwiseLinearConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=use_bias,
            key=flow_1_key,
        )
        flow_conv_2 = PointwiseLinearConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=out_channels,
            out_channels=num_spatial_dims * num_heads,
            use_bias=use_bias,
            key=flow_2_key,
        )
        if zero_init_displacement:
            flow_conv_2 = eqx.tree_at(
                lambda leaf: leaf.weight,
                flow_conv_2,
                jnp.zeros_like(flow_conv_2.weight),
            )
            if use_bias:
                flow_conv_2 = eqx.tree_at(
                    lambda leaf: leaf.bias,
                    flow_conv_2,
                    jnp.zeros_like(flow_conv_2.bias),
                )
        self.flow_conv_2 = flow_conv_2

    def __call__(self, x: Float[Array, "C_i ..."]) -> Float[Array, "C_o ..."]:
        spatial_shape = x.shape[1:]
        if len(spatial_shape) != self.num_spatial_dims:
            raise ValueError(
                f"Expected an input with {self.num_spatial_dims} spatial "
                f"dimensions, got an array of shape {x.shape}"
            )

        # (C_o, *spatial) -> (num_heads, C_o // num_heads, *spatial)
        value = self.value_conv(x)
        value = value.reshape(self.num_heads, -1, *spatial_shape)

        # (D * num_heads, *spatial) -> (num_heads, D, *spatial)
        displacement = self.flow_conv_2(self.activation(self.flow_conv_1(x)))
        displacement = displacement.reshape(
            self.num_heads, self.num_spatial_dims, *spatial_shape
        )

        # The identity grid in index units, derived from the input shape so that
        # this layer stays agnostic to the spatial resolution.
        base_grid = jnp.stack(
            jnp.meshgrid(
                *[jnp.arange(n, dtype=displacement.dtype) for n in spatial_shape],
                indexing="ij",
            )
        )

        # Displacements are predicted in normalized domain units, so convert
        # them to index units. Broadcasts as (D, 1, ..., 1) against (H, D, ...).
        half_extent = jnp.array(
            [(n - 1) / 2 for n in spatial_shape], dtype=displacement.dtype
        ).reshape(self.num_spatial_dims, *((1,) * self.num_spatial_dims))
        coordinates = base_grid + self.displacement_scale * half_extent * displacement

        # Each head warps its own slice of the value field along its own
        # displacement field.
        warped = jax.vmap(
            lambda value_head, coordinates_head: _pullback(
                value_head, coordinates_head, boundary_mode=self.boundary_mode
            )
        )(value, coordinates)

        return warped.reshape(-1, *spatial_shape)

    @property
    def receptive_field(self) -> tuple[tuple[float, float], ...]:
        # Displacements are unconstrained, so a head can in principle sample
        # anywhere in the domain. As with `SpectralConv`, the receptive field is
        # unbounded - though unlike `SpectralConv` it is also data-dependent,
        # and small at initialisation.
        return tuple(((jnp.inf, jnp.inf),) * self.num_spatial_dims)
