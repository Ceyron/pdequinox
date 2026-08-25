"""
Tests for the multi-head warp primitive of the Flower architecture.

The centerpiece is `test_analytic_advection_*`: for 1d linear advection the
exact solution operator *is* a pullback, `u(t + Δt, x) = u(t, x - c Δt)`, so a
single-head `SelfWarp` with an identity value map and a constant displacement
reproduces it exactly. That checks the primitive against a closed form rather
than against itself.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import pdequinox as pdeqx


def _constant_displacement_warp(warp, *, displacement: float):
    """
    Rewire a single-head, single-channel `SelfWarp` into a pure shift operator:
    an identity value map and a spatially constant displacement field.
    """
    # Value map becomes the identity
    warp = eqx.tree_at(
        lambda w: w.value_conv.weight, warp, jnp.ones_like(warp.value_conv.weight)
    )
    warp = eqx.tree_at(
        lambda w: w.value_conv.bias, warp, jnp.zeros_like(warp.value_conv.bias)
    )
    # First layer of the displacement MLP outputs zero everywhere ...
    warp = eqx.tree_at(
        lambda w: w.flow_conv_1.weight, warp, jnp.zeros_like(warp.flow_conv_1.weight)
    )
    warp = eqx.tree_at(
        lambda w: w.flow_conv_1.bias, warp, jnp.zeros_like(warp.flow_conv_1.bias)
    )
    # ... so the second layer's bias alone sets the (constant) displacement
    warp = eqx.tree_at(
        lambda w: w.flow_conv_2.weight, warp, jnp.zeros_like(warp.flow_conv_2.weight)
    )
    warp = eqx.tree_at(
        lambda w: w.flow_conv_2.bias,
        warp,
        jnp.full_like(warp.flow_conv_2.bias, displacement),
    )
    return warp


def _shift_warp(num_points: int, shift: float, boundary_mode: str):
    """A `SelfWarp` that shifts a 1d field by `shift` grid cells."""
    warp = pdeqx.conv.SelfWarp(
        num_spatial_dims=1,
        in_channels=1,
        out_channels=1,
        num_heads=1,
        boundary_mode=boundary_mode,
        key=jax.random.PRNGKey(0),
    )
    # Displacements are in normalized domain units, i.e. index units divided by
    # (N - 1) / 2
    displacement = 2.0 * shift / (num_points - 1)
    return _constant_displacement_warp(warp, displacement=displacement)


@pytest.mark.parametrize("shift", [1, 2, 5])
def test_analytic_advection_integer_shift(shift: int):
    """An integer shift must reproduce `jnp.roll` exactly."""
    num_points = 16
    warp = _shift_warp(num_points, shift, "periodic")

    u = jax.random.normal(jax.random.PRNGKey(42), (1, num_points))

    assert jnp.allclose(warp(u), jnp.roll(u, -shift, axis=-1), atol=1e-5)


@pytest.mark.parametrize("shift", [0.25, 0.5, 1.5, 2.75])
def test_analytic_advection_fractional_shift(shift: float):
    """
    A fractional shift must reproduce linear interpolation between the two
    neighbouring grid points.
    """
    num_points = 16
    warp = _shift_warp(num_points, shift, "periodic")

    u = jax.random.normal(jax.random.PRNGKey(42), (1, num_points))

    lower = int(jnp.floor(shift))
    fraction = shift - lower
    reference = (1.0 - fraction) * jnp.roll(u, -lower, axis=-1) + fraction * jnp.roll(
        u, -(lower + 1), axis=-1
    )

    assert jnp.allclose(warp(u), reference, atol=1e-5)


@pytest.mark.parametrize("boundary_mode", ["periodic", "dirichlet", "neumann"])
def test_boundary_handling(boundary_mode: str):
    """
    Shifting a ramp by one cell reads out of the domain at the last grid point.
    Each boundary mode must extend the field the same way the corresponding
    `PhysicsConv` padding would.
    """
    num_points = 8
    warp = _shift_warp(num_points, 1, boundary_mode)

    # Deliberately starts at 1 (not 0) so that the periodic and dirichlet cases
    # produce different values at the boundary
    u = jnp.arange(1, num_points + 1, dtype=jnp.float32).reshape(1, num_points)

    correct_output = jnp.arange(2, num_points + 2, dtype=jnp.float32)
    if boundary_mode == "periodic":
        # wraps around to the first entry
        correct_output = correct_output.at[-1].set(1.0)
    elif boundary_mode == "dirichlet":
        # reads zero outside of the domain
        correct_output = correct_output.at[-1].set(0.0)
    elif boundary_mode == "neumann":
        # mirrors without repeating the edge sample
        correct_output = correct_output.at[-1].set(float(num_points - 1))

    assert jnp.allclose(warp(u)[0], correct_output, atol=1e-5)


def test_gradient_reaches_displacement_mlp():
    """
    The gradient must flow into the displacement MLP, not just the value map.

    This guards a failure mode that would otherwise be silent: if the
    coordinates were detached, the layer would degrade into a purely pointwise
    map and still train, only much worse.
    """
    warp = pdeqx.conv.SelfWarp(
        num_spatial_dims=1,
        in_channels=2,
        out_channels=4,
        num_heads=2,
        boundary_mode="periodic",
        key=jax.random.PRNGKey(0),
    )

    u = jax.random.normal(jax.random.PRNGKey(1), (2, 16))

    grad = eqx.filter_grad(lambda model: jnp.sum(model(u) ** 2))(warp)

    assert jnp.any(grad.flow_conv_2.weight != 0.0)
    assert jnp.any(grad.flow_conv_2.bias != 0.0)
    assert jnp.any(grad.flow_conv_1.weight != 0.0)
    assert jnp.any(grad.value_conv.weight != 0.0)
    assert jnp.all(jnp.isfinite(grad.flow_conv_2.weight))


def test_zero_init_displacement_is_pure_value_map():
    """With a zero displacement field the layer reduces to its value map."""
    warp = pdeqx.conv.SelfWarp(
        num_spatial_dims=2,
        in_channels=3,
        out_channels=8,
        num_heads=4,
        zero_init_displacement=True,
        boundary_mode="periodic",
        key=jax.random.PRNGKey(0),
    )

    u = jax.random.normal(jax.random.PRNGKey(1), (3, 8, 8))

    assert jnp.allclose(warp(u), warp.value_conv(u), atol=1e-5)


@pytest.mark.parametrize(
    "num_spatial_dims,spatial_shape",
    [
        (1, (16,)),
        (2, (8, 12)),  # deliberately not square
        (3, (4, 6, 8)),
    ],
)
def test_shapes(num_spatial_dims: int, spatial_shape: tuple[int, ...]):
    warp = pdeqx.conv.SelfWarp(
        num_spatial_dims=num_spatial_dims,
        in_channels=2,
        out_channels=6,
        num_heads=3,
        boundary_mode="periodic",
        key=jax.random.PRNGKey(0),
    )

    u = jax.random.normal(jax.random.PRNGKey(1), (2, *spatial_shape))

    assert warp(u).shape == (6, *spatial_shape)


def test_output_grid_is_decoupled_from_input_grid():
    """
    The pullback samples onto whatever grid the coordinates describe, which is
    what would allow a warping layer to also change the resolution.
    """
    from pdequinox.conv._self_warp import _pullback

    u = jnp.arange(2 * 16, dtype=jnp.float32).reshape(2, 16)
    coordinates = jnp.linspace(0.0, 15.0, 8).reshape(1, 8)

    assert _pullback(u, coordinates, boundary_mode="periodic").shape == (2, 8)


def test_out_channels_must_be_divisible_by_num_heads():
    with pytest.raises(ValueError, match="divisible"):
        pdeqx.conv.SelfWarp(
            num_spatial_dims=2,
            in_channels=3,
            out_channels=7,
            num_heads=2,
            boundary_mode="periodic",
            key=jax.random.PRNGKey(0),
        )


@pytest.mark.parametrize(
    "num_spatial_dims,in_channels,out_channels,num_heads",
    [
        (1, 1, 4, 2),
        (2, 3, 8, 4),
        (3, 2, 12, 3),
    ],
)
def test_count_parameters(
    num_spatial_dims: int, in_channels: int, out_channels: int, num_heads: int
):
    warp = pdeqx.conv.SelfWarp(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        num_heads=num_heads,
        boundary_mode="periodic",
        key=jax.random.PRNGKey(0),
    )

    num_displacements = num_spatial_dims * num_heads
    correct_num_parameters = (
        # value map
        in_channels * out_channels
        + out_channels
        # displacement MLP, first layer
        + in_channels * out_channels
        + out_channels
        # displacement MLP, second layer
        + out_channels * num_displacements
        + num_displacements
    )

    assert pdeqx.count_parameters(warp) == correct_num_parameters
