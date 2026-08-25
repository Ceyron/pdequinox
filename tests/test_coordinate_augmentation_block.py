"""
Tests for coordinate augmentation.

The important one is `test_fourier_grid_has_period_exactly_num_points`: the
whole reason the fourier encoding exists is that it is seamless across a
periodic boundary, and building it on `linspace` (which includes both endpoints)
would silently break that while passing every shape and parameter-count check.
"""

import jax
import jax.numpy as jnp
import pytest

import pdequinox as pdeqx
from pdequinox.blocks._coordinate_augmentation_block import _coordinate_grid


@pytest.mark.parametrize("num_points", [4, 8, 16, 33])
def test_fourier_grid_has_period_exactly_num_points(num_points: int):
    """
    One full period must span exactly `num_points` samples, so that the encoding
    joins up seamlessly across a periodic boundary.
    """
    grid = _coordinate_grid((num_points,), "fourier", jnp.float32)

    assert grid.shape == (2, num_points)

    sin_part, cos_part = grid[0], grid[1]
    angle = jnp.arctan2(sin_part, cos_part)

    # Angular steps, including the step that wraps from the last sample back
    # around to the first one
    wrapped = jnp.concatenate([angle, angle[:1]])
    step = jnp.diff(wrapped)
    # Bring each step into (-pi, pi] so the branch cut of arctan2 cancels
    step = (step + jnp.pi) % (2 * jnp.pi) - jnp.pi

    assert jnp.allclose(step, 2 * jnp.pi / num_points, atol=1e-5)

    # The failure mode this guards against: with a linspace-based grid the first
    # and the last sample land on the same angle and are encoded identically.
    assert not jnp.allclose(grid[:, 0], grid[:, -1], atol=1e-5)


def test_linear_grid_matches_the_reference():
    """The linear ramp reproduces the reference's `create_grid`."""
    grid = _coordinate_grid((5,), "linear", jnp.float32)

    assert grid.shape == (1, 5)
    assert jnp.allclose(grid[0], jnp.linspace(-1.0, 1.0, 5))
    assert grid[0, 0] == -1.0
    assert grid[0, -1] == 1.0


def test_linear_grid_axis_ordering():
    """Channel i varies along spatial axis i (array-index order)."""
    grid = _coordinate_grid((3, 5), "linear", jnp.float32)

    assert grid.shape == (2, 3, 5)
    assert jnp.allclose(grid[0][:, 0], jnp.linspace(-1.0, 1.0, 3))
    assert jnp.allclose(grid[1][0, :], jnp.linspace(-1.0, 1.0, 5))


@pytest.mark.parametrize(
    "coordinate_encoding,channels_per_dim",
    [("linear", 1), ("fourier", 2), ("none", 0)],
)
@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_number_of_added_channels(
    coordinate_encoding: str, channels_per_dim: int, num_spatial_dims: int
):
    factory = pdeqx.blocks.CoordinateAugmentationBlockFactory(
        coordinate_encoding=coordinate_encoding
    )
    block = factory(
        num_spatial_dims=num_spatial_dims,
        in_channels=3,
        out_channels=7,
        activation=jax.nn.relu,
        boundary_mode="periodic",
        key=jax.random.PRNGKey(0),
    )

    num_added = channels_per_dim * num_spatial_dims
    # The inner block is a 1x1 convolution over the widened input
    assert pdeqx.count_parameters(block) == (3 + num_added) * 7 + 7


@pytest.mark.parametrize(
    "boundary_mode,correct_encoding",
    [
        ("periodic", "fourier"),
        ("dirichlet", "linear"),
        ("neumann", "linear"),
    ],
)
def test_auto_resolves_from_the_boundary_mode(
    boundary_mode: str, correct_encoding: str
):
    factory = pdeqx.blocks.CoordinateAugmentationBlockFactory(
        coordinate_encoding="auto"
    )
    block = factory(
        num_spatial_dims=2,
        in_channels=3,
        out_channels=7,
        activation=jax.nn.relu,
        boundary_mode=boundary_mode,
        key=jax.random.PRNGKey(0),
    )

    assert block.coordinate_encoding == correct_encoding


@pytest.mark.parametrize("coordinate_encoding", ["linear", "fourier", "auto", "none"])
@pytest.mark.parametrize(
    "num_spatial_dims,spatial_shape",
    [
        (1, (32,)),
        (2, (16, 24)),  # deliberately not square
        (3, (8, 8, 12)),
    ],
)
def test_shapes(
    coordinate_encoding: str,
    num_spatial_dims: int,
    spatial_shape: tuple[int, ...],
):
    factory = pdeqx.blocks.CoordinateAugmentationBlockFactory(
        coordinate_encoding=coordinate_encoding
    )
    block = factory(
        num_spatial_dims=num_spatial_dims,
        in_channels=3,
        out_channels=7,
        activation=jax.nn.relu,
        boundary_mode="periodic",
        key=jax.random.PRNGKey(0),
    )

    dummy_input = jax.random.normal(jax.random.PRNGKey(1), (3, *spatial_shape))

    assert block(dummy_input).shape == (7, *spatial_shape)


def test_receptive_field_delegates_to_the_inner_block():
    factory = pdeqx.blocks.CoordinateAugmentationBlockFactory(
        pdeqx.blocks.ClassicResBlockFactory()
    )
    block = factory(
        num_spatial_dims=2,
        in_channels=3,
        out_channels=7,
        activation=jax.nn.relu,
        boundary_mode="periodic",
        key=jax.random.PRNGKey(0),
    )

    assert block.receptive_field == block.inner.receptive_field
    # and it is actually non-trivial, so the delegation is meaningful
    assert block.receptive_field != ((0, 0),) * 2


def test_wrapper_adds_no_parameters_of_its_own():
    kwargs = dict(
        num_spatial_dims=2,
        out_channels=7,
        activation=jax.nn.relu,
        boundary_mode="periodic",
        key=jax.random.PRNGKey(0),
    )

    wrapped = pdeqx.blocks.CoordinateAugmentationBlockFactory(
        coordinate_encoding="linear"
    )(in_channels=3, **kwargs)
    # The same inner block, built directly over the widened input
    inner_directly = pdeqx.blocks.LinearChannelAdjustBlockFactory()(
        in_channels=3 + 2, **kwargs
    )

    assert pdeqx.count_parameters(wrapped) == pdeqx.count_parameters(inner_directly)


@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_composes_as_a_lifting_factory(num_spatial_dims: int):
    """Stage 4 uses this in exactly these two slots."""
    lifting_factory = pdeqx.blocks.CoordinateAugmentationBlockFactory()

    sequential = pdeqx.Sequential(
        num_spatial_dims=num_spatial_dims,
        in_channels=2,
        out_channels=3,
        hidden_channels=16,
        num_blocks=2,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
        lifting_factory=lifting_factory,
    )

    hierarchical = pdeqx.Hierarchical(
        num_spatial_dims=num_spatial_dims,
        in_channels=2,
        out_channels=3,
        hidden_channels=16,
        num_levels=2,
        num_blocks=1,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
        lifting_factory=lifting_factory,
    )

    shape = (2,) + (16,) * num_spatial_dims
    dummy_input = jax.random.normal(jax.random.PRNGKey(1), shape)

    assert sequential(dummy_input).shape == (3,) + (16,) * num_spatial_dims
    assert hierarchical(dummy_input).shape == (3,) + (16,) * num_spatial_dims


def test_block_rejects_the_unresolved_auto_encoding():
    inner = pdeqx.blocks.LinearChannelAdjustBlockFactory()(
        num_spatial_dims=2,
        in_channels=5,
        out_channels=7,
        activation=jax.nn.relu,
        boundary_mode="periodic",
        key=jax.random.PRNGKey(0),
    )

    with pytest.raises(ValueError, match="concrete"):
        pdeqx.blocks.CoordinateAugmentationBlock(
            inner,
            num_spatial_dims=2,
            coordinate_encoding="auto",
        )


def test_factory_rejects_an_unknown_encoding():
    factory = pdeqx.blocks.CoordinateAugmentationBlockFactory(
        coordinate_encoding="polynomial"
    )

    with pytest.raises(ValueError, match="coordinate_encoding"):
        factory(
            num_spatial_dims=2,
            in_channels=3,
            out_channels=7,
            activation=jax.nn.relu,
            boundary_mode="periodic",
            key=jax.random.PRNGKey(0),
        )
