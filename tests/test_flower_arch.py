"""
Tests for the two Flower architectures.

The headline tests are the translation ones. Coordinate augmentation is
documented in several places as deliberately breaking translation equivariance,
which is a claim about behaviour that no shape or parameter-count test would
ever exercise.
"""

import jax
import jax.numpy as jnp
import pytest

import pdequinox as pdeqx


def _relative_translation_error(model, x, shift: int) -> float:
    """How far the model is from commuting with a periodic shift."""
    shifted_then_applied = model(jnp.roll(x, shift, axis=-1))
    applied_then_shifted = jnp.roll(model(x), shift, axis=-1)
    return float(
        jnp.max(jnp.abs(shifted_then_applied - applied_then_shifted))
        / jnp.max(jnp.abs(applied_then_shifted))
    )


@pytest.mark.parametrize("shift", [8, 16])
def test_flower_is_translation_equivariant_without_coordinates(shift: int):
    """
    With `coordinate_encoding="none"` everything in the architecture commutes
    with a periodic shift, *except* that the strided convolutions only commute
    with shifts that are a multiple of the total stride `2 ** num_levels`.
    """
    num_levels = 3
    flower = pdeqx.arch.Flower(
        1,
        2,
        2,
        hidden_channels=16,
        num_levels=num_levels,
        num_heads=8,
        coordinate_encoding="none",
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
    )

    x = jax.random.normal(jax.random.PRNGKey(1), (2, 64))

    assert shift % 2**num_levels == 0
    assert _relative_translation_error(flower, x, shift) < 1e-3


def test_flower_is_not_equivariant_to_a_shift_below_the_stride():
    """A shift that is not a multiple of the total stride is not respected."""
    flower = pdeqx.arch.Flower(
        1,
        2,
        2,
        hidden_channels=16,
        num_levels=3,
        num_heads=8,
        coordinate_encoding="none",
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
    )

    x = jax.random.normal(jax.random.PRNGKey(1), (2, 64))

    assert _relative_translation_error(flower, x, 3) > 0.1


def test_coordinate_augmentation_breaks_translation_equivariance():
    """
    The documented cost of handing the network absolute positions: even a shift
    that the scaffold would otherwise respect is no longer respected.
    """

    def build(coordinate_encoding):
        return pdeqx.arch.Flower(
            1,
            2,
            2,
            hidden_channels=16,
            num_levels=3,
            num_heads=8,
            coordinate_encoding=coordinate_encoding,
            key=jax.random.PRNGKey(0),
            boundary_mode="periodic",
        )

    x = jax.random.normal(jax.random.PRNGKey(1), (2, 64))

    assert _relative_translation_error(build("none"), x, 8) < 1e-3
    assert _relative_translation_error(build("linear"), x, 8) > 0.1


@pytest.mark.parametrize("shift", [1, 5, 7])
def test_sequential_flower_is_equivariant_to_any_shift(shift: int):
    """
    Without the strided convolutions there is no stride to respect, so the
    single-scale variant commutes with *every* periodic shift.
    """
    sequential_flower = pdeqx.arch.SequentialFlower(
        1,
        2,
        2,
        hidden_channels=16,
        num_blocks=3,
        num_heads=8,
        coordinate_encoding="none",
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
    )

    x = jax.random.normal(jax.random.PRNGKey(1), (2, 64))

    assert _relative_translation_error(sequential_flower, x, shift) < 1e-3


@pytest.mark.parametrize("coordinate_encoding", ["linear", "fourier", "auto", "none"])
@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_shapes(coordinate_encoding: str, num_spatial_dims: int):
    shape = (2,) + (16,) * num_spatial_dims
    dummy_input = jax.random.normal(jax.random.PRNGKey(1), shape)

    for arch in (pdeqx.arch.Flower, pdeqx.arch.SequentialFlower):
        kwargs = dict(
            hidden_channels=16,
            num_heads=8,
            coordinate_encoding=coordinate_encoding,
            key=jax.random.PRNGKey(0),
            boundary_mode="periodic",
        )
        if arch is pdeqx.arch.Flower:
            kwargs["num_levels"] = 2
        else:
            kwargs["num_blocks"] = 2

        model = arch(num_spatial_dims, 2, 3, **kwargs)

        assert model(dummy_input).shape == (3,) + (16,) * num_spatial_dims


@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_receptive_field_is_unbounded(num_spatial_dims: int):
    flower = pdeqx.arch.Flower(
        num_spatial_dims,
        1,
        1,
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
    )

    assert flower.receptive_field == ((jnp.inf, jnp.inf),) * num_spatial_dims


def test_parameter_count_at_the_configuration_from_the_paper():
    """
    Pins the figure quoted in the docstring, so that a change to the scaffold
    cannot silently drift away from it.
    """
    flower = pdeqx.arch.Flower(
        2,
        4,
        4,
        hidden_channels=160,
        num_heads=40,
        num_levels=3,
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
    )

    num_parameters = pdeqx.count_parameters(flower)

    # The reference implementation reports 17,329,362 for this configuration;
    # see the deviations listed in the `Flower` docstring for why ours is
    # smaller
    assert 13_000_000 < num_parameters < 13_500_000
    assert num_parameters / 17_329_362 == pytest.approx(0.76, abs=0.01)


def test_num_heads_must_divide_hidden_channels():
    with pytest.raises(ValueError, match="divisible"):
        pdeqx.arch.Flower(
            2,
            1,
            1,
            hidden_channels=16,
            num_heads=5,
            key=jax.random.PRNGKey(0),
            boundary_mode="periodic",
        )
