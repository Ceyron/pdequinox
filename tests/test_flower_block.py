"""
Tests for the residual pullback block of the Flower architecture.

The composition tests are the important ones here: a block that works in
isolation but does not slot into `Sequential`/`Hierarchical` would otherwise
only surface once the architectures are built on top of it.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import pdequinox as pdeqx


@pytest.mark.parametrize(
    "num_spatial_dims,spatial_shape",
    [
        (1, (32,)),
        (2, (16, 24)),  # deliberately not square
        (3, (8, 8, 12)),
    ],
)
@pytest.mark.parametrize("in_channels,out_channels", [(8, 8), (3, 8)])
def test_shapes(
    num_spatial_dims: int,
    spatial_shape: tuple[int, ...],
    in_channels: int,
    out_channels: int,
):
    block = pdeqx.blocks.FlowerBlock(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        num_heads=4,
        boundary_mode="periodic",
        key=jax.random.PRNGKey(0),
    )

    dummy_input = jax.random.normal(
        jax.random.PRNGKey(1), (in_channels, *spatial_shape)
    )

    assert block(dummy_input).shape == (out_channels, *spatial_shape)


@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_composes_with_sequential(num_spatial_dims: int):
    """The factory must be usable as a drop-in block factory."""
    net = pdeqx.Sequential(
        num_spatial_dims=num_spatial_dims,
        in_channels=2,
        out_channels=3,
        hidden_channels=16,
        num_blocks=3,
        activation=jax.nn.gelu,
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
        block_factory=pdeqx.blocks.FlowerBlockFactory(num_heads=8),
    )

    shape = (2,) + (16,) * num_spatial_dims
    dummy_input = jax.random.normal(jax.random.PRNGKey(1), shape)

    assert net(dummy_input).shape == (3,) + (16,) * num_spatial_dims


@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_composes_with_hierarchical(num_spatial_dims: int):
    """
    The channel counts a block is constructed with inside `Hierarchical` are
    integer multiples of `hidden_channels`, so a `num_heads` dividing
    `hidden_channels` has to work at every level.
    """
    net = pdeqx.Hierarchical(
        num_spatial_dims=num_spatial_dims,
        in_channels=2,
        out_channels=3,
        hidden_channels=16,
        num_levels=3,
        num_blocks=1,
        activation=jax.nn.gelu,
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
        left_arc_factory=pdeqx.blocks.FlowerBlockFactory(num_heads=8),
        right_arc_factory=pdeqx.blocks.FlowerBlockFactory(num_heads=8),
    )

    shape = (2,) + (32,) * num_spatial_dims
    dummy_input = jax.random.normal(jax.random.PRNGKey(1), shape)

    assert net(dummy_input).shape == (3,) + (32,) * num_spatial_dims


@pytest.mark.parametrize("use_norm", [True, False])
@pytest.mark.parametrize(
    "num_spatial_dims,in_channels,out_channels,num_heads",
    [
        (1, 3, 8, 4),
        (2, 4, 12, 3),
        (3, 2, 6, 2),
    ],
)
def test_count_parameters(
    use_norm: bool,
    num_spatial_dims: int,
    in_channels: int,
    out_channels: int,
    num_heads: int,
):
    block = pdeqx.blocks.FlowerBlock(
        num_spatial_dims=num_spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        num_heads=num_heads,
        use_norm=use_norm,
        boundary_mode="periodic",
        key=jax.random.PRNGKey(0),
    )

    num_displacements = num_spatial_dims * num_heads
    correct_num_parameters = (
        # SelfWarp: value map, first and second layer of the displacement MLP
        (in_channels * out_channels + out_channels)
        + (in_channels * out_channels + out_channels)
        + (out_channels * num_displacements + num_displacements)
        # IdProj
        + (in_channels * out_channels + out_channels)
    )
    if use_norm:
        # GroupNorm weight and bias
        correct_num_parameters += 2 * out_channels

    assert pdeqx.count_parameters(block) == correct_num_parameters


def test_use_norm_false_is_identity():
    block = pdeqx.blocks.FlowerBlock(
        num_spatial_dims=2,
        in_channels=3,
        out_channels=8,
        num_heads=4,
        use_norm=False,
        boundary_mode="periodic",
        key=jax.random.PRNGKey(0),
    )

    assert isinstance(block.norm, eqx.nn.Identity)


@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_receptive_field_is_unbounded(num_spatial_dims: int):
    block = pdeqx.blocks.FlowerBlock(
        num_spatial_dims=num_spatial_dims,
        in_channels=3,
        out_channels=8,
        num_heads=4,
        boundary_mode="periodic",
        key=jax.random.PRNGKey(0),
    )

    assert block.receptive_field == ((jnp.inf, jnp.inf),) * num_spatial_dims


def test_activation_is_not_routed_into_the_displacement_mlp():
    """
    The block uses two different activations: `activation` for the block
    nonlinearity and `mlp_activation` inside the displacement MLP. The block
    factory protocol only passes a single `activation`, which must reach the
    former and *not* the latter.
    """
    factory = pdeqx.blocks.FlowerBlockFactory(
        num_heads=4,
        mlp_activation=jax.nn.relu,
    )

    block = factory(
        num_spatial_dims=2,
        in_channels=3,
        out_channels=8,
        activation=jax.nn.tanh,
        boundary_mode="periodic",
        key=jax.random.PRNGKey(0),
    )

    assert block.activation is jax.nn.tanh
    assert block.self_warp.activation is jax.nn.relu


def test_out_channels_must_be_divisible_by_num_heads():
    with pytest.raises(ValueError, match="divisible"):
        pdeqx.blocks.FlowerBlock(
            num_spatial_dims=2,
            in_channels=3,
            out_channels=7,
            num_heads=2,
            boundary_mode="periodic",
            key=jax.random.PRNGKey(0),
        )
