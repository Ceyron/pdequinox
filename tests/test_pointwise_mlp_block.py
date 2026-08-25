import jax
import jax.numpy as jnp
import pytest

import pdequinox as pdeqx


@pytest.mark.parametrize(
    "num_spatial_dims,spatial_shape",
    [
        (1, (16,)),
        (2, (8, 12)),
        (3, (4, 6, 8)),
    ],
)
def test_shapes(num_spatial_dims: int, spatial_shape: tuple[int, ...]):
    block = pdeqx.blocks.PointwiseMLPBlock(
        num_spatial_dims=num_spatial_dims,
        in_channels=3,
        out_channels=5,
        key=jax.random.PRNGKey(0),
    )

    dummy_input = jax.random.normal(jax.random.PRNGKey(1), (3, *spatial_shape))

    assert block(dummy_input).shape == (5, *spatial_shape)


@pytest.mark.parametrize("hidden_channels", [None, 1, 11])
def test_count_parameters(hidden_channels):
    in_channels, out_channels = 3, 5
    block = pdeqx.blocks.PointwiseMLPBlock(
        num_spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        key=jax.random.PRNGKey(0),
    )

    # `None` means "as many as there are input channels"
    hidden = in_channels if hidden_channels is None else hidden_channels

    correct_num_parameters = (
        in_channels * hidden + hidden + hidden * out_channels + out_channels
    )

    assert pdeqx.count_parameters(block) == correct_num_parameters


@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_receptive_field_is_pointwise(num_spatial_dims: int):
    block = pdeqx.blocks.PointwiseMLPBlock(
        num_spatial_dims=num_spatial_dims,
        in_channels=3,
        out_channels=5,
        key=jax.random.PRNGKey(0),
    )

    assert block.receptive_field == ((0, 0),) * num_spatial_dims


def test_no_activation_on_the_output():
    """
    The block is meant to be usable as a projection head, so the last operation
    must be linear. With a ReLU that would otherwise clamp the output to be
    non-negative.
    """
    block = pdeqx.blocks.PointwiseMLPBlock(
        num_spatial_dims=1,
        in_channels=4,
        out_channels=4,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
    )

    dummy_input = jax.random.normal(jax.random.PRNGKey(1), (4, 32))

    assert jnp.any(block(dummy_input) < 0.0)
