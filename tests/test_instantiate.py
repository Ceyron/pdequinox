import jax

import pdequinox as pdeqx


def test_instantiate():
    """
    Test whether the default architectures instantiate under their default
    configurations.
    """

    pdeqx.BlockNet(
        num_spatial_dims=1,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        num_blocks=3,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
    )

    pdeqx.ConvNet(
        num_spatial_dims=1,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        depth=3,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
    )

    pdeqx.MLP(
        num_spatial_dims=1,
        num_points=48,
        in_channels=1,
        out_channels=1,
        width_size=16,
        depth=3,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
    )

    pdeqx.UNet(
        num_spatial_dims=1,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        num_levels=3,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
    )
