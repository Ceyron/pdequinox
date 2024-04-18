import jax
import pytest

import pdequinox as pdeqx


def test_instantiate():
    """
    Test whether the default architectures instantiate under their default
    configurations.
    """

    pdeqx.Sequential(
        num_spatial_dims=1,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        num_blocks=3,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
    )

    pdeqx.Hierarchical(
        num_spatial_dims=1,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        num_levels=3,
        num_blocks=1,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
    )

    pdeqx.arch.ConvNet(
        num_spatial_dims=1,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        depth=3,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
    )

    pdeqx.arch.MLP(
        num_spatial_dims=1,
        num_points=48,
        in_channels=1,
        out_channels=1,
        width_size=16,
        depth=3,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
    )

    pdeqx.arch.ClassicUNet(
        num_spatial_dims=1,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        num_levels=3,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
    )


@pytest.mark.parametrize(
    "num_spatial_dims,arch,boundary_mode",
    [
        (num_spatial_dims, arch, boundary_mode)
        for num_spatial_dims in [1, 2, 3]
        for arch in [
            pdeqx.arch.ClassicFNO,
            pdeqx.arch.ClassicUNet,
            pdeqx.arch.ClassicResNet,
            pdeqx.arch.ConvNet,
            pdeqx.arch.DilatedResNet,
            pdeqx.arch.ModernResNet,
            pdeqx.arch.ModernUNet,
        ]
        for boundary_mode in ["periodic", "dirichlet", "neumann"]
    ],
)
def test_default_config(num_spatial_dims, arch, boundary_mode):
    arch(1, 2, 5, key=jax.random.PRNGKey(0))
