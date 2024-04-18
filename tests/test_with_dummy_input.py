import jax
import pytest

import pdequinox as pdeqx


@pytest.mark.parametrize(
    "block,num_spatial_dims",
    [
        (block, D)
        for block in [
            pdeqx.blocks.ClassicDoubleConvBlock,
            pdeqx.blocks.ClassicResBlock,
            pdeqx.blocks.DilatedResBlock,
            pdeqx.blocks.ModernResBlock,
        ]
        for D in [1, 2, 3]
    ],
)
def test_block_with_dummy_input(block, num_spatial_dims):
    instantiated_block = block(
        num_spatial_dims=num_spatial_dims,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
    )

    shape = (1,) + (32,) * num_spatial_dims
    dummy_input = jax.random.normal(jax.random.PRNGKey(0), shape)

    instantiated_block(dummy_input)


@pytest.mark.parametrize(
    "num_spatial_dims",
    [1, 2, 3],
)
def test_classic_spectral_block_with_dummy_input(num_spatial_dims):
    instantiated_block = pdeqx.blocks.ClassicSpectralBlock(
        num_spatial_dims=num_spatial_dims,
        in_channels=1,
        out_channels=1,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
    )

    shape = (1,) + (32,) * num_spatial_dims

    dummy_input = jax.random.normal(jax.random.PRNGKey(0), shape)

    instantiated_block(dummy_input)


@pytest.mark.parametrize(
    "block_factory,num_spatial_dims",
    [
        (block_factory, D)
        for block_factory in [
            pdeqx.blocks.ClassicDoubleConvBlockFactory,
            pdeqx.blocks.ClassicResBlockFactory,
            pdeqx.blocks.ClassicSpectralBlockFactory,
            pdeqx.blocks.LinearChannelAdjustBlockFactory,
            pdeqx.blocks.LinearConvBlockFactory,
            pdeqx.blocks.LinearConvDownBlockFactory,
            pdeqx.blocks.LinearConvUpBlockFactory,
            pdeqx.blocks.ModernResBlockFactory,
        ]
        for D in [1, 2, 3]
    ],
)
def test_block_factory_with_dummy_input(block_factory, num_spatial_dims):
    factory = block_factory()

    instantiated_block = factory(
        num_spatial_dims=num_spatial_dims,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
    )

    shape = (1,) + (32,) * num_spatial_dims

    dummy_input = jax.random.normal(jax.random.PRNGKey(0), shape)

    instantiated_block(dummy_input)


@pytest.mark.parametrize(
    "arch,num_spatial_dims",
    [
        (arch, D)
        for arch in [
            pdeqx.arch.ClassicResNet,
            pdeqx.arch.ClassicUNet,
            pdeqx.arch.ConvNet,
            pdeqx.arch.DilatedResNet,
            pdeqx.arch.ModernResNet,
            pdeqx.arch.ModernUNet,
        ]
        for D in [1, 2, 3]
    ],
)
def test_arch_with_dummy_input(arch, num_spatial_dims):
    instantiated_arch = arch(
        num_spatial_dims=num_spatial_dims,
        in_channels=1,
        out_channels=1,
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
    )

    shape = (1,) + (32,) * num_spatial_dims

    dummy_input = jax.random.normal(jax.random.PRNGKey(0), shape)

    instantiated_arch(dummy_input)


@pytest.mark.parametrize(
    "num_spatial_dims",
    [1, 2, 3],
)
def test_fno_with_dummy_input(num_spatial_dims):
    instantiated_arch = pdeqx.arch.ClassicFNO(
        num_spatial_dims=num_spatial_dims,
        in_channels=1,
        out_channels=1,
        key=jax.random.PRNGKey(0),
    )

    shape = (1,) + (32,) * num_spatial_dims

    dummy_input = jax.random.normal(jax.random.PRNGKey(0), shape)

    instantiated_arch(dummy_input)


@pytest.mark.parametrize(
    "num_spatial_dims",
    [1, 2, 3],
)
def test_mlp_with_dummy_input(num_spatial_dims):
    instantiated_arch = pdeqx.arch.MLP(
        num_spatial_dims=num_spatial_dims,
        in_channels=1,
        out_channels=1,
        num_points=32,
        key=jax.random.PRNGKey(0),
    )

    shape = (1,) + (32,) * num_spatial_dims

    dummy_input = jax.random.normal(jax.random.PRNGKey(0), shape)

    instantiated_arch(dummy_input)
