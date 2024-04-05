import jax
import pytest

import pdequinox as pdeqx


@pytest.mark.parametrize(
    "block",
    [
        pdeqx.blocks.ClassicDoubleConvBlock,
        pdeqx.blocks.ClassicResBlock,
        pdeqx.blocks.DilatedResBlock,
        pdeqx.blocks.ModernResBlock,
    ],
)
def test_block_with_dummy_input(block):
    instantiated_block = block(
        num_spatial_dims=1,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
    )

    dummy_input = jax.random.normal(jax.random.PRNGKey(0), (1, 32))

    instantiated_block(dummy_input)


def test_classic_spectral_block_with_dummy_input():
    instantiated_block = pdeqx.blocks.ClassicSpectralBlock(
        num_spatial_dims=1,
        in_channels=1,
        out_channels=1,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
    )

    dummy_input = jax.random.normal(jax.random.PRNGKey(0), (1, 32))

    instantiated_block(dummy_input)


@pytest.mark.parametrize(
    "block_factory",
    [
        pdeqx.blocks.ClassicDoubleConvBlockFactory,
        pdeqx.blocks.ClassicResBlockFactory,
        pdeqx.blocks.ClassicSpectralBlockFactory,
        pdeqx.blocks.LinearChannelAdjustBlockFactory,
        pdeqx.blocks.LinearConvBlockFactory,
        pdeqx.blocks.LinearConvDownBlockFactory,
        pdeqx.blocks.LinearConvUpBlockFactory,
        pdeqx.blocks.ModernResBlockFactory,
    ],
)
def test_block_factory_with_dummy_input(block_factory):
    factory = block_factory()

    instantiated_block = factory(
        num_spatial_dims=1,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
    )

    dummy_input = jax.random.normal(jax.random.PRNGKey(0), (1, 32))

    instantiated_block(dummy_input)


@pytest.mark.parametrize(
    "arch",
    [
        # pdeqx.arch.ClassicFNO,
        pdeqx.arch.ClassicResNet,
        pdeqx.arch.ClassicUNet,
        pdeqx.arch.ConvNet,
        pdeqx.arch.DilatedResNet,
        # pdeqx.arch.MLP,
        pdeqx.arch.ModernResNet,
    ],
)
def test_arch_with_dummy_input(arch):
    instantiated_arch = arch(
        num_spatial_dims=1,
        in_channels=1,
        out_channels=1,
        key=jax.random.PRNGKey(0),
        boundary_mode="periodic",
    )

    dummy_input = jax.random.normal(jax.random.PRNGKey(0), (1, 32))

    instantiated_arch(dummy_input)


def test_fno_with_dummy_input():
    instantiated_arch = pdeqx.arch.ClassicFNO(
        num_spatial_dims=1,
        in_channels=1,
        out_channels=1,
        key=jax.random.PRNGKey(0),
    )

    dummy_input = jax.random.normal(jax.random.PRNGKey(0), (1, 32))

    instantiated_arch(dummy_input)


def test_mlp_with_dummy_input():
    instantiated_arch = pdeqx.arch.MLP(
        num_spatial_dims=1,
        in_channels=1,
        out_channels=1,
        num_points=32,
        key=jax.random.PRNGKey(0),
    )

    dummy_input = jax.random.normal(jax.random.PRNGKey(0), (1, 32))

    instantiated_arch(dummy_input)
