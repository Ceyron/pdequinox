import jax
import jax.numpy as jnp
import pytest

import pdequinox as pdeqx

# Use hidden_channels=16 and num_heads=2 so the RoPE divisibility constraint
# (hidden_channels // num_heads) % (2 * num_spatial_dims) == 0 is satisfied
# for all dims: 16//2=8, 8 % 2=0, 8 % 4=0, 8 % 6=? -> need 6 | 8 which fails for 3D.
# Use hidden_channels=24, num_heads=2: 24//2=12, 12%2=0, 12%4=0, 12%6=0. Works!
HIDDEN = 24
HEADS = 2
BLOCKS = 2
SPATIAL_SIZE = 16  # divisible by patch_size=4 and window_size=4


@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_dense_transformer(num_spatial_dims):
    net = pdeqx.arch.DenseTransformer(
        num_spatial_dims=num_spatial_dims,
        in_channels=3,
        out_channels=3,
        hidden_channels=HIDDEN,
        num_heads=HEADS,
        num_blocks=BLOCKS,
        key=jax.random.PRNGKey(0),
    )
    shape = (3,) + (SPATIAL_SIZE,) * num_spatial_dims
    x = jax.random.normal(jax.random.PRNGKey(1), shape)
    y = net(x)
    assert y.shape == shape


@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_swin_transformer(num_spatial_dims):
    net = pdeqx.arch.SwinTransformer(
        num_spatial_dims=num_spatial_dims,
        in_channels=3,
        out_channels=3,
        hidden_channels=HIDDEN,
        num_heads=HEADS,
        num_blocks=BLOCKS,
        window_size=4,
        boundary_mode="periodic",
        key=jax.random.PRNGKey(0),
    )
    shape = (3,) + (SPATIAL_SIZE,) * num_spatial_dims
    x = jax.random.normal(jax.random.PRNGKey(1), shape)
    y = net(x)
    assert y.shape == shape


@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_patch_transformer(num_spatial_dims):
    net = pdeqx.arch.PatchTransformer(
        num_spatial_dims=num_spatial_dims,
        in_channels=3,
        out_channels=3,
        hidden_channels=HIDDEN,
        num_heads=HEADS,
        num_blocks=BLOCKS,
        patch_size=4,
        key=jax.random.PRNGKey(0),
    )
    shape = (3,) + (SPATIAL_SIZE,) * num_spatial_dims
    x = jax.random.normal(jax.random.PRNGKey(1), shape)
    y = net(x)
    assert y.shape == shape


@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_axial_transformer(num_spatial_dims):
    net = pdeqx.arch.AxialTransformer(
        num_spatial_dims=num_spatial_dims,
        in_channels=3,
        out_channels=3,
        hidden_channels=HIDDEN,
        num_heads=HEADS,
        num_blocks=BLOCKS,
        key=jax.random.PRNGKey(0),
    )
    shape = (3,) + (SPATIAL_SIZE,) * num_spatial_dims
    x = jax.random.normal(jax.random.PRNGKey(1), shape)
    y = net(x)
    assert y.shape == shape


# Block-level tests
@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_dense_attention_block(num_spatial_dims):
    block = pdeqx.blocks.DenseAttentionBlock(
        num_spatial_dims=num_spatial_dims,
        in_channels=HIDDEN,
        num_heads=HEADS,
        key=jax.random.PRNGKey(0),
    )
    shape = (HIDDEN,) + (SPATIAL_SIZE,) * num_spatial_dims
    x = jax.random.normal(jax.random.PRNGKey(1), shape)
    y = block(x)
    assert y.shape == shape
    assert all(
        r == (float("inf"), float("inf")) for r in block.receptive_field
    )


@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_swin_attention_block(num_spatial_dims):
    block = pdeqx.blocks.SwinAttentionBlock(
        num_spatial_dims=num_spatial_dims,
        in_channels=HIDDEN,
        num_heads=HEADS,
        window_size=4,
        shift=0,
        boundary_mode="periodic",
        key=jax.random.PRNGKey(0),
    )
    shape = (HIDDEN,) + (SPATIAL_SIZE,) * num_spatial_dims
    x = jax.random.normal(jax.random.PRNGKey(1), shape)
    y = block(x)
    assert y.shape == shape
    assert block.receptive_field == ((3.0, 3.0),) * num_spatial_dims


@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_axial_attention_block(num_spatial_dims):
    block = pdeqx.blocks.AxialAttentionBlock(
        num_spatial_dims=num_spatial_dims,
        in_channels=HIDDEN,
        num_heads=HEADS,
        key=jax.random.PRNGKey(0),
    )
    shape = (HIDDEN,) + (SPATIAL_SIZE,) * num_spatial_dims
    x = jax.random.normal(jax.random.PRNGKey(1), shape)
    y = block(x)
    assert y.shape == shape


@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_patchify_unpatchify_blocks(num_spatial_dims):
    patch_size = 4
    lift = pdeqx.blocks.PatchifyLiftingBlock(
        num_spatial_dims=num_spatial_dims,
        in_channels=3,
        out_channels=HIDDEN,
        patch_size=patch_size,
        key=jax.random.PRNGKey(0),
    )
    proj = pdeqx.blocks.UnpatchifyProjectionBlock(
        num_spatial_dims=num_spatial_dims,
        in_channels=HIDDEN,
        out_channels=3,
        patch_size=patch_size,
        key=jax.random.PRNGKey(1),
    )
    shape_in = (3,) + (SPATIAL_SIZE,) * num_spatial_dims
    x = jax.random.normal(jax.random.PRNGKey(2), shape_in)
    h = lift(x)
    expected_hidden_shape = (HIDDEN,) + (SPATIAL_SIZE // patch_size,) * num_spatial_dims
    assert h.shape == expected_hidden_shape
    y = proj(h)
    assert y.shape == shape_in


# Factory tests
@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_dense_attention_block_factory(num_spatial_dims):
    factory = pdeqx.blocks.DenseAttentionBlockFactory(
        num_heads=HEADS,
    )
    block = factory(
        num_spatial_dims=num_spatial_dims,
        in_channels=HIDDEN,
        out_channels=HIDDEN,
        activation=jax.nn.gelu,
        boundary_mode="periodic",
        key=jax.random.PRNGKey(0),
    )
    shape = (HIDDEN,) + (SPATIAL_SIZE,) * num_spatial_dims
    x = jax.random.normal(jax.random.PRNGKey(1), shape)
    block(x)


@pytest.mark.parametrize("num_spatial_dims", [1, 2, 3])
def test_swin_attention_block_factory_alternates_shift(num_spatial_dims):
    factory = pdeqx.blocks.SwinAttentionBlockFactory(
        num_heads=HEADS,
        window_size=4,
    )
    b0 = factory(
        num_spatial_dims=num_spatial_dims,
        in_channels=HIDDEN,
        out_channels=HIDDEN,
        activation=jax.nn.gelu,
        boundary_mode="periodic",
        key=jax.random.PRNGKey(0),
    )
    b1 = factory(
        num_spatial_dims=num_spatial_dims,
        in_channels=HIDDEN,
        out_channels=HIDDEN,
        activation=jax.nn.gelu,
        boundary_mode="periodic",
        key=jax.random.PRNGKey(1),
    )
    assert b0.shift == 0
    assert b1.shift == 2  # window_size // 2


# Swin boundary modes
@pytest.mark.parametrize("boundary_mode", ["periodic", "dirichlet", "neumann"])
def test_swin_boundary_modes(boundary_mode):
    block = pdeqx.blocks.SwinAttentionBlock(
        num_spatial_dims=1,
        in_channels=HIDDEN,
        num_heads=HEADS,
        window_size=4,
        shift=2,
        boundary_mode=boundary_mode,
        key=jax.random.PRNGKey(0),
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (HIDDEN, SPATIAL_SIZE))
    y = block(x)
    assert y.shape == x.shape
