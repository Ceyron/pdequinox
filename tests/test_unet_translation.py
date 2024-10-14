import jax
import jax.numpy as jnp
import pytest

import pdequinox as pdeqx


@pytest.mark.parametrize(
    "num_spatial_dims,num_points,num_levels",
    [
        (num_spatial_dims, num_points, num_levels)
        for num_spatial_dims in (1, 2, 3)
        for num_points in (36,)
        for num_levels in (0, 1, 2)
    ],
)
def test_unet_translation(num_spatial_dims: int, num_points: int, num_levels: int):
    grid_1d = jnp.linspace(0, 1.0, num_points, endpoint=False)

    grid = jnp.stack(
        jnp.meshgrid(*[grid_1d for _ in range(num_spatial_dims)], indexing="ij")
    )

    u_0 = jnp.sin(2 * jnp.pi * grid[0:1])

    if num_spatial_dims >= 2:
        u_0 *= jnp.sin(4 * jnp.pi * grid[1:2])

    if num_spatial_dims >= 3:
        u_0 *= jnp.sin(6 * jnp.pi * grid[2:3])

    unet = pdeqx.arch.ClassicUNet(
        num_spatial_dims,
        1,
        1,
        hidden_channels=1,
        num_levels=num_levels,
        use_norm=False,
        key=jax.random.PRNGKey(0),
    )

    # Change all kernels to identity and biases to zero
    arr, structure = jax.tree.flatten(unet)

    arr_new = []

    for a in arr:
        if isinstance(a, jnp.ndarray):
            if a.ndim == num_spatial_dims + 1:
                # Bias
                arr_new.append(jnp.zeros_like(a))
            elif a.ndim == num_spatial_dims + 2:
                # Convolutional kernel
                new_kernel = jnp.zeros_like(a)
                if a.shape[-1] == 3:
                    if num_spatial_dims == 1:
                        new_kernel = new_kernel.at[..., 1].set(1.0)
                    elif num_spatial_dims == 2:
                        new_kernel = new_kernel.at[..., 1, 1].set(1.0)
                    elif num_spatial_dims == 3:
                        new_kernel = new_kernel.at[..., 1, 1, 1].set(1.0)
                elif a.shape[-1] == 1:
                    new_kernel = new_kernel.at[..., 0].set(1.0)
                else:
                    raise ValueError(f"Invalid kernel shape {a.shape}")
                arr_new.append(new_kernel)
        else:
            arr_new.append(a)

    print(arr_new)

    unet_new = jax.tree_util.tree_unflatten(structure, arr_new)

    u_0_processed = unet_new(u_0)

    if num_spatial_dims == 1:
        assert u_0_processed[..., 1] == pytest.approx(u_0[..., 1])
    elif num_spatial_dims == 2:
        assert u_0_processed[..., 1, 1] == pytest.approx(u_0[..., 1, 1])
    elif num_spatial_dims == 3:
        assert u_0_processed[..., 1, 1, 1] == pytest.approx(u_0[..., 1, 1, 1])
