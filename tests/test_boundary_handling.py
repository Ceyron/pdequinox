from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import pdequinox as pdeqx


@pytest.mark.parametrize(
    "boundary_mode",
    ["periodic", "dirichlet", "neumann"],
)
def test_periodic_boundary_handling(
    boundary_mode: Literal["periodic", "dirichlet", "neumann"],
):
    physics_conv = pdeqx.conv.PhysicsConv(
        num_spatial_dims=1,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        use_bias=False,  # Important!
        key=jax.random.PRNGKey(0),
        boundary_mode=boundary_mode,
    )

    # Reset the weights array
    physics_conv = eqx.tree_at(
        lambda leaf: leaf.weight,
        physics_conv,
        jnp.ones((1, 1, 3)),
    )

    input_array = jnp.ones((1, 7))

    if boundary_mode == "periodic":
        correct_output = jnp.ones((1, 7)) * 3
    elif boundary_mode == "dirichlet":
        correct_output = jnp.ones((1, 7)) * 3
        correct_output = correct_output.at[0, 0].set(2)
        correct_output = correct_output.at[0, -1].set(2)
    elif boundary_mode == "neumann":
        correct_output = jnp.ones((1, 7)) * 3
        correct_output = correct_output.at[0, 0].set(3)
        correct_output = correct_output.at[0, -1].set(3)
    else:
        raise ValueError(f"Unknown boundary mode: {boundary_mode}")

    output_array = physics_conv(input_array)

    assert output_array == pytest.approx(correct_output)
