import jax
import jax.numpy as jnp
import pytest

import pdequinox as pdeqx


@pytest.mark.parametrize(
    "component,correct_receptive_field",
    [
        # 1d
        (
            pdeqx.conv.PhysicsConv(
                1, 1, 1, 3, key=jax.random.PRNGKey(0), boundary_mode="periodic"
            ),
            ((1, 1),),
        ),
        (
            pdeqx.conv.SpectralConv(1, 1, 1, 12, key=jax.random.PRNGKey(0)),
            ((jnp.inf, jnp.inf),),
        ),
        (
            pdeqx.conv.PointwiseLinearConv(1, 1, 1, key=jax.random.PRNGKey(0)),
            ((0, 0),),
        ),
        (
            # Displacements are unconstrained, hence unbounded (and, unlike the
            # SpectralConv, data-dependent) reach
            pdeqx.conv.SelfWarp(
                1, 1, 2, 2, key=jax.random.PRNGKey(0), boundary_mode="periodic"
            ),
            ((jnp.inf, jnp.inf),),
        ),
    ],
)
def test_query_receptive_field(component, correct_receptive_field):
    assert component.receptive_field == correct_receptive_field
