import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray


def poisson_1d_dirichlet(
    num_points: int = 64,
    num_samples: int = 1000,
    *,
    domain_extent: float = 5.0,
    key: PRNGKeyArray,
):
    """
    Produces pairs of force fields and displacement fields for a 1D Poisson
    equation with Dirichlet boundary conditions.

    The force fields are created by sampling random discontinuities in the
    domain. The displacement fields are the solutions to the Poisson equation
    using a three-point finite difference stencil (second order accurate).

    **Arguments:**

    - `num_points`: int. The number of interior degrees of freedom.
    - `num_samples`: int. The number of samples to generate.
    - `domain_extent`: float. The extent of the domain. (keyword-only argument)
    - `key`: PRNGKeyArray. The random key. (keyword-only argument)


    **Returns:**

    - `force_fields`: Array[num_samples, 1, num_points]. The force fields.
    - `displacement_fields`: Array[num_samples, 1, num_points]. The displacement
        fields.
    """
    grid = jnp.linspace(0, domain_extent, num_points + 2)[1:-1]
    dx = grid[1] - grid[0]

    A = (
        jnp.diag(jnp.ones(num_points - 1), -1)
        - 2 * jnp.diag(jnp.ones(num_points), 0)
        + jnp.diag(jnp.ones(num_points - 1), 1)
    )
    A = A / dx**2

    def solve_poisson(f):
        return jnp.linalg.solve(A, -f)

    def create_discontinuity(key):
        limit_1_key, limit_2_key = jax.random.split(key)
        lower_limit = jax.random.uniform(
            limit_1_key, (), minval=0.2 * domain_extent, maxval=0.4 * domain_extent
        )
        upper_limit = jax.random.uniform(
            limit_2_key, (), minval=0.6 * domain_extent, maxval=0.8 * domain_extent
        )

        discontinuity = jnp.where(
            (grid >= lower_limit) & (grid <= upper_limit), 1.0, 0.0
        )

        return discontinuity

    keys = jax.random.split(key, num_samples)
    force_fields = jax.vmap(create_discontinuity)(keys)
    displacement_fields = jax.vmap(solve_poisson)(force_fields)

    # Add a singleton channel axis
    force_fields = force_fields[:, None, :]
    displacement_fields = displacement_fields[:, None, :]

    return force_fields, displacement_fields
