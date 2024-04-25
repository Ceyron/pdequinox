import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import pdequinox as pdeqx

# Uncomment the following line to run the tests on the CPU
# jax.config.update("jax_platform_name", "cpu")

# Uncomment the following line to run the tests in double precision
# jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "considered_mode,derivative_order",
    [
        (2, 1),
        (2, 2),
        (5, 1),
        (5, 2),
    ],
)
def test_spectral_conv_derivative(considered_mode, derivative_order):
    NUM_POINTS = 48
    NUM_MODES = 10

    grid = jnp.linspace(0, 2 * jnp.pi, NUM_POINTS, endpoint=False)

    considered_sine_mode = jnp.sin(considered_mode * grid).reshape((1, -1))
    if derivative_order == 1:
        considered_sine_mode_derivative = considered_mode * jnp.cos(
            considered_mode * grid
        ).reshape((1, -1))
    elif derivative_order == 2:
        considered_sine_mode_derivative = -(considered_mode**2) * jnp.sin(
            considered_mode * grid
        ).reshape((1, -1))
    else:
        raise ValueError("derivative_order must be 1 or 2")

    wavenumbers = jnp.fft.rfftfreq(NUM_POINTS, 1 / NUM_POINTS)
    derivative_operator = (1j * wavenumbers) ** derivative_order

    derivative_operator_stripped = derivative_operator[:NUM_MODES]
    derivative_operator_stripped_reshaped = derivative_operator_stripped.reshape(
        (1, 1, 1, -1)
    )

    spectral_conv = pdeqx.conv.SpectralConv(
        1, 1, 1, NUM_MODES, key=jax.random.PRNGKey(0)  # dummy key
    )

    spectral_conv = eqx.tree_at(
        lambda leaf: leaf.weights_real,
        spectral_conv,
        derivative_operator_stripped_reshaped.real,
    )
    spectral_conv = eqx.tree_at(
        lambda leaf: leaf.weights_imag,
        spectral_conv,
        derivative_operator_stripped_reshaped.imag,
    )

    pred = spectral_conv(considered_sine_mode)

    assert pred == pytest.approx(considered_sine_mode_derivative, abs=3e-5)


@pytest.mark.parametrize(
    "considered_mode_x,considered_mode_y,derivative_order",
    [
        (2, 2, 1),
        (2, 2, 2),
        (5, 5, 1),
        (5, 5, 2),
    ],
)
def test_spectral_conv_derivative_2d(
    considered_mode_x, considered_mode_y, derivative_order
):
    NUM_POINTS_X = 48
    NUM_POINTS_Y = 36
    NUM_MODES = 10

    grid_x = jnp.linspace(0, 2 * jnp.pi, NUM_POINTS_X, endpoint=False)
    grid_y = jnp.linspace(0, 2 * jnp.pi, NUM_POINTS_Y, endpoint=False)
    grid = jnp.stack(jnp.meshgrid(grid_x, grid_y, indexing="ij"))

    considered_sine_mode = (
        jnp.sin(considered_mode_x * grid[0]) * jnp.sin(considered_mode_y * grid[1])
    )[None, ...]
    if derivative_order == 1:
        considered_sine_mode_derivative = jnp.stack(
            [
                considered_mode_x
                * jnp.cos(considered_mode_x * grid[0])
                * jnp.sin(considered_mode_y * grid[1]),
                considered_mode_y
                * jnp.sin(considered_mode_x * grid[0])
                * jnp.cos(considered_mode_y * grid[1]),
            ]
        )
    elif derivative_order == 2:
        considered_sine_mode_derivative = jnp.stack(
            [
                -(considered_mode_x**2)
                * jnp.sin(considered_mode_x * grid[0])
                * jnp.sin(considered_mode_y * grid[1]),
                -(considered_mode_y**2)
                * jnp.sin(considered_mode_x * grid[0])
                * jnp.sin(considered_mode_y * grid[1]),
            ]
        )
    else:
        raise ValueError("derivative_order must be 1 or 2")

    wavenumbers_x = jnp.fft.fftfreq(NUM_POINTS_X, 1 / NUM_POINTS_X)
    wavenumbers_y = jnp.fft.fftfreq(NUM_POINTS_Y, 1 / NUM_POINTS_Y)
    wavenumbers = jnp.stack(jnp.meshgrid(wavenumbers_x, wavenumbers_y, indexing="ij"))

    derivative_operator = (1j * wavenumbers) ** derivative_order

    derivative_operator_lower = derivative_operator[..., :NUM_MODES, :NUM_MODES]
    derivative_operator_upper = derivative_operator[..., -NUM_MODES:, :NUM_MODES]

    # Should now be of shape (2, 1, 2, 48, 36//2 + 1)
    derivative_operator_stripped_reshaped = jnp.stack(
        [derivative_operator_lower, derivative_operator_upper]
    )[:, :, None]

    spectral_conv = pdeqx.conv.SpectralConv(
        2, 1, 2, NUM_MODES, key=jax.random.PRNGKey(0)  # dummy key
    )

    spectral_conv = eqx.tree_at(
        lambda leaf: leaf.weights_real,
        spectral_conv,
        derivative_operator_stripped_reshaped.real,
    )
    spectral_conv = eqx.tree_at(
        lambda leaf: leaf.weights_imag,
        spectral_conv,
        derivative_operator_stripped_reshaped.imag,
    )

    pred = spectral_conv(considered_sine_mode)

    assert pred == pytest.approx(considered_sine_mode_derivative, abs=3e-5, rel=1e-5)
