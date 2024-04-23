import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import pdequinox as pdeqx


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
