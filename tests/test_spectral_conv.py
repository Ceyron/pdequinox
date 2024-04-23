import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import pdequinox as pdeqx


def test_spectral_conv_derivative():
    NUM_POINTS = 48
    NUM_MODES = 10

    grid = jnp.linspace(0, 2 * jnp.pi, NUM_POINTS, endpoint=False)

    second_sine_mode = jnp.sin(2 * grid).reshape((1, -1))
    second_sine_mode_derivative = 2 * jnp.cos(2 * grid).reshape((1, -1))

    wavenumbers = jnp.fft.rfftfreq(NUM_POINTS, 1 / NUM_POINTS)
    derivative_operator = 1j * wavenumbers

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

    pred = spectral_conv(second_sine_mode)

    assert pred == pytest.approx(second_sine_mode_derivative, abs=1e-5)
