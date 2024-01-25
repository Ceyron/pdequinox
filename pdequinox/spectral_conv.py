import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Complex, PRNGKeyArray
import jax.random as jr

class SpectralConv(eqx.Module):
    num_spatial_dims: int
    num_modes: int
    weights_real: Complex[Array, "C_i C_o *K"]
    weights_imag: Complex[Array, "C_i C_o *K"]

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_modes: int,
        *,
        key: PRNGKeyArray,
    ):
        self.num_spatial_dims = num_spatial_dims
        self.num_modes = num_modes

        weight_shape = (in_channels, out_channels,) + (num_modes,) * num_spatial_dims

        real_key, imag_key = jr.split(key)
        scale = 1 / (in_channels * out_channels)
        self.weights_real = scale * jr.normal(real_key, weight_shape)
        self.weights_imag = scale * jr.normal(imag_key, weight_shape)

    def __call__(self, x: Float[Array, "C *N"]) -> Float[Array, "C *N"]:
        """Spectral convolution.

        Args:
            x: Input array with shape (C, *N).

        Returns:
            Array with shape (C, *N).
        """
        spatial_shape = x.shape[1:]
        x_hat = jnp.fft.rfftn(x, axes=tuple(range(1, self.num_spatial_dims + 1)))
        
        right_most_wavenumbers = jnp.fft.rfftfreq(spatial_shape[-1], d=1 / spatial_shape[-1])
        other_wave_numbers = [jnp.fft.fftfreq(spatial_shape[i], d=1 / spatial_shape[i]) for i in range(self.num_spatial_dims - 1)]
        wavenumbers = other_wave_numbers + [right_most_wavenumbers]
        wavenumbers = jnp.stack(jnp.meshgrid(*wavenumbers, indexing="ij", sparse=True))

        mask = True
        for one_wavenumber in wavenumbers:
            mask = mask & (jnp.abs(one_wavenumber) <= self.num_modes)

        # ToDo!
        

