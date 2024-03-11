from itertools import product

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray


class SpectralConv(eqx.Module):
    """
    Huge credit to the Serket library for this implementation:
    https://github.com/ASEM000/serket
    """

    num_spatial_dims: int
    num_modes: tuple[int]
    weights_real: Float[Array, "G C_o C_i ..."]
    weights_imag: Float[Array, "G C_o C_i ..."]

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_modes: tuple[int, ...] or int,
        *,
        key: PRNGKeyArray,
    ):
        if isinstance(num_modes, int):
            num_modes = (num_modes,) * num_spatial_dims

        if len(num_modes) != num_spatial_dims:
            raise ValueError("num_modes must have the same length as num_spatial_dims")

        self.num_spatial_dims = num_spatial_dims
        self.num_modes = num_modes

        weight_shape = (
            2 ** (num_spatial_dims - 1),
            in_channels,
            out_channels,
        ) + num_modes

        real_key, imag_key = jr.split(key)
        scale = 1 / (in_channels * out_channels)
        self.weights_real = scale * jr.normal(real_key, weight_shape)
        self.weights_imag = scale * jr.normal(imag_key, weight_shape)

    def __call__(self, x: Float[Array, "C_i ..."]) -> Float[Array, "C_o ..."]:
        return spectral_conv_nd(x, self.weights_real, self.weights_imag, self.num_modes)


def spectral_conv_nd(
    input: Float[Array, "C_i ..."],
    weight_r: Float[Array, "G C_o C_i ..."],
    weight_i: Float[Array, "G C_o C_i ..."],
    modes: tuple[int, ...],
) -> Float[Array, "C_o ..."]:
    """fourier neural operator convolution function.

    Full credit to the Serket library for this function:
    https://github.com/ASEM000/serket/blob/fc6e754b8d5b22075e09b2bceff497a4e8c57fad/serket/_src/nn/convolution.py#L464

    Args:
        input: input array. shape is ``(in_features, spatial size)``. weight_r:
        real convolutional kernel. shape is ``(2 ** (dim-1), out_features,
        in_features, modes)``.
            where dim is the number of spatial dimensions.
        weight_i: convolutional kernel. shape is ``(2 ** (dim-1), out_features,
        in_features, modes)``.
            where dim is the number of spatial dimensions.
        modes: number of modes included in the fft representation of the input.
    """

    def generate_modes_slices(modes: tuple[int, ...]):
        *ms, ml = modes
        slices_ = [[slice(None, ml)]]
        slices_ += [[slice(None, mode), slice(-mode, None)] for mode in reversed(ms)]
        return [[slice(None)] + list(reversed(i)) for i in product(*slices_)]

    _, *si, sl = input.shape
    weight = weight_r + 1j * weight_i
    _, o, *_ = weight.shape
    x_fft = jnp.fft.rfftn(input, s=(*si, sl))
    out = jnp.zeros([o, *si, sl // 2 + 1], dtype=input.dtype) + 0j
    for i, slice_i in enumerate(generate_modes_slices(modes)):
        matmul_out = jnp.einsum("i...,oi...->o...", x_fft[tuple(slice_i)], weight[i])
        out = out.at[tuple(slice_i)].set(matmul_out)
    return jnp.fft.irfftn(out, s=(*si, sl))
