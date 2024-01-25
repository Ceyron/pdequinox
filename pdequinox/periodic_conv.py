import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

class PeriodicPadding(eqx.Module):
    num_spatial_dims: int
    padding: int

    def __call__(self, x: Float[Array, "C *N"]) -> Float[Array, "C *(N+2*padding)"]:
        """Periodic padding for convolution.

        Args:
            x: Input array with shape (C, *N).

        Returns:
            Array with shape (C, *N + 2 * padding).
        """
        pad_width = [(0, 0),] + [(self.padding, self.padding) for _ in range(self.num_spatial_dims)]
        return jnp.pad(x, pad_width, mode="wrap")

    
class PeriodicConv(eqx.Module):
    pad: PeriodicPadding
    conv: eqx.nn.Conv

    def __init__(
        self,
        num_spacial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        *,
        key,
        use_bias: bool = True,
        zero_bias_init: bool = False,
    ):
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        # padding_with = ((kernel_size - 1) // 2) * dilation

        self.pad = PeriodicPadding(num_spatial_dims, padding_width)
        self.conv = eqx.nn.Conv(
            in_channels,
            out_channels,
            kernel_size,
            dilation,
            key=key,
            use_bias=use_bias,
        )
        if use_bias and zero_bias_init:
            zero_bias = jnp.zeros_like(self.conv.bias)
            self.conv = eqx.tree_at(lambda l: l.bias, self.conv, zero_bias)

    def __call__(self, x: Float[Array, "C *N"]) -> Float[Array, "C *N"]:
        """Periodic convolution.

        Args:
            x: Input array with shape (C, *N).

        Returns:
            Array with shape (C, *N).
        """
        return self.conv(self.pad(x))
        

