import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

class PointwiseLinearConv(eqx.Module):
    """
    aka 1x1 Convolution; used primarily for channel adjustment
    """
    
    conv: eqx.nn.Conv

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        key,
        use_bias: bool = True,
        zero_bias_init: bool = False,
    ):
        self.conv = eqx.nn.Conv(
            num_spatial_dims,
            in_channels,
            out_channels,
            1,
            key=key,
            use_bias=use_bias,
        )
        if use_bias and zero_bias_init:
            zero_bias = jnp.zeros_like(self.conv.bias)
            self.conv = eqx.tree_at(lambda l: l.bias, self.conv, zero_bias)

    def __call__(self, x: Float[Array, "C ..."]) -> Float[Array, "C ..."]:
        return self.conv(x)
