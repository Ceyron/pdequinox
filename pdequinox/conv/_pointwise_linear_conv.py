import equinox as eqx
import jax.numpy as jnp


class PointwiseLinearConv(eqx.nn.Conv):
    """
    aka 1x1 Convolution; used primarily for channel adjustment
    """

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        use_bias: bool = True,
        *,
        key,
        zero_bias_init: bool = False,
    ):
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            padding=0,
            use_bias=use_bias,
            key=key,
        )
        if use_bias and zero_bias_init:
            zero_bias = jnp.zeros_like(self.conv.bias)
            self.conv = eqx.tree_at(lambda leaf: leaf.bias, self.conv, zero_bias)

    @property
    def receptive_field(self) -> tuple[tuple[float, float], ...]:
        return tuple(((0.0, 0.0),) * self.num_spatial_dims)