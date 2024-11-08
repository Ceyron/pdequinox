import equinox as eqx
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray


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
        zero_bias_init: bool = False,
        key: PRNGKeyArray,
    ):
        """
        General n-dimensional pointwise linear convolution (=1x1 convolution).
        This is primarily used for channel adjustment.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example
            traditional, convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `use_bias`: Whether to use a bias term. (Default: `True`)
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        - `zero_bias_init`: Whether to initialise the bias to zero. (Default:
            `False`)
        """
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
            self.bias = jnp.zeros_like(self.bias)

    @property
    def receptive_field(self) -> tuple[tuple[float, float], ...]:
        return tuple(((0.0, 0.0),) * self.num_spatial_dims)
