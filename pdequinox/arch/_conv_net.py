from typing import Callable

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from .._utils import sum_receptive_fields
from ..conv import PhysicsConv


def _identity(x):
    return x


class ConvNet(eqx.Module):
    layers: tuple[PhysicsConv, ...]
    activation: Callable
    final_activation: Callable

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: int = 16,
        depth: int = 10,
        activation: Callable = jax.nn.relu,
        kernel_size: int = 3,
        final_activation: Callable = _identity,
        use_bias: bool = True,
        use_final_bias: bool = True,
        boundary_mode: str = "periodic",
        key: PRNGKeyArray,
        zero_bias_init: bool = False,
    ):
        """
        A simple feed-forward convolutional neural network.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example
            traditional convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `hidden_channels`: The number of channels in the hidden layers.
            Default is `16`.
        - `depth`: The number of hidden layers. Default is `10`. If `depth ==
            0`, there will only be one **linear** convolution from the input
            channels to the output channels. Hence, `depth` denotes the number
            of hidden layers. The number of convolutions performed is `depth +
            1`.
        - `activation`: The activation function to use in the hidden layers.
            Default is `jax.nn.relu`.
        - `kernel_size`: The size of the convolutional kernel. Default is `3`.
        - `final_activation`: The activation function to use in the final layer.
            Default is the identity function.
        - `use_bias`: If `True`, uses bias in the hidden layers. Default is
            `True`.
        - `use_final_bias`: If `True`, uses bias in the final layer. Default is
            `True`.
        - `boundary_mode`: The boundary mode to use. Default is `periodic`.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        - `zero_bias_init`: If `True`, initialises the bias to zero. Default is
            `False`.
        """

        self.activation = activation
        self.final_activation = final_activation

        keys = jax.random.split(key, depth + 1)

        def conv_constructor(i, o, b, k):
            return PhysicsConv(
                num_spatial_dims=num_spatial_dims,
                in_channels=i,
                out_channels=o,
                kernel_size=kernel_size,
                stride=1,
                dilation=1,
                use_bias=b,
                zero_bias_init=zero_bias_init,
                boundary_mode=boundary_mode,
                key=k,
            )

        layers = []
        if depth == 0:
            layers.append(
                conv_constructor(in_channels, out_channels, use_final_bias, keys[0])
            )
        else:
            layers.append(
                conv_constructor(in_channels, hidden_channels, use_bias, keys[0])
            )
            for i in range(depth - 1):
                layers.append(
                    conv_constructor(
                        hidden_channels, hidden_channels, use_bias, keys[i + 1]
                    )
                )
            layers.append(
                conv_constructor(
                    hidden_channels, out_channels, use_final_bias, keys[-1]
                )
            )

        self.layers = tuple(layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.final_activation(self.layers[-1](x))

    @property
    def receptive_field(self) -> tuple[tuple[float, float], ...]:
        individual_receptive_fields = tuple(
            conv.receptive_field for conv in self.layers
        )
        return sum_receptive_fields(individual_receptive_fields)
