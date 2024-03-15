from typing import Callable

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from ._utils import sum_receptive_fields
from .conv import PhysicsConv


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
        hidden_channels: int,
        depth: int,
        activation: Callable,
        kernel_size: int = 3,
        final_activation: Callable = _identity,
        use_bias: bool = True,
        use_final_bias: bool = True,
        *,
        boundary_mode: str,
        key: PRNGKeyArray,
        zero_bias_init: bool = False,
        **boundary_kwargs,
    ):
        """
        Depth denotes how many hidden layers there are
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
                **boundary_kwargs,
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
        individual_receptive_fields = (conv.receptive_field for conv in self.layers)
        return sum_receptive_fields(individual_receptive_fields)
