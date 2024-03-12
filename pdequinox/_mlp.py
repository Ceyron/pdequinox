from math import prod
from typing import Callable

import equinox as eqx
from jaxtyping import PRNGKeyArray


def _identity(x):
    return x


class MLP(eqx.Module):
    num_spatial_dims: int
    num_points: int
    in_channels: int
    out_channels: int
    flat_mlp: eqx.nn.MLP

    _in_shape: tuple[int, ...]
    _out_shape: tuple[int, ...]

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        in_channels: int,
        out_channels: int,
        width_size: int,
        depth: int,
        activation: Callable,
        final_activation: Callable = _identity,
        use_bias: bool = True,
        use_final_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        self.num_spatial_dims = num_spatial_dims
        self.num_points = num_points
        self.in_channels = in_channels
        self.out_channels = out_channels

        self._in_shape = (in_channels,) + (1, num_points) * num_spatial_dims
        self._out_shape = (out_channels,) + (1, num_points) * num_spatial_dims
        flat_in_size = prod(self._in_shape)
        flat_out_size = prod(self._out_shape)

        self.flat_mlp = eqx.nn.MLP(
            in_size=flat_in_size,
            out_size=flat_out_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            final_activation=final_activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            key=key,
        )

    def __call__(self, x):
        if x.shape != self._in_shape:
            raise ValueError(
                f"Input shape {x.shape} does not match expected shape {self._in_shape}. For batched operation use jax.vmap"
            )
        x_flat = x.flatten()
        x_flat = self.flat_mlp(x_flat)
        x = x_flat.reshape(self._out_shape)
        return x
