from abc import ABC, abstractmethod
from typing import Callable

import equinox as eqx
from jaxtyping import PRNGKeyArray


class Block(eqx.Module, ABC):
    pass


class BlockFactory(eqx.Module, ABC):
    @abstractmethod
    def __call__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        activation: Callable,
        *,
        boundary_mode: str,
        key: PRNGKeyArray,
    ) -> Block:
        """
        Construct a block (= `equinox.Module`)

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example
            traditional, convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `activation`: The activation function to use. For example
            `jax.nn.relu`.
        - `boundary_mode`: The boundary mode to use. For example `"periodic"`.
            (Keyword only argument.)
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        - ` `: Additional keyword arguments to pass to the boundary
            mode constructor.
        """
        pass
