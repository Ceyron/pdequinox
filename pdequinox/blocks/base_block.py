from abc import ABC, abstractmethod

import jax
import equinox as eqx

from typing import Any, Callable
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
        **boundary_kwargs,
    ) -> Block:
        pass