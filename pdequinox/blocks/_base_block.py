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
        **boundary_kwargs,
    ) -> Block:
        pass
