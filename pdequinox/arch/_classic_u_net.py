from typing import Callable

import jax
from jaxtyping import PRNGKeyArray

from .._hierarchical import Hierarchical


# ToDo change to maxpool
class ClassicUNet(Hierarchical):
    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: int = 16,
        num_levels: int = 4,
        activation: Callable = jax.nn.relu,
        key: PRNGKeyArray,
        boundary_mode: str = "periodic",
        **boundary_kwargs,
    ):
        """
        Classic U-Net
        """
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            activation=activation,
            key=key,
            boundary_mode=boundary_mode,
            boundary_kwargs=boundary_kwargs,
        )
