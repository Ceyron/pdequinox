from typing import Callable

import jax
from jaxtyping import PRNGKeyArray

from .._base_block_net import BaseBlockNet
from ..blocks import ClassicSpectralBlockFactory, LinearChannelAdjustBlockFactory


class ClassicFNO(BaseBlockNet):
    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: int = 32,
        num_modes: int = 12,
        num_blocks: int = 4,
        activation: Callable = jax.nn.gelu,
        key: PRNGKeyArray,
    ):
        """
        Vanilla FNO

        https://github.com/neuraloperator/neuraloperator/blob/af93f781d5e013f8ba5c52baa547f2ada304ffb0/fourier_1d.py#L62
        """
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            activation=activation,
            key=key,
            boundary_mode="periodic",  # Does not matter
            lifting_factory=LinearChannelAdjustBlockFactory(
                use_bias=True,
                zero_bias_init=False,
            ),
            block_factory=ClassicSpectralBlockFactory(
                num_modes=num_modes,
                use_bias=True,
                zero_bias_init=False,
            ),
            projection_factory=LinearChannelAdjustBlockFactory(
                use_bias=True,
                zero_bias_init=False,
            ),
        )
