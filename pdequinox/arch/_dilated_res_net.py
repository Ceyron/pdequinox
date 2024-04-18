from typing import Callable

import jax
from jaxtyping import PRNGKeyArray

from .._sequential import Sequential
from ..blocks import DilatedResBlockFactory, LinearChannelAdjustBlockFactory


class DilatedResNet(Sequential):
    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        hidden_channels: int = 32,
        num_blocks: int = 2,
        dilation_rates: tuple[int, ...] = (1, 2, 4, 8, 4, 2, 1),
        use_norm: bool = True,
        activation: Callable = jax.nn.relu,
        boundary_mode: str = "periodic",
        key: PRNGKeyArray,
    ):
        """
        ResNet with varying dilation rates; very close to publication of
        Stachenfeld et al. (2021), based on the implementation of PDEArena.

        Each block consists of a sequence of convolutions with different
        dilation rates as defined by `dilation_rates`. There is an addiitonal
        skip connection.


        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example
            traditional convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `hidden_channels`: The number of channels in the hidden layers.
            Default is `32`.
        - `num_blocks`: The number of blocks to use. Default is `2`.
        - `dilation_rates`: The dilation rates to use. Default is `(1, 2, 4, 8,
            4, 2, 1)`.
        - `use_norm`: If `True`, uses group norm.
        - `activation`: The activation function to use in the blocks. Default is
            `jax.nn.relu`.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        - `boundary_mode`: The boundary mode to use. Default is `periodic`.
        """
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            activation=activation,
            key=key,
            boundary_mode=boundary_mode,
            lifting_factory=LinearChannelAdjustBlockFactory(
                use_bias=True,
                zero_bias_init=False,
            ),
            block_factory=DilatedResBlockFactory(
                dilation_rates=dilation_rates,
                use_norm=use_norm,
                use_bias=True,
                zero_bias_init=False,
            ),
            projection_factory=LinearChannelAdjustBlockFactory(
                use_bias=True,
                zero_bias_init=False,
            ),
        )
