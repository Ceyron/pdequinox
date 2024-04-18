from typing import Callable, Literal, Optional

import jax
from jaxtyping import PRNGKeyArray

from .._sequential import Sequential
from ..blocks import ClassicSpectralBlockFactory, LinearChannelAdjustBlockFactory


class ClassicFNO(Sequential):
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
        boundary_mode: Optional[
            Literal["periodic", "dirichlet", "neumann"]
        ] = None,  # unused
        key: PRNGKeyArray,
    ):
        """
        The vanilla Fourier Neural Operator very close to the original Li et al.
        (2020) paper.

        Performs spectral convolution in Fourier to obtain global receptive
        field.

        Note that this architecture does not take a `boundary_mode` argument.
        The authors argue that arbitrary boundary conditions can be learned.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example
            traditional convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `hidden_channels`: The number of channels in the hidden layers.
          Default
            is `32`.
        - `num_modes`: The number of modes to use in the Fourier basis. Think of
            modes as the equivalence of kernel size for classical convolutions.
            Default is `12`.
        - `num_blocks`: The number of blocks to use. One block consists of one
            spectral convolution with a byass by a 1x1 convolution, followed by
            the activation function. Default is `4`.
        - `activation`: The activation function to use in the blocks. Default is
            `jax.nn.gelu`. This is often preferrable over `jax.nn.relu` because
            it recovers more higher modes.
        - `boundary_mode`: Unused, just for compatibility with other architectures.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        See also the reference implementation in PyTorch:

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
