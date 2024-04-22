from typing import Callable, Literal, Union

import jax
from jaxtyping import PRNGKeyArray

from ..conv import PointwiseLinearConv, SpectralConv
from ._base_block import Block, BlockFactory


class ClassicSpectralBlock(Block):
    spectral_conv: SpectralConv
    by_pass_conv: PointwiseLinearConv
    activation: Callable

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        # Uses gelu because it likely recovers more modes
        activation: Callable = jax.nn.gelu,
        num_modes: int = 8,
        use_bias: bool = True,
        zero_bias_init: bool = False,
        key: PRNGKeyArray,
    ):
        """
        Residual-style block as used in vanilla FNOs; combines a spectral
        convolution with a bypass.

        Does not have argument `boundary_mode` because it would not respect it.
        In the original FNO paper it is argued that the bypass helps recover the
        boundary condition.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example
            traditional convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        - `activation`: The activation function to use after each convolution.
            Default is `jax.nn.relu`.
        - `num_modes`: How many modes to consider in Fourier space. At max this
            can be N//2+1, with N being the number of spatial points. Think of
            it as the analogy of the kernel size.
        - `use_bias`: Whether to use a bias in the bypass convolution. Default
            `True`.
        - `zero_bias_init`: Whether to initialise the bias to zero. Default is
            `False`.
        """
        k_1, k_2 = jax.random.split(key)
        self.spectral_conv = SpectralConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_modes=num_modes,
            key=k_1,
        )
        self.by_pass_conv = PointwiseLinearConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=use_bias,
            zero_bias_init=zero_bias_init,
            key=k_2,
        )
        self.activation = activation

    def __call__(self, x):
        x = self.spectral_conv(x) + self.by_pass_conv(x)
        x = self.activation(x)
        return x

    @property
    def receptive_field(self) -> tuple[tuple[float, float], ...]:
        return self.spectral_conv.receptive_field


class ClassicSpectralBlockFactory(BlockFactory):
    num_modes: Union[int, tuple[int, ...]]
    use_bias: bool
    zero_bias_init: bool

    def __init__(
        self,
        *,
        num_modes: int = 8,
        use_bias: bool = True,
        zero_bias_init: bool = False,
    ):
        """
        Factory for creating `ClassicSpectralBlock` instances.

        **Arguments:**

        - `num_modes`: How many modes to consider in Fourier space. At max this
            can be N//2+1, with N being the number of spatial points. Think of
            it as the analogy of the kernel size.
        - `use_bias`: Whether to use a bias in the bypass convolution. Default
            `True`.
        - `zero_bias_init`: Whether to initialise the bias to zero. Default is
            `False`.
        """
        self.num_modes = num_modes
        self.use_bias = use_bias
        self.zero_bias_init = zero_bias_init

    def __call__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        activation: Callable,
        boundary_mode: Literal["periodic", "dirichlet", "neumann"],  # unused
        key: PRNGKeyArray,
        # unused
    ):
        return ClassicSpectralBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_modes=self.num_modes,
            activation=activation,
            key=key,
            use_bias=self.use_bias,
            zero_bias_init=self.zero_bias_init,
        )
