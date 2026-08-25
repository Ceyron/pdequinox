from typing import Callable, Literal, Optional

import jax
from jaxtyping import PRNGKeyArray

from ..conv import PointwiseLinearConv
from ._base_block import Block, BlockFactory


class PointwiseMLPBlock(Block):
    conv_1: PointwiseLinearConv
    conv_2: PointwiseLinearConv
    activation: Callable

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        activation: Callable = jax.nn.relu,
        hidden_channels: Optional[int] = None,
        use_bias: bool = True,
        zero_bias_init: bool = False,
        key: PRNGKeyArray,
    ):
        """
        A two-layer pointwise (1x1 convolution) MLP, applied independently at
        each spatial position.

        Computes `x -> W_2(activation(W_1(x)))`. Note that there is **no**
        activation on the output; this is meant to be usable as a projection
        head at the end of an architecture.

        Does not have an argument `boundary_mode` because a pointwise operation
        never reaches beyond its own position.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example
            traditional convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `activation`: The activation function applied between the two
            convolutions. Default is `jax.nn.relu`.
        - `hidden_channels`: The number of channels in between the two
            convolutions. If `None` (the default), `in_channels` is used.
        - `use_bias`: Whether to use a bias in the convolutions. Default is
            `True`.
        - `zero_bias_init`: Whether to initialise the biases to zero. Default is
            `False`.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        if hidden_channels is None:
            hidden_channels = in_channels

        key_1, key_2 = jax.random.split(key)

        self.conv_1 = PointwiseLinearConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=hidden_channels,
            use_bias=use_bias,
            zero_bias_init=zero_bias_init,
            key=key_1,
        )
        self.conv_2 = PointwiseLinearConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=hidden_channels,
            out_channels=out_channels,
            use_bias=use_bias,
            zero_bias_init=zero_bias_init,
            key=key_2,
        )
        self.activation = activation

    def __call__(self, x):
        return self.conv_2(self.activation(self.conv_1(x)))

    @property
    def receptive_field(self) -> tuple[tuple[float, float], ...]:
        return self.conv_1.receptive_field


class PointwiseMLPBlockFactory(BlockFactory):
    hidden_channels: Optional[int]
    use_bias: bool
    zero_bias_init: bool

    def __init__(
        self,
        *,
        hidden_channels: Optional[int] = None,
        use_bias: bool = True,
        zero_bias_init: bool = False,
    ):
        """
        Factory for creating `PointwiseMLPBlock` instances.

        **Arguments:**

        - `hidden_channels`: The number of channels in between the two
            convolutions. If `None` (the default), the number of input channels
            is used.
        - `use_bias`: Whether to use a bias in the convolutions. Default is
            `True`.
        - `zero_bias_init`: Whether to initialise the biases to zero. Default is
            `False`.
        """
        self.hidden_channels = hidden_channels
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
    ):
        return PointwiseMLPBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            hidden_channels=self.hidden_channels,
            use_bias=self.use_bias,
            zero_bias_init=self.zero_bias_init,
            key=key,
        )
