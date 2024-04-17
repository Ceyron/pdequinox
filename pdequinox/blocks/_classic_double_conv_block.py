from typing import Callable

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from ..conv import PhysicsConv
from ._base_block import Block, BlockFactory


class ClassicDoubleConvBlock(Block):
    conv_1: PhysicsConv
    norm_1: eqx.nn.GroupNorm
    conv_2: PhysicsConv
    norm_2: eqx.nn.GroupNorm
    activation: Callable

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        boundary_mode: str,
        key: PRNGKeyArray,
        activation: Callable = jax.nn.relu,
        kernel_size: int = 3,
        use_norm: bool = True,
        num_groups: int = 1,  # for GroupNorm
        use_bias: bool = True,
        zero_bias_init: bool = False,
        **boundary_kwargs,
    ):
        """
        Block that performs two sequential convolutions with activation and
        optional group normalization in between.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example
            traditional convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `boundary_mode`: The boundary mode to use for the convolution.
            (Keyword only argument)
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        - `activation`: The activation function to use after each convolution.
            Default is `jax.nn.relu`.
        - `kernel_size`: The size of the convolutional kernel. Default is `3`.
        - `use_norm`: Whether to use group normalization. Default is `True`.
        - `num_groups`: The number of groups to use for group normalization.
            Default is `1`.
        - `use_bias`: Whether to use bias in the convolutional layers. Default is
            `True`.
        - `zero_bias_init`: Whether to initialise the bias to zero. Default is
            `False`.
        """

        def conv_constructor(i, o, b, k):
            return PhysicsConv(
                num_spatial_dims=num_spatial_dims,
                in_channels=i,
                out_channels=o,
                kernel_size=kernel_size,
                stride=1,
                dilation=1,
                boundary_mode=boundary_mode,
                use_bias=b,
                zero_bias_init=zero_bias_init,
                key=k,
                **boundary_kwargs,
            )

        k_1, k_2 = jax.random.split(key)
        self.conv_1 = conv_constructor(in_channels, out_channels, use_bias, k_1)
        if use_norm:
            self.norm_1 = eqx.nn.GroupNorm(groups=num_groups, channels=out_channels)
        else:
            self.norm_1 = eqx.nn.Identity()
        self.conv_2 = conv_constructor(out_channels, out_channels, use_bias, k_2)
        if use_norm:
            self.norm_2 = eqx.nn.GroupNorm(groups=num_groups, channels=out_channels)
        else:
            self.norm_2 = eqx.nn.Identity()

        self.activation = activation

    def __call__(self, x):
        x = self.activation(self.norm_1(self.conv_1(x)))
        x = self.activation(self.norm_2(self.conv_2(x)))
        return x

    @property
    def receptive_field(self) -> tuple[tuple[float, float], ...]:
        conv_1_receptive_field = self.conv_1.receptive_field
        conv_2_receptive_field = self.conv_2.receptive_field
        return tuple(
            (
                c_1_i_backward + c_2_i_backward,
                c_1_i_forward + c_2_i_forward,
            )
            for (c_1_i_backward, c_1_i_forward), (c_2_i_backward, c_2_i_forward) in zip(
                conv_1_receptive_field, conv_2_receptive_field
            )
        )


class ClassicDoubleConvBlockFactory(BlockFactory):
    kernel_size: int
    use_norm: bool
    num_groups: int
    use_bias: bool
    zero_bias_init: bool

    def __init__(
        self,
        *,
        kernel_size: int = 3,
        use_norm: bool = True,
        num_groups: int = 1,
        use_bias: bool = True,
        zero_bias_init: bool = False,
    ):
        """
        Factory for creating `ClassicDoubleConvBlock` instances.

        **Arguments:**

        - `kernel_size`: The size of the convolutional kernel. Default is `3`.
        - `use_norm`: Whether to use group normalization. Default is `True`.
        - `num_groups`: The number of groups to use for group normalization.
            Default is `1`.
        - `use_bias`: Whether to use bias in the convolutional layers. Default is
            `True`.
        - `zero_bias_init`: Whether to initialise the bias to zero. Default is
            `False`.
        """
        self.kernel_size = kernel_size
        self.use_norm = use_norm
        self.num_groups = num_groups
        self.use_bias = use_bias
        self.zero_bias_init = zero_bias_init

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
    ):
        return ClassicDoubleConvBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            boundary_mode=boundary_mode,
            key=key,
            kernel_size=self.kernel_size,
            use_norm=self.use_norm,
            num_groups=self.num_groups,
            use_bias=self.use_bias,
            zero_bias_init=self.zero_bias_init,
            **boundary_kwargs,
        )
