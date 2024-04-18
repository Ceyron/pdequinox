"""
Uses the modifications as in PDEArena:
https://github.com/microsoft/pdearena/blob/22360a766387c3995220b4a1265a936ab9a81b88/pdearena/modules/twod_resnet.py#L15

most importantly, it does pre-activation instead of post-activation (a
re-ordering of the operations to allow for a clean bypass/residual connection)

ToDo: check if we also need the no-bias in the bypass
"""

from typing import Callable, Literal

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from ..conv import PhysicsConv, PointwiseLinearConv


class ModernResBlock(eqx.Module):
    conv_1: eqx.Module
    norm_1: eqx.Module
    conv_2: eqx.Module
    norm_2: eqx.Module
    bypass_conv: eqx.Module
    bypass_norm: eqx.Module
    activation: Callable

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        boundary_mode: Literal["periodic", "dirichlet", "neumann"],
        key,
        activation: Callable = jax.nn.relu,
        kernel_size: int = 3,
        use_norm: bool = True,
        num_groups: int = 1,  # for GroupNorm
        use_bias: bool = True,
        zero_bias_init: bool = False,
    ):
        """
        Block that performs two sequential convolutions with activation and
        optional group normalization in between. The order of operations is
        based on "pre-activation" to allow for a clean bypass/residual
        connection.

        If the number of input channels is different from the number of output
        channels, a pointwise convolution (without bias) is used to match the
        number of channels.

        If `use_norm` is `True`, group normalization is used after each
        convolution. If there is a convolution that matches the number of
        channels, the bypass will also have group normalization.

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
        - `use_bias`: Whether to use bias in the convolutional layers. Default
            is `True`.
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
            )

        conv_1_key, conv_2_key, key = jax.random.split(key, 3)

        if use_norm:
            self.norm_1 = eqx.nn.GroupNorm(groups=num_groups, channels=in_channels)
        else:
            self.norm_1 = eqx.nn.Identity()
        self.conv_1 = conv_constructor(in_channels, out_channels, use_bias, conv_1_key)

        # In the PDEArena, for some reason, there is always a second group norm
        # even if use_norm is False
        if use_norm:
            self.norm_2 = eqx.nn.GroupNorm(groups=num_groups, channels=out_channels)
        else:
            self.norm_2 = eqx.nn.Identity()
        self.conv_2 = conv_constructor(out_channels, out_channels, use_bias, conv_2_key)

        self.activation = activation

        if out_channels != in_channels:
            bypass_conv_key, _ = jax.random.split(key)

            if use_norm:
                self.bypass_norm = eqx.nn.GroupNorm(
                    groups=num_groups, channels=in_channels
                )
            else:
                self.bypass_norm = eqx.nn.Identity()

            self.bypass_conv = PointwiseLinearConv(
                num_spatial_dims=num_spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=False,  # Following PDEArena
                key=bypass_conv_key,
            )
        else:
            self.bypass_norm = eqx.nn.Identity()
            self.bypass_conv = eqx.nn.Identity()

    def __call__(self, x):
        x_skip = x
        # Using pre-activation instead of post-activation
        x = self.conv_1(self.activation(self.norm_1(x)))
        x = self.conv_2(self.activation(self.norm_2(x)))

        x = x + self.bypass_conv(self.bypass_norm(x_skip))
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


class ModernResBlockFactory(eqx.Module):
    kernel_size: int
    use_norm: bool
    num_groups: int
    use_bias: bool
    zero_bias_init: bool

    def __init__(
        self,
        kernel_size: int = 3,
        *,
        use_norm: bool = True,
        num_groups: int = 1,
        use_bias: bool = True,
        zero_bias_init: bool = False,
    ):
        """
        Factory for creating `ModernResBlock` instances.

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
        boundary_mode: Literal["periodic", "dirichlet", "neumann"],
        key: PRNGKeyArray,
    ):
        return ModernResBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            kernel_size=self.kernel_size,
            boundary_mode=boundary_mode,
            key=key,
            use_norm=self.use_norm,
            num_groups=self.num_groups,
            use_bias=self.use_bias,
            zero_bias_init=self.zero_bias_init,
        )
