"""
Uses the modifications as in PDEArena:
https://github.com/microsoft/pdearena/blob/22360a766387c3995220b4a1265a936ab9a81b88/pdearena/modules/twod_resnet.py#L15

most importantly, it oes pre-activation instead of post-activation

ToDo: check if we also need the no-bias in the bypass
"""

from typing import Callable

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from .._physics_conv import PhysicsConv
from .._pointwise_linear_conv import PointwiseLinearConv


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
        activation: Callable,
        kernel_size: int = 3,
        *,
        boundary_mode: str,
        key,
        use_norm: bool = True,
        num_groups: int = 1,  # for GroupNorm
        use_bias: bool = True,
        zero_bias_init: bool = False,
        **boundary_kwargs,
    ):
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

        conv_1_key, conv_2_key, key = jax.random.split(key, 3)
        self.conv_1 = conv_constructor(in_channels, out_channels, use_bias, conv_1_key)
        if use_norm:
            self.norm_1 = eqx.nn.GroupNorm(groups=num_groups, channels=out_channels)
        else:
            self.norm_1 = eqx.nn.Identity()
        self.conv_2 = conv_constructor(out_channels, out_channels, use_bias, conv_2_key)
        # In the PDEArena, for some reason, there is always a second group norm
        # even if use_norm is False
        if use_norm:
            self.norm_2 = eqx.nn.GroupNorm(groups=num_groups, channels=out_channels)
        else:
            self.norm_2 = eqx.nn.Identity()
        self.activation = activation

        if out_channels != in_channels:
            bypass_conv_key, _ = jax.random.split(key)
            self.bypass_conv = PointwiseLinearConv(
                num_spatial_dims=num_spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=False,  # Following PDEArena
                key=bypass_conv_key,
            )
            if use_norm:
                self.bypass_norm = eqx.nn.GroupNorm(
                    groups=num_groups, channels=out_channels
                )
            else:
                self.bypass_norm = eqx.nn.Identity()
        else:
            self.bypass_conv = eqx.nn.Identity()
            self.bypass_norm = eqx.nn.Identity()

    def __call__(self, x):
        x_skip = x
        # Using pre-activation instead of post-activation
        x = self.conv_1(self.activation(self.norm_1(x)))
        x = self.conv_2(self.activation(self.norm_2(x)))

        x = x + self.bypass_conv(self.bypass_norm(x_skip))
        return x


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
            **boundary_kwargs,
        )
