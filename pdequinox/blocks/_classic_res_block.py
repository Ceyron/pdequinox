from typing import Callable

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from ..conv import PhysicsConv, PointwiseLinearConv


class ClassicResBlock(eqx.Module):
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
        boundary_mode: str,
        key: PRNGKeyArray,
        activation: Callable = jax.nn.relu,
        kernel_size: int = 3,
        use_norm: bool = False,
        num_groups: int = 1,  # for group norm
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

        k_1, k_2, key = jax.random.split(key, 3)
        self.conv_1 = conv_constructor(in_channels, out_channels, use_bias, k_1)
        if use_norm:
            self.norm_1 = eqx.nn.GroupNorm(groups=num_groups, channels=out_channels)
        else:
            self.norm_1 = eqx.nn.Identity

        self.conv_2 = conv_constructor(out_channels, out_channels, use_bias, k_2)
        if use_norm:
            self.norm_2 = eqx.nn.GroupNorm(groups=num_groups, channels=out_channels)
        else:
            self.norm_2 = eqx.nn.Identity

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

        self.activation = activation

    def __call__(self, x):
        x_skip = x
        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = self.norm_2(x)
        x = x + x_skip
        x = self.activation(x)
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


class ClassicResBlockFactory(eqx.Module):
    kernel_size: int
    use_bias: bool
    zero_bias_init: bool

    def __init__(
        self,
        kernel_size: int = 3,
        *,
        use_bias: bool = True,
        zero_bias_init: bool = False,
    ):
        self.kernel_size = kernel_size
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
        return ClassicResBlock(
            num_spatial_dims,
            in_channels,
            out_channels,
            activation=activation,
            kernel_size=self.kernel_size,
            boundary_mode=boundary_mode,
            key=key,
            use_bias=self.use_bias,
            zero_bias_init=self.zero_bias_init,
            **boundary_kwargs,
        )
