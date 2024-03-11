"""
Following
https://github.com/microsoft/pdearena/blob/22360a766387c3995220b4a1265a936ab9a81b88/pdearena/modules/twod_resnet.py#L56

but correctly does the channel matching
"""

from typing import Callable

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from ..conv import PhysicsConv, PointwiseLinearConv


class DilatedResBlock(eqx.Module):
    norm_layers: tuple[eqx.nn.GroupNorm]
    conv_layers: tuple[PhysicsConv]
    activation: Callable
    bypass_conv: eqx.Module
    bypass_norm: eqx.Module

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        activation: Callable,
        kernel_size: int = 3,
        dilation_rates: tuple[int] = (1, 2, 4, 8, 4, 2, 1),
        *,
        boundary_mode: str,
        key,
        use_norm: bool = True,
        num_groups: int = 1,  # for GroupNorm
        use_bias: bool = True,
        zero_bias_init: bool = False,
        **boundary_kwargs,
    ):
        def conv_constructor(i, o, d, b, k):
            return PhysicsConv(
                num_spatial_dims=num_spatial_dims,
                in_channels=i,
                out_channels=o,
                kernel_size=kernel_size,
                stride=1,
                dilation=d,
                boundary_mode=boundary_mode,
                use_bias=b,
                zero_bias_init=zero_bias_init,
                key=k,
                **boundary_kwargs,
            )

        if use_norm:
            norm_layers = []
            norm_layers.append(
                eqx.nn.GroupNorm(groups=num_groups, channels=in_channels)
            )

            for _ in dilation_rates[1:]:
                norm_layers.append(
                    eqx.nn.GroupNorm(groups=num_groups, channels=out_channels)
                )

            self.norm_layers = tuple(norm_layers)
        else:
            self.norm_layers = tuple(eqx.nn.Identity() for _ in dilation_rates)

        key, *keys = jax.random.split(key, len(dilation_rates) + 1)

        conv_layers = []
        conv_layers.append(
            conv_constructor(
                in_channels, out_channels, dilation_rates[0], use_bias, keys[0]
            )
        )
        for d, k in zip(dilation_rates[1:], keys[1:]):
            conv_layers.append(
                conv_constructor(out_channels, out_channels, d, use_bias, k)
            )

        self.conv_layers = tuple(conv_layers)

        self.activation = activation

        if out_channels != in_channels:
            bypass_conv_key, _ = jax.random.split(key)
            self.bypass_conv = PointwiseLinearConv(
                num_spatial_dims=num_spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=use_bias,  # Todo: should this be True or False by default?
                zero_bias_init=zero_bias_init,
                key=bypass_conv_key,
                **boundary_kwargs,
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
        for norm, conv in zip(self.norm_layers, self.conv_layers):
            x = norm(x)
            x = conv(x)
            x = self.activation(x)

        x_skip = self.bypass_conv(self.bypass_norm(x_skip))
        x = x + x_skip

        return x


class DilatedResBlockFactory(eqx.Module):
    kernel_size: int
    dilation_rates: tuple[int]
    use_norm: bool
    num_groups: int
    use_bias: bool
    zero_bias_init: bool

    def __init__(
        self,
        kernel_size: int = 3,
        dilation_rates: tuple[int] = (1, 2, 4, 8, 4, 2, 1),
        *,
        use_norm: bool = True,
        num_groups: int = 1,
        use_bias: bool = True,
        zero_bias_init: bool = False,
    ):
        self.kernel_size = kernel_size
        self.dilation_rates = dilation_rates
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
    ) -> DilatedResBlock:
        return DilatedResBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation,
            kernel_size=self.kernel_size,
            dilation_rates=self.dilation_rates,
            boundary_mode=boundary_mode,
            key=key,
            use_norm=self.use_norm,
            num_groups=self.num_groups,
            use_bias=self.use_bias,
            zero_bias_init=self.zero_bias_init,
            **boundary_kwargs,
        )
