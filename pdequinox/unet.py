"""
This is NOT! a good implementation yet
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Callable, List

class DoubleConv(eqx.Module):
    conv_1: eqx.nn.Conv
    norm_1: eqx.nn.GroupNorm
    conv_2: eqx.nn.Conv
    norm_2: eqx.nn.GroupNorm
    activation: Callable

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        activation: Callable,
        *,
        key,
    ):
        key_1, key_2 = jax.random.split(key)
        self.conv_1 = eqx.nn.Conv(
            num_spatial_dims,
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            key=key_1,
            use_bias=False,
        )
        self.norm_1 = eqx.nn.GroupNorm(out_channels, out_channels)
        self.conv_2 = eqx.nn.Conv(
            num_spatial_dims,
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            key=key_2,
            use_bias=False,
        )
        self.norm_2 = eqx.nn.GroupNorm(out_channels, out_channels)
        self.activation = activation

    def __call__(self, x):
        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = self.norm_2(x)
        x = self.activation(x)

        return x
    
class UNet(eqx.Module):
    """
    Uses convolution for downsampling instead of max pooling
    """
    lifting: DoubleConv
    down_sample_convs: List[eqx.nn.Conv]
    up_sample_convs: List[eqx.nn.Conv]
    double_conv_down: List[DoubleConv]
    double_conv_up: List[DoubleConv]
    # bottleneck: DoubleConv
    projection: eqx.nn.Conv
    reduction_factor: int
    levels: int

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        activation: Callable,
        hidden_channels: int = 16,
        *,
        reduction_factor: int = 2,
        levels = 4,
        key,
    ):
        self.down_sample_convs = []
        self.up_sample_convs = []
        self.double_conv_down = []
        self.double_conv_up = []
        self.reduction_factor = reduction_factor
        self.levels = levels

        key, lifting_key, projection_key = jax.random.split(key, 3)

        self.lifting = DoubleConv(
            num_spatial_dims,
            in_channels,
            hidden_channels,
            activation,
            key=lifting_key,
        )
        self.projection = eqx.nn.Conv(
            num_spatial_dims,
            hidden_channels,
            out_channels,
            kernel_size=1,
            key=projection_key,
            use_bias=False,
        )

        channel_list = [hidden_channels * self.reduction_factor**i for i in range(levels)]

        for (fan_in, fan_out) in zip(channel_list[:-1], channel_list[1:]):
            key, down_sample_key, down_key, up_sample_key, up_key = jax.random.split(key, 5)
            self.down_sample_convs.append(eqx.nn.Conv(
                num_spatial_dims,
                fan_in,
                fan_in,
                kernel_size=3,
                stride=2,
                padding=1,
                key=down_sample_key,
                use_bias=False,
            ))
            self.up_sample_convs.append(eqx.nn.ConvTranspose(
                num_spatial_dims,
                fan_out,
                fan_out // self.reduction_factor,
                kernel_size=2,
                stride=2,
                key=up_sample_key,
                use_bias=False,
            ))
            self.double_conv_down.append(DoubleConv(
                num_spatial_dims,
                fan_in,
                fan_out,
                activation,
                key=down_key,
            ))
            self.double_conv_up.append(DoubleConv(
                num_spatial_dims,
                self.reduction_factor * fan_in,
                fan_in,
                activation,
                key=up_key,
            ))


    def __call__(self, x):
        spatial_shape = x.shape[1:]
        for dims in spatial_shape:
            if dims % self.reduction_factor**self.levels != 0:
                raise ValueError("Spatial dim issue")
            
        x = self.lifting(x)

        skips = []
        for (down_conv, double_conv) in zip(self.down_sample_convs, self.double_conv_down):
            skips.append(x)
            x = down_conv(x)
            x = double_conv(x)

        for (up_conv, double_conv) in zip(
            reversed(self.up_sample_convs),
            reversed(self.double_conv_up)
        ):
            skip = skips.pop()
            x = up_conv(x)
            x = jnp.concatenate((skip, x), axis=0)
            x = double_conv(x)

        x = self.projection(x)

        return x
        
