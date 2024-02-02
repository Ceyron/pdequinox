"""
This is NOT! a good implementation yet
"""

import equinox as eqx
import jax
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
    down_convs: List[DoubleConv]
    up_sample: List[eqx.nn.ConvTranspose]
    up_convs: List[DoubleConv]
    bottleneck: DoubleConv
    max_pool: eqx.nn.MaxPool
    final_conv: eqx.nn.Conv
    levels: int

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        activation: Callable,
        hidden_channels: int = 16,
        levels = 4,
        *,
        key,
    ):
        self.down_convs = []
        self.up_convs = []
        self.max_pool = eqx.nn.MaxPool(num_spatial_dims, kernel_size=2)
        self.up_sample = []
        self.levels = levels

        channel_list = [in_channels] + [hidden_channels * 2**i for i in range(levels)]

        for (fan_in, fan_out) in zip(channel_list[:-1], channel_list[1:]):
            key, down_key, up_key, up_sample_key = jax.random.split(key, 4)
            self.down_convs.append(DoubleConv(
                num_spatial_dims,
                fan_in,
                fan_out,
                activation,
                key=down_key,
            ))
            self.up_sample.append(eqx.nn.ConvTranspose(
                num_spatial_dims,
                fan_out,
                fan_out,
                kernel_size=2,
                stride=2,
                key=up_sample_key,
                use_bias=False,
            ))

            # Todo!


        
