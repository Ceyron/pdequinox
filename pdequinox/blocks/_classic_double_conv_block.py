from typing import Callable

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from .._physics_conv import PhysicsConv
from ._base_block import Block, BlockFactory


class ClassicDoubleConvBlock(Block):
    """Also does channel adjustment"""

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
        activation: Callable,
        kernel_size: int = 3,
        *,
        boundary_mode: str,
        key: PRNGKeyArray,
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


class ClassicDoubleConvBlockFactory(BlockFactory):
    kernel_size: int = 3
    use_norm: bool = True
    num_groups: int = 1
    use_bias: bool = True
    zero_bias_init: bool = False

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
            **boundary_kwargs,
        )
