import jax.numpy as jnp
import equinox as eqx

from physics_conv import PhysicsConv
from typing import Any, Callable
from jaxtyping import PRNGKeyArray

# Base Classes

class ChannelAdjuster(eqx.Module):
    pass

class ChannelAdjusterFactory(eqx.Module):
    def __call__(
        self,
        num_spacial_dims: int,
        in_channels: int,
        out_channels: int,
        activation: Callable,
        *,
        key: PRNGKeyArray,
    ) -> ChannelAdjuster:
        raise NotImplementedError("Must be implemented by subclass")
    
# Linear Adjusters
    
class LinearAdjuster(ChannelAdjuster):
    conv: eqx.Module

    def __init__(
        self,
        num_spacial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        key: PRNGKeyArray,
        use_bias: bool = True,
        zero_bias_init: bool = False,
    ):
        self.conv = eqx.nn.Conv(
            num_spatial_dims,
            in_channels,
            out_channels,
            1,
            1,
            key=key,
            use_bias=use_bias,
        )
        if use_bias and zero_bias_init:
            zero_bias = jnp.zeros_like(self.conv.bias)
            self.conv = eqx.tree_at(lambda l: l.bias, self.conv, zero_bias)

    def __call__(self, x):
        return self.conv(x)
    
class LinearAdjusterFactory(ChannelAdjusterFactory):
    use_bias: bool = True
    zero_bias_init: bool = False

    def __call__(
        self,
        num_spacial_dims: int,
        in_channels: int,
        out_channels: int,
        activation: Callable, # unused
        *,
        key: PRNGKeyArray,
    ) -> ChannelAdjuster:
        return LinearAdjuster(
            num_spacial_dims,
            in_channels,
            out_channels,
            key=key,
            use_bias=self.use_bias,
            zero_bias_init=self.zero_bias_init,
        )