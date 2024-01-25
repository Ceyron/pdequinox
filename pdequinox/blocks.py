import equinox as eqx

from physics_conv import PhysicsConv
from typing import Any, Callable
from jaxtyping import PRNGKeyArray

# Base Classes

class Block(eqx.Module):
    pass

class BlockFactory(eqx.Module):
    def __call__(
        self,
        num_spacial_dims: int,
        channels: int,
        *,
        key: PRNGKeyArray,
    ) -> Block:
        raise NotImplementedError("Must be implemented by subclass")

# Classic ResNet Blocks

class ClassicResBlock(eqx.Module):
    conv_1: eqx.Module
    conv_2: eqx.Module
    activation: Callable

    def __init__(
        self,
        num_spacial_dims: int,
        channels: int,
        activation: Callable,
        kernel_size: int = 3,
        dilation: int = 1,
        *,
        boundary_mode: str,
        key,
        use_bias: bool = True,
        zero_bias_init: bool = False,
        **boundary_kwargs,
    ):
        self.conv_1 = PhysicsConv(
            num_spacial_dims,
            channels,
            channels,
            kernel_size,
            dilation,
            boundary_mode=boundary_mode,
            key=key,
            use_bias=use_bias,
            zero_bias_init=zero_bias_init,
            **boundary_kwargs,
        )
        self.conv_2 = PhysicsConv(
            num_spacial_dims,
            channels,
            channels,
            kernel_size,
            dilation,
            boundary_mode=boundary_mode,
            key=key,
            use_bias=use_bias,
            zero_bias_init=zero_bias_init,
            **boundary_kwargs,
        )
        self.activation = activation

    def __call__(self, x):
        x_skip = x
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = x + x_skip
        x = self.activation(x)
        return x

class ClassicResBlockFactory(eqx.Module):
    kernel_size: int
    dilation: int
    boundary_mode: str
    use_bias: bool
    zero_bias_init: bool
    boundary_kwargs: dict

    def __init__(
        self,
        activation: Callable,
        kernel_size: int = 3,
        dilation: int = 1,
        *,
        boundary_mode: str,
        use_bias: bool = True,
        zero_bias_init: bool = False,
        **boundary_kwargs,
    ):
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.boundary_mode = boundary_mode
        self.use_bias = use_bias
        self.zero_bias_init = zero_bias_init
        self.boundary_kwargs = boundary_kwargs
        self.activation = activation

    def __call__(
        self,
        num_spacial_dims: int,
        channels: int,
        *,
        key: PRNGKeyArray,
    ):
        return ClassicResBlock(
            num_spacial_dims,
            channels,
            self.activation,
            self.kernel_size,
            self.dilation,
            boundary_mode=self.boundary_mode,
            key=key,
            use_bias=self.use_bias,
            zero_bias_init=self.zero_bias_init,
            **self.boundary_kwargs,
        )
