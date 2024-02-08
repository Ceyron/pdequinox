import jax
import equinox as eqx

from .physics_conv import PhysicsConv
from .spectral_conv import SpectralConv
from .pointwise_linear_conv import PointwiseLinearConv
from typing import Any, Callable
from jaxtyping import PRNGKeyArray

# Base Classes

class Block(eqx.Module):
    pass

class BlockFactory(eqx.Module):
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
    ) -> Block:
        raise NotImplementedError("Must be implemented by subclass")
    
# Classic Double Conv (as used in unets)

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
        *,
        boundary_mode: str,
        key: PRNGKeyArray,
        **boundary_kwargs,
    ):
        k_1, k_2 = jax.random.split(key)
        self.conv_1 = PhysicsConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            use_bias=False,
            key=k_1,
            boundary_mode=boundary_mode,
            **boundary_kwargs,
        )
        self.norm_1 = eqx.nn.GroupNorm(out_channels, out_channels)
        self.conv_2 = PhysicsConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            use_bias=False,
            key=k_2,
            boundary_mode=boundary_mode,
            **boundary_kwargs,
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
    
class ClassicDoubleConvBlockFactory(BlockFactory):
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

# Classic ResNet Blocks

class ClassicResBlock(eqx.Module):
    conv_1: eqx.Module
    conv_2: eqx.Module
    activation: Callable

    def __init__(
        self,
        num_spatial_dims: int,
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
        k_1, k_2 = jax.random.split(key)
        self.conv_1 = PhysicsConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            boundary_mode=boundary_mode,
            use_bias=use_bias,
            zero_bias_init=zero_bias_init,
            key=k_1,
            **boundary_kwargs,
        )
        self.conv_2 = PhysicsConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            boundary_mode=boundary_mode,
            use_bias=use_bias,
            zero_bias_init=zero_bias_init,
            key=k_2,
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
    use_bias: bool
    zero_bias_init: bool

    def __init__(
        self,
        kernel_size: int = 3,
        dilation: int = 1,
        *,
        use_bias: bool = True,
        zero_bias_init: bool = False,
    ):
        self.kernel_size = kernel_size
        self.dilation = dilation
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
        if in_channels != out_channels:
            raise ValueError("ClassicResBlock only supports in_channels == out_channels")
        else:
            channels = in_channels
        return ClassicResBlock(
            num_spatial_dims,
            channels,
            activation,
            self.kernel_size,
            self.dilation,
            boundary_mode=boundary_mode,
            key=key,
            use_bias=self.use_bias,
            zero_bias_init=self.zero_bias_init,
            **boundary_kwargs,
        )
    
### Spectral ResNet Blocks (aka FNO-Block)
    
class ClassicSpectralBlock(Block):
    spectral_conv: SpectralConv
    by_pass_conv: PointwiseLinearConv
    activation: Callable

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_modes: int,
        activation: Callable,
        *,
        use_bias: bool = True,
        zero_bias_init: bool = False,
        key: PRNGKeyArray,
    ):
        k_1, k_2 = jax.random.split(key)
        self.spectral_conv = SpectralConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_modes=num_modes,
            key=k_1,
        )
        self.by_pass_conv = PointwiseLinearConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=use_bias,
            zero_bias_init=zero_bias_init,
            key=k_2,
        )
        self.activation = activation

    def __call__(self, x):
        x = self.spectral_conv(x) + self.by_pass_conv(x)
        x = self.activation(x)
        return x
    
class ClassicSpectralBlockFactory(BlockFactory):
    num_modes: int or tuple[int, ...]
    use_bias: bool = True
    zero_bias_init: bool = False

    def __call__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        activation: Callable,
        *,
        boundary_mode: str,  # unused
        key: PRNGKeyArray,
        **boundary_kwargs,  # unused
    ):
        return ClassicSpectralBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_modes=self.num_modes,
            activation=activation,
            key=key,
            use_bias=self.use_bias,
            zero_bias_init=self.zero_bias_init,
        )

### Linear channel adjuster

LinearChannelAdjustmentBlock = PointwiseLinearConv

class LinearChannelAdjustmentBlockFactory(BlockFactory):
    use_bias: bool = True
    zero_bias_init: bool = False

    def __call__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        activation: Callable,  # unused
        *,
        boundary_mode: str,  # unused
        key: PRNGKeyArray,
        **boundary_kwargs,  # unused
    ):
        return PointwiseLinearConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=self.use_bias,
            zero_bias_init=self.zero_bias_init,
            key=key,
        )
