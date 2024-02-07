import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
from typing import Tuple, Sequence
from equinox import field

class BasePadding(eqx.Module):
    pass

class PeriodicPadding(BasePadding):
    pad_width: tuple[tuple[int, int], ...] = field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        padding: tuple[tuple[int, int], ...],
    ):
        self.pad_width = ((0, 0),) + padding

    def __call__(self, x: Float[Array, "C *N"]):
        """Periodic padding for convolution.

        Args:
            x: Input array with shape (C, *N).

        Returns:
            Array with shape (C, *(N + 2 * padding)).
        """
        return jnp.pad(x, self.pad_width, mode="wrap")

def compute_valid_padding(
    kernel_size: tuple[int, ...],
    dilation: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    padding_width = tuple(
        ((k - 1) // 2) * d
        for k, d in zip(kernel_size, dilation)
    )
    return padding_width

def compute_valid_transpose_padding(
    kernel_size: tuple[int, ...],
    dilation: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
    output_padding: tuple[int, ...]
) -> tuple[tuple[int, int], ...]:
    padding_width = tuple(
        (d * (k - 1) - p_0, d * (k - 1) - p_1 + o)
        for k, d, (p_0, p_1), o in zip(kernel_size, dilation, padding, output_padding)
    )
    return padding_width


class PhysicsConv(eqx.Module):
    pad: BasePadding
    conv: eqx.nn.Conv
    boundary_mode: str
    boundary_kwargs: dict

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        *,
        boundary_mode: str,
        use_bias: bool = True,
        zero_bias_init: bool = False,
        key,
        **boundary_kwargs,
    ):
        """
        Only way to reduce the spatial dimensions is to use stride. Otherwise,
        the convolutions will always be 'valid'.
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.conv = eqx.nn.Conv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=0,  # padding is handled by the pad module
            groups=1,  # no groups for now
            key=key,
            use_bias=use_bias,
        )
        if use_bias and zero_bias_init:
            zero_bias = jnp.zeros_like(self.conv.bias)
            self.conv = eqx.tree_at(lambda l: l.bias, self.conv, zero_bias)

        self.boundary_mode = boundary_mode.lower()
        self.boundary_kwargs = boundary_kwargs
        self._setup_padder()

    def _setup_padder(self):
        padding_width = compute_valid_padding(
            self.conv.kernel_size,
            self.conv.dilation,
        )

        if self.boundary_mode == "periodic":
            self.pad = PeriodicPadding(
                self.conv.num_spatial_dims,
                padding_width,
                **self.boundary_kwargs,
            )
        else:
            raise ValueError(f"boundary_mode={self.boundary_mode} not implemented")


    def __call__(self, x: Float[Array, "C *N"]) -> Float[Array, "C *N"]:
        """Periodic convolution.

        Args:
            x: Input array with shape (C, *N).

        Returns:
            Array with shape (C, *N).
        """
        return self.conv(self.pad(x))
        
class PhysicsConvTranspose(eqx.Module):
    pad: BasePadding
    conv: eqx.nn.ConvTranspose
    boundary_mode: str
    boundary_kwargs: dict

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        output_padding: int = 0,
        dilation: int = 1,
        *,
        boundary_mode: str,
        use_bias: bool = True,
        zero_bias_init: bool = False,
        key,
        **boundary_kwargs,
    ):
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.conv = eqx.nn.ConvTranspose(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=0, # padding is handled by the pad module
            dilation=dilation,
            padding=0,  # padding is handled by the pad module
            groups=1,  # no groups for now
            key=key,
            use_bias=use_bias,
        )
        if use_bias and zero_bias_init:
            zero_bias = jnp.zeros_like(self.conv.bias)
            self.conv = eqx.tree_at(lambda l: l.bias, self.conv, zero_bias)

        self.boundary_mode = boundary_mode.lower()
        self.boundary_kwargs = boundary_kwargs
        self.pad = self._setup_padder(output_padding)
    
    def _setup_padder(self, output_padding: int):
        valid_padding_width = compute_valid_padding(
            self.conv.kernel_size,
            self.conv.dilation,
        )
        transpose_padding_width = compute_valid_transpose_padding(
            self.conv.kernel_size, self.conv.dilation, valid_padding_width, output_padding
        )
        if self.boundary_mode == "periodic":
            padder = PeriodicPadding(
                self.conv.num_spatial_dims,
                transpose_padding_width,
                **self.boundary_kwargs,
            )
        else:
            raise ValueError(f"boundary_mode={self.boundary_mode} not implemented")
        
        return padder


    def __call__(
        self,
        x: Float[Array, "C *N"],
        *,
        output_padding: int or None = None,
    ) -> Float[Array, "C *N"]:
        """
        Use `output_padding` to overwrite the output padding specified in the
        constructor.
        """
        if output_padding is not None:
            padder = self._setup_padder(output_padding)
        else:
            padder = self.pad

        return self.conv(padder(x))

