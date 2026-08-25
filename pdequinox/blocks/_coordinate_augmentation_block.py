"""
Augments a field with its own coordinates before handing it to another block.

This is what the Flower architecture (Muser et al., ICML 2026) does in its
lifting layer. It matters there because every learned map in Flower is pointwise
(1x1 convolutions), so without coordinate channels a block has no way of knowing
*where* in the domain it is acting.
"""

from typing import Callable, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ._base_block import Block, BlockFactory
from ._linear_channel_adjust_block import LinearChannelAdjustBlockFactory

CONCRETE_COORDINATE_ENCODINGS = ("linear", "fourier", "none")


def _resolve_coordinate_encoding(
    coordinate_encoding: Literal["linear", "fourier", "auto", "none"],
    boundary_mode: Literal["periodic", "dirichlet", "neumann"],
) -> str:
    """
    Turn `"auto"` into a concrete encoding. Done once at construction time, so
    that the block itself never has to carry the boundary mode around.
    """
    if coordinate_encoding == "auto":
        return "fourier" if boundary_mode == "periodic" else "linear"
    if coordinate_encoding not in CONCRETE_COORDINATE_ENCODINGS:
        raise ValueError(
            "coordinate_encoding must be one of 'linear', 'fourier', 'auto', "
            f"'none', got {coordinate_encoding}"
        )
    return coordinate_encoding


def _num_coordinate_channels(
    num_spatial_dims: int,
    coordinate_encoding: str,
) -> int:
    if coordinate_encoding == "none":
        return 0
    if coordinate_encoding == "linear":
        return num_spatial_dims
    if coordinate_encoding == "fourier":
        return 2 * num_spatial_dims
    raise ValueError(f"Unknown coordinate_encoding {coordinate_encoding}")


def _coordinate_grid(
    spatial_shape: tuple[int, ...],
    coordinate_encoding: str,
    dtype,
) -> Float[Array, "C ..."]:
    """
    Build the coordinate channels for a given spatial shape.

    The grid is derived from the shape at call time, so this stays agnostic to
    the spatial resolution. Under `jax.jit` it is a compile-time constant.
    """
    if coordinate_encoding == "linear":
        # Endpoints included, matching the reference implementation
        axes = [jnp.linspace(-1.0, 1.0, n, dtype=dtype) for n in spatial_shape]
        return jnp.stack(jnp.meshgrid(*axes, indexing="ij"))

    if coordinate_encoding == "fourier":
        # NOT linspace: with both endpoints included, the first and the last
        # sample would land on the same angle and therefore receive an identical
        # encoding. Stepping by 2/n instead makes one full period span exactly
        # n samples, so the encoding is seamless across a periodic boundary.
        axes = [-1.0 + 2.0 * jnp.arange(n, dtype=dtype) / n for n in spatial_shape]
        grid = jnp.stack(jnp.meshgrid(*axes, indexing="ij"))
        return jnp.concatenate(
            [jnp.sin(jnp.pi * grid), jnp.cos(jnp.pi * grid)],
            axis=0,
        )

    raise ValueError(f"Cannot build a coordinate grid for {coordinate_encoding}")


class CoordinateAugmentationBlock(Block):
    inner: Block
    num_spatial_dims: int = eqx.field(static=True)
    coordinate_encoding: str = eqx.field(static=True)

    def __init__(
        self,
        inner: Block,
        *,
        num_spatial_dims: int,
        coordinate_encoding: Literal["linear", "fourier", "none"],
    ):
        """
        Concatenates coordinate channels to the input, then applies another
        block to the result.

        Since a block is expected to map `in_channels` to `out_channels`, and
        concatenating coordinates only maps `in_channels` to `in_channels + d`,
        this block *wraps* another block rather than standing on its own. The
        wrapped block has to have been constructed for the widened number of
        input channels; use
        [`pdequinox.blocks.CoordinateAugmentationBlockFactory`][], which takes
        care of that bookkeeping.

        **Arguments:**

        - `inner`: The block to apply after concatenating the coordinates. It
            must accept `in_channels + n` channels, where `n` is the number of
            coordinate channels for the chosen encoding.
        - `num_spatial_dims`: The number of spatial dimensions. (Keyword only
            argument.)
        - `coordinate_encoding`: One of `"linear"`, `"fourier"` or `"none"`; see
            the factory for what these mean. Must already be concrete: `"auto"`
            is resolved by the factory. (Keyword only argument.)

        !!! warning

            Handing a network absolute coordinates deliberately **breaks
            translation equivariance**. Every other architecture in this package
            is either translation-equivariant by construction (convolutions) or
            global (spectral). This is useful when the dynamics genuinely are
            not translation invariant - spatially varying media, walls, complex
            geometry - and is likely to be a liability when they are.
        """
        if coordinate_encoding not in CONCRETE_COORDINATE_ENCODINGS:
            raise ValueError(
                "coordinate_encoding must be concrete, i.e. one of 'linear', "
                f"'fourier', 'none', got {coordinate_encoding}. ('auto' is "
                "resolved by CoordinateAugmentationBlockFactory.)"
            )
        self.inner = inner
        self.num_spatial_dims = num_spatial_dims
        self.coordinate_encoding = coordinate_encoding

    def __call__(self, x):
        if len(x.shape) - 1 != self.num_spatial_dims:
            raise ValueError(
                f"Expected an input with {self.num_spatial_dims} spatial "
                f"dimensions, got an array of shape {x.shape}"
            )

        if self.coordinate_encoding == "none":
            return self.inner(x)

        grid = _coordinate_grid(x.shape[1:], self.coordinate_encoding, x.dtype)
        return self.inner(jnp.concatenate([x, grid], axis=0))

    @property
    def receptive_field(self) -> tuple[tuple[float, float], ...]:
        # Concatenating channels contributes no spatial reach of its own.
        return self.inner.receptive_field


class CoordinateAugmentationBlockFactory(BlockFactory):
    inner_factory: BlockFactory
    coordinate_encoding: str

    def __init__(
        self,
        inner_factory: BlockFactory = LinearChannelAdjustBlockFactory(),
        *,
        coordinate_encoding: Literal["linear", "fourier", "auto", "none"] = "linear",
    ):
        """
        Factory for creating `CoordinateAugmentationBlock` instances.

        Widens the number of input channels handed to `inner_factory` by the
        number of coordinate channels, so that from the outside the resulting
        block still maps `in_channels` to `out_channels` and can be dropped into
        a lifting slot.

        With its defaults, this reproduces the lifting layer of the Flower
        architecture: concatenate a linear coordinate ramp, then apply a 1x1
        convolution.

        **Arguments:**

        - `inner_factory`: The factory for the block applied after the
            coordinates have been concatenated. Default is
            `LinearChannelAdjustBlockFactory`, i.e. a 1x1 convolution.
        - `coordinate_encoding`: How to encode positions. (Keyword only
            argument.) One of:

            - `"linear"` (default): a ramp on `[-1, 1]` per axis, adding
                `num_spatial_dims` channels. This is what the reference
                implementation of Flower uses, for every boundary condition.
            - `"fourier"`: `(sin(pi x), cos(pi x))` per axis, adding
                `2 * num_spatial_dims` channels. Unlike the ramp, this is
                continuous across a periodic boundary.
            - `"auto"`: `"fourier"` for a `periodic` boundary mode, `"linear"`
                otherwise.
            - `"none"`: no coordinates at all; the input is passed straight
                through to the inner block.

        !!! info

            The default is `"linear"` rather than `"auto"` so that the Flower
            architecture reproduces its reference implementation out of the box.

            Note though that a ramp on a periodic axis is discontinuous at the
            wrap point, and samples the same physical location at both ends of
            the axis. `"fourier"` avoids both. Whether this actually matters for
            training is untested, which is why it is opt-in.
        """
        self.inner_factory = inner_factory
        self.coordinate_encoding = coordinate_encoding

    def __call__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        activation: Callable,
        boundary_mode: Literal["periodic", "dirichlet", "neumann"],
        key: PRNGKeyArray,
    ):
        coordinate_encoding = _resolve_coordinate_encoding(
            self.coordinate_encoding, boundary_mode
        )
        num_coordinate_channels = _num_coordinate_channels(
            num_spatial_dims, coordinate_encoding
        )

        inner = self.inner_factory(
            num_spatial_dims=num_spatial_dims,
            # The inner block sees the widened input
            in_channels=in_channels + num_coordinate_channels,
            out_channels=out_channels,
            activation=activation,
            boundary_mode=boundary_mode,
            key=key,
        )

        return CoordinateAugmentationBlock(
            inner,
            num_spatial_dims=num_spatial_dims,
            coordinate_encoding=coordinate_encoding,
        )
