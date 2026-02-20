from typing import Callable, Literal

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray

from .._transformer_utils import patchify, to_channel_first, to_channel_last, unpatchify
from ._base_block import Block, BlockFactory


class PatchifyLiftingBlock(Block):
    num_spatial_dims: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    linear: eqx.nn.Linear

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        *,
        key: PRNGKeyArray,
    ):
        self.num_spatial_dims = num_spatial_dims
        self.patch_size = patch_size
        self.linear = eqx.nn.Linear(
            in_features=in_channels * patch_size**num_spatial_dims,
            out_features=out_channels,
            key=key,
        )

    def __call__(self, x):
        # x: (in_channels, *spatial)
        x = to_channel_last(x)  # (*spatial, in_channels)

        # Patchify each channel
        x_patched = jax.vmap(
            lambda _x: patchify(_x, patch_size=self.patch_size),
            in_axes=-1,
            out_axes=-1,
        )(x)
        # x_patched: (*num_patches, *patch_sizes, in_channels)

        # Merge patch content and channels into one dim
        num_patches_shape = tuple(
            x.shape[i] // self.patch_size for i in range(self.num_spatial_dims)
        )
        x_patched = x_patched.reshape(*num_patches_shape, -1)
        # (*num_patches, in_channels * patch_size^d)

        # Apply linear projection vmapped over patch grid
        linear_vmapped = self.linear
        for _ in range(self.num_spatial_dims):
            linear_vmapped = jax.vmap(linear_vmapped)
        x_out = linear_vmapped(x_patched)
        # (*num_patches, out_channels)

        x_out = to_channel_first(x_out)  # (out_channels, *num_patches)
        return x_out

    @property
    def receptive_field(self) -> tuple[tuple[float, float], ...]:
        p = float(self.patch_size - 1)
        return ((p, p),) * self.num_spatial_dims


class PatchifyLiftingBlockFactory(BlockFactory):
    patch_size: int

    def __init__(self, *, patch_size: int = 4):
        self.patch_size = patch_size

    def __call__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        activation: Callable,  # unused
        boundary_mode: Literal["periodic", "dirichlet", "neumann"],  # unused
        key: PRNGKeyArray,
    ):
        return PatchifyLiftingBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=self.patch_size,
            key=key,
        )


class UnpatchifyProjectionBlock(Block):
    num_spatial_dims: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    linear: eqx.nn.Linear

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        *,
        key: PRNGKeyArray,
    ):
        self.num_spatial_dims = num_spatial_dims
        self.patch_size = patch_size
        self.linear = eqx.nn.Linear(
            in_features=in_channels,
            out_features=out_channels * patch_size**num_spatial_dims,
            key=key,
        )

    def __call__(self, x):
        # x: (in_channels, *num_patches)
        x = to_channel_last(x)  # (*num_patches, in_channels)

        # Apply linear projection vmapped over patch grid
        linear_vmapped = self.linear
        for _ in range(self.num_spatial_dims):
            linear_vmapped = jax.vmap(linear_vmapped)
        x = linear_vmapped(x)
        # (*num_patches, out_channels * patch_size^d)

        num_patches_shape = x.shape[: self.num_spatial_dims]
        out_channels = x.shape[-1] // self.patch_size**self.num_spatial_dims

        # Reshape to separate out channels and patch dims
        patch_shape = (self.patch_size,) * self.num_spatial_dims
        x = x.reshape(*num_patches_shape, out_channels, *patch_shape)
        # (*num_patches, out_channels, *patch_sizes)

        # Unpatchify each channel
        x = jax.vmap(unpatchify, in_axes=self.num_spatial_dims, out_axes=-1)(x)
        # (*spatial, out_channels)

        x = to_channel_first(x)  # (out_channels, *spatial)
        return x

    @property
    def receptive_field(self) -> tuple[tuple[float, float], ...]:
        return ((0.0, 0.0),) * self.num_spatial_dims


class UnpatchifyProjectionBlockFactory(BlockFactory):
    patch_size: int

    def __init__(self, *, patch_size: int = 4):
        self.patch_size = patch_size

    def __call__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        activation: Callable,  # unused
        boundary_mode: Literal["periodic", "dirichlet", "neumann"],  # unused
        key: PRNGKeyArray,
    ):
        return UnpatchifyProjectionBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=self.patch_size,
            key=key,
        )
