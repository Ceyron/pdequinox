from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox._misc import default_floating_dtype
from jaxtyping import Array, Complex, Float, PRNGKeyArray


class RoPENd(eqx.Module):
    """N-dimensional Rotary Positional Embedding."""

    num_spatial_dims: int = eqx.field(static=True)
    embedding_size: int = eqx.field(static=True)
    theta: float = eqx.field(static=True, default=10_000.0)
    dtype: Any = eqx.field(static=True, default_factory=default_floating_dtype)

    def __check_init__(self) -> None:
        if self.embedding_size < 0:
            raise ValueError("`embedding_size` must not be negative.")
        if (self.embedding_size % (2 * self.num_spatial_dims)) != 0:
            raise ValueError(
                f"embedding_size ({self.embedding_size}) must be divisible "
                f"by 2 * num_spatial_dims ({2 * self.num_spatial_dims})"
            )

    @staticmethod
    def precompute_rotations(
        spatial_dims: tuple[int, ...], feature_dim: int, theta: float, dtype: Any
    ) -> Complex[Array, " seq_length_x ... embedding_size/2"]:
        num_spatial_dims = len(spatial_dims)
        k_max = feature_dim // (2 * num_spatial_dims)
        theta_ks = 1.0 / (theta ** (jnp.arange(0, k_max, dtype=jnp.float32) / k_max))
        positions = jnp.meshgrid(
            *[jnp.arange(d) for d in spatial_dims], indexing="ij"
        )
        angles = jnp.concatenate(
            [pos.astype(jnp.float32)[..., None] * theta_ks for pos in positions],
            axis=-1,
        )
        rotations_complex = jnp.exp(1j * angles)
        return rotations_complex

    def __call__(
        self,
        x: Float[Array, " seq_length_x ... embedding_size"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " seq_length_x ... embedding_size"]:
        spatial_dims, feature_dim = x.shape[:-1], x.shape[-1]
        if feature_dim != self.embedding_size:
            raise ValueError(
                f"x.shape[-1] must match self.embedding_size, "
                f"but {feature_dim} != {self.embedding_size}"
            )
        rotations_complex = self.precompute_rotations(
            spatial_dims, feature_dim, self.theta, self.dtype
        )
        orig_dtype = x.dtype
        x = x.reshape(*spatial_dims, -1, 2)
        x = x[..., 0] + 1j * x[..., 1]
        x = rotations_complex * x
        x = jnp.stack([jnp.real(x), jnp.imag(x)], axis=-1)
        return x.reshape(*spatial_dims, feature_dim).astype(orig_dtype)


class SpatialAttention(eqx.Module):
    """Multi-head self-attention over flattened spatial dimensions with RoPE."""

    attention: eqx.nn.MultiheadAttention
    rope_nd: RoPENd
    feed_forward: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(
        self,
        num_spatial_dims: int,
        embedding_size: int,
        num_heads: int,
        *,
        feed_forward_multiplier: int = 4,
        activation=jax.nn.gelu,
        key: PRNGKeyArray,
    ):
        attn_key, ff_key = jax.random.split(key, 2)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=embedding_size,
            key=attn_key,
        )
        self.rope_nd = RoPENd(
            num_spatial_dims=num_spatial_dims,
            embedding_size=embedding_size // num_heads,
        )
        self.feed_forward = eqx.nn.MLP(
            in_size=embedding_size,
            out_size=embedding_size,
            width_size=feed_forward_multiplier * embedding_size,
            depth=1,
            activation=activation,
            key=ff_key,
        )
        self.norm1 = eqx.nn.LayerNorm(embedding_size)
        self.norm2 = eqx.nn.LayerNorm(embedding_size)

    def __call__(
        self,
        x: Float[Array, " seq_length_x ... embedding_size"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " seq_length_x ... embedding_size"]:
        spatial_dims = x.shape[:-1]

        def _flatten(_x):
            return _x.reshape(-1, _x.shape[-1])

        def _unflatten(_x):
            return _x.reshape(*spatial_dims, _x.shape[-1])

        def apply_rope(_x):
            _x = _unflatten(_x)
            _x = self.rope_nd(_x)
            _x = _flatten(_x)
            return _x

        def process_heads(query_heads, key_heads, value_heads):
            query_heads = jax.vmap(apply_rope, in_axes=-2, out_axes=-2)(query_heads)
            key_heads = jax.vmap(apply_rope, in_axes=-2, out_axes=-2)(key_heads)
            return query_heads, key_heads, value_heads

        x = _flatten(x)

        residual = x
        x = jax.vmap(self.norm1)(x)
        x = self.attention(x, x, x, process_heads=process_heads)
        x = x + residual

        residual = x
        x = jax.vmap(self.norm2)(x)
        x = jax.vmap(self.feed_forward)(x)
        x = x + residual

        x = _unflatten(x)
        return x


class AxialSpatialAttention(eqx.Module):
    """1D attention along a single spatial axis with RoPE."""

    attention: eqx.nn.MultiheadAttention
    rope_nd: RoPENd
    feed_forward: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    axis: int = eqx.field(static=True)
    num_spatial_dims: int = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims: int,
        embedding_size: int,
        num_heads: int,
        axis: int,
        *,
        feed_forward_multiplier: int = 4,
        activation=jax.nn.gelu,
        key: PRNGKeyArray,
    ):
        attn_key, ff_key = jax.random.split(key, 2)
        self.axis = axis
        self.num_spatial_dims = num_spatial_dims
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=embedding_size,
            key=attn_key,
        )
        self.rope_nd = RoPENd(
            num_spatial_dims=1,
            embedding_size=embedding_size // num_heads,
        )
        self.feed_forward = eqx.nn.MLP(
            in_size=embedding_size,
            out_size=embedding_size,
            width_size=feed_forward_multiplier * embedding_size,
            depth=1,
            activation=activation,
            key=ff_key,
        )
        self.norm1 = eqx.nn.LayerNorm(embedding_size)
        self.norm2 = eqx.nn.LayerNorm(embedding_size)

    def __call__(
        self,
        x: Float[Array, " *spatial embedding_size"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " *spatial embedding_size"]:
        spatial_dims = x.shape[:-1]
        embedding_size = x.shape[-1]

        # Move target axis to front, flatten other spatial dims into batch
        perm = (
            [self.axis]
            + list(range(0, self.axis))
            + list(range(self.axis + 1, self.num_spatial_dims))
            + [self.num_spatial_dims]
        )
        x = x.transpose(perm)  # (axis_len, *other_spatial, embedding_size)
        axis_len = x.shape[0]
        batch_size = 1
        for d in x.shape[1:-1]:
            batch_size *= d
        x = x.reshape(axis_len, batch_size, embedding_size)

        def _attend_1d(x_slice):
            # x_slice: (axis_len, embedding_size)
            def apply_rope(_x):
                return self.rope_nd(_x)

            def process_heads(query_heads, key_heads, value_heads):
                query_heads = jax.vmap(apply_rope, in_axes=-2, out_axes=-2)(
                    query_heads
                )
                key_heads = jax.vmap(apply_rope, in_axes=-2, out_axes=-2)(key_heads)
                return query_heads, key_heads, value_heads

            residual = x_slice
            x_out = jax.vmap(self.norm1)(x_slice)
            x_out = self.attention(x_out, x_out, x_out, process_heads=process_heads)
            x_out = x_out + residual

            residual = x_out
            x_out = jax.vmap(self.norm2)(x_out)
            x_out = jax.vmap(self.feed_forward)(x_out)
            x_out = x_out + residual
            return x_out

        # vmap over batch dim (axis=1)
        x = jax.vmap(_attend_1d, in_axes=1, out_axes=1)(x)

        # Restore shape
        other_spatial = tuple(spatial_dims[i] for i in range(self.num_spatial_dims) if i != self.axis)
        x = x.reshape(axis_len, *other_spatial, embedding_size)

        # Inverse transpose
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        x = x.transpose(inv_perm)
        return x


def patchify(
    x: Float[Array, " num_points_x ..."],
    patch_size: int,
) -> Array:
    """Reshape spatial dims into (num_patches, patch_size) per dimension.

    Input: (*spatial) -> Output: (*num_patches, *patch_sizes)
    """
    num_spatial_dims = x.ndim

    if num_spatial_dims == 1:
        n = x.shape[0]
        return x.reshape(n // patch_size, patch_size)
    elif num_spatial_dims == 2:
        n0, n1 = x.shape
        return x.reshape(
            n0 // patch_size, patch_size, n1 // patch_size, patch_size
        ).transpose(0, 2, 1, 3)
    elif num_spatial_dims == 3:
        n0, n1, n2 = x.shape
        return x.reshape(
            n0 // patch_size, patch_size,
            n1 // patch_size, patch_size,
            n2 // patch_size, patch_size,
        ).transpose(0, 2, 4, 1, 3, 5)
    else:
        raise ValueError("patchify only supports up to 3 spatial dimensions.")


def unpatchify(x: Array) -> Float[Array, " num_points_x ..."]:
    """Inverse of patchify. Input: (*num_patches, *patch_sizes) -> Output: (*spatial)"""
    num_spatial_dims = x.ndim // 2

    if num_spatial_dims == 1:
        np0, ps0 = x.shape
        return x.reshape(np0 * ps0)
    elif num_spatial_dims == 2:
        np0, np1, ps0, ps1 = x.shape
        return x.transpose(0, 2, 1, 3).reshape(np0 * ps0, np1 * ps1)
    elif num_spatial_dims == 3:
        np0, np1, np2, ps0, ps1, ps2 = x.shape
        return x.transpose(0, 3, 1, 4, 2, 5).reshape(
            np0 * ps0, np1 * ps1, np2 * ps2
        )
    else:
        raise ValueError("unpatchify only supports up to 3 spatial dimensions.")


def to_channel_last(x: Float[Array, "C *spatial"]) -> Float[Array, "*spatial C"]:
    """Transpose from channel-first (C, *spatial) to channel-last (*spatial, C)."""
    num_spatial_dims = x.ndim - 1
    return x.transpose(*range(1, num_spatial_dims + 1), 0)


def to_channel_first(x: Float[Array, "*spatial C"]) -> Float[Array, "C *spatial"]:
    """Transpose from channel-last (*spatial, C) to channel-first (C, *spatial)."""
    num_spatial_dims = x.ndim - 1
    return x.transpose(num_spatial_dims, *range(0, num_spatial_dims))


def boundary_shift(x, shift, axis, boundary_mode):
    """Shift spatial data along an axis respecting boundary conditions."""
    if shift == 0:
        return x
    if boundary_mode == "periodic":
        return jnp.roll(x, shift=-shift, axis=axis)
    mode = "constant" if boundary_mode == "dirichlet" else "reflect"
    pad_width = [(0, 0)] * x.ndim
    if shift > 0:
        pad_width[axis] = (0, shift)
    else:
        pad_width[axis] = (-shift, 0)
    x_padded = jnp.pad(x, pad_width, mode=mode)
    slc = [slice(None)] * x.ndim
    if shift > 0:
        slc[axis] = slice(shift, shift + x.shape[axis])
    else:
        slc[axis] = slice(0, x.shape[axis])
    return x_padded[tuple(slc)]


def boundary_unshift(x, shift, axis, boundary_mode):
    """Reverse the boundary-aware shift."""
    return boundary_shift(x, -shift, axis, boundary_mode)
