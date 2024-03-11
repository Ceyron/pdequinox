"""
Convolution routines for pdequinox, partly sugarcoated on eqx.nn modules.
"""

from ._physics_conv import PhysicsConv, PhysicsConvTranspose
from ._pointwise_linear_conv import PointwiseLinearConv
from ._spectral_conv import SpectralConv

__all__ = [
    "PhysicsConv",
    "PhysicsConvTranspose",
    "PointwiseLinearConv",
    "SpectralConv",
]
