"""
A collection of sample architectures used in papers on physics-based deep
learning.
"""

from ._classic_fno import ClassicFNO
from ._classic_u_net import ClassicUNet
from ._conv_net import ConvNet
from ._mlp import MLP

__all__ = [
    "ClassicFNO",
    "ConvNet",
    "ClassicUNet",
    "MLP",
]
