"""
A collection of sample architectures used in papers on physics-based deep
learning.
"""

from ._classic_fno import ClassicFNO
from ._classic_res_net import ClassicResNet
from ._classic_u_net import ClassicUNet
from ._conv_net import ConvNet
from ._dilated_res_net import DilatedResNet
from ._mlp import MLP
from ._modern_res_net import ModernResNet

__all__ = [
    "ClassicFNO",
    "ClassicResNet",
    "ConvNet",
    "DilatedResNet",
    "ClassicUNet",
    "MLP",
    "ModernResNet",
]
