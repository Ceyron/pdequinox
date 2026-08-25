"""
A collection of sample architectures used in papers on physics-based deep
learning.
"""

from ._classic_fno import ClassicFNO
from ._classic_res_net import ClassicResNet
from ._classic_u_net import ClassicUNet
from ._conv_net import ConvNet
from ._dilated_res_net import DilatedResNet
from ._flower import Flower
from ._mlp import MLP
from ._modern_res_net import ModernResNet
from ._modern_u_net import ModernUNet
from ._sequential_flower import SequentialFlower

__all__ = [
    "ClassicFNO",
    "ClassicResNet",
    "ConvNet",
    "DilatedResNet",
    "ClassicUNet",
    "Flower",
    "MLP",
    "ModernResNet",
    "ModernUNet",
    "SequentialFlower",
]
