"""
Quantum-Inspired Portfolio Optimization System

A comprehensive platform for applying quantum computing principles to portfolio management problems.
This package provides quantum-inspired algorithms, classical benchmarks, and comprehensive
analysis tools for cutting-edge portfolio optimization research.

Author: Research Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "research@example.com"

from .quantum import *
from .classical import *
from .data import *
from .utils import *

__all__ = [
    "quantum",
    "classical", 
    "data",
    "utils",
]