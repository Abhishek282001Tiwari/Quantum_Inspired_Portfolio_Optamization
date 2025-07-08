"""
Classical portfolio optimization algorithms.

This module implements classical benchmark algorithms for portfolio optimization
including Markowitz mean-variance optimization, Black-Litterman model,
and various risk-based optimization techniques.
"""

from .markowitz import MarkowitzOptimizer

__all__ = [
    "MarkowitzOptimizer",
]