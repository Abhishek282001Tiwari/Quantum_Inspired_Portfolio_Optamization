"""
Quantum algorithms module for portfolio optimization.

This module implements quantum-inspired algorithms including:
- Quantum Approximate Optimization Algorithm (QAOA)
- Variational Quantum Eigensolver (VQE)
- Quantum Annealing simulation
- Grover's algorithm adaptation
- Quantum-inspired evolutionary algorithms
"""

try:
    from .qaoa import QAOAPortfolioOptimizer
except ImportError:
    from .qaoa_simple import QAOAPortfolioOptimizer

try:
    from .vqe import VQEPortfolioOptimizer
except ImportError:
    VQEPortfolioOptimizer = None

from .quantum_annealing import QuantumAnnealingOptimizer
from .grover import GroverAssetSelector
from .quantum_evolution import QuantumEvolutionaryOptimizer
from .quantum_ml import QuantumMLPortfolio

__all__ = [
    "QAOAPortfolioOptimizer",
    "VQEPortfolioOptimizer", 
    "QuantumAnnealingOptimizer",
    "GroverAssetSelector",
    "QuantumEvolutionaryOptimizer",
    "QuantumMLPortfolio",
]