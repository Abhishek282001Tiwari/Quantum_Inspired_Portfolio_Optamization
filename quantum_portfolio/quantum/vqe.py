"""
Variational Quantum Eigensolver (VQE) for Portfolio Optimization.

This module implements VQE for solving portfolio optimization problems,
focusing on finding the ground state of the portfolio Hamiltonian.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
try:
    from qiskit.algorithms import VQE
    from qiskit.algorithms.optimizers import COBYLA, SPSA, L_BFGS_B, SLSQP
    from qiskit.circuit.library import TwoLocal, RealAmplitudes, EfficientSU2
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import Estimator
except ImportError:
    # Fallback for different Qiskit versions
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.circuit.library import TwoLocal, RealAmplitudes, EfficientSU2
    from scipy.optimize import minimize
    VQE = None
    COBYLA = None
    SPSA = None
    L_BFGS_B = None
    SLSQP = None
    Estimator = None
import pandas as pd
from scipy.optimize import minimize
from ..utils.portfolio_utils import validate_portfolio_inputs
from ..utils.quantum_utils import create_pauli_operator


class VQEPortfolioOptimizer:
    """
    VQE-based portfolio optimization solver.
    
    This class uses Variational Quantum Eigensolver to find optimal
    portfolio allocations by minimizing the portfolio Hamiltonian.
    """
    
    def __init__(
        self,
        num_assets: int,
        ansatz: str = "RealAmplitudes",
        optimizer: str = "COBYLA",
        max_iterations: int = 1000,
        shots: int = 1024,
        seed: int = 42,
        reps: int = 2
    ):
        """
        Initialize VQE portfolio optimizer.
        
        Args:
            num_assets: Number of assets in the portfolio
            ansatz: Quantum circuit ansatz type
            optimizer: Classical optimizer for parameter optimization
            max_iterations: Maximum iterations for classical optimizer
            shots: Number of quantum circuit shots
            seed: Random seed for reproducibility
            reps: Number of repetitions in the ansatz
        """
        self.num_assets = num_assets
        self.shots = shots
        self.seed = seed
        self.reps = reps
        self.max_iterations = max_iterations
        
        # Set up ansatz
        self.ansatz = self._setup_ansatz(ansatz)
        
        # Set up classical optimizer
        self.optimizer = self._setup_optimizer(optimizer)
        
        # Initialize quantum components
        self.estimator = Estimator()
        
        # Initialize results storage
        self.results = {}
        self.optimization_history = []
        
    def _setup_ansatz(self, ansatz_name: str) -> QuantumCircuit:
        """Set up the variational ansatz."""
        ansatz_map = {
            "RealAmplitudes": RealAmplitudes(self.num_assets, reps=self.reps),
            "EfficientSU2": EfficientSU2(self.num_assets, reps=self.reps),
            "TwoLocal": TwoLocal(
                self.num_assets, 
                ['ry', 'rz'], 
                'cz', 
                reps=self.reps, 
                entanglement='circular'
            )
        }
        
        return ansatz_map.get(ansatz_name, RealAmplitudes(self.num_assets, reps=self.reps))
    
    def _setup_optimizer(self, optimizer_name: str):
        """Set up the classical optimizer for VQE parameters."""
        optimizers = {
            "COBYLA": COBYLA(maxiter=self.max_iterations),
            "SPSA": SPSA(maxiter=self.max_iterations),
            "L_BFGS_B": L_BFGS_B(maxiter=self.max_iterations),
            "SLSQP": SLSQP(maxiter=self.max_iterations)
        }
        return optimizers.get(optimizer_name, COBYLA(maxiter=self.max_iterations))
    
    def create_portfolio_hamiltonian(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 1.0,
        return_weight: float = 1.0,
        constraints: Optional[Dict] = None
    ) -> SparsePauliOp:
        """
        Create the Hamiltonian for portfolio optimization.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            risk_aversion: Risk aversion parameter
            return_weight: Weight for return maximization
            constraints: Additional constraints dict
            
        Returns:
            SparsePauliOp: Quantum Hamiltonian for the portfolio problem
        """
        validate_portfolio_inputs(expected_returns, covariance_matrix)
        
        # Create Pauli operators
        pauli_list = []
        
        # Return terms (to be maximized, so negative coefficients)
        for i in range(self.num_assets):
            pauli_list.append((f"Z_{i}", -return_weight * expected_returns[i]))
        
        # Risk terms (quadratic in weights)
        for i in range(self.num_assets):
            for j in range(i, self.num_assets):
                coeff = risk_aversion * covariance_matrix[i, j]
                if i == j:
                    pauli_list.append((f"I_{i}", coeff))  # Diagonal terms
                else:
                    pauli_list.append((f"Z_{i}Z_{j}", coeff))  # Off-diagonal terms
        
        # Add constraint terms if specified
        if constraints:
            pauli_list.extend(self._add_constraint_terms(constraints))
        
        # Convert to SparsePauliOp
        hamiltonian = SparsePauliOp.from_list(pauli_list)
        
        return hamiltonian
    
    def _add_constraint_terms(self, constraints: Dict) -> List[Tuple[str, float]]:
        """Add constraint terms to the Hamiltonian."""
        constraint_terms = []
        
        # Budget constraint (sum of weights = 1)
        if 'budget' in constraints:
            penalty = constraints['budget'].get('penalty', 10.0)
            target = constraints['budget'].get('target', 1.0)
            
            # Add penalty term: penalty * (sum_i w_i - target)^2
            # This becomes: penalty * (sum_i (1-Z_i)/2 - target)^2
            
            # Linear terms
            for i in range(self.num_assets):
                constraint_terms.append((f"Z_{i}", -penalty * target))
            
            # Quadratic terms
            for i in range(self.num_assets):
                for j in range(i, self.num_assets):
                    coeff = penalty / 4 if i == j else penalty / 2
                    constraint_terms.append((f"Z_{i}Z_{j}", coeff))
            
            # Constant term
            constraint_terms.append(("I", penalty * target**2))
        
        # Cardinality constraint (maximum number of assets)
        if 'cardinality' in constraints:
            penalty = constraints['cardinality'].get('penalty', 5.0)
            max_assets = constraints['cardinality'].get('max_assets', self.num_assets)
            
            # Add penalty for selecting more than max_assets
            for i in range(self.num_assets):
                constraint_terms.append((f"Z_{i}", -penalty))
            
            # Add penalty for pairs beyond limit
            constraint_terms.append(("I", penalty * max_assets))
        
        return constraint_terms
    
    def optimize_portfolio(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 1.0,
        return_weight: float = 1.0,
        constraints: Optional[Dict] = None,
        initial_params: Optional[np.ndarray] = None,
        callback: Optional[Callable] = None
    ) -> Dict:
        """
        Optimize portfolio using VQE.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            risk_aversion: Risk aversion parameter
            return_weight: Weight for return maximization
            constraints: Additional constraints dict
            initial_params: Initial parameters for VQE
            callback: Callback function for optimization progress
            
        Returns:
            Dict: Optimization results including weights and metrics
        """
        # Create Hamiltonian
        hamiltonian = self.create_portfolio_hamiltonian(
            expected_returns, covariance_matrix, risk_aversion, return_weight, constraints
        )
        
        # Initialize parameters
        if initial_params is None:
            initial_params = np.random.uniform(0, 2*np.pi, self.ansatz.num_parameters)
        
        # Set up VQE
        vqe = VQE(
            estimator=self.estimator,
            ansatz=self.ansatz,
            optimizer=self.optimizer,
            initial_point=initial_params,
            callback=callback
        )
        
        # Run optimization
        result = vqe.compute_minimum_eigenvalue(hamiltonian)
        
        # Process results
        self.results = self._process_results(
            result, expected_returns, covariance_matrix, risk_aversion, hamiltonian
        )
        
        return self.results
    
    def _process_results(
        self,
        result,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float,
        hamiltonian: SparsePauliOp
    ) -> Dict:
        """Process VQE optimization results."""
        # Extract optimal parameters
        optimal_params = result.optimal_point
        
        # Get the optimal quantum state
        optimal_circuit = self.ansatz.bind_parameters(optimal_params)
        
        # Extract portfolio weights from quantum state
        # This is a simplified approach - in practice, you'd measure the state
        # and extract weights from the measurement statistics
        weights = self._extract_weights_from_state(optimal_circuit)
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        # Calculate objective value
        objective_value = result.eigenvalue
        
        results = {
            'weights': weights,
            'optimal_parameters': optimal_params,
            'portfolio_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'objective_value': objective_value,
            'optimization_result': result,
            'optimal_circuit': optimal_circuit,
            'hamiltonian': hamiltonian,
            'num_function_evaluations': result.cost_function_evals,
            'convergence_info': {
                'converged': hasattr(result, 'converged') and result.converged,
                'final_gradient': getattr(result, 'final_gradient', None),
                'optimization_time': getattr(result, 'optimization_time', None)
            }
        }
        
        return results
    
    def _extract_weights_from_state(self, circuit: QuantumCircuit) -> np.ndarray:
        """
        Extract portfolio weights from quantum state.
        
        This is a simplified implementation. In practice, you would:
        1. Execute the circuit with measurements
        2. Collect statistics from multiple shots
        3. Convert measurement outcomes to portfolio weights
        """
        # For now, return uniform weights as placeholder
        # In a real implementation, you'd measure the quantum state
        # and extract weights based on measurement probabilities
        
        # Placeholder implementation
        weights = np.ones(self.num_assets) / self.num_assets
        
        # Add some variation based on circuit parameters
        if hasattr(circuit, 'parameters') and len(circuit.parameters) > 0:
            param_values = [p for p in circuit.parameters if hasattr(p, 'value')]
            if param_values:
                # Use parameter values to create weight variations
                variations = np.sin(param_values[:self.num_assets])
                weights = np.abs(variations)
                weights = weights / np.sum(weights)
        
        return weights
    
    def get_ansatz_properties(self) -> Dict:
        """Get properties of the VQE ansatz."""
        return {
            'num_qubits': self.ansatz.num_qubits,
            'num_parameters': self.ansatz.num_parameters,
            'circuit_depth': self.ansatz.depth(),
            'reps': self.reps,
            'ansatz_type': type(self.ansatz).__name__,
            'entanglement': getattr(self.ansatz, 'entanglement', None),
            'rotation_blocks': getattr(self.ansatz, 'rotation_blocks', None),
            'entanglement_blocks': getattr(self.ansatz, 'entanglement_blocks', None)
        }
    
    def analyze_energy_landscape(
        self,
        hamiltonian: SparsePauliOp,
        param_ranges: Optional[Dict] = None,
        resolution: int = 50
    ) -> Dict:
        """
        Analyze the energy landscape of the VQE optimization.
        
        Args:
            hamiltonian: The portfolio Hamiltonian
            param_ranges: Parameter ranges for analysis
            resolution: Resolution for parameter grid
            
        Returns:
            Dict: Energy landscape analysis
        """
        if param_ranges is None:
            param_ranges = {'param_0': (0, 2*np.pi), 'param_1': (0, 2*np.pi)}
        
        # Create parameter grid
        param_names = list(param_ranges.keys())
        param_values = []
        energies = []
        
        for name, (min_val, max_val) in param_ranges.items():
            param_values.append(np.linspace(min_val, max_val, resolution))
        
        # Calculate energy for each parameter combination
        # This is simplified - in practice, you'd do a more sophisticated analysis
        param_grid = np.meshgrid(*param_values[:2])  # Focus on first two parameters
        
        for i in range(resolution):
            for j in range(resolution):
                params = np.zeros(self.ansatz.num_parameters)
                params[0] = param_grid[0][i, j]
                params[1] = param_grid[1][i, j] if len(param_grid) > 1 else 0
                
                # Bind parameters and calculate energy
                circuit = self.ansatz.bind_parameters(params)
                
                # Simplified energy calculation
                energy = np.random.normal(0, 1)  # Placeholder
                energies.append(energy)
        
        return {
            'parameter_grid': param_grid,
            'energies': np.array(energies).reshape(resolution, resolution),
            'param_ranges': param_ranges,
            'resolution': resolution,
            'min_energy': np.min(energies),
            'max_energy': np.max(energies),
            'energy_variance': np.var(energies)
        }
    
    def compare_ansatz_performance(
        self,
        ansatz_list: List[str],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        **kwargs
    ) -> Dict:
        """
        Compare performance of different ansatz types.
        
        Args:
            ansatz_list: List of ansatz types to compare
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            **kwargs: Additional arguments for optimization
            
        Returns:
            Dict: Comparison results
        """
        comparison_results = {}
        
        for ansatz_type in ansatz_list:
            # Create new optimizer with different ansatz
            temp_optimizer = VQEPortfolioOptimizer(
                num_assets=self.num_assets,
                ansatz=ansatz_type,
                optimizer=type(self.optimizer).__name__,
                max_iterations=self.max_iterations,
                shots=self.shots,
                seed=self.seed,
                reps=self.reps
            )
            
            # Optimize with this ansatz
            result = temp_optimizer.optimize_portfolio(
                expected_returns, covariance_matrix, **kwargs
            )
            
            comparison_results[ansatz_type] = {
                'objective_value': result['objective_value'],
                'sharpe_ratio': result['sharpe_ratio'],
                'portfolio_return': result['portfolio_return'],
                'portfolio_risk': result['portfolio_risk'],
                'num_parameters': temp_optimizer.ansatz.num_parameters,
                'circuit_depth': temp_optimizer.ansatz.depth(),
                'function_evaluations': result['num_function_evaluations']
            }
        
        return comparison_results