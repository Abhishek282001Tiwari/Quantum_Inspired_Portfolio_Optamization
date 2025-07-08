"""
Quantum Approximate Optimization Algorithm (QAOA) for Portfolio Optimization.

This module implements QAOA for solving portfolio optimization problems,
including cardinality constraints and multi-objective optimization.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
try:
    from qiskit.algorithms import QAOA
    from qiskit.algorithms.optimizers import COBYLA, SPSA, L_BFGS_B
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import Sampler
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_optimization.converters import QuadraticProgramToQubo
    HAS_QISKIT_OPTIMIZATION = True
except ImportError:
    # Fallback for different Qiskit versions
    from qiskit.quantum_info import SparsePauliOp
    from scipy.optimize import minimize
    QAOA = None
    COBYLA = None
    SPSA = None
    L_BFGS_B = None
    QuadraticProgram = None
    MinimumEigenOptimizer = None
    QuadraticProgramToQubo = None
    Sampler = None
    HAS_QISKIT_OPTIMIZATION = False
import pandas as pd
from scipy.optimize import minimize
from ..utils.portfolio_utils import validate_portfolio_inputs
from ..utils.quantum_utils import create_pauli_operator, encode_constraints


class QAOAPortfolioOptimizer:
    """
    QAOA-based portfolio optimization solver.
    
    This class implements the Quantum Approximate Optimization Algorithm
    to solve portfolio optimization problems with various constraints.
    """
    
    def __init__(
        self,
        num_assets: int,
        p_layers: int = 2,
        optimizer: str = "COBYLA",
        max_iterations: int = 1000,
        shots: int = 1024,
        seed: int = 42
    ):
        """
        Initialize QAOA portfolio optimizer.
        
        Args:
            num_assets: Number of assets in the portfolio
            p_layers: Number of QAOA layers (depth)
            optimizer: Classical optimizer for parameter optimization
            max_iterations: Maximum iterations for classical optimizer
            shots: Number of quantum circuit shots
            seed: Random seed for reproducibility
        """
        self.num_assets = num_assets
        self.p_layers = p_layers
        self.shots = shots
        self.seed = seed
        self.max_iterations = max_iterations
        
        # Initialize quantum components
        self.qubits = QuantumRegister(num_assets, 'q')
        self.cbits = ClassicalRegister(num_assets, 'c')
        
        # Set up classical optimizer
        self.optimizer = self._setup_optimizer(optimizer)
        
        # Initialize results storage
        self.results = {}
        self.optimization_history = []
        
    def _setup_optimizer(self, optimizer_name: str):
        """Set up the classical optimizer for QAOA parameters."""
        optimizers = {
            "COBYLA": COBYLA(maxiter=self.max_iterations),
            "SPSA": SPSA(maxiter=self.max_iterations),
            "L_BFGS_B": L_BFGS_B(maxiter=self.max_iterations)
        }
        return optimizers.get(optimizer_name, COBYLA(maxiter=self.max_iterations))
    
    def create_portfolio_hamiltonian(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 1.0,
        cardinality_constraint: Optional[int] = None
    ) -> SparsePauliOp:
        """
        Create the Hamiltonian for portfolio optimization.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            risk_aversion: Risk aversion parameter
            cardinality_constraint: Maximum number of assets to select
            
        Returns:
            SparsePauliOp: Quantum Hamiltonian for the portfolio problem
        """
        # Validate inputs
        validate_portfolio_inputs(expected_returns, covariance_matrix)
        
        # Create quadratic program
        qp = QuadraticProgram()
        
        # Add binary variables for asset selection
        for i in range(self.num_assets):
            qp.binary_var(f'x_{i}')
        
        # Objective: maximize return - risk_aversion * risk
        # Convert to minimization: minimize -return + risk_aversion * risk
        linear_coeffs = -expected_returns  # Negative for maximization
        quadratic_coeffs = risk_aversion * covariance_matrix
        
        qp.minimize(
            linear=linear_coeffs,
            quadratic=quadratic_coeffs
        )
        
        # Add cardinality constraint if specified
        if cardinality_constraint:
            qp.linear_constraint(
                linear=[1] * self.num_assets,
                sense='<=',
                rhs=cardinality_constraint,
                name='cardinality'
            )
        
        # Convert to QUBO
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)
        
        # Convert to Pauli operator
        hamiltonian = create_pauli_operator(qubo)
        
        return hamiltonian
    
    def create_qaoa_circuit(
        self,
        hamiltonian: SparsePauliOp,
        parameters: np.ndarray
    ) -> QuantumCircuit:
        """
        Create QAOA quantum circuit.
        
        Args:
            hamiltonian: Problem Hamiltonian
            parameters: QAOA parameters [gamma, beta] for each layer
            
        Returns:
            QuantumCircuit: QAOA circuit
        """
        circuit = QuantumCircuit(self.qubits, self.cbits)
        
        # Initialize in superposition
        circuit.h(self.qubits)
        
        # Apply QAOA layers
        for p in range(self.p_layers):
            gamma = parameters[p]
            beta = parameters[p + self.p_layers]
            
            # Apply problem Hamiltonian (cost layer)
            self._apply_cost_layer(circuit, hamiltonian, gamma)
            
            # Apply mixer Hamiltonian (mixer layer)
            self._apply_mixer_layer(circuit, beta)
        
        # Measure
        circuit.measure(self.qubits, self.cbits)
        
        return circuit
    
    def _apply_cost_layer(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        gamma: float
    ):
        """Apply the cost layer of QAOA."""
        # This is a simplified implementation
        # In practice, you would decompose the Hamiltonian into Pauli gates
        for i in range(self.num_assets):
            circuit.rz(2 * gamma, self.qubits[i])
            
        # Add two-qubit interactions based on covariance
        for i in range(self.num_assets):
            for j in range(i + 1, self.num_assets):
                circuit.rzz(2 * gamma, self.qubits[i], self.qubits[j])
    
    def _apply_mixer_layer(self, circuit: QuantumCircuit, beta: float):
        """Apply the mixer layer of QAOA."""
        for i in range(self.num_assets):
            circuit.rx(2 * beta, self.qubits[i])
    
    def optimize_portfolio(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 1.0,
        cardinality_constraint: Optional[int] = None,
        initial_params: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Optimize portfolio using QAOA.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            risk_aversion: Risk aversion parameter
            cardinality_constraint: Maximum number of assets to select
            initial_params: Initial parameters for QAOA
            
        Returns:
            Dict: Optimization results including weights and metrics
        """
        # Create Hamiltonian
        hamiltonian = self.create_portfolio_hamiltonian(
            expected_returns, covariance_matrix, risk_aversion, cardinality_constraint
        )
        
        # Initialize parameters
        if initial_params is None:
            initial_params = np.random.uniform(0, 2*np.pi, 2 * self.p_layers)
        
        # Set up QAOA algorithm
        sampler = Sampler()
        qaoa = QAOA(
            sampler=sampler,
            optimizer=self.optimizer,
            reps=self.p_layers,
            initial_point=initial_params
        )
        
        # Solve the optimization problem
        minimum_eigen_optimizer = MinimumEigenOptimizer(qaoa)
        
        # Create the quadratic program for the optimizer
        qp = self._create_quadratic_program(
            expected_returns, covariance_matrix, risk_aversion, cardinality_constraint
        )
        
        # Optimize
        result = minimum_eigen_optimizer.solve(qp)
        
        # Process results
        self.results = self._process_results(
            result, expected_returns, covariance_matrix, risk_aversion
        )
        
        return self.results
    
    def _create_quadratic_program(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float,
        cardinality_constraint: Optional[int]
    ) -> QuadraticProgram:
        """Create quadratic program for optimization."""
        qp = QuadraticProgram()
        
        # Add variables
        for i in range(self.num_assets):
            qp.binary_var(f'x_{i}')
        
        # Add objective
        linear_coeffs = -expected_returns
        quadratic_coeffs = risk_aversion * covariance_matrix
        
        qp.minimize(linear=linear_coeffs, quadratic=quadratic_coeffs)
        
        # Add constraints
        if cardinality_constraint:
            qp.linear_constraint(
                linear=[1] * self.num_assets,
                sense='<=',
                rhs=cardinality_constraint,
                name='cardinality'
            )
        
        return qp
    
    def _process_results(
        self,
        result,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float
    ) -> Dict:
        """Process optimization results."""
        # Extract solution
        solution = result.x if hasattr(result, 'x') else result.samples[0].x
        
        # Convert to portfolio weights
        weights = np.array([solution[f'x_{i}'] for i in range(self.num_assets)])
        
        # Normalize weights if needed
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        # Count selected assets
        num_selected = np.sum(weights > 1e-6)
        
        results = {
            'weights': weights,
            'selected_assets': weights > 1e-6,
            'portfolio_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'num_selected_assets': num_selected,
            'optimization_value': result.fval if hasattr(result, 'fval') else None,
            'quantum_result': result
        }
        
        return results
    
    def get_circuit_properties(self) -> Dict:
        """Get properties of the QAOA circuit."""
        # Create a dummy circuit to analyze
        dummy_hamiltonian = SparsePauliOp.from_list([("Z", 1.0)])
        dummy_params = np.ones(2 * self.p_layers)
        circuit = self.create_qaoa_circuit(dummy_hamiltonian, dummy_params)
        
        return {
            'num_qubits': circuit.num_qubits,
            'num_parameters': len(dummy_params),
            'circuit_depth': circuit.depth(),
            'num_layers': self.p_layers,
            'gate_count': len(circuit.data)
        }
    
    def analyze_quantum_advantage(
        self,
        classical_results: Dict,
        tolerance: float = 1e-3
    ) -> Dict:
        """
        Analyze quantum advantage over classical methods.
        
        Args:
            classical_results: Results from classical optimization
            tolerance: Tolerance for comparing results
            
        Returns:
            Dict: Quantum advantage analysis
        """
        if not self.results:
            raise ValueError("No quantum results available. Run optimization first.")
        
        quantum_obj = self.results['optimization_value']
        classical_obj = classical_results.get('optimization_value', float('inf'))
        
        advantage = {
            'quantum_objective': quantum_obj,
            'classical_objective': classical_obj,
            'improvement': classical_obj - quantum_obj if quantum_obj else 0,
            'relative_improvement': (classical_obj - quantum_obj) / abs(classical_obj) if classical_obj != 0 else 0,
            'quantum_superior': quantum_obj < classical_obj - tolerance if quantum_obj else False,
            'quantum_sharpe': self.results['sharpe_ratio'],
            'classical_sharpe': classical_results.get('sharpe_ratio', 0),
            'sharpe_improvement': self.results['sharpe_ratio'] - classical_results.get('sharpe_ratio', 0)
        }
        
        return advantage