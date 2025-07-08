"""
Quantum Annealing simulation for Portfolio Optimization.

This module implements quantum annealing algorithms for solving 
portfolio optimization problems using Ising model formulations.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from numba import jit
import networkx as nx
from ..utils.portfolio_utils import validate_portfolio_inputs
from ..utils.quantum_utils import ising_to_qubo, qubo_to_ising


class QuantumAnnealingOptimizer:
    """
    Quantum Annealing simulator for portfolio optimization.
    
    This class implements simulated quantum annealing to solve
    portfolio optimization problems formulated as Ising models.
    """
    
    def __init__(
        self,
        num_assets: int,
        temperature_schedule: str = "linear",
        max_iterations: int = 10000,
        initial_temperature: float = 10.0,
        final_temperature: float = 0.01,
        num_reads: int = 100,
        seed: int = 42,
        parallel: bool = True
    ):
        """
        Initialize Quantum Annealing optimizer.
        
        Args:
            num_assets: Number of assets in the portfolio
            temperature_schedule: Temperature cooling schedule
            max_iterations: Maximum annealing iterations
            initial_temperature: Starting temperature
            final_temperature: Final temperature
            num_reads: Number of independent annealing runs
            seed: Random seed for reproducibility
            parallel: Whether to run parallel annealing
        """
        self.num_assets = num_assets
        self.temperature_schedule = temperature_schedule
        self.max_iterations = max_iterations
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.num_reads = num_reads
        self.seed = seed
        self.parallel = parallel
        
        # Set random seed
        np.random.seed(seed)
        
        # Initialize results storage
        self.results = {}
        self.annealing_history = []
        
    def create_ising_model(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 1.0,
        constraints: Optional[Dict] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create Ising model for portfolio optimization.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            risk_aversion: Risk aversion parameter
            constraints: Additional constraints dict
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (h, J) where h is local fields 
                                         and J is coupling matrix
        """
        validate_portfolio_inputs(expected_returns, covariance_matrix)
        
        # Initialize Ising parameters
        h = np.zeros(self.num_assets)  # Local magnetic fields
        J = np.zeros((self.num_assets, self.num_assets))  # Coupling matrix
        
        # Return terms (convert to minimization)
        # Spin variables: s_i ∈ {-1, +1}
        # Portfolio weight: w_i = (1 + s_i) / 2
        # Expected return: sum_i w_i * r_i = sum_i (1 + s_i) * r_i / 2
        
        for i in range(self.num_assets):
            h[i] -= expected_returns[i] / 2  # Negative for maximization
        
        # Risk terms: risk_aversion * sum_ij w_i * Σ_ij * w_j
        # = risk_aversion * sum_ij (1 + s_i)(1 + s_j) * Σ_ij / 4
        # = risk_aversion * sum_ij (1 + s_i + s_j + s_i*s_j) * Σ_ij / 4
        
        for i in range(self.num_assets):
            for j in range(self.num_assets):
                if i == j:
                    # Diagonal terms
                    h[i] += risk_aversion * covariance_matrix[i, i] / 4
                else:
                    # Off-diagonal terms
                    J[i, j] += risk_aversion * covariance_matrix[i, j] / 4
                    h[i] += risk_aversion * covariance_matrix[i, j] / 4
        
        # Add constraint terms
        if constraints:
            h, J = self._add_constraint_terms(h, J, constraints)
        
        return h, J
    
    def _add_constraint_terms(
        self,
        h: np.ndarray,
        J: np.ndarray,
        constraints: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add constraint terms to Ising model."""
        
        # Budget constraint: sum_i w_i = budget
        if 'budget' in constraints:
            penalty = constraints['budget'].get('penalty', 10.0)
            target_budget = constraints['budget'].get('target', 1.0)
            
            # sum_i w_i = sum_i (1 + s_i)/2 = (N + sum_i s_i)/2 = target_budget
            # => sum_i s_i = 2*target_budget - N
            # Penalty: penalty * (sum_i s_i - (2*target_budget - N))^2
            
            target_spin_sum = 2 * target_budget - self.num_assets
            
            # Linear terms
            for i in range(self.num_assets):
                h[i] -= penalty * target_spin_sum
            
            # Quadratic terms
            for i in range(self.num_assets):
                for j in range(i + 1, self.num_assets):
                    J[i, j] += penalty
                    J[j, i] += penalty
        
        # Cardinality constraint: at most k assets selected
        if 'cardinality' in constraints:
            penalty = constraints['cardinality'].get('penalty', 5.0)
            max_assets = constraints['cardinality'].get('max_assets', self.num_assets)
            
            # Number of selected assets: sum_i w_i = sum_i (1 + s_i)/2
            # We want: sum_i (1 + s_i)/2 <= max_assets
            # => sum_i s_i <= 2*max_assets - N
            
            # Use slack variable approach or penalty method
            # For simplicity, add penalty for exceeding limit
            for i in range(self.num_assets):
                h[i] += penalty / max_assets
        
        return h, J
    
    def temperature_schedule_func(self, iteration: int) -> float:
        """
        Calculate temperature at given iteration.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            float: Temperature at this iteration
        """
        progress = iteration / self.max_iterations
        
        if self.temperature_schedule == "linear":
            return self.initial_temperature * (1 - progress) + self.final_temperature * progress
        elif self.temperature_schedule == "exponential":
            return self.initial_temperature * np.exp(-5 * progress)
        elif self.temperature_schedule == "logarithmic":
            return self.initial_temperature / (1 + np.log(1 + iteration))
        elif self.temperature_schedule == "power":
            return self.initial_temperature * (1 - progress)**2
        else:
            return self.initial_temperature * (1 - progress) + self.final_temperature * progress
    
    def _calculate_energy(
        self,
        spins: np.ndarray,
        h: np.ndarray,
        J: np.ndarray
    ) -> float:
        """Calculate energy of spin configuration."""
        energy = 0.0
        n = len(spins)
        
        # Local field terms
        for i in range(n):
            energy += h[i] * spins[i]
        
        # Coupling terms
        for i in range(n):
            for j in range(i + 1, n):
                energy += J[i, j] * spins[i] * spins[j]
        
        return energy
    
    def simulated_annealing(
        self,
        h: np.ndarray,
        J: np.ndarray,
        initial_state: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Perform simulated annealing optimization.
        
        Args:
            h: Local magnetic fields
            J: Coupling matrix
            initial_state: Initial spin configuration
            
        Returns:
            Dict: Annealing results
        """
        # Initialize spin configuration
        if initial_state is None:
            spins = np.random.choice([-1, 1], size=self.num_assets)
        else:
            spins = initial_state.copy()
        
        best_spins = spins.copy()
        current_energy = self._calculate_energy(spins, h, J)
        best_energy = current_energy
        
        # Store annealing history
        energies = []
        temperatures = []
        acceptance_rates = []
        
        accepted_moves = 0
        
        for iteration in range(self.max_iterations):
            temperature = self.temperature_schedule_func(iteration)
            
            # Propose random spin flip
            flip_index = np.random.randint(0, self.num_assets)
            old_spin = spins[flip_index]
            new_spin = -old_spin
            
            # Calculate energy change
            energy_change = 0.0
            
            # Local field contribution
            energy_change += h[flip_index] * (new_spin - old_spin)
            
            # Coupling contributions
            for j in range(self.num_assets):
                if j != flip_index:
                    energy_change += J[flip_index, j] * spins[j] * (new_spin - old_spin)
            
            # Accept or reject move
            if energy_change < 0 or np.random.random() < np.exp(-energy_change / temperature):
                spins[flip_index] = new_spin
                current_energy += energy_change
                accepted_moves += 1
                
                # Update best solution
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_spins = spins.copy()
            
            # Store history
            if iteration % 100 == 0:
                energies.append(current_energy)
                temperatures.append(temperature)
                acceptance_rates.append(accepted_moves / (iteration + 1))
        
        return {
            'best_spins': best_spins,
            'best_energy': best_energy,
            'final_spins': spins,
            'final_energy': current_energy,
            'energy_history': energies,
            'temperature_history': temperatures,
            'acceptance_history': acceptance_rates,
            'total_accepted': accepted_moves,
            'acceptance_rate': accepted_moves / self.max_iterations
        }
    
    def optimize_portfolio(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 1.0,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """
        Optimize portfolio using quantum annealing.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            risk_aversion: Risk aversion parameter
            constraints: Additional constraints dict
            
        Returns:
            Dict: Optimization results
        """
        # Create Ising model
        h, J = self.create_ising_model(
            expected_returns, covariance_matrix, risk_aversion, constraints
        )
        
        # Perform multiple annealing runs
        all_results = []
        
        for run in range(self.num_reads):
            # Set different random seed for each run
            np.random.seed(self.seed + run)
            
            # Perform annealing
            result = self.simulated_annealing(h, J)
            all_results.append(result)
        
        # Find best result across all runs
        best_result = min(all_results, key=lambda x: x['best_energy'])
        
        # Process results
        self.results = self._process_results(
            best_result, all_results, expected_returns, covariance_matrix, risk_aversion
        )
        
        return self.results
    
    def _process_results(
        self,
        best_result: Dict,
        all_results: List[Dict],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float
    ) -> Dict:
        """Process annealing results."""
        # Convert spins to portfolio weights
        best_spins = best_result['best_spins']
        weights = (1 + best_spins) / 2  # Convert from {-1, +1} to {0, 1}
        
        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        # Statistics across all runs
        best_energies = [result['best_energy'] for result in all_results]
        final_energies = [result['final_energy'] for result in all_results]
        
        results = {
            'weights': weights,
            'selected_assets': weights > 1e-6,
            'portfolio_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'best_energy': best_result['best_energy'],
            'best_spins': best_spins,
            'annealing_statistics': {
                'num_reads': self.num_reads,
                'best_energy_mean': np.mean(best_energies),
                'best_energy_std': np.std(best_energies),
                'best_energy_min': np.min(best_energies),
                'best_energy_max': np.max(best_energies),
                'final_energy_mean': np.mean(final_energies),
                'final_energy_std': np.std(final_energies),
                'success_rate': sum(1 for e in best_energies if e == min(best_energies)) / len(best_energies)
            },
            'optimization_parameters': {
                'max_iterations': self.max_iterations,
                'temperature_schedule': self.temperature_schedule,
                'initial_temperature': self.initial_temperature,
                'final_temperature': self.final_temperature,
                'num_reads': self.num_reads
            },
            'all_results': all_results
        }
        
        return results
    
    def analyze_solution_landscape(
        self,
        h: np.ndarray,
        J: np.ndarray,
        num_samples: int = 1000
    ) -> Dict:
        """
        Analyze the solution landscape of the Ising model.
        
        Args:
            h: Local magnetic fields
            J: Coupling matrix
            num_samples: Number of random configurations to sample
            
        Returns:
            Dict: Landscape analysis results
        """
        energies = []
        configurations = []
        
        for _ in range(num_samples):
            # Generate random spin configuration
            spins = np.random.choice([-1, 1], size=self.num_assets)
            energy = self._calculate_energy(spins, h, J)
            
            energies.append(energy)
            configurations.append(spins.copy())
        
        energies = np.array(energies)
        
        # Find energy statistics
        min_energy = np.min(energies)
        max_energy = np.max(energies)
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)
        
        # Find ground state configurations
        ground_state_energy = min_energy
        ground_states = [configs for configs, energy in zip(configurations, energies) 
                        if abs(energy - ground_state_energy) < 1e-10]
        
        return {
            'energy_statistics': {
                'min': min_energy,
                'max': max_energy,
                'mean': mean_energy,
                'std': std_energy,
                'median': np.median(energies),
                'quantiles': np.percentile(energies, [25, 50, 75, 90, 95, 99])
            },
            'ground_states': ground_states,
            'num_ground_states': len(ground_states),
            'ground_state_degeneracy': len(ground_states) / num_samples,
            'energy_gap': np.sort(np.unique(energies))[1] - ground_state_energy if len(np.unique(energies)) > 1 else 0,
            'all_energies': energies,
            'all_configurations': configurations
        }
    
    def create_interaction_graph(self, J: np.ndarray, threshold: float = 1e-6) -> nx.Graph:
        """
        Create interaction graph from coupling matrix.
        
        Args:
            J: Coupling matrix
            threshold: Threshold for edge creation
            
        Returns:
            nx.Graph: Interaction graph
        """
        G = nx.Graph()
        
        # Add nodes
        for i in range(self.num_assets):
            G.add_node(i)
        
        # Add edges based on coupling strength
        for i in range(self.num_assets):
            for j in range(i + 1, self.num_assets):
                if abs(J[i, j]) > threshold:
                    G.add_edge(i, j, weight=abs(J[i, j]))
        
        return G
    
    def get_annealing_schedule_analysis(self) -> Dict:
        """Analyze the annealing temperature schedule."""
        iterations = np.arange(self.max_iterations)
        temperatures = [self.temperature_schedule_func(i) for i in iterations]
        
        return {
            'schedule_type': self.temperature_schedule,
            'iterations': iterations,
            'temperatures': temperatures,
            'initial_temperature': self.initial_temperature,
            'final_temperature': self.final_temperature,
            'cooling_rate': (self.initial_temperature - self.final_temperature) / self.max_iterations,
            'effective_cooling_steps': np.sum(np.diff(temperatures) < 0)
        }