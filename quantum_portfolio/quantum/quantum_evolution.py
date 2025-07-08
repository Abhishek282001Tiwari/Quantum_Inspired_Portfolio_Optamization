"""
Quantum-Inspired Evolutionary Algorithm for Portfolio Optimization.

This module implements quantum-inspired evolutionary algorithms
that use quantum principles for enhanced optimization performance.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import random


class QuantumEvolutionaryOptimizer:
    """
    Quantum-inspired evolutionary optimizer for portfolio problems.
    
    This class implements evolutionary algorithms enhanced with
    quantum-inspired operators for improved search capabilities.
    """
    
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        quantum_rotation_angle: float = 0.01
    ):
        """
        Initialize quantum evolutionary optimizer.
        
        Args:
            population_size: Size of the population
            generations: Number of generations
            mutation_rate: Mutation probability
            quantum_rotation_angle: Quantum rotation angle for updates
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.quantum_rotation_angle = quantum_rotation_angle
    
    def optimize_portfolio(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 1.0
    ) -> Dict:
        """
        Optimize portfolio using quantum-inspired evolution.
        
        Args:
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            risk_aversion: Risk aversion parameter
            
        Returns:
            Dict: Optimization results
        """
        num_assets = len(expected_returns)
        
        # Initialize quantum population
        population = self._initialize_quantum_population(num_assets)
        
        best_individual = None
        best_fitness = float('-inf')
        fitness_history = []
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [
                self._evaluate_fitness(individual, expected_returns, covariance_matrix, risk_aversion)
                for individual in population
            ]
            
            # Track best individual
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_individual = population[max_fitness_idx].copy()
            
            fitness_history.append(best_fitness)
            
            # Quantum-inspired selection and evolution
            population = self._quantum_evolution_step(
                population, fitness_scores, expected_returns, covariance_matrix
            )
        
        # Convert best individual to portfolio weights
        weights = self._decode_individual(best_individual)
        weights = weights / np.sum(weights)  # Normalize
        
        # Calculate final metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        return {
            'weights': weights,
            'portfolio_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'best_fitness': best_fitness,
            'fitness_history': fitness_history,
            'generations': self.generations
        }
    
    def _initialize_quantum_population(self, num_assets: int) -> List[np.ndarray]:
        """Initialize population with quantum-inspired encoding."""
        population = []
        
        for _ in range(self.population_size):
            # Quantum-inspired individual: superposition of basis states
            individual = np.random.uniform(0, 2*np.pi, num_assets)
            population.append(individual)
        
        return population
    
    def _decode_individual(self, individual: np.ndarray) -> np.ndarray:
        """Decode quantum individual to portfolio weights."""
        # Convert quantum angles to probabilities/weights
        weights = np.sin(individual)**2
        return weights
    
    def _evaluate_fitness(
        self,
        individual: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float
    ) -> float:
        """Evaluate fitness of individual."""
        weights = self._decode_individual(individual)
        
        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            return float('-inf')
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.dot(weights, np.dot(covariance_matrix, weights))
        
        # Fitness function: return - risk_aversion * risk
        fitness = portfolio_return - risk_aversion * portfolio_risk
        
        return fitness
    
    def _quantum_evolution_step(
        self,
        population: List[np.ndarray],
        fitness_scores: List[float],
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> List[np.ndarray]:
        """Perform one step of quantum-inspired evolution."""
        new_population = []
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        
        # Elite preservation
        elite_size = max(1, self.population_size // 10)
        for i in range(elite_size):
            new_population.append(population[sorted_indices[i]].copy())
        
        # Quantum-inspired reproduction
        while len(new_population) < self.population_size:
            # Select parents using quantum-inspired selection
            parent1 = self._quantum_selection(population, fitness_scores)
            parent2 = self._quantum_selection(population, fitness_scores)
            
            # Quantum crossover
            child = self._quantum_crossover(parent1, parent2)
            
            # Quantum mutation
            if random.random() < self.mutation_rate:
                child = self._quantum_mutation(child)
            
            new_population.append(child)
        
        return new_population[:self.population_size]
    
    def _quantum_selection(
        self,
        population: List[np.ndarray],
        fitness_scores: List[float]
    ) -> np.ndarray:
        """Quantum-inspired selection operator."""
        # Convert fitness to probabilities
        fitness_array = np.array(fitness_scores)
        
        # Shift to positive values
        min_fitness = np.min(fitness_array)
        if min_fitness < 0:
            fitness_array = fitness_array - min_fitness + 1e-8
        
        # Quantum-inspired probability amplification
        probabilities = fitness_array**2
        probabilities = probabilities / np.sum(probabilities)
        
        # Select individual
        selected_idx = np.random.choice(len(population), p=probabilities)
        return population[selected_idx]
    
    def _quantum_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> np.ndarray:
        """Quantum-inspired crossover operator."""
        # Quantum interference-based crossover
        alpha = np.random.uniform(0, 1)
        
        # Quantum superposition of parents
        child = alpha * parent1 + (1 - alpha) * parent2
        
        # Add quantum phase
        phase = np.random.uniform(0, 2*np.pi, len(child))
        child = child + self.quantum_rotation_angle * np.sin(phase)
        
        return child
    
    def _quantum_mutation(self, individual: np.ndarray) -> np.ndarray:
        """Quantum-inspired mutation operator."""
        mutated = individual.copy()
        
        # Quantum rotation mutation
        for i in range(len(individual)):
            if random.random() < 0.1:  # Mutation probability per gene
                rotation_angle = np.random.normal(0, self.quantum_rotation_angle)
                mutated[i] += rotation_angle
        
        return mutated