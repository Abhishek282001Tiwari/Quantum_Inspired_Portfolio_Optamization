"""
Markowitz Mean-Variance Optimization.

This module implements the classic Markowitz mean-variance optimization
for portfolio construction, including efficient frontier generation
and various constraint handling.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint, Bounds
from scipy.linalg import inv, pinv
import cvxpy as cp
from ..utils.portfolio_utils import validate_portfolio_inputs, calculate_portfolio_metrics
import warnings


class MarkowitzOptimizer:
    """
    Markowitz mean-variance portfolio optimizer.
    
    This class implements the classical Markowitz optimization framework
    for portfolio construction based on mean-variance optimization.
    """
    
    def __init__(
        self,
        risk_aversion: float = 1.0,
        allow_short_selling: bool = False,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        solver: str = "CVXPY",
        verbose: bool = False
    ):
        """
        Initialize Markowitz optimizer.
        
        Args:
            risk_aversion: Risk aversion parameter (λ)
            allow_short_selling: Whether to allow short selling
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            solver: Optimization solver to use
            verbose: Whether to print optimization details
        """
        self.risk_aversion = risk_aversion
        self.allow_short_selling = allow_short_selling
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.solver = solver
        self.verbose = verbose
        
        # Results storage
        self.results = {}
        self.optimization_history = []
    
    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """
        Optimize portfolio using Markowitz mean-variance optimization.
        
        Args:
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix of returns
            target_return: Target portfolio return (optional)
            target_risk: Target portfolio risk (optional)
            constraints: Additional constraints dictionary
            
        Returns:
            Dict: Optimization results
        """
        validate_portfolio_inputs(expected_returns, covariance_matrix)
        
        if self.solver == "CVXPY":
            results = self._optimize_cvxpy(
                expected_returns, covariance_matrix, target_return, target_risk, constraints
            )
        else:
            results = self._optimize_scipy(
                expected_returns, covariance_matrix, target_return, target_risk, constraints
            )
        
        self.results = results
        return results
    
    def _optimize_cvxpy(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        target_return: Optional[float],
        target_risk: Optional[float],
        constraints: Optional[Dict]
    ) -> Dict:
        """Optimize using CVXPY."""
        n_assets = len(expected_returns)
        
        # Define optimization variables
        weights = cp.Variable(n_assets)
        
        # Define objective function
        portfolio_return = expected_returns.T @ weights
        portfolio_risk = cp.quad_form(weights, covariance_matrix)
        
        if target_return is not None:
            # Minimize risk for target return
            objective = cp.Minimize(portfolio_risk)
        elif target_risk is not None:
            # Maximize return for target risk
            objective = cp.Maximize(portfolio_return)
        else:
            # Standard mean-variance optimization
            objective = cp.Maximize(portfolio_return - 0.5 * self.risk_aversion * portfolio_risk)
        
        # Define constraints
        constraint_list = []
        
        # Budget constraint
        constraint_list.append(cp.sum(weights) == 1)
        
        # Box constraints
        if self.allow_short_selling:
            constraint_list.append(weights >= -self.max_weight)
        else:
            constraint_list.append(weights >= self.min_weight)
        
        constraint_list.append(weights <= self.max_weight)
        
        # Target return constraint
        if target_return is not None:
            constraint_list.append(portfolio_return >= target_return)
        
        # Target risk constraint
        if target_risk is not None:
            constraint_list.append(portfolio_risk <= target_risk**2)
        
        # Additional constraints
        if constraints:
            constraint_list.extend(self._add_cvxpy_constraints(weights, constraints))
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraint_list)
        
        try:
            problem.solve(verbose=self.verbose)
            
            if problem.status not in ["infeasible", "unbounded"]:
                optimal_weights = weights.value
                
                # Calculate portfolio metrics
                portfolio_metrics = calculate_portfolio_metrics(
                    optimal_weights, expected_returns, covariance_matrix
                )
                
                results = {
                    'weights': optimal_weights,
                    'status': problem.status,
                    'objective_value': problem.value,
                    'portfolio_return': portfolio_metrics['return'],
                    'portfolio_risk': portfolio_metrics['volatility'],
                    'sharpe_ratio': portfolio_metrics['sharpe_ratio'],
                    'effective_assets': portfolio_metrics['effective_assets'],
                    'concentration_ratio': portfolio_metrics['concentration_ratio'],
                    'optimization_method': 'CVXPY',
                    'solver_info': {
                        'solver_name': problem.solver_stats.solver_name,
                        'solve_time': problem.solver_stats.solve_time,
                        'num_iters': problem.solver_stats.num_iters
                    }
                }
                
            else:
                results = {
                    'weights': None,
                    'status': problem.status,
                    'error': 'Optimization failed',
                    'optimization_method': 'CVXPY'
                }
                
        except Exception as e:
            results = {
                'weights': None,
                'status': 'error',
                'error': str(e),
                'optimization_method': 'CVXPY'
            }
        
        return results
    
    def _optimize_scipy(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        target_return: Optional[float],
        target_risk: Optional[float],
        constraints: Optional[Dict]
    ) -> Dict:
        """Optimize using SciPy."""
        n_assets = len(expected_returns)
        
        # Define objective function
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            
            if target_return is not None:
                return portfolio_risk**2  # Minimize risk
            elif target_risk is not None:
                return -portfolio_return  # Maximize return
            else:
                return -(portfolio_return - 0.5 * self.risk_aversion * portfolio_risk**2)
        
        # Define constraints
        constraint_list = []
        
        # Budget constraint
        constraint_list.append({
            'type': 'eq',
            'fun': lambda weights: np.sum(weights) - 1
        })
        
        # Target return constraint
        if target_return is not None:
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda weights: np.dot(weights, expected_returns) - target_return
            })
        
        # Target risk constraint
        if target_risk is not None:
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda weights: target_risk**2 - np.dot(weights, np.dot(covariance_matrix, weights))
            })
        
        # Additional constraints
        if constraints:
            constraint_list.extend(self._add_scipy_constraints(constraints))
        
        # Define bounds
        if self.allow_short_selling:
            bounds = [(-self.max_weight, self.max_weight) for _ in range(n_assets)]
        else:
            bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets
        
        # Solve optimization
        try:
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_list,
                options={'disp': self.verbose}
            )
            
            if result.success:
                optimal_weights = result.x
                
                # Calculate portfolio metrics
                portfolio_metrics = calculate_portfolio_metrics(
                    optimal_weights, expected_returns, covariance_matrix
                )
                
                results = {
                    'weights': optimal_weights,
                    'status': 'optimal',
                    'objective_value': result.fun,
                    'portfolio_return': portfolio_metrics['return'],
                    'portfolio_risk': portfolio_metrics['volatility'],
                    'sharpe_ratio': portfolio_metrics['sharpe_ratio'],
                    'effective_assets': portfolio_metrics['effective_assets'],
                    'concentration_ratio': portfolio_metrics['concentration_ratio'],
                    'optimization_method': 'SciPy',
                    'solver_info': {
                        'nit': result.nit,
                        'nfev': result.nfev,
                        'success': result.success,
                        'message': result.message
                    }
                }
                
            else:
                results = {
                    'weights': None,
                    'status': 'failed',
                    'error': result.message,
                    'optimization_method': 'SciPy'
                }
                
        except Exception as e:
            results = {
                'weights': None,
                'status': 'error',
                'error': str(e),
                'optimization_method': 'SciPy'
            }
        
        return results
    
    def _add_cvxpy_constraints(self, weights, constraints: Dict) -> List:
        """Add additional constraints for CVXPY optimization."""
        constraint_list = []
        
        # Turnover constraint
        if 'turnover' in constraints:
            current_weights = constraints['turnover']['current_weights']
            max_turnover = constraints['turnover']['max_turnover']
            
            turnover = cp.norm(weights - current_weights, 1)
            constraint_list.append(turnover <= max_turnover)
        
        # Sector constraints
        if 'sector' in constraints:
            sector_matrix = constraints['sector']['sector_matrix']
            sector_bounds = constraints['sector']['sector_bounds']
            
            for i, (min_bound, max_bound) in enumerate(sector_bounds):
                sector_weight = sector_matrix[i, :] @ weights
                if min_bound is not None:
                    constraint_list.append(sector_weight >= min_bound)
                if max_bound is not None:
                    constraint_list.append(sector_weight <= max_bound)
        
        # Cardinality constraint (approximation)
        if 'cardinality' in constraints:
            max_assets = constraints['cardinality']['max_assets']
            # This is a relaxation - exact cardinality requires integer programming
            constraint_list.append(cp.norm(weights, 1) <= max_assets * self.max_weight)
        
        return constraint_list
    
    def _add_scipy_constraints(self, constraints: Dict) -> List:
        """Add additional constraints for SciPy optimization."""
        constraint_list = []
        
        # Turnover constraint
        if 'turnover' in constraints:
            current_weights = constraints['turnover']['current_weights']
            max_turnover = constraints['turnover']['max_turnover']
            
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda weights: max_turnover - np.sum(np.abs(weights - current_weights))
            })
        
        # Sector constraints
        if 'sector' in constraints:
            sector_matrix = constraints['sector']['sector_matrix']
            sector_bounds = constraints['sector']['sector_bounds']
            
            for i, (min_bound, max_bound) in enumerate(sector_bounds):
                if min_bound is not None:
                    constraint_list.append({
                        'type': 'ineq',
                        'fun': lambda weights, i=i: np.dot(sector_matrix[i, :], weights) - min_bound
                    })
                if max_bound is not None:
                    constraint_list.append({
                        'type': 'ineq',
                        'fun': lambda weights, i=i: max_bound - np.dot(sector_matrix[i, :], weights)
                    })
        
        return constraint_list
    
    def generate_efficient_frontier(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        num_points: int = 100,
        return_range: Optional[Tuple[float, float]] = None
    ) -> Dict:
        """
        Generate efficient frontier.
        
        Args:
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            num_points: Number of points on the frontier
            return_range: Range of returns to consider
            
        Returns:
            Dict: Efficient frontier data
        """
        validate_portfolio_inputs(expected_returns, covariance_matrix)
        
        if return_range is None:
            min_return = np.min(expected_returns)
            max_return = np.max(expected_returns)
        else:
            min_return, max_return = return_range
        
        target_returns = np.linspace(min_return, max_return, num_points)
        
        frontier_data = {
            'returns': [],
            'risks': [],
            'weights': [],
            'sharpe_ratios': []
        }
        
        for target_return in target_returns:
            try:
                result = self.optimize(
                    expected_returns,
                    covariance_matrix,
                    target_return=target_return
                )
                
                if result['weights'] is not None:
                    frontier_data['returns'].append(result['portfolio_return'])
                    frontier_data['risks'].append(result['portfolio_risk'])
                    frontier_data['weights'].append(result['weights'])
                    frontier_data['sharpe_ratios'].append(result['sharpe_ratio'])
                
            except Exception as e:
                if self.verbose:
                    warnings.warn(f"Failed to optimize for target return {target_return}: {e}")
                continue
        
        return {
            'returns': np.array(frontier_data['returns']),
            'risks': np.array(frontier_data['risks']),
            'weights': np.array(frontier_data['weights']),
            'sharpe_ratios': np.array(frontier_data['sharpe_ratios']),
            'num_points': len(frontier_data['returns'])
        }
    
    def get_minimum_variance_portfolio(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """
        Get minimum variance portfolio.
        
        Args:
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            constraints: Additional constraints
            
        Returns:
            Dict: Minimum variance portfolio results
        """
        # Temporarily set risk aversion to very high value
        original_risk_aversion = self.risk_aversion
        self.risk_aversion = 1e6
        
        try:
            result = self.optimize(
                expected_returns,
                covariance_matrix,
                constraints=constraints
            )
        finally:
            self.risk_aversion = original_risk_aversion
        
        return result
    
    def get_maximum_return_portfolio(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        max_risk: float,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """
        Get maximum return portfolio for given risk level.
        
        Args:
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            max_risk: Maximum allowed risk
            constraints: Additional constraints
            
        Returns:
            Dict: Maximum return portfolio results
        """
        return self.optimize(
            expected_returns,
            covariance_matrix,
            target_risk=max_risk,
            constraints=constraints
        )
    
    def get_tangency_portfolio(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_free_rate: float = 0.0,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """
        Get tangency portfolio (maximum Sharpe ratio).
        
        Args:
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            risk_free_rate: Risk-free rate
            constraints: Additional constraints
            
        Returns:
            Dict: Tangency portfolio results
        """
        n_assets = len(expected_returns)
        
        # Define excess returns
        excess_returns = expected_returns - risk_free_rate
        
        # Analytical solution for tangency portfolio (if no constraints)
        if constraints is None and self.allow_short_selling:
            try:
                inv_cov = inv(covariance_matrix)
                weights = inv_cov @ excess_returns
                weights = weights / np.sum(weights)
                
                portfolio_metrics = calculate_portfolio_metrics(
                    weights, expected_returns, covariance_matrix, risk_free_rate
                )
                
                return {
                    'weights': weights,
                    'portfolio_return': portfolio_metrics['return'],
                    'portfolio_risk': portfolio_metrics['volatility'],
                    'sharpe_ratio': portfolio_metrics['sharpe_ratio'],
                    'method': 'analytical'
                }
            except np.linalg.LinAlgError:
                # Fallback to numerical optimization
                pass
        
        # Numerical optimization for tangency portfolio
        def negative_sharpe_ratio(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            
            if portfolio_risk == 0:
                return -np.inf
            
            return -(portfolio_return - risk_free_rate) / portfolio_risk
        
        # Constraints
        constraint_list = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
        
        if constraints:
            constraint_list.extend(self._add_scipy_constraints(constraints))
        
        # Bounds
        if self.allow_short_selling:
            bounds = [(-self.max_weight, self.max_weight) for _ in range(n_assets)]
        else:
            bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(
                negative_sharpe_ratio,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_list,
                options={'disp': self.verbose}
            )
            
            if result.success:
                optimal_weights = result.x
                portfolio_metrics = calculate_portfolio_metrics(
                    optimal_weights, expected_returns, covariance_matrix, risk_free_rate
                )
                
                return {
                    'weights': optimal_weights,
                    'portfolio_return': portfolio_metrics['return'],
                    'portfolio_risk': portfolio_metrics['volatility'],
                    'sharpe_ratio': portfolio_metrics['sharpe_ratio'],
                    'method': 'numerical'
                }
            else:
                return {
                    'weights': None,
                    'error': result.message,
                    'method': 'numerical'
                }
        except Exception as e:
            return {
                'weights': None,
                'error': str(e),
                'method': 'numerical'
            }