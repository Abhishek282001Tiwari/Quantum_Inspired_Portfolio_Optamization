"""
Portfolio optimization utility functions.

This module provides common utilities for portfolio optimization
including validation, calculation, and analysis functions.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from scipy.stats import norm
import warnings


def validate_portfolio_inputs(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> None:
    """
    Validate inputs for portfolio optimization.
    
    Args:
        expected_returns: Expected returns vector
        covariance_matrix: Covariance matrix
        weights: Portfolio weights (optional)
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Check expected returns
    if not isinstance(expected_returns, np.ndarray):
        raise ValueError("Expected returns must be a numpy array")
    
    if expected_returns.ndim != 1:
        raise ValueError("Expected returns must be a 1D array")
    
    if len(expected_returns) == 0:
        raise ValueError("Expected returns cannot be empty")
    
    # Check covariance matrix
    if not isinstance(covariance_matrix, np.ndarray):
        raise ValueError("Covariance matrix must be a numpy array")
    
    if covariance_matrix.ndim != 2:
        raise ValueError("Covariance matrix must be 2D")
    
    if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise ValueError("Covariance matrix must be square")
    
    if covariance_matrix.shape[0] != len(expected_returns):
        raise ValueError("Covariance matrix dimensions must match expected returns length")
    
    # Check if covariance matrix is positive semi-definite
    eigenvalues = np.linalg.eigvals(covariance_matrix)
    if np.any(eigenvalues < -1e-8):
        warnings.warn("Covariance matrix is not positive semi-definite")
    
    # Check weights if provided
    if weights is not None:
        if not isinstance(weights, np.ndarray):
            raise ValueError("Weights must be a numpy array")
        
        if weights.ndim != 1:
            raise ValueError("Weights must be a 1D array")
        
        if len(weights) != len(expected_returns):
            raise ValueError("Weights length must match expected returns length")
        
        if np.any(weights < 0):
            warnings.warn("Negative weights detected")


def calculate_portfolio_metrics(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Calculate portfolio performance metrics.
    
    Args:
        weights: Portfolio weights
        expected_returns: Expected returns vector
        covariance_matrix: Covariance matrix
        risk_free_rate: Risk-free rate for Sharpe ratio
        
    Returns:
        Dict: Portfolio metrics
    """
    validate_portfolio_inputs(expected_returns, covariance_matrix, weights)
    
    # Basic metrics
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Risk-adjusted metrics
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
    
    # Diversification metrics
    effective_assets = 1 / np.sum(weights**2) if np.sum(weights**2) > 0 else 0
    concentration_ratio = np.max(weights) if len(weights) > 0 else 0
    
    return {
        'return': portfolio_return,
        'volatility': portfolio_volatility,
        'variance': portfolio_variance,
        'sharpe_ratio': sharpe_ratio,
        'effective_assets': effective_assets,
        'concentration_ratio': concentration_ratio,
        'sum_weights': np.sum(weights),
        'num_assets': len(weights),
        'num_selected': np.sum(weights > 1e-6),
        'max_weight': np.max(weights),
        'min_weight': np.min(weights)
    }


def calculate_var_es(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    confidence_level: float = 0.05,
    time_horizon: int = 1
) -> Dict[str, float]:
    """
    Calculate Value at Risk (VaR) and Expected Shortfall (ES).
    
    Args:
        weights: Portfolio weights
        expected_returns: Expected returns vector
        covariance_matrix: Covariance matrix
        confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
        time_horizon: Time horizon in periods
        
    Returns:
        Dict: VaR and ES metrics
    """
    validate_portfolio_inputs(expected_returns, covariance_matrix, weights)
    
    # Portfolio parameters
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
    
    # Scale for time horizon
    scaled_return = portfolio_return * time_horizon
    scaled_volatility = portfolio_volatility * np.sqrt(time_horizon)
    
    # Calculate VaR (assuming normal distribution)
    var_parametric = scaled_return - norm.ppf(1 - confidence_level) * scaled_volatility
    
    # Calculate Expected Shortfall
    es_parametric = scaled_return - scaled_volatility * norm.pdf(norm.ppf(confidence_level)) / confidence_level
    
    return {
        'var_parametric': var_parametric,
        'es_parametric': es_parametric,
        'var_absolute': -var_parametric,
        'es_absolute': -es_parametric,
        'confidence_level': confidence_level,
        'time_horizon': time_horizon
    }


def calculate_tracking_error(
    portfolio_weights: np.ndarray,
    benchmark_weights: np.ndarray,
    covariance_matrix: np.ndarray
) -> float:
    """
    Calculate tracking error relative to benchmark.
    
    Args:
        portfolio_weights: Portfolio weights
        benchmark_weights: Benchmark weights
        covariance_matrix: Covariance matrix
        
    Returns:
        float: Tracking error
    """
    active_weights = portfolio_weights - benchmark_weights
    tracking_variance = np.dot(active_weights, np.dot(covariance_matrix, active_weights))
    return np.sqrt(tracking_variance)


def calculate_maximum_drawdown(returns: np.ndarray) -> Dict[str, float]:
    """
    Calculate maximum drawdown from return series.
    
    Args:
        returns: Time series of returns
        
    Returns:
        Dict: Maximum drawdown metrics
    """
    if len(returns) == 0:
        return {'max_drawdown': 0.0, 'max_drawdown_duration': 0, 'current_drawdown': 0.0}
    
    # Calculate cumulative returns
    cumulative = np.cumprod(1 + returns)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative)
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    
    # Find maximum drawdown
    max_drawdown = np.min(drawdown)
    
    # Find maximum drawdown duration
    drawdown_periods = []
    in_drawdown = False
    start_idx = 0
    
    for i, dd in enumerate(drawdown):
        if dd < 0 and not in_drawdown:
            in_drawdown = True
            start_idx = i
        elif dd >= 0 and in_drawdown:
            in_drawdown = False
            drawdown_periods.append(i - start_idx)
    
    if in_drawdown:
        drawdown_periods.append(len(drawdown) - start_idx)
    
    max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
    current_drawdown = drawdown[-1]
    
    return {
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': max_drawdown_duration,
        'current_drawdown': current_drawdown,
        'drawdown_series': drawdown
    }


def calculate_information_ratio(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray
) -> float:
    """
    Calculate information ratio.
    
    Args:
        portfolio_returns: Portfolio return series
        benchmark_returns: Benchmark return series
        
    Returns:
        float: Information ratio
    """
    if len(portfolio_returns) != len(benchmark_returns):
        raise ValueError("Portfolio and benchmark returns must have same length")
    
    active_returns = portfolio_returns - benchmark_returns
    
    if len(active_returns) == 0:
        return 0.0
    
    excess_return = np.mean(active_returns)
    tracking_error = np.std(active_returns)
    
    return excess_return / tracking_error if tracking_error > 0 else 0.0


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    target_return: Optional[float] = None
) -> float:
    """
    Calculate Sortino ratio.
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate
        target_return: Target return (if None, uses risk-free rate)
        
    Returns:
        float: Sortino ratio
    """
    if target_return is None:
        target_return = risk_free_rate
    
    excess_returns = returns - target_return
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0
    
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    
    return np.mean(excess_returns) / downside_deviation if downside_deviation > 0 else 0.0


def calculate_calmar_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate
        
    Returns:
        float: Calmar ratio
    """
    if len(returns) == 0:
        return 0.0
    
    annual_return = np.mean(returns) * 252 - risk_free_rate  # Assuming daily returns
    max_dd = calculate_maximum_drawdown(returns)['max_drawdown']
    
    return annual_return / abs(max_dd) if max_dd != 0 else 0.0


def create_efficient_frontier(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    num_points: int = 100,
    risk_free_rate: float = 0.0,
    allow_short_selling: bool = False
) -> Dict[str, np.ndarray]:
    """
    Create efficient frontier.
    
    Args:
        expected_returns: Expected returns vector
        covariance_matrix: Covariance matrix
        num_points: Number of points on the frontier
        risk_free_rate: Risk-free rate
        allow_short_selling: Whether to allow short selling
        
    Returns:
        Dict: Efficient frontier data
    """
    validate_portfolio_inputs(expected_returns, covariance_matrix)
    
    from scipy.optimize import minimize
    
    n_assets = len(expected_returns)
    
    # Define optimization constraints
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    if not allow_short_selling:
        bounds = [(0, 1) for _ in range(n_assets)]
    else:
        bounds = [(-1, 1) for _ in range(n_assets)]
    
    # Calculate minimum variance portfolio
    def portfolio_variance(weights):
        return np.dot(weights, np.dot(covariance_matrix, weights))
    
    min_var_result = minimize(
        portfolio_variance,
        np.ones(n_assets) / n_assets,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    min_var_return = np.dot(min_var_result.x, expected_returns)
    max_return = np.max(expected_returns)
    
    # Generate target returns
    target_returns = np.linspace(min_var_return, max_return, num_points)
    
    frontier_volatilities = []
    frontier_weights = []
    
    for target_return in target_returns:
        # Add return constraint
        return_constraint = {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_return}
        all_constraints = constraints + [return_constraint]
        
        # Minimize variance for given return
        result = minimize(
            portfolio_variance,
            np.ones(n_assets) / n_assets,
            method='SLSQP',
            bounds=bounds,
            constraints=all_constraints
        )
        
        if result.success:
            frontier_volatilities.append(np.sqrt(result.fun))
            frontier_weights.append(result.x)
        else:
            frontier_volatilities.append(np.nan)
            frontier_weights.append(np.full(n_assets, np.nan))
    
    return {
        'returns': target_returns,
        'volatilities': np.array(frontier_volatilities),
        'weights': np.array(frontier_weights),
        'sharpe_ratios': (target_returns - risk_free_rate) / np.array(frontier_volatilities)
    }


def calculate_portfolio_attribution(
    portfolio_weights: np.ndarray,
    benchmark_weights: np.ndarray,
    asset_returns: np.ndarray,
    benchmark_returns: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Calculate portfolio performance attribution.
    
    Args:
        portfolio_weights: Portfolio weights
        benchmark_weights: Benchmark weights
        asset_returns: Asset returns matrix (time x assets)
        benchmark_returns: Benchmark returns vector
        
    Returns:
        Dict: Attribution analysis
    """
    if asset_returns.shape[1] != len(portfolio_weights):
        raise ValueError("Asset returns dimensions must match portfolio weights")
    
    # Calculate active weights
    active_weights = portfolio_weights - benchmark_weights
    
    # Calculate attribution components
    allocation_effect = np.sum(active_weights * np.mean(asset_returns, axis=0))
    
    # Selection effect (simplified)
    portfolio_return = np.mean(np.dot(asset_returns, portfolio_weights))
    benchmark_return = np.mean(benchmark_returns)
    
    selection_effect = portfolio_return - benchmark_return - allocation_effect
    
    return {
        'allocation_effect': allocation_effect,
        'selection_effect': selection_effect,
        'total_active_return': allocation_effect + selection_effect,
        'active_weights': active_weights
    }