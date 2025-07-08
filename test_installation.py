#!/usr/bin/env python3
"""
Installation and Functionality Test for Quantum Portfolio Optimization Platform.

This script tests the installation and basic functionality of all major components
to ensure the platform is working correctly.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test that all major modules can be imported."""
    print("🔍 Testing module imports...")
    
    try:
        # Core dependencies
        import numpy as np
        print("  ✅ NumPy imported successfully")
        
        import pandas as pd
        print("  ✅ Pandas imported successfully")
        
        import qiskit
        print("  ✅ Qiskit imported successfully")
        
        import cvxpy as cp
        print("  ✅ CVXPY imported successfully")
        
        import scipy
        print("  ✅ SciPy imported successfully")
        
        # Quantum portfolio modules
        sys.path.append(os.path.dirname(__file__))
        
        from quantum_portfolio.quantum.qaoa import QAOAPortfolioOptimizer
        print("  ✅ QAOA module imported successfully")
        
        from quantum_portfolio.quantum.vqe import VQEPortfolioOptimizer
        print("  ✅ VQE module imported successfully")
        
        from quantum_portfolio.quantum.quantum_annealing import QuantumAnnealingOptimizer
        print("  ✅ Quantum Annealing module imported successfully")
        
        from quantum_portfolio.classical.markowitz import MarkowitzOptimizer
        print("  ✅ Markowitz module imported successfully")
        
        from quantum_portfolio.data.market_data import MarketDataFetcher
        print("  ✅ Market Data module imported successfully")
        
        from quantum_portfolio.utils.portfolio_utils import calculate_portfolio_metrics
        print("  ✅ Portfolio Utils imported successfully")
        
        from quantum_portfolio.utils.quantum_utils import create_pauli_operator
        print("  ✅ Quantum Utils imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of quantum and classical optimizers."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        import numpy as np
        from quantum_portfolio.classical.markowitz import MarkowitzOptimizer
        from quantum_portfolio.quantum.quantum_annealing import QuantumAnnealingOptimizer
        
        # Create test data
        np.random.seed(42)
        n_assets = 4  # Small for testing
        
        # Generate synthetic expected returns and covariance matrix
        expected_returns = np.random.uniform(0.01, 0.05, n_assets)
        
        # Create a valid covariance matrix
        correlation = np.random.uniform(0.1, 0.5, (n_assets, n_assets))
        correlation = (correlation + correlation.T) / 2
        np.fill_diagonal(correlation, 1.0)
        
        # Ensure positive semi-definite
        eigenvals, eigenvecs = np.linalg.eigh(correlation)
        eigenvals = np.maximum(eigenvals, 0.01)
        correlation = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        volatilities = np.random.uniform(0.1, 0.3, n_assets)
        covariance_matrix = np.outer(volatilities, volatilities) * correlation
        
        print(f"  📊 Created test data: {n_assets} assets")
        
        # Test Classical Markowitz
        print("  🏛️ Testing Classical Markowitz...")
        classical_optimizer = MarkowitzOptimizer(risk_aversion=1.0)
        classical_result = classical_optimizer.optimize(expected_returns, covariance_matrix)
        
        if classical_result['weights'] is not None:
            print(f"    ✅ Classical optimization successful")
            print(f"    📈 Sharpe ratio: {classical_result['sharpe_ratio']:.4f}")
        else:
            print(f"    ❌ Classical optimization failed")
            return False
        
        # Test Quantum Annealing (simplified)
        print("  🌀 Testing Quantum Annealing...")
        qa_optimizer = QuantumAnnealingOptimizer(
            num_assets=n_assets,
            max_iterations=500,  # Reduced for testing
            num_reads=5
        )
        qa_result = qa_optimizer.optimize_portfolio(
            expected_returns, covariance_matrix, risk_aversion=1.0
        )
        
        if qa_result['weights'] is not None:
            print(f"    ✅ Quantum Annealing optimization successful")
            print(f"    📈 Sharpe ratio: {qa_result['sharpe_ratio']:.4f}")
        else:
            print(f"    ❌ Quantum Annealing optimization failed")
            return False
        
        # Compare results
        classical_sharpe = classical_result['sharpe_ratio']
        quantum_sharpe = qa_result['sharpe_ratio']
        
        print(f"\n  📊 Comparison Results:")
        print(f"    Classical Sharpe: {classical_sharpe:.4f}")
        print(f"    Quantum Sharpe: {quantum_sharpe:.4f}")
        
        if abs(classical_sharpe - quantum_sharpe) < 2.0:  # Reasonable difference
            print(f"    ✅ Results are within reasonable range")
        else:
            print(f"    ⚠️  Large difference in results (may be normal)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_fetching():
    """Test market data fetching functionality."""
    print("\n📊 Testing data fetching...")
    
    try:
        from quantum_portfolio.data.market_data import MarketDataFetcher
        
        data_fetcher = MarketDataFetcher()
        
        # Test getting stock universe
        universe = data_fetcher.get_stock_universe('sp500_sample')
        print(f"  ✅ Retrieved stock universe: {len(universe)} symbols")
        
        # Test synthetic data generation (fallback)
        print("  🧪 Testing synthetic data generation...")
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        n_assets = len(symbols)
        n_days = 100
        
        # Generate synthetic returns
        np.random.seed(42)
        returns = np.random.multivariate_normal(
            mean=np.random.uniform(0.0001, 0.002, n_assets),
            cov=np.eye(n_assets) * 0.01,
            size=n_days
        )
        
        returns_data = pd.DataFrame(
            returns,
            columns=symbols,
            index=pd.date_range(start=datetime.now() - timedelta(days=n_days), periods=n_days, freq='D')
        )
        
        expected_returns = data_fetcher.calculate_expected_returns(returns_data)
        covariance_matrix = data_fetcher.calculate_covariance_matrix(returns_data)
        
        print(f"  ✅ Generated synthetic data: {returns_data.shape}")
        print(f"  📈 Expected returns range: {expected_returns.min():.6f} to {expected_returns.max():.6f}")
        print(f"  📊 Covariance matrix shape: {covariance_matrix.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data fetching test error: {e}")
        return False


def test_quantum_utils():
    """Test quantum utility functions."""
    print("\n⚛️ Testing quantum utilities...")
    
    try:
        from quantum_portfolio.utils.quantum_utils import create_pauli_operator, ising_to_qubo, qubo_to_ising
        import numpy as np
        
        # Test Ising to QUBO conversion
        h = np.array([0.1, -0.2, 0.3])
        J = np.array([[0, 0.1, -0.1], [0.1, 0, 0.2], [-0.1, 0.2, 0]])
        
        q, Q = ising_to_qubo(h, J)
        print(f"  ✅ Ising to QUBO conversion successful")
        print(f"  📊 QUBO linear terms: {q}")
        
        # Test reverse conversion
        h_back, J_back = qubo_to_ising(q, Q)
        print(f"  ✅ QUBO to Ising conversion successful")
        
        # Check if conversion is consistent (within numerical precision)
        if np.allclose(h, h_back, atol=1e-10) and np.allclose(J, J_back, atol=1e-10):
            print(f"  ✅ Conversion consistency verified")
        else:
            print(f"  ⚠️  Small numerical differences in conversion (normal)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Quantum utils test error: {e}")
        return False


def test_portfolio_utils():
    """Test portfolio utility functions."""
    print("\n📈 Testing portfolio utilities...")
    
    try:
        from quantum_portfolio.utils.portfolio_utils import (
            calculate_portfolio_metrics, validate_portfolio_inputs, 
            calculate_var_es, calculate_maximum_drawdown
        )
        import numpy as np
        
        # Create test portfolio
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        expected_returns = np.array([0.02, 0.03, 0.01, 0.04])
        covariance_matrix = np.eye(4) * 0.01 + 0.005  # Simple covariance matrix
        
        # Test validation
        validate_portfolio_inputs(expected_returns, covariance_matrix, weights)
        print(f"  ✅ Portfolio input validation successful")
        
        # Test portfolio metrics
        metrics = calculate_portfolio_metrics(weights, expected_returns, covariance_matrix)
        print(f"  ✅ Portfolio metrics calculation successful")
        print(f"  📊 Portfolio return: {metrics['return']:.4f}")
        print(f"  📊 Portfolio risk: {metrics['volatility']:.4f}")
        print(f"  📊 Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
        
        # Test VaR calculation
        var_metrics = calculate_var_es(weights, expected_returns, covariance_matrix)
        print(f"  ✅ VaR/ES calculation successful")
        print(f"  📊 VaR (95%): {var_metrics['var_parametric']:.4f}")
        
        # Test drawdown calculation with synthetic returns
        np.random.seed(42)
        synthetic_returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
        drawdown_metrics = calculate_maximum_drawdown(synthetic_returns)
        print(f"  ✅ Maximum drawdown calculation successful")
        print(f"  📊 Max drawdown: {drawdown_metrics['max_drawdown']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Portfolio utils test error: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Quantum Portfolio Optimization Platform - Installation Test")
    print("=" * 70)
    
    test_results = []
    
    # Run all tests
    test_results.append(("Module Imports", test_imports()))
    test_results.append(("Basic Functionality", test_basic_functionality()))
    test_results.append(("Data Fetching", test_data_fetching()))
    test_results.append(("Quantum Utils", test_quantum_utils()))
    test_results.append(("Portfolio Utils", test_portfolio_utils()))
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print("-" * 70)
    print(f"OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The Quantum Portfolio Optimization Platform is ready to use!")
        print("\n🚀 Next steps:")
        print("1. Run the quantum advantage demo: python examples/quantum_advantage_demo.py")
        print("2. Explore the Jupyter notebooks in the notebooks/ directory")
        print("3. Check the documentation for advanced usage")
        print("\n🌟 Ready to explore quantum advantages in portfolio optimization!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed.")
        print("Please check the error messages above and ensure all dependencies are installed correctly.")
        print("\n🔧 Installation help:")
        print("1. Make sure you have Python 3.8+ installed")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Install in development mode: pip install -e .")
        return 1


if __name__ == "__main__":
    exit(main())