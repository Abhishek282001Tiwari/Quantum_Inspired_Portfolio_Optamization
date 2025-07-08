"""
Data processing and integration module.

This module provides data fetching, processing, and integration capabilities
for market data, economic indicators, and alternative data sources.
"""

from .market_data import MarketDataFetcher

__all__ = [
    "MarketDataFetcher",
]