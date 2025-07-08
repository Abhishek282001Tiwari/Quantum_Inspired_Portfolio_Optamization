"""
Market data fetching and processing.

This module provides functionality to fetch and process market data
from various sources including Yahoo Finance, Alpha Vantage, and Quandl.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


class MarketDataFetcher:
    """
    Market data fetcher for various financial instruments.
    
    This class provides methods to fetch market data from multiple sources
    and process it for portfolio optimization.
    """
    
    def __init__(
        self,
        alpha_vantage_key: Optional[str] = None,
        quandl_key: Optional[str] = None,
        cache_data: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize market data fetcher.
        
        Args:
            alpha_vantage_key: Alpha Vantage API key
            quandl_key: Quandl API key
            cache_data: Whether to cache fetched data
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries (seconds)
        """
        self.alpha_vantage_key = alpha_vantage_key
        self.quandl_key = quandl_key
        self.cache_data = cache_data
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Cache for storing fetched data
        self.cache = {}
        
        # Default stock universes
        self.stock_universes = {
            'sp500_sample': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'JNJ', 'V'],
            'tech_stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX', 'ADBE', 'CRM', 'ORCL'],
            'etfs': ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'GLD', 'VTI', 'BND', 'VNQ'],
            'sectors': ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE']
        }
    
    def fetch_stock_data(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        source: str = 'yahoo',
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch stock price data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data fetch
            end_date: End date for data fetch
            source: Data source ('yahoo', 'alpha_vantage')
            interval: Data interval ('1d', '1wk', '1mo')
            
        Returns:
            pd.DataFrame: Stock price data
        """
        if source == 'yahoo':
            return self._fetch_yahoo_data(symbols, start_date, end_date, interval)
        elif source == 'alpha_vantage':
            return self._fetch_alpha_vantage_data(symbols, start_date, end_date, interval)
        else:
            raise ValueError(f"Unsupported data source: {source}")
    
    def _fetch_yahoo_data(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str
    ) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        try:
            # Download data for all symbols
            data = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                interval=interval,
                group_by='ticker',
                auto_adjust=True,
                prepost=True,
                threads=True,
                progress=False
            )
            
            # If single symbol, restructure data
            if len(symbols) == 1:
                data.columns = pd.MultiIndex.from_product([symbols, data.columns])
            
            # Extract adjusted close prices
            close_prices = pd.DataFrame()
            
            for symbol in symbols:
                try:
                    if len(symbols) == 1:
                        close_prices[symbol] = data[symbol]['Close']
                    else:
                        close_prices[symbol] = data[symbol]['Close']
                except KeyError:
                    warnings.warn(f"No data found for symbol: {symbol}")
                    continue
            
            # Remove rows with all NaN values
            close_prices = close_prices.dropna(how='all')
            
            return close_prices
            
        except Exception as e:
            warnings.warn(f"Error fetching Yahoo Finance data: {e}")
            return pd.DataFrame()
    
    def _fetch_alpha_vantage_data(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str
    ) -> pd.DataFrame:
        """Fetch data from Alpha Vantage."""
        if not self.alpha_vantage_key:
            raise ValueError("Alpha Vantage API key not provided")
        
        base_url = "https://www.alphavantage.co/query"
        
        def fetch_symbol_data(symbol: str) -> pd.Series:
            """Fetch data for a single symbol."""
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': 'full',
                'apikey': self.alpha_vantage_key
            }
            
            for attempt in range(self.max_retries):
                try:
                    response = requests.get(base_url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    if 'Time Series (Daily)' in data:
                        time_series = data['Time Series (Daily)']
                        df = pd.DataFrame.from_dict(time_series, orient='index')
                        df.index = pd.to_datetime(df.index)
                        df = df.sort_index()
                        
                        # Filter by date range
                        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
                        df = df[mask]
                        
                        return df['5. adjusted close'].astype(float)
                    else:
                        warnings.warn(f"No data returned for symbol: {symbol}")
                        return pd.Series(dtype=float)
                        
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        warnings.warn(f"Error fetching data for {symbol}: {e}")
                        return pd.Series(dtype=float)
        
        # Fetch data for all symbols
        all_data = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {executor.submit(fetch_symbol_data, symbol): symbol for symbol in symbols}
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if not data.empty:
                        all_data[symbol] = data
                except Exception as e:
                    warnings.warn(f"Error processing data for {symbol}: {e}")
        
        # Combine all data
        if all_data:
            combined_df = pd.DataFrame(all_data)
            return combined_df.fillna(method='ffill')
        else:
            return pd.DataFrame()
    
    def calculate_returns(
        self,
        price_data: pd.DataFrame,
        method: str = 'simple',
        period: int = 1
    ) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            price_data: DataFrame with price data
            method: Return calculation method ('simple', 'log')
            period: Period for return calculation
            
        Returns:
            pd.DataFrame: Returns data
        """
        if method == 'simple':
            returns = price_data.pct_change(periods=period)
        elif method == 'log':
            returns = np.log(price_data / price_data.shift(period))
        else:
            raise ValueError(f"Unsupported return method: {method}")
        
        return returns.dropna()
    
    def calculate_expected_returns(
        self,
        returns_data: pd.DataFrame,
        method: str = 'historical',
        lookback_period: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate expected returns.
        
        Args:
            returns_data: DataFrame with returns data
            method: Method for calculating expected returns
            lookback_period: Number of periods to look back
            
        Returns:
            pd.Series: Expected returns
        """
        if lookback_period:
            returns_data = returns_data.tail(lookback_period)
        
        if method == 'historical':
            return returns_data.mean()
        elif method == 'exponential':
            # Exponentially weighted returns
            return returns_data.ewm(span=30).mean().iloc[-1]
        elif method == 'shrinkage':
            # James-Stein shrinkage estimator
            historical_mean = returns_data.mean()
            overall_mean = historical_mean.mean()
            
            # Shrinkage factor (simplified)
            shrinkage_factor = 0.1
            
            return (1 - shrinkage_factor) * historical_mean + shrinkage_factor * overall_mean
        else:
            raise ValueError(f"Unsupported expected returns method: {method}")
    
    def calculate_covariance_matrix(
        self,
        returns_data: pd.DataFrame,
        method: str = 'historical',
        lookback_period: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate covariance matrix.
        
        Args:
            returns_data: DataFrame with returns data
            method: Method for calculating covariance matrix
            lookback_period: Number of periods to look back
            
        Returns:
            pd.DataFrame: Covariance matrix
        """
        if lookback_period:
            returns_data = returns_data.tail(lookback_period)
        
        if method == 'historical':
            return returns_data.cov()
        elif method == 'exponential':
            # Exponentially weighted covariance
            return returns_data.ewm(span=30).cov().iloc[-len(returns_data.columns):]
        elif method == 'shrinkage':
            # Ledoit-Wolf shrinkage
            return self._ledoit_wolf_shrinkage(returns_data)
        else:
            raise ValueError(f"Unsupported covariance method: {method}")
    
    def _ledoit_wolf_shrinkage(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Apply Ledoit-Wolf shrinkage to covariance matrix."""
        from sklearn.covariance import LedoitWolf
        
        lw = LedoitWolf()
        lw.fit(returns_data.fillna(0))
        
        shrunk_cov = pd.DataFrame(
            lw.covariance_,
            index=returns_data.columns,
            columns=returns_data.columns
        )
        
        return shrunk_cov
    
    def get_stock_universe(self, universe_name: str) -> List[str]:
        """
        Get predefined stock universe.
        
        Args:
            universe_name: Name of the stock universe
            
        Returns:
            List[str]: List of stock symbols
        """
        return self.stock_universes.get(universe_name, [])
    
    def fetch_market_data_for_optimization(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        return_method: str = 'simple',
        expected_returns_method: str = 'historical',
        covariance_method: str = 'historical'
    ) -> Dict[str, Union[pd.DataFrame, pd.Series, np.ndarray]]:
        """
        Fetch and process market data for portfolio optimization.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data
            return_method: Method for calculating returns
            expected_returns_method: Method for expected returns
            covariance_method: Method for covariance matrix
            
        Returns:
            Dict: Processed market data
        """
        # Fetch price data
        price_data = self.fetch_stock_data(symbols, start_date, end_date)
        
        if price_data.empty:
            raise ValueError("No price data could be fetched")
        
        # Calculate returns
        returns_data = self.calculate_returns(price_data, method=return_method)
        
        # Calculate expected returns
        expected_returns = self.calculate_expected_returns(
            returns_data, method=expected_returns_method
        )
        
        # Calculate covariance matrix
        covariance_matrix = self.calculate_covariance_matrix(
            returns_data, method=covariance_method
        )
        
        return {
            'price_data': price_data,
            'returns_data': returns_data,
            'expected_returns': expected_returns,
            'covariance_matrix': covariance_matrix,
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date
        }
    
    def get_data_quality_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate data quality metrics.
        
        Args:
            data: DataFrame with data to analyze
            
        Returns:
            Dict: Data quality metrics
        """
        metrics = {}
        
        # Missing data percentage
        metrics['missing_data_pct'] = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        
        # Data availability per symbol
        metrics['symbol_availability'] = (data.count() / len(data)).to_dict()
        
        # Date range coverage
        metrics['date_range_days'] = (data.index.max() - data.index.min()).days
        
        # Data freshness (days since last data point)
        metrics['data_freshness_days'] = (datetime.now() - data.index.max()).days
        
        return metrics