"""
Stock data fetching and preprocessing module for StockPulse dashboard.
Handles yfinance API calls and data cleaning operations.
"""

import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import numpy as np


class StockDataFetcher:
    """Class to handle stock data fetching and preprocessing."""
    
    def __init__(self):
        self.data_cache = {}
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def fetch_stock_data(_self, ticker: str, period: str = "1y", start_date=None, end_date=None):
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Period for data ('1y', '6mo', '3mo', etc.)
            start_date: Custom start date
            end_date: Custom end date
            
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        try:
            stock = yf.Ticker(ticker.upper())
            
            if start_date and end_date:
                data = stock.history(start=start_date, end=end_date)
            else:
                data = stock.history(period=period)
            
            if data.empty:
                st.error(f"No data found for ticker: {ticker}")
                return pd.DataFrame()
            
            # Clean column names
            data.columns = [col.replace(' ', '_').lower() for col in data.columns]
            # Reset index to make Date a column
            data = data.reset_index()
            # Handle both 'Date' and 'date' column names
            if 'Date' in data.columns:
                data.rename(columns={'Date': 'date'}, inplace=True)
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            else:
                st.error("No 'date' column found in fetched data.")
                return pd.DataFrame()
            return data
            
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=600)  # Cache for 10 minutes
    def get_stock_info(_self, ticker: str):
        """
        Get basic stock information.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Stock information including company name, sector, etc.
        """
        try:
            stock = yf.Ticker(ticker.upper())
            info = stock.info
            
            return {
                'company_name': info.get('longName', ticker.upper()),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0)
            }
            
        except Exception as e:
            return {
                'company_name': ticker.upper(),
                'sector': 'N/A',
                'industry': 'N/A',
                'market_cap': 0,
                'pe_ratio': 0,
                'dividend_yield': 0
            }
    
    def calculate_returns(self, data: pd.DataFrame):
        """
        Calculate daily returns and cumulative returns.
        
        Args:
            data (pd.DataFrame): Stock data
            
        Returns:
            pd.DataFrame: Data with returns columns added
        """
        if data.empty or 'close' not in data.columns:
            return data
        
        data['daily_return'] = data['close'].pct_change()
        data['cumulative_return'] = (1 + data['daily_return']).cumprod() - 1
        
        return data
    
    def get_price_change(self, data: pd.DataFrame):
        """
        Calculate price change and percentage change.
        
        Args:
            data (pd.DataFrame): Stock data
            
        Returns:
            tuple: (price_change, percentage_change)
        """
        if data.empty or len(data) < 2:
            return 0, 0
        
        current_price = data['close'].iloc[-1]
        previous_price = data['close'].iloc[-2]
        
        price_change = current_price - previous_price
        percentage_change = (price_change / previous_price) * 100
        
        return price_change, percentage_change
    
    def get_key_metrics(self, data: pd.DataFrame):
        """
        Calculate key metrics for the stock.
        
        Args:
            data (pd.DataFrame): Stock data
            
        Returns:
            dict: Key metrics including latest price, volume, etc.
        """
        if data.empty:
            return {}
        
        latest_data = data.iloc[-1]
        price_change, pct_change = self.get_price_change(data)
        
        return {
            'latest_price': latest_data['close'],
            'price_change': price_change,
            'percentage_change': pct_change,
            'volume': latest_data['volume'],
            'avg_volume': data['volume'].mean(),
            'high_52w': data['high'].max(),
            'low_52w': data['low'].min(),
            'volatility': data['daily_return'].std() * np.sqrt(252) * 100  # Annualized volatility
        }
    
    def fetch_multiple_stocks(self, tickers: list, period: str = "1y"):
        """
        Fetch data for multiple stocks for comparison.
        
        Args:
            tickers (list): List of ticker symbols
            period (str): Period for data
            
        Returns:
            dict: Dictionary with ticker as key and data as value
        """
        stock_data = {}
        
        for ticker in tickers:
            data = self.fetch_stock_data(ticker, period)
            if not data.empty:
                stock_data[ticker.upper()] = data
        
        return stock_data