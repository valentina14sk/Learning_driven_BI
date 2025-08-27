"""
Technical indicators calculation module for StockPulse dashboard.
Implements popular technical analysis indicators like SMA, EMA, RSI, MACD, and Bollinger Bands.
"""

import pandas as pd
import numpy as np
import streamlit as st


class TechnicalIndicators:
    """Class to calculate various technical indicators for stock analysis."""
    
    def __init__(self):
        pass
    
    def simple_moving_average(self, data: pd.Series, window: int = 20):
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            data (pd.Series): Price data (typically closing price)
            window (int): Period for SMA calculation
            
        Returns:
            pd.Series: SMA values
        """
        return data.rolling(window=window).mean()
    
    def exponential_moving_average(self, data: pd.Series, window: int = 20):
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            data (pd.Series): Price data (typically closing price)
            window (int): Period for EMA calculation
            
        Returns:
            pd.Series: EMA values
        """
        return data.ewm(span=window).mean()
    
    def relative_strength_index(self, data: pd.Series, window: int = 14):
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data (pd.Series): Price data (typically closing price)
            window (int): Period for RSI calculation
            
        Returns:
            pd.Series: RSI values
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data (pd.Series): Price data (typically closing price)
            fast (int): Fast EMA period
            slow (int): Slow EMA period
            signal (int): Signal line EMA period
            
        Returns:
            tuple: (MACD line, Signal line, Histogram)
        """
        ema_fast = self.exponential_moving_average(data, fast)
        ema_slow = self.exponential_moving_average(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.exponential_moving_average(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def bollinger_bands(self, data: pd.Series, window: int = 20, num_std: float = 2):
        """
        Calculate Bollinger Bands.
        
        Args:
            data (pd.Series): Price data (typically closing price)
            window (int): Period for moving average
            num_std (float): Number of standard deviations
            
        Returns:
            tuple: (Upper band, Middle band (SMA), Lower band)
        """
        sma = self.simple_moving_average(data, window)
        std = data.rolling(window=window).std()
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return upper_band, sma, lower_band
    
    def stochastic_oscillator(self, high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3):
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high (pd.Series): High prices
            low (pd.Series): Low prices
            close (pd.Series): Closing prices
            k_window (int): %K period
            d_window (int): %D period
            
        Returns:
            tuple: (%K, %D)
        """
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return k_percent, d_percent
    
    def calculate_all_indicators(self, data: pd.DataFrame):
        """
        Calculate all technical indicators for the given data.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with all indicators added
        """
        if data.empty or 'close' not in data.columns:
            return data
        
        # Moving Averages
        data['sma_20'] = self.simple_moving_average(data['close'], 20)
        data['sma_50'] = self.simple_moving_average(data['close'], 50)
        data['ema_12'] = self.exponential_moving_average(data['close'], 12)
        data['ema_26'] = self.exponential_moving_average(data['close'], 26)
        
        # RSI
        data['rsi'] = self.relative_strength_index(data['close'])
        
        # MACD
        macd_line, signal_line, histogram = self.macd(data['close'])
        data['macd'] = macd_line
        data['macd_signal'] = signal_line
        data['macd_histogram'] = histogram
        
        # Bollinger Bands
        upper_bb, middle_bb, lower_bb = self.bollinger_bands(data['close'])
        data['bb_upper'] = upper_bb
        data['bb_middle'] = middle_bb
        data['bb_lower'] = lower_bb
        
        # Stochastic Oscillator (if OHLC data available)
        if all(col in data.columns for col in ['high', 'low', 'close']):
            k_percent, d_percent = self.stochastic_oscillator(data['high'], data['low'], data['close'])
            data['stoch_k'] = k_percent
            data['stoch_d'] = d_percent
        
        return data
    
    def get_trading_signals(self, data: pd.DataFrame):
        """
        Generate basic trading signals based on technical indicators.
        
        Args:
            data (pd.DataFrame): Data with calculated indicators
            
        Returns:
            dict: Trading signals and recommendations
        """
        if data.empty or len(data) < 50:
            return {'signal': 'HOLD', 'strength': 0, 'reasons': []}
        
        latest = data.iloc[-1]
        signals = []
        score = 0
        
        # RSI signals
        if 'rsi' in data.columns and not pd.isna(latest['rsi']):
            if latest['rsi'] < 30:
                signals.append("RSI indicates oversold condition (potential buy)")
                score += 1
            elif latest['rsi'] > 70:
                signals.append("RSI indicates overbought condition (potential sell)")
                score -= 1
        
        # Moving Average signals
        if all(col in data.columns for col in ['close', 'sma_20', 'sma_50']):
            if latest['close'] > latest['sma_20'] > latest['sma_50']:
                signals.append("Price above both 20 & 50 SMA (bullish trend)")
                score += 1
            elif latest['close'] < latest['sma_20'] < latest['sma_50']:
                signals.append("Price below both 20 & 50 SMA (bearish trend)")
                score -= 1
        
        # MACD signals
        if all(col in data.columns for col in ['macd', 'macd_signal']):
            if latest['macd'] > latest['macd_signal']:
                signals.append("MACD above signal line (bullish momentum)")
                score += 1
            else:
                signals.append("MACD below signal line (bearish momentum)")
                score -= 1
        
        # Bollinger Bands signals
        if all(col in data.columns for col in ['close', 'bb_upper', 'bb_lower']):
            if latest['close'] > latest['bb_upper']:
                signals.append("Price above upper Bollinger Band (overbought)")
                score -= 1
            elif latest['close'] < latest['bb_lower']:
                signals.append("Price below lower Bollinger Band (oversold)")
                score += 1
        
        # Determine overall signal
        if score >= 2:
            signal = 'BUY'
            strength = min(score * 20, 100)
        elif score <= -2:
            signal = 'SELL'
            strength = min(abs(score) * 20, 100)
        else:
            signal = 'HOLD'
            strength = abs(score) * 10
        
        return {
            'signal': signal,
            'strength': strength,
            'reasons': signals,
            'score': score
        }