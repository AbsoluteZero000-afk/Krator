"""Technical analysis indicators using TA-Lib with NaN-safe wrappers.

This module provides robust wrappers around TA-Lib indicators with proper
NaN handling, warm-up period management, and trading-specific enhancements.
"""

import numpy as np
import pandas as pd
import talib
from typing import Tuple, Optional, Dict, Any, Union
from decimal import Decimal
from loguru import logger


class IndicatorError(Exception):
    """Custom exception for indicator calculation errors."""
    pass


def validate_price_data(data: Union[np.ndarray, pd.Series], min_length: int = 1) -> np.ndarray:
    """Validate and convert price data to numpy array.
    
    Args:
        data: Price data as numpy array or pandas Series
        min_length: Minimum required data length
        
    Returns:
        Validated numpy array
        
    Raises:
        IndicatorError: If data is invalid or too short
    """
    if data is None:
        raise IndicatorError("Price data cannot be None")
    
    # Convert to numpy array if pandas Series
    if isinstance(data, pd.Series):
        data = data.values
    elif isinstance(data, list):
        data = np.array(data, dtype=np.float64)
    
    if not isinstance(data, np.ndarray):
        raise IndicatorError(f"Unsupported data type: {type(data)}")
    
    if len(data) < min_length:
        raise IndicatorError(f"Insufficient data: need at least {min_length} points, got {len(data)}")
    
    # Ensure float64 dtype for TA-Lib compatibility
    return data.astype(np.float64)


def get_warmup_period(indicator_name: str, **params) -> int:
    """Get the warm-up period required for an indicator.
    
    Args:
        indicator_name: Name of the indicator
        **params: Indicator parameters
        
    Returns:
        Number of periods needed for indicator to produce valid values
    """
    warmup_periods = {
        'SMA': params.get('timeperiod', 20),
        'EMA': params.get('timeperiod', 20),
        'RSI': params.get('timeperiod', 14),
        'MACD': max(params.get('fastperiod', 12), params.get('slowperiod', 26)) + params.get('signalperiod', 9),
        'BBANDS': params.get('timeperiod', 20),
        'ATR': params.get('timeperiod', 14),
        'STOCH': max(params.get('fastk_period', 5), params.get('slowk_period', 3), params.get('slowd_period', 3)),
        'WILLR': params.get('timeperiod', 14),
        'ADX': params.get('timeperiod', 14),
        'CCI': params.get('timeperiod', 14),
        'MFI': params.get('timeperiod', 14),
    }
    
    return warmup_periods.get(indicator_name.upper(), 50)  # Default 50 period buffer


def safe_sma(close: Union[np.ndarray, pd.Series], timeperiod: int = 20) -> np.ndarray:
    """Calculate Simple Moving Average with NaN safety.
    
    Args:
        close: Closing prices
        timeperiod: Number of periods for SMA
        
    Returns:
        SMA values with NaN for insufficient data
    """
    try:
        close_data = validate_price_data(close, min_length=1)
        
        if len(close_data) < timeperiod:
            # Return array filled with NaN if insufficient data
            return np.full(len(close_data), np.nan)
        
        result = talib.SMA(close_data, timeperiod=timeperiod)
        return result
        
    except Exception as e:
        logger.error(f"Error calculating SMA: {e}", timeperiod=timeperiod, data_length=len(close))
        return np.full(len(close), np.nan)


def safe_ema(close: Union[np.ndarray, pd.Series], timeperiod: int = 20) -> np.ndarray:
    """Calculate Exponential Moving Average with NaN safety.
    
    Args:
        close: Closing prices
        timeperiod: Number of periods for EMA
        
    Returns:
        EMA values with NaN for insufficient data
    """
    try:
        close_data = validate_price_data(close, min_length=timeperiod)
        result = talib.EMA(close_data, timeperiod=timeperiod)
        return result
        
    except Exception as e:
        logger.error(f"Error calculating EMA: {e}", timeperiod=timeperiod, data_length=len(close))
        return np.full(len(close), np.nan)


def safe_rsi(close: Union[np.ndarray, pd.Series], timeperiod: int = 14) -> np.ndarray:
    """Calculate Relative Strength Index with NaN safety.
    
    Args:
        close: Closing prices
        timeperiod: Number of periods for RSI
        
    Returns:
        RSI values (0-100) with NaN for insufficient data
    """
    try:
        close_data = validate_price_data(close, min_length=timeperiod + 1)
        result = talib.RSI(close_data, timeperiod=timeperiod)
        return result
        
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}", timeperiod=timeperiod, data_length=len(close))
        return np.full(len(close), np.nan)


def safe_macd(
    close: Union[np.ndarray, pd.Series],
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate MACD with NaN safety.
    
    Args:
        close: Closing prices
        fastperiod: Fast EMA period
        slowperiod: Slow EMA period
        signalperiod: Signal line EMA period
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    try:
        close_data = validate_price_data(close, min_length=slowperiod + signalperiod)
        macd, signal, histogram = talib.MACD(
            close_data, 
            fastperiod=fastperiod, 
            slowperiod=slowperiod, 
            signalperiod=signalperiod
        )
        return macd, signal, histogram
        
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}", 
                    fastperiod=fastperiod, slowperiod=slowperiod, 
                    signalperiod=signalperiod, data_length=len(close))
        nan_array = np.full(len(close), np.nan)
        return nan_array, nan_array, nan_array


def safe_bollinger_bands(
    close: Union[np.ndarray, pd.Series],
    timeperiod: int = 20,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
    matype: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Bollinger Bands with NaN safety.
    
    Args:
        close: Closing prices
        timeperiod: Number of periods for moving average
        nbdevup: Number of standard deviations for upper band
        nbdevdn: Number of standard deviations for lower band
        matype: Moving average type (0=SMA)
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    try:
        close_data = validate_price_data(close, min_length=timeperiod)
        upper, middle, lower = talib.BBANDS(
            close_data,
            timeperiod=timeperiod,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn,
            matype=matype
        )
        return upper, middle, lower
        
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}", 
                    timeperiod=timeperiod, data_length=len(close))
        nan_array = np.full(len(close), np.nan)
        return nan_array, nan_array, nan_array


def safe_atr(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    timeperiod: int = 14
) -> np.ndarray:
    """Calculate Average True Range with NaN safety.
    
    Args:
        high: High prices
        low: Low prices
        close: Closing prices
        timeperiod: Number of periods for ATR
        
    Returns:
        ATR values with NaN for insufficient data
    """
    try:
        high_data = validate_price_data(high, min_length=timeperiod)
        low_data = validate_price_data(low, min_length=timeperiod)
        close_data = validate_price_data(close, min_length=timeperiod)
        
        # Ensure all arrays have same length
        min_length = min(len(high_data), len(low_data), len(close_data))
        high_data = high_data[-min_length:]
        low_data = low_data[-min_length:]
        close_data = close_data[-min_length:]
        
        result = talib.ATR(high_data, low_data, close_data, timeperiod=timeperiod)
        return result
        
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}", timeperiod=timeperiod)
        return np.full(len(close), np.nan)


def safe_stochastic(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    fastk_period: int = 5,
    slowk_period: int = 3,
    slowk_matype: int = 0,
    slowd_period: int = 3,
    slowd_matype: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Stochastic Oscillator with NaN safety.
    
    Args:
        high: High prices
        low: Low prices
        close: Closing prices
        fastk_period: %K period
        slowk_period: Slow %K period
        slowk_matype: Slow %K MA type
        slowd_period: %D period
        slowd_matype: %D MA type
        
    Returns:
        Tuple of (slowk, slowd) values
    """
    try:
        min_required = max(fastk_period, slowk_period, slowd_period)
        high_data = validate_price_data(high, min_length=min_required)
        low_data = validate_price_data(low, min_length=min_required)
        close_data = validate_price_data(close, min_length=min_required)
        
        # Ensure all arrays have same length
        min_length = min(len(high_data), len(low_data), len(close_data))
        high_data = high_data[-min_length:]
        low_data = low_data[-min_length:]
        close_data = close_data[-min_length:]
        
        slowk, slowd = talib.STOCH(
            high_data, low_data, close_data,
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowk_matype=slowk_matype,
            slowd_period=slowd_period,
            slowd_matype=slowd_matype
        )
        return slowk, slowd
        
    except Exception as e:
        logger.error(f"Error calculating Stochastic: {e}")
        nan_array = np.full(len(close), np.nan)
        return nan_array, nan_array


def safe_adx(
    high: Union[np.ndarray, pd.Series],
    low: Union[np.ndarray, pd.Series],
    close: Union[np.ndarray, pd.Series],
    timeperiod: int = 14
) -> np.ndarray:
    """Calculate Average Directional Index with NaN safety.
    
    Args:
        high: High prices
        low: Low prices
        close: Closing prices
        timeperiod: Number of periods for ADX
        
    Returns:
        ADX values with NaN for insufficient data
    """
    try:
        high_data = validate_price_data(high, min_length=timeperiod * 2)
        low_data = validate_price_data(low, min_length=timeperiod * 2)
        close_data = validate_price_data(close, min_length=timeperiod * 2)
        
        # Ensure all arrays have same length
        min_length = min(len(high_data), len(low_data), len(close_data))
        high_data = high_data[-min_length:]
        low_data = low_data[-min_length:]
        close_data = close_data[-min_length:]
        
        result = talib.ADX(high_data, low_data, close_data, timeperiod=timeperiod)
        return result
        
    except Exception as e:
        logger.error(f"Error calculating ADX: {e}", timeperiod=timeperiod)
        return np.full(len(close), np.nan)


class TechnicalIndicators:
    """Class for managing multiple technical indicators with caching."""
    
    def __init__(self, cache_size: int = 1000):
        """Initialize indicators manager.
        
        Args:
            cache_size: Maximum number of cached indicator results
        """
        self.cache = {}
        self.cache_size = cache_size
        
    def calculate_all(
        self,
        high: Union[np.ndarray, pd.Series],
        low: Union[np.ndarray, pd.Series],
        close: Union[np.ndarray, pd.Series],
        volume: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Dict[str, Any]:
        """Calculate all common technical indicators.
        
        Args:
            high: High prices
            low: Low prices
            close: Closing prices
            volume: Volume data (optional)
            
        Returns:
            Dictionary of indicator values
        """
        try:
            indicators = {}
            
            # Moving averages
            indicators['sma_20'] = safe_sma(close, 20)
            indicators['sma_50'] = safe_sma(close, 50)
            indicators['ema_12'] = safe_ema(close, 12)
            indicators['ema_26'] = safe_ema(close, 26)
            
            # Momentum indicators
            indicators['rsi'] = safe_rsi(close, 14)
            macd, signal, histogram = safe_macd(close)
            indicators['macd'] = macd
            indicators['macd_signal'] = signal
            indicators['macd_histogram'] = histogram
            
            # Volatility indicators
            upper, middle, lower = safe_bollinger_bands(close)
            indicators['bb_upper'] = upper
            indicators['bb_middle'] = middle
            indicators['bb_lower'] = lower
            indicators['atr'] = safe_atr(high, low, close)
            
            # Oscillators
            slowk, slowd = safe_stochastic(high, low, close)
            indicators['stoch_k'] = slowk
            indicators['stoch_d'] = slowd
            
            # Trend indicators
            indicators['adx'] = safe_adx(high, low, close)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}
    
    def get_signal_summary(self, indicators: Dict[str, Any], close_price: float) -> Dict[str, str]:
        """Get trading signals summary from indicators.
        
        Args:
            indicators: Dictionary of indicator values
            close_price: Current close price
            
        Returns:
            Dictionary of signal summaries
        """
        signals = {}
        
        try:
            # RSI signals
            if 'rsi' in indicators and not np.isnan(indicators['rsi'][-1]):
                rsi_value = indicators['rsi'][-1]
                if rsi_value > 70:
                    signals['rsi'] = 'OVERBOUGHT'
                elif rsi_value < 30:
                    signals['rsi'] = 'OVERSOLD'
                else:
                    signals['rsi'] = 'NEUTRAL'
            
            # MACD signals
            if all(k in indicators for k in ['macd', 'macd_signal']):
                macd_val = indicators['macd'][-1]
                signal_val = indicators['macd_signal'][-1]
                if not (np.isnan(macd_val) or np.isnan(signal_val)):
                    if macd_val > signal_val:
                        signals['macd'] = 'BULLISH'
                    else:
                        signals['macd'] = 'BEARISH'
            
            # Bollinger Bands signals
            if all(k in indicators for k in ['bb_upper', 'bb_lower']):
                bb_upper = indicators['bb_upper'][-1]
                bb_lower = indicators['bb_lower'][-1]
                if not (np.isnan(bb_upper) or np.isnan(bb_lower)):
                    if close_price > bb_upper:
                        signals['bollinger'] = 'OVERBOUGHT'
                    elif close_price < bb_lower:
                        signals['bollinger'] = 'OVERSOLD'
                    else:
                        signals['bollinger'] = 'NEUTRAL'
                        
        except Exception as e:
            logger.error(f"Error generating signal summary: {e}")
            
        return signals
