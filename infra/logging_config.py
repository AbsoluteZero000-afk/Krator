"""Logging configuration for Krator trading system.

This module configures Loguru for structured logging with JSON output,
rotation, and trading-specific log formatting.
"""

import sys
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
from config.settings import get_settings


def serialize_record(record: Dict[str, Any]) -> str:
    """Custom serializer for trading-specific log records.
    
    Produces clean JSON logs with trading context while excluding
    heavy fields that aren't needed in production.
    """
    subset = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
    }
    
    # Add extra fields (trading context) directly to top level
    if record["extra"]:
        # Filter out internal loguru fields
        extra_fields = {
            k: v for k, v in record["extra"].items() 
            if not k.startswith('_')
        }
        subset.update(extra_fields)
    
    # Add exception information if present
    if record["exception"]:
        exc = record["exception"]
        subset["exception"] = {
            "type": exc.type.__name__ if exc.type else "Unknown",
            "value": str(exc.value) if exc.value else "",
            "traceback": traceback.format_exception(
                exc.type, exc.value, exc.traceback
            ) if exc.traceback else []
        }
    
    return json.dumps(subset, default=str)


def patching_function(record: Dict[str, Any]) -> None:
    """Patch function to add serialized JSON to record."""
    record["serialized"] = serialize_record(record)


def configure_logging() -> None:
    """Configure Loguru logging for the Krator trading system."""
    settings = get_settings()
    
    # Remove default handler
    logger.remove()
    
    # Create logs directory if it doesn't exist
    log_file_path = Path(settings.logging.file_path)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure the logger with custom patching for JSON output
    logger_with_patch = logger.patch(patching_function)
    
    # Console handler - structured JSON in production, readable in development
    if settings.logging.enable_json:
        console_format = "{serialized}"
    else:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Add console handler
    logger_with_patch.add(
        sys.stderr,
        level=settings.logging.level,
        format=console_format,
        colorize=not settings.logging.enable_json,
        backtrace=True,
        diagnose=True,
    )
    
    # File handler with rotation and retention
    logger_with_patch.add(
        str(log_file_path),
        level=settings.logging.level,
        format="{serialized}" if settings.logging.enable_json else console_format,
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
        compression="gz",
        backtrace=True,
        diagnose=True,
        enqueue=True,  # Thread-safe logging
    )
    
    # Add error-specific handler for critical issues
    error_log_path = log_file_path.parent / "errors.log"
    logger_with_patch.add(
        str(error_log_path),
        level="ERROR",
        format="{serialized}" if settings.logging.enable_json else console_format,
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )
    
    # Log startup message
    logger.info(
        "Logging configured successfully",
        environment=settings.environment,
        log_level=settings.logging.level,
        json_logging=settings.logging.enable_json,
        log_file=str(log_file_path)
    )


def get_trading_logger(component: str) -> "logger":
    """Get a logger with trading-specific context.
    
    Args:
        component: Component name (e.g., 'strategy', 'order_manager', 'risk')
        
    Returns:
        Logger instance with component context
    """
    return logger.bind(component=component)


def log_trade_execution(
    symbol: str,
    side: str,
    quantity: int,
    price: float,
    order_id: str,
    strategy: Optional[str] = None,
    **kwargs
) -> None:
    """Log trade execution with structured data.
    
    Args:
        symbol: Trading symbol
        side: BUY or SELL
        quantity: Number of shares
        price: Execution price
        order_id: Order identifier
        strategy: Strategy name if applicable
        **kwargs: Additional trading context
    """
    logger.info(
        "Trade executed",
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        order_id=order_id,
        strategy=strategy,
        event_type="trade_execution",
        **kwargs
    )


def log_signal_generated(
    symbol: str,
    signal_type: str,
    strength: float,
    strategy: str,
    indicators: Optional[Dict[str, float]] = None,
    **kwargs
) -> None:
    """Log trading signal generation with structured data.
    
    Args:
        symbol: Trading symbol
        signal_type: Type of signal (BUY, SELL, HOLD)
        strength: Signal strength (0.0 to 1.0)
        strategy: Strategy name
        indicators: Technical indicator values
        **kwargs: Additional signal context
    """
    logger.info(
        "Signal generated",
        symbol=symbol,
        signal_type=signal_type,
        strength=strength,
        strategy=strategy,
        indicators=indicators or {},
        event_type="signal_generation",
        **kwargs
    )


def log_risk_event(
    risk_type: str,
    severity: str,
    message: str,
    symbol: Optional[str] = None,
    current_value: Optional[float] = None,
    threshold: Optional[float] = None,
    action_taken: Optional[str] = None,
    **kwargs
) -> None:
    """Log risk management events with structured data.
    
    Args:
        risk_type: Type of risk (drawdown, exposure, etc.)
        severity: Risk severity (LOW, MEDIUM, HIGH, CRITICAL)
        message: Human-readable risk message
        symbol: Affected symbol if applicable
        current_value: Current risk metric value
        threshold: Risk threshold that was breached
        action_taken: Action taken in response
        **kwargs: Additional risk context
    """
    log_level = "WARNING" if severity in ["LOW", "MEDIUM"] else "ERROR"
    
    logger.log(
        log_level,
        message,
        risk_type=risk_type,
        severity=severity,
        symbol=symbol,
        current_value=current_value,
        threshold=threshold,
        action_taken=action_taken,
        event_type="risk_event",
        **kwargs
    )


def log_system_health(
    component: str,
    status: str,
    response_time_ms: float,
    details: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """Log system health check results.
    
    Args:
        component: Component name being checked
        status: Health status (HEALTHY, DEGRADED, UNHEALTHY)
        response_time_ms: Response time in milliseconds
        details: Additional health check details
        **kwargs: Additional health context
    """
    log_level = "INFO" if status == "HEALTHY" else "WARNING"
    if status == "UNHEALTHY":
        log_level = "ERROR"
    
    logger.log(
        log_level,
        f"Health check: {component} is {status}",
        component=component,
        status=status,
        response_time_ms=response_time_ms,
        details=details or {},
        event_type="health_check",
        **kwargs
    )


def log_market_data(
    symbol: str,
    price: float,
    volume: int,
    exchange: str = "ALPACA",
    **kwargs
) -> None:
    """Log market data updates (use sparingly to avoid log spam).
    
    Args:
        symbol: Trading symbol
        price: Current price
        volume: Trading volume
        exchange: Exchange name
        **kwargs: Additional market data context
    """
    logger.debug(
        "Market data update",
        symbol=symbol,
        price=price,
        volume=volume,
        exchange=exchange,
        event_type="market_data",
        **kwargs
    )


def log_performance_metrics(
    strategy: str,
    total_return: float,
    sharpe_ratio: float,
    max_drawdown: float,
    win_rate: float,
    trades_count: int,
    **kwargs
) -> None:
    """Log strategy performance metrics.
    
    Args:
        strategy: Strategy name
        total_return: Total return percentage
        sharpe_ratio: Sharpe ratio
        max_drawdown: Maximum drawdown percentage
        win_rate: Win rate percentage
        trades_count: Total number of trades
        **kwargs: Additional performance metrics
    """
    logger.info(
        "Performance metrics updated",
        strategy=strategy,
        total_return=total_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        trades_count=trades_count,
        event_type="performance_metrics",
        **kwargs
    )


# Convenience aliases for common logging patterns
trade_logger = get_trading_logger("trading")
strategy_logger = get_trading_logger("strategy")
risk_logger = get_trading_logger("risk")
data_logger = get_trading_logger("data")
system_logger = get_trading_logger("system")
