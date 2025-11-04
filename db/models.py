"""SQLAlchemy database models for Krator trading system.

This module defines all database models for storing trading data, orders,
positions, and system metrics with proper indexing and relationships.
"""

from datetime import datetime
from decimal import Decimal
from sqlalchemy import (
    Column, Integer, String, Numeric, DateTime, Boolean, Text,
    ForeignKey, Index, BigInteger, Float, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum
import uuid

Base = declarative_base()


class OrderSideEnum(str, Enum):
    """Order side enumeration for database."""
    BUY = "BUY"
    SELL = "SELL"


class OrderTypeEnum(str, Enum):
    """Order type enumeration for database."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatusEnum(str, Enum):
    """Order status enumeration for database."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL_FILLED = "PARTIAL_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class Symbol(Base):
    """Trading symbol master data."""
    __tablename__ = 'symbols'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    exchange = Column(String(20), nullable=False, default='ALPACA')
    name = Column(String(200))
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(BigInteger)
    active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    bars = relationship("OHLCVBar", back_populates="symbol_ref", cascade="all, delete-orphan")
    orders = relationship("Order", back_populates="symbol_ref", cascade="all, delete-orphan")
    positions = relationship("Position", back_populates="symbol_ref", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Symbol(symbol='{self.symbol}', exchange='{self.exchange}')>"


class OHLCVBar(Base):
    """OHLCV price data with time-series optimized structure."""
    __tablename__ = 'ohlcv_bars'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)  # 1Min, 5Min, 1Hour, 1Day
    open_price = Column(Numeric(precision=10, scale=4), nullable=False)
    high_price = Column(Numeric(precision=10, scale=4), nullable=False)
    low_price = Column(Numeric(precision=10, scale=4), nullable=False)
    close_price = Column(Numeric(precision=10, scale=4), nullable=False)
    volume = Column(BigInteger, nullable=False)
    trade_count = Column(Integer)
    vwap = Column(Numeric(precision=10, scale=4))  # Volume Weighted Average Price
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    symbol_ref = relationship("Symbol", back_populates="bars")
    
    # Composite index for efficient time-series queries
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol_id', 'timestamp'),
        Index('idx_symbol_timeframe_timestamp', 'symbol_id', 'timeframe', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<OHLCVBar(symbol_id={self.symbol_id}, timestamp='{self.timestamp}', close={self.close_price})>"


class Order(Base):
    """Order tracking with comprehensive audit trail."""
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(100), unique=True, nullable=False, index=True)
    client_order_id = Column(String(100), index=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id'), nullable=False)
    side = Column(SQLEnum(OrderSideEnum), nullable=False)
    order_type = Column(SQLEnum(OrderTypeEnum), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Numeric(precision=10, scale=4))  # NULL for market orders
    stop_price = Column(Numeric(precision=10, scale=4))  # For stop orders
    time_in_force = Column(String(10), default='DAY')  # DAY, GTC, IOC, FOK
    status = Column(SQLEnum(OrderStatusEnum), nullable=False, default=OrderStatusEnum.PENDING)
    filled_quantity = Column(Integer, default=0, nullable=False)
    remaining_quantity = Column(Integer, nullable=False)
    avg_fill_price = Column(Numeric(precision=10, scale=4))
    commission = Column(Numeric(precision=8, scale=4), default=0)
    strategy = Column(String(100))  # Strategy that generated this order
    submitted_at = Column(DateTime, index=True)
    filled_at = Column(DateTime)
    cancelled_at = Column(DateTime)
    reject_reason = Column(Text)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    symbol_ref = relationship("Symbol", back_populates="orders")
    fills = relationship("Fill", back_populates="order_ref", cascade="all, delete-orphan")
    
    # Indexes for efficient queries
    __table_args__ = (
        Index('idx_symbol_status', 'symbol_id', 'status'),
        Index('idx_strategy_submitted', 'strategy', 'submitted_at'),
        Index('idx_status_submitted', 'status', 'submitted_at'),
    )
    
    def __repr__(self):
        return f"<Order(order_id='{self.order_id}', symbol_id={self.symbol_id}, side='{self.side}', status='{self.status}')>"


class Fill(Base):
    """Individual fill records for partial and complete order executions."""
    __tablename__ = 'fills'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fill_id = Column(String(100), unique=True, nullable=False, index=True)
    order_id = Column(String(100), ForeignKey('orders.order_id'), nullable=False)
    execution_id = Column(String(100), nullable=False)
    fill_quantity = Column(Integer, nullable=False)
    fill_price = Column(Numeric(precision=10, scale=4), nullable=False)
    commission = Column(Numeric(precision=8, scale=4), default=0)
    liquidity_flag = Column(String(10))  # 'A'dd liquidity, 'R'emove liquidity
    executed_at = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    order_ref = relationship("Order", back_populates="fills")
    
    def __repr__(self):
        return f"<Fill(fill_id='{self.fill_id}', quantity={self.fill_quantity}, price={self.fill_price})>"


class Position(Base):
    """Current and historical position tracking."""
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id'), nullable=False)
    quantity = Column(Integer, nullable=False)  # Positive for long, negative for short
    avg_price = Column(Numeric(precision=10, scale=4), nullable=False)
    market_value = Column(Numeric(precision=12, scale=4), nullable=False)
    cost_basis = Column(Numeric(precision=12, scale=4), nullable=False)
    unrealized_pnl = Column(Numeric(precision=12, scale=4), nullable=False)
    realized_pnl = Column(Numeric(precision=12, scale=4), default=0, nullable=False)
    last_price = Column(Numeric(precision=10, scale=4))
    side = Column(String(10), nullable=False)  # 'LONG' or 'SHORT'
    opened_at = Column(DateTime, nullable=False)
    closed_at = Column(DateTime)  # NULL for open positions
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    symbol_ref = relationship("Symbol", back_populates="positions")
    
    # Indexes
    __table_args__ = (
        Index('idx_symbol_opened', 'symbol_id', 'opened_at'),
        Index('idx_open_positions', 'closed_at'),  # NULL values for open positions
    )
    
    def __repr__(self):
        return f"<Position(symbol_id={self.symbol_id}, quantity={self.quantity}, unrealized_pnl={self.unrealized_pnl})>"


class Trade(Base):
    """Completed round-trip trades for performance analysis."""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(100), unique=True, nullable=False, index=True)
    symbol_id = Column(Integer, ForeignKey('symbols.id'), nullable=False)
    strategy = Column(String(100), nullable=False, index=True)
    side = Column(SQLEnum(OrderSideEnum), nullable=False)
    quantity = Column(Integer, nullable=False)
    entry_price = Column(Numeric(precision=10, scale=4), nullable=False)
    exit_price = Column(Numeric(precision=10, scale=4), nullable=False)
    gross_pnl = Column(Numeric(precision=12, scale=4), nullable=False)
    commission = Column(Numeric(precision=8, scale=4), default=0)
    net_pnl = Column(Numeric(precision=12, scale=4), nullable=False)
    return_percent = Column(Float)  # (exit_price - entry_price) / entry_price * 100
    hold_time_minutes = Column(Integer)  # Duration of trade in minutes
    opened_at = Column(DateTime, nullable=False, index=True)
    closed_at = Column(DateTime, nullable=False, index=True)
    entry_reason = Column(Text)  # Why trade was entered
    exit_reason = Column(Text)   # Why trade was exited
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    symbol_ref = relationship("Symbol")
    
    # Indexes for performance analysis
    __table_args__ = (
        Index('idx_strategy_closed', 'strategy', 'closed_at'),
        Index('idx_symbol_strategy', 'symbol_id', 'strategy'),
        Index('idx_pnl_analysis', 'strategy', 'net_pnl', 'closed_at'),
    )
    
    def __repr__(self):
        return f"<Trade(trade_id='{self.trade_id}', symbol_id={self.symbol_id}, net_pnl={self.net_pnl})>"


class Portfolio(Base):
    """Portfolio snapshots for equity curve and performance tracking."""
    __tablename__ = 'portfolio_snapshots'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    total_equity = Column(Numeric(precision=15, scale=4), nullable=False)
    cash = Column(Numeric(precision=15, scale=4), nullable=False)
    buying_power = Column(Numeric(precision=15, scale=4), nullable=False)
    long_market_value = Column(Numeric(precision=15, scale=4), default=0)
    short_market_value = Column(Numeric(precision=15, scale=4), default=0)
    unrealized_pnl = Column(Numeric(precision=12, scale=4), default=0)
    realized_pnl_daily = Column(Numeric(precision=12, scale=4), default=0)
    realized_pnl_total = Column(Numeric(precision=12, scale=4), default=0)
    day_trade_count = Column(Integer, default=0)
    positions_count = Column(Integer, default=0)
    orders_pending = Column(Integer, default=0)
    max_drawdown = Column(Float)  # Maximum drawdown from peak
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Daily index for performance analysis
    __table_args__ = (
        Index('idx_daily_snapshots', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<Portfolio(timestamp='{self.timestamp}', equity={self.total_equity}, pnl={self.unrealized_pnl})>"


class SystemEvent(Base):
    """System events and alerts for monitoring and debugging."""
    __tablename__ = 'system_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(100), unique=True, nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    component = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False)  # INFO, WARNING, ERROR, CRITICAL
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    details = Column(Text)  # JSON serialized details
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    created_at = Column(DateTime, default=func.now(), nullable=False, index=True)
    
    # Indexes for monitoring queries
    __table_args__ = (
        Index('idx_component_severity', 'component', 'severity'),
        Index('idx_type_created', 'event_type', 'created_at'),
        Index('idx_unresolved_events', 'resolved', 'created_at'),
    )
    
    def __repr__(self):
        return f"<SystemEvent(event_type='{self.event_type}', severity='{self.severity}', title='{self.title[:50]}...')>"


class StrategyMetrics(Base):
    """Strategy performance metrics and statistics."""
    __tablename__ = 'strategy_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(100), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float)  # Percentage
    gross_profit = Column(Numeric(precision=12, scale=4), default=0)
    gross_loss = Column(Numeric(precision=12, scale=4), default=0)
    net_profit = Column(Numeric(precision=12, scale=4), default=0)
    profit_factor = Column(Float)  # Gross profit / Gross loss
    avg_win = Column(Numeric(precision=12, scale=4))
    avg_loss = Column(Numeric(precision=12, scale=4))
    max_win = Column(Numeric(precision=12, scale=4))
    max_loss = Column(Numeric(precision=12, scale=4))
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)  # Percentage
    avg_hold_time_minutes = Column(Integer)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Unique constraint and indexes
    __table_args__ = (
        Index('idx_strategy_date', 'strategy_name', 'date'),
        Index('idx_performance_ranking', 'net_profit', 'sharpe_ratio'),
    )
    
    def __repr__(self):
        return f"<StrategyMetrics(strategy='{self.strategy_name}', date='{self.date}', net_profit={self.net_profit})>"


# Utility functions for database operations

def create_all_tables(engine):
    """Create all database tables.
    
    Args:
        engine: SQLAlchemy engine instance
    """
    Base.metadata.create_all(engine)


def drop_all_tables(engine):
    """Drop all database tables (use with caution).
    
    Args:
        engine: SQLAlchemy engine instance
    """
    Base.metadata.drop_all(engine)
