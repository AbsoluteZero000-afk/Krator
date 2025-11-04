"""Tests for core event system.

This module tests the event system including proper dataclass field ordering,
event creation, and validation to prevent common bugs.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from core.events import (
    BaseEvent, EventType, OrderSide, OrderType, OrderStatus,
    MarketDataEvent, PriceUpdateEvent, SignalEvent, OrderEvent,
    FillEvent, PositionEvent, PortfolioEvent, RiskEvent,
    SystemEvent, HealthCheckEvent,
    create_market_data_event, create_signal_event, create_order_event,
    create_system_event
)


class TestEventDataclassOrdering:
    """Test dataclass field ordering to prevent argument bugs."""
    
    def test_base_event_field_order(self):
        """Test BaseEvent has non-default fields first."""
        # This should work - all required fields provided
        event = BaseEvent(
            event_id=UUID('12345678-1234-5678-1234-567812345678'),
            event_type=EventType.SYSTEM_STARTUP,
            timestamp=datetime.utcnow(),
            source="test"
        )
        
        assert event.event_id is not None
        assert event.event_type == EventType.SYSTEM_STARTUP
        assert event.timestamp is not None
        assert event.source == "test"
        assert event.metadata == {}  # Default value
    
    def test_market_data_event_field_order(self):
        """Test MarketDataEvent field ordering."""
        event = MarketDataEvent(
            event_id=UUID('12345678-1234-5678-1234-567812345678'),
            event_type=EventType.MARKET_DATA_RECEIVED,
            timestamp=datetime.utcnow(),
            source="test",
            symbol="AAPL",
            price=Decimal('150.25'),
            volume=1000,
            exchange="NASDAQ"
        )
        
        assert event.symbol == "AAPL"
        assert event.price == Decimal('150.25')
        assert event.volume == 1000
        assert event.exchange == "NASDAQ"
        assert event.bid is None  # Default value
    
    def test_signal_event_field_order(self):
        """Test SignalEvent field ordering."""
        event = SignalEvent(
            event_id=UUID('12345678-1234-5678-1234-567812345678'),
            event_type=EventType.SIGNAL_GENERATED,
            timestamp=datetime.utcnow(),
            source="strategy",
            symbol="MSFT",
            signal_type="BUY",
            side=OrderSide.BUY,
            strength=0.8,
            strategy_name="RSI_Strategy"
        )
        
        assert event.symbol == "MSFT"
        assert event.signal_type == "BUY"
        assert event.side == OrderSide.BUY
        assert event.strength == 0.8
        assert event.strategy_name == "RSI_Strategy"
        assert event.confidence == 0.0  # Default value
    
    def test_order_event_field_order(self):
        """Test OrderEvent field ordering."""
        event = OrderEvent(
            event_id=UUID('12345678-1234-5678-1234-567812345678'),
            event_type=EventType.ORDER_SUBMITTED,
            timestamp=datetime.utcnow(),
            source="order_manager",
            order_id="ORD123",
            symbol="TSLA",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        assert event.order_id == "ORD123"
        assert event.symbol == "TSLA"
        assert event.side == OrderSide.SELL
        assert event.order_type == OrderType.MARKET
        assert event.quantity == 100
        assert event.status == OrderStatus.PENDING  # Default value


class TestEventCreationFactories:
    """Test event factory functions."""
    
    def test_create_market_data_event(self):
        """Test market data event factory."""
        event = create_market_data_event(
            symbol="AAPL",
            price=Decimal('145.50'),
            volume=2000,
            exchange="NYSE"
        )
        
        assert isinstance(event, MarketDataEvent)
        assert isinstance(event.event_id, UUID)
        assert event.event_type == EventType.MARKET_DATA_RECEIVED
        assert isinstance(event.timestamp, datetime)
        assert event.source == "market_data_feed"
        assert event.symbol == "AAPL"
        assert event.price == Decimal('145.50')
        assert event.volume == 2000
        assert event.exchange == "NYSE"
    
    def test_create_signal_event(self):
        """Test signal event factory."""
        event = create_signal_event(
            symbol="GOOGL",
            signal_type="SELL",
            side=OrderSide.SELL,
            strength=0.9,
            strategy_name="MACD_Crossover",
            confidence=0.85
        )
        
        assert isinstance(event, SignalEvent)
        assert event.event_type == EventType.SIGNAL_GENERATED
        assert event.symbol == "GOOGL"
        assert event.signal_type == "SELL"
        assert event.side == OrderSide.SELL
        assert event.strength == 0.9
        assert event.strategy_name == "MACD_Crossover"
        assert event.confidence == 0.85
    
    def test_create_order_event(self):
        """Test order event factory."""
        event = create_order_event(
            order_id="ORD456",
            symbol="NVDA",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=50,
            price=Decimal('220.00')
        )
        
        assert isinstance(event, OrderEvent)
        assert event.event_type == EventType.ORDER_SUBMITTED
        assert event.order_id == "ORD456"
        assert event.symbol == "NVDA"
        assert event.side == OrderSide.BUY
        assert event.order_type == OrderType.LIMIT
        assert event.quantity == 50
        assert event.price == Decimal('220.00')
    
    def test_create_system_event(self):
        """Test system event factory."""
        event = create_system_event(
            component="database",
            status="HEALTHY",
            message="Database connection established",
            event_type=EventType.SYSTEM_STARTUP
        )
        
        assert isinstance(event, SystemEvent)
        assert event.event_type == EventType.SYSTEM_STARTUP
        assert event.component == "database"
        assert event.status == "HEALTHY"
        assert event.message == "Database connection established"


class TestEventValidation:
    """Test event validation and error handling."""
    
    def test_base_event_validation(self):
        """Test BaseEvent validation."""
        with pytest.raises(ValueError, match="event_id must be a UUID"):
            BaseEvent(
                event_id="not-a-uuid",  # Invalid UUID
                event_type=EventType.SYSTEM_STARTUP,
                timestamp=datetime.utcnow(),
                source="test"
            )
        
        with pytest.raises(ValueError, match="timestamp must be a datetime"):
            BaseEvent(
                event_id=UUID('12345678-1234-5678-1234-567812345678'),
                event_type=EventType.SYSTEM_STARTUP,
                timestamp="not-a-datetime",  # Invalid datetime
                source="test"
            )
    
    def test_frozen_dataclass(self):
        """Test that events are immutable (frozen)."""
        event = create_market_data_event(
            symbol="AAPL",
            price=Decimal('150.00'),
            volume=1000
        )
        
        # Should not be able to modify frozen dataclass
        with pytest.raises(Exception):  # FrozenInstanceError in Python 3.7+
            event.symbol = "MSFT"
        
        with pytest.raises(Exception):
            event.price = Decimal('160.00')


class TestEventTypes:
    """Test specific event type implementations."""
    
    def test_price_update_event(self):
        """Test PriceUpdateEvent creation and fields."""
        event = PriceUpdateEvent(
            event_id=UUID('12345678-1234-5678-1234-567812345678'),
            event_type=EventType.PRICE_UPDATE,
            timestamp=datetime.utcnow(),
            source="price_feed",
            symbol="BTC-USD",
            open_price=Decimal('45000.00'),
            high_price=Decimal('45500.00'),
            low_price=Decimal('44800.00'),
            close_price=Decimal('45200.00'),
            volume=125000,
            timeframe="1Min"
        )
        
        assert event.symbol == "BTC-USD"
        assert event.open_price == Decimal('45000.00')
        assert event.high_price == Decimal('45500.00')
        assert event.low_price == Decimal('44800.00')
        assert event.close_price == Decimal('45200.00')
        assert event.volume == 125000
        assert event.timeframe == "1Min"
        assert event.exchange == "ALPACA"  # Default value
    
    def test_fill_event(self):
        """Test FillEvent creation and fields."""
        event = FillEvent(
            event_id=UUID('12345678-1234-5678-1234-567812345678'),
            event_type=EventType.ORDER_FILLED,
            timestamp=datetime.utcnow(),
            source="broker",
            order_id="ORD789",
            symbol="SPY",
            side=OrderSide.BUY,
            fill_quantity=100,
            fill_price=Decimal('420.50'),
            commission=Decimal('1.00'),
            execution_id="EXEC123"
        )
        
        assert event.order_id == "ORD789"
        assert event.symbol == "SPY"
        assert event.side == OrderSide.BUY
        assert event.fill_quantity == 100
        assert event.fill_price == Decimal('420.50')
        assert event.commission == Decimal('1.00')
        assert event.execution_id == "EXEC123"
        assert event.remaining_quantity == 0  # Default value
    
    def test_position_event(self):
        """Test PositionEvent creation and fields."""
        event = PositionEvent(
            event_id=UUID('12345678-1234-5678-1234-567812345678'),
            event_type=EventType.POSITION_UPDATED,
            timestamp=datetime.utcnow(),
            source="portfolio",
            symbol="QQQ",
            quantity=200,
            avg_price=Decimal('350.75'),
            market_value=Decimal('70150.00'),
            unrealized_pnl=Decimal('500.00')
        )
        
        assert event.symbol == "QQQ"
        assert event.quantity == 200
        assert event.avg_price == Decimal('350.75')
        assert event.market_value == Decimal('70150.00')
        assert event.unrealized_pnl == Decimal('500.00')
        assert event.side == "LONG"  # Default value
    
    def test_portfolio_event(self):
        """Test PortfolioEvent creation and fields."""
        event = PortfolioEvent(
            event_id=UUID('12345678-1234-5678-1234-567812345678'),
            event_type=EventType.PORTFOLIO_UPDATED,
            timestamp=datetime.utcnow(),
            source="portfolio",
            total_value=Decimal('100000.00'),
            cash=Decimal('25000.00'),
            buying_power=Decimal('50000.00'),
            equity=Decimal('100000.00'),
            unrealized_pnl=Decimal('2500.00'),
            realized_pnl=Decimal('1200.00')
        )
        
        assert event.total_value == Decimal('100000.00')
        assert event.cash == Decimal('25000.00')
        assert event.buying_power == Decimal('50000.00')
        assert event.equity == Decimal('100000.00')
        assert event.unrealized_pnl == Decimal('2500.00')
        assert event.realized_pnl == Decimal('1200.00')
        assert event.day_trade_count == 0  # Default value
    
    def test_risk_event(self):
        """Test RiskEvent creation and fields."""
        event = RiskEvent(
            event_id=UUID('12345678-1234-5678-1234-567812345678'),
            event_type=EventType.RISK_LIMIT_BREACH,
            timestamp=datetime.utcnow(),
            source="risk_manager",
            risk_type="max_drawdown",
            severity="HIGH",
            message="Maximum drawdown exceeded",
            current_value=0.06,
            threshold_value=0.05,
            action_taken="Position reduced"
        )
        
        assert event.risk_type == "max_drawdown"
        assert event.severity == "HIGH"
        assert event.message == "Maximum drawdown exceeded"
        assert event.current_value == 0.06
        assert event.threshold_value == 0.05
        assert event.action_taken == "Position reduced"
    
    def test_health_check_event(self):
        """Test HealthCheckEvent creation and fields."""
        event = HealthCheckEvent(
            event_id=UUID('12345678-1234-5678-1234-567812345678'),
            event_type=EventType.HEARTBEAT,
            timestamp=datetime.utcnow(),
            source="sentinel",
            component="database",
            status="HEALTHY",
            response_time_ms=25.5,
            details={"connections": 5, "pool_size": 20}
        )
        
        assert event.component == "database"
        assert event.status == "HEALTHY"
        assert event.response_time_ms == 25.5
        assert event.details == {"connections": 5, "pool_size": 20}
        assert event.is_healthy == True  # Default value


class TestEventEnums:
    """Test event-related enumerations."""
    
    def test_event_type_enum(self):
        """Test EventType enum values."""
        assert EventType.MARKET_DATA_RECEIVED.value == "market_data_received"
        assert EventType.SIGNAL_GENERATED.value == "signal_generated"
        assert EventType.ORDER_SUBMITTED.value == "order_submitted"
        assert EventType.ORDER_FILLED.value == "order_filled"
        assert EventType.RISK_LIMIT_BREACH.value == "risk_limit_breach"
        assert EventType.SYSTEM_STARTUP.value == "system_startup"
    
    def test_order_side_enum(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"
    
    def test_order_type_enum(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"
        assert OrderType.STOP.value == "STOP"
        assert OrderType.STOP_LIMIT.value == "STOP_LIMIT"
    
    def test_order_status_enum(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "PENDING"
        assert OrderStatus.SUBMITTED.value == "SUBMITTED"
        assert OrderStatus.PARTIAL_FILLED.value == "PARTIAL_FILLED"
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.CANCELLED.value == "CANCELLED"
        assert OrderStatus.REJECTED.value == "REJECTED"


if __name__ == "__main__":
    # Run tests manually if script is executed directly
    pytest.main([__file__, "-v"])
