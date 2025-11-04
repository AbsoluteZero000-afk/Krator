"""Celery application configuration for Krator trading system.

This module sets up Celery for distributed task processing with Redis backend,
proper task routing, and trading-specific task definitions.
"""

import os
import asyncio
from celery import Celery, Task
from celery.signals import task_prerun, task_postrun, task_failure
from kombu import Queue
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from config.settings import get_settings
from infra.logging_config import system_logger

# Get settings
settings = get_settings()

# Create Celery application
app = Celery('krator')

# Configure Celery
app.conf.update(
    # Broker settings
    broker_url=settings.celery.broker_url,
    result_backend=settings.celery.result_backend,
    
    # Serialization
    accept_content=[settings.celery.accept_content],
    task_serializer=settings.celery.task_serializer,
    result_serializer=settings.celery.result_serializer,
    
    # Timezone
    timezone=settings.celery.timezone,
    enable_utc=settings.celery.enable_utc,
    
    # Task settings
    task_track_started=settings.celery.task_track_started,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Worker settings
    worker_prefetch_multiplier=settings.celery.worker_prefetch_multiplier,
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=False,
    
    # Result backend settings
    result_expires=3600,  # 1 hour
    result_persistent=True,
    
    # Task routing
    task_routes={
        'krator.tasks.execute_order': {'queue': 'critical'},
        'krator.tasks.update_position': {'queue': 'critical'},
        'krator.tasks.process_fill': {'queue': 'critical'},
        'krator.tasks.risk_check': {'queue': 'high'},
        'krator.tasks.calculate_indicators': {'queue': 'normal'},
        'krator.tasks.update_portfolio': {'queue': 'normal'},
        'krator.tasks.send_alert': {'queue': 'low'},
        'krator.tasks.cleanup_data': {'queue': 'low'},
        'krator.tasks.generate_report': {'queue': 'low'},
    },
    
    # Queue definitions
    task_default_queue='normal',
    task_queues=(
        Queue('critical', routing_key='critical', queue_arguments={'x-max-priority': 10}),
        Queue('high', routing_key='high', queue_arguments={'x-max-priority': 8}),
        Queue('normal', routing_key='normal', queue_arguments={'x-max-priority': 5}),
        Queue('low', routing_key='low', queue_arguments={'x-max-priority': 1}),
    ),
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        'update-portfolio-metrics': {
            'task': 'krator.tasks.update_portfolio_metrics',
            'schedule': 60.0,  # Every minute
            'options': {'queue': 'normal'}
        },
        'calculate-strategy-metrics': {
            'task': 'krator.tasks.calculate_strategy_metrics',
            'schedule': 300.0,  # Every 5 minutes
            'options': {'queue': 'normal'}
        },
        'system-health-check': {
            'task': 'krator.tasks.system_health_check',
            'schedule': 30.0,  # Every 30 seconds
            'options': {'queue': 'high'}
        },
        'cleanup-old-data': {
            'task': 'krator.tasks.cleanup_old_data',
            'schedule': timedelta(hours=6),  # Every 6 hours
            'options': {'queue': 'low'}
        },
        'send-daily-report': {
            'task': 'krator.tasks.send_daily_report',
            'schedule': timedelta(hours=24),  # Daily
            'options': {'queue': 'low'}
        },
    },
    beat_schedule_filename='celerybeat-schedule',
)


class CallbackTask(Task):
    """Base task class with callbacks and error handling."""
    
    def on_success(self, retval: Any, task_id: str, args: tuple, kwargs: dict) -> None:
        """Called when task succeeds."""
        system_logger.info(
            "Task completed successfully",
            task_id=task_id,
            task_name=self.name,
            result=str(retval)[:200] if retval else None,
            duration_ms=getattr(self.request, 'duration_ms', None)
        )
    
    def on_failure(self, exc, task_id: str, args: tuple, kwargs: dict, einfo) -> None:
        """Called when task fails."""
        system_logger.error(
            "Task failed",
            task_id=task_id,
            task_name=self.name,
            error=str(exc),
            args=args,
            kwargs=kwargs,
            traceback=str(einfo),
            duration_ms=getattr(self.request, 'duration_ms', None)
        )
    
    def on_retry(self, exc, task_id: str, args: tuple, kwargs: dict, einfo) -> None:
        """Called when task is retried."""
        system_logger.warning(
            "Task retry",
            task_id=task_id,
            task_name=self.name,
            error=str(exc),
            retry=self.request.retries,
            max_retries=self.max_retries
        )


# Set base task class
app.Task = CallbackTask


# Signal handlers for task monitoring
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Called before task execution."""
    task.request.start_time = datetime.utcnow()
    
    system_logger.debug(
        "Task started",
        task_id=task_id,
        task_name=sender,
        args=args,
        kwargs=kwargs
    )


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, 
                        retval=None, state=None, **kwds):
    """Called after task execution."""
    if hasattr(task.request, 'start_time'):
        duration = datetime.utcnow() - task.request.start_time
        task.request.duration_ms = duration.total_seconds() * 1000
    
    system_logger.debug(
        "Task completed",
        task_id=task_id,
        task_name=sender,
        state=state,
        duration_ms=getattr(task.request, 'duration_ms', None)
    )


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds):
    """Called when task fails."""
    system_logger.error(
        "Task failure signal",
        task_id=task_id,
        task_name=sender,
        exception=str(exception),
        traceback=str(traceback)
    )


# Trading Tasks

@app.task(bind=True, max_retries=3, default_retry_delay=60)
def execute_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a trading order.
    
    Args:
        order_data: Order information dictionary
        
    Returns:
        Execution result dictionary
    """
    try:
        from core.broker_alpaca import AlpacaBroker
        
        broker = AlpacaBroker()
        result = broker.submit_order(
            symbol=order_data['symbol'],
            side=order_data['side'],
            order_type=order_data['order_type'],
            quantity=order_data['quantity'],
            price=order_data.get('price'),
            time_in_force=order_data.get('time_in_force', 'DAY')
        )
        
        system_logger.info(
            "Order executed via Celery",
            order_id=result.get('id'),
            symbol=order_data['symbol'],
            side=order_data['side'],
            quantity=order_data['quantity']
        )
        
        return {
            'success': True,
            'order_id': result.get('id'),
            'status': result.get('status'),
            'message': 'Order executed successfully'
        }
        
    except Exception as exc:
        system_logger.error(f"Order execution failed: {exc}")
        raise self.retry(exc=exc, countdown=60, max_retries=3)


@app.task(bind=True)
def process_fill(self, fill_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process order fill and update position.
    
    Args:
        fill_data: Fill information dictionary
        
    Returns:
        Processing result
    """
    try:
        from core.portfolio import PortfolioManager
        
        portfolio = PortfolioManager()
        result = portfolio.process_fill(
            symbol=fill_data['symbol'],
            side=fill_data['side'],
            quantity=fill_data['quantity'],
            price=fill_data['price'],
            order_id=fill_data['order_id']
        )
        
        system_logger.info(
            "Fill processed",
            symbol=fill_data['symbol'],
            quantity=fill_data['quantity'],
            price=fill_data['price']
        )
        
        return {'success': True, 'position_updated': True}
        
    except Exception as exc:
        system_logger.error(f"Fill processing failed: {exc}")
        raise


@app.task(bind=True)
def calculate_indicators(self, symbol: str, timeframe: str = '1Min') -> Dict[str, Any]:
    """Calculate technical indicators for a symbol.
    
    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        
    Returns:
        Calculated indicators
    """
    try:
        from data.indicators import TechnicalIndicators
        from data.feed_historical import get_historical_data
        
        # Get historical data
        data = get_historical_data(symbol, timeframe, limit=200)
        
        if data is None or len(data) < 20:
            return {'success': False, 'message': 'Insufficient data'}
        
        # Calculate indicators
        indicators = TechnicalIndicators()
        results = indicators.calculate_all(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            volume=data['volume']
        )
        
        system_logger.debug(
            "Indicators calculated",
            symbol=symbol,
            indicators_count=len(results)
        )
        
        return {
            'success': True,
            'symbol': symbol,
            'indicators': results,
            'data_points': len(data)
        }
        
    except Exception as exc:
        system_logger.error(f"Indicator calculation failed: {exc}")
        raise


@app.task(bind=True)
def update_portfolio_metrics(self) -> Dict[str, Any]:
    """Update portfolio metrics and equity curve.
    
    Returns:
        Update result
    """
    try:
        from core.portfolio import PortfolioManager
        
        portfolio = PortfolioManager()
        metrics = portfolio.get_current_metrics()
        
        # Save to database
        portfolio.save_snapshot(metrics)
        
        system_logger.info(
            "Portfolio metrics updated",
            total_equity=float(metrics.get('total_equity', 0)),
            unrealized_pnl=float(metrics.get('unrealized_pnl', 0))
        )
        
        return {'success': True, 'metrics': metrics}
        
    except Exception as exc:
        system_logger.error(f"Portfolio metrics update failed: {exc}")
        raise


@app.task(bind=True)
def send_alert(self, alert_type: str, message: str, **kwargs) -> Dict[str, Any]:
    """Send alert notification.
    
    Args:
        alert_type: Type of alert (trade, risk, system)
        message: Alert message
        **kwargs: Additional alert data
        
    Returns:
        Send result
    """
    try:
        from infra.alerts import send_slack_alert
        
        result = send_slack_alert(
            alert_type=alert_type,
            message=message,
            **kwargs
        )
        
        system_logger.info(
            "Alert sent",
            alert_type=alert_type,
            message=message[:100]
        )
        
        return {'success': True, 'sent': result}
        
    except Exception as exc:
        system_logger.error(f"Alert sending failed: {exc}")
        raise


@app.task(bind=True)
def system_health_check(self) -> Dict[str, Any]:
    """Perform system health check.
    
    Returns:
        Health check results
    """
    try:
        from infra.sentinel import SystemSentinel
        
        sentinel = SystemSentinel()
        health_status = sentinel.check_system_health()
        
        system_logger.debug(
            "System health check completed",
            overall_status=health_status['overall_status'],
            components_checked=len(health_status['components'])
        )
        
        return {'success': True, 'health_status': health_status}
        
    except Exception as exc:
        system_logger.error(f"Health check failed: {exc}")
        raise


@app.task(bind=True)
def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, Any]:
    """Clean up old system data.
    
    Args:
        days_to_keep: Number of days of data to retain
        
    Returns:
        Cleanup results
    """
    try:
        from db.models import SystemEvent, OHLCVBar
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        # Create database session
        engine = create_engine(settings.database.url)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        # Clean up old system events
        events_deleted = session.query(SystemEvent).filter(
            SystemEvent.created_at < cutoff_date,
            SystemEvent.resolved == True
        ).delete()
        
        # Clean up old minute bars (keep daily and higher)
        bars_deleted = session.query(OHLCVBar).filter(
            OHLCVBar.timestamp < cutoff_date,
            OHLCVBar.timeframe == '1Min'
        ).delete()
        
        session.commit()
        session.close()
        
        system_logger.info(
            "Data cleanup completed",
            events_deleted=events_deleted,
            bars_deleted=bars_deleted,
            cutoff_date=cutoff_date.isoformat()
        )
        
        return {
            'success': True,
            'events_deleted': events_deleted,
            'bars_deleted': bars_deleted
        }
        
    except Exception as exc:
        system_logger.error(f"Data cleanup failed: {exc}")
        raise


@app.task(bind=True)
def send_daily_report(self) -> Dict[str, Any]:
    """Generate and send daily trading report.
    
    Returns:
        Report generation result
    """
    try:
        from core.analytics import generate_daily_report
        from infra.alerts import send_report_alert
        
        # Generate report
        report = generate_daily_report()
        
        # Send via Slack
        send_report_alert(report)
        
        system_logger.info(
            "Daily report sent",
            trades_count=report.get('trades_count', 0),
            net_pnl=report.get('net_pnl', 0)
        )
        
        return {'success': True, 'report': report}
        
    except Exception as exc:
        system_logger.error(f"Daily report failed: {exc}")
        raise


# Utility functions

def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get status of a Celery task.
    
    Args:
        task_id: Task ID
        
    Returns:
        Task status information
    """
    result = app.AsyncResult(task_id)
    return {
        'task_id': task_id,
        'state': result.state,
        'current': result.info.get('current', 0) if result.info else 0,
        'total': result.info.get('total', 1) if result.info else 1,
        'result': result.result if result.ready() else None
    }


def cancel_task(task_id: str) -> bool:
    """Cancel a running task.
    
    Args:
        task_id: Task ID to cancel
        
    Returns:
        True if cancellation was successful
    """
    try:
        app.control.revoke(task_id, terminate=True)
        system_logger.info("Task cancelled", task_id=task_id)
        return True
    except Exception as e:
        system_logger.error(f"Failed to cancel task {task_id}: {e}")
        return False


# Export the Celery app
__all__ = ['app', 'get_task_status', 'cancel_task']
