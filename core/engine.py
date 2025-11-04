"""Core trading engine for Krator system.

This module implements the main async trading engine that orchestrates
market data, strategy execution, risk management, and order execution
with proper backpressure handling and error recovery.
"""

import asyncio
from asyncio import Queue, Event, Task
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set, Any, Callable, Awaitable
from uuid import uuid4
from dataclasses import dataclass
from enum import Enum

from loguru import logger
from core.events import (
    BaseEvent, EventType, MarketDataEvent, SignalEvent, OrderEvent,
    SystemEvent, create_system_event
)
from config.settings import get_settings
from infra.logging_config import system_logger


class EngineState(Enum):
    """Engine state enumeration."""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"
    RECOVERY = "RECOVERY"


@dataclass
class EngineMetrics:
    """Engine performance metrics."""
    events_processed: int = 0
    events_per_second: float = 0.0
    queue_depth: int = 0
    active_tasks: int = 0
    uptime_seconds: float = 0.0
    last_heartbeat: Optional[datetime] = None
    errors_count: int = 0
    recoveries_count: int = 0


class TradingEngine:
    """Main async trading engine with event-driven architecture.
    
    The engine processes events in the following order:
    1. Market Data Events → Strategy Processing
    2. Strategy Signals → Risk Management
    3. Risk-Approved Orders → Broker Execution
    4. Fill Events → Portfolio Updates
    """
    
    def __init__(
        self,
        max_queue_size: int = 10000,
        heartbeat_interval: int = 3600,
        recovery_timeout: int = 30
    ):
        """Initialize the trading engine.
        
        Args:
            max_queue_size: Maximum event queue size for backpressure
            heartbeat_interval: Heartbeat interval in seconds
            recovery_timeout: Recovery timeout in seconds
        """
        self.settings = get_settings()
        
        # Core state
        self.state = EngineState.STOPPED
        self.engine_id = str(uuid4())
        self.start_time: Optional[datetime] = None
        
        # Event processing
        self.event_queue: Queue[BaseEvent] = Queue(maxsize=max_queue_size)
        self.event_handlers: Dict[EventType, List[Callable]] = {}
        self.running_tasks: Set[Task] = set()
        
        # Control events
        self.shutdown_event = Event()
        self.recovery_event = Event()
        
        # Configuration
        self.heartbeat_interval = heartbeat_interval
        self.recovery_timeout = recovery_timeout
        
        # Metrics
        self.metrics = EngineMetrics()
        self._last_metrics_update = datetime.utcnow()
        
        # Recovery state
        self._recovery_attempts = 0
        self._max_recovery_attempts = 3
        
        # Register core event handlers
        self._register_core_handlers()
        
        system_logger.info(
            "Trading engine initialized",
            engine_id=self.engine_id,
            max_queue_size=max_queue_size,
            heartbeat_interval=heartbeat_interval
        )
    
    def _register_core_handlers(self) -> None:
        """Register core system event handlers."""
        self.register_handler(EventType.SYSTEM_STARTUP, self._handle_startup)
        self.register_handler(EventType.SYSTEM_SHUTDOWN, self._handle_shutdown)
        self.register_handler(EventType.HEARTBEAT, self._handle_heartbeat)
        self.register_handler(EventType.ERROR_OCCURRED, self._handle_error)
        self.register_handler(EventType.RECOVERY_COMPLETED, self._handle_recovery)
    
    def register_handler(
        self, 
        event_type: EventType, 
        handler: Callable[[BaseEvent], Awaitable[None]]
    ) -> None:
        """Register an event handler for a specific event type.
        
        Args:
            event_type: Type of event to handle
            handler: Async handler function
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        
        system_logger.debug(
            "Event handler registered",
            event_type=event_type.value,
            handler=handler.__name__,
            total_handlers=len(self.event_handlers[event_type])
        )
    
    async def publish_event(self, event: BaseEvent) -> None:
        """Publish an event to the processing queue.
        
        Args:
            event: Event to publish
            
        Raises:
            asyncio.QueueFull: If queue is full (backpressure)
        """
        try:
            # Non-blocking put with immediate feedback
            self.event_queue.put_nowait(event)
            
            system_logger.debug(
                "Event published",
                event_type=event.event_type.value,
                event_id=str(event.event_id),
                queue_size=self.event_queue.qsize()
            )
            
        except asyncio.QueueFull:
            system_logger.error(
                "Event queue full - applying backpressure",
                event_type=event.event_type.value,
                queue_size=self.event_queue.qsize()
            )
            
            # Publish queue full event for monitoring
            await self._handle_backpressure()
            raise
    
    async def _handle_backpressure(self) -> None:
        """Handle queue backpressure by implementing flow control."""
        self.metrics.errors_count += 1
        
        # Create backpressure event
        backpressure_event = create_system_event(
            component="engine",
            status="WARNING",
            message="Event queue full - backpressure applied",
            event_type=EventType.ERROR_OCCURRED,
            details={
                "queue_size": self.event_queue.qsize(),
                "max_queue_size": self.event_queue.maxsize
            }
        )
        
        # Try to handle backpressure event immediately
        await self._process_event(backpressure_event)
    
    async def start(self) -> None:
        """Start the trading engine."""
        if self.state != EngineState.STOPPED:
            raise RuntimeError(f"Engine is not in STOPPED state: {self.state}")
        
        self.state = EngineState.STARTING
        self.start_time = datetime.utcnow()
        self.shutdown_event.clear()
        self.recovery_event.clear()
        
        system_logger.info("Starting trading engine", engine_id=self.engine_id)
        
        try:
            # Start core tasks
            tasks = [
                self._event_processor(),
                self._heartbeat_task(),
                self._metrics_task()
            ]
            
            # Create and track tasks
            for task_coro in tasks:
                task = asyncio.create_task(task_coro)
                self.running_tasks.add(task)
                # Remove completed tasks automatically
                task.add_done_callback(self.running_tasks.discard)
            
            self.state = EngineState.RUNNING
            
            # Publish startup event
            startup_event = create_system_event(
                component="engine",
                status="RUNNING",
                message="Trading engine started successfully",
                event_type=EventType.SYSTEM_STARTUP
            )
            await self.publish_event(startup_event)
            
            system_logger.info(
                "Trading engine started successfully",
                engine_id=self.engine_id,
                active_tasks=len(self.running_tasks)
            )
            
        except Exception as e:
            self.state = EngineState.ERROR
            system_logger.error(
                "Failed to start trading engine",
                engine_id=self.engine_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def stop(self, timeout: float = 30.0) -> None:
        """Stop the trading engine gracefully.
        
        Args:
            timeout: Maximum time to wait for graceful shutdown
        """
        if self.state in [EngineState.STOPPED, EngineState.STOPPING]:
            return
        
        self.state = EngineState.STOPPING
        
        system_logger.info(
            "Stopping trading engine", 
            engine_id=self.engine_id,
            timeout=timeout
        )
        
        try:
            # Signal shutdown
            self.shutdown_event.set()
            
            # Publish shutdown event
            shutdown_event = create_system_event(
                component="engine",
                status="STOPPING",
                message="Trading engine shutdown initiated",
                event_type=EventType.SYSTEM_SHUTDOWN
            )
            await self.publish_event(shutdown_event)
            
            # Wait for tasks to complete gracefully
            if self.running_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self.running_tasks, return_exceptions=True),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    system_logger.warning(
                        "Graceful shutdown timeout - cancelling tasks",
                        active_tasks=len(self.running_tasks)
                    )
                    
                    # Cancel remaining tasks
                    for task in self.running_tasks:
                        if not task.done():
                            task.cancel()
                    
                    # Wait a bit for cancellations
                    await asyncio.sleep(1.0)
            
            self.state = EngineState.STOPPED
            
            system_logger.info(
                "Trading engine stopped",
                engine_id=self.engine_id,
                uptime_seconds=self.get_uptime_seconds()
            )
            
        except Exception as e:
            self.state = EngineState.ERROR
            system_logger.error(
                "Error during engine shutdown",
                engine_id=self.engine_id,
                error=str(e),
                exc_info=True
            )
            raise
    
    async def _event_processor(self) -> None:
        """Main event processing loop."""
        system_logger.info("Event processor started")
        
        while not self.shutdown_event.is_set():
            try:
                # Wait for event with timeout to check shutdown periodically
                try:
                    event = await asyncio.wait_for(
                        self.event_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue  # Check shutdown event and continue
                
                # Process the event
                await self._process_event(event)
                self.metrics.events_processed += 1
                
                # Mark task as done
                self.event_queue.task_done()
                
            except asyncio.CancelledError:
                system_logger.info("Event processor cancelled")
                break
            except Exception as e:
                self.metrics.errors_count += 1
                system_logger.error(
                    "Error in event processor",
                    error=str(e),
                    exc_info=True
                )
                
                # Try to recover from error
                await self._attempt_recovery(e)
        
        system_logger.info("Event processor stopped")
    
    async def _process_event(self, event: BaseEvent) -> None:
        """Process a single event by calling registered handlers.
        
        Args:
            event: Event to process
        """
        handlers = self.event_handlers.get(event.event_type, [])
        
        if not handlers:
            system_logger.debug(
                "No handlers registered for event type",
                event_type=event.event_type.value,
                event_id=str(event.event_id)
            )
            return
        
        system_logger.debug(
            "Processing event",
            event_type=event.event_type.value,
            event_id=str(event.event_id),
            handlers_count=len(handlers)
        )
        
        # Execute all handlers concurrently
        try:
            handler_tasks = [handler(event) for handler in handlers]
            await asyncio.gather(*handler_tasks, return_exceptions=True)
            
        except Exception as e:
            system_logger.error(
                "Error processing event",
                event_type=event.event_type.value,
                event_id=str(event.event_id),
                error=str(e),
                exc_info=True
            )
            raise
    
    async def _heartbeat_task(self) -> None:
        """Send periodic heartbeat events."""
        system_logger.info("Heartbeat task started", interval=self.heartbeat_interval)
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                if self.shutdown_event.is_set():
                    break
                
                # Create heartbeat event
                heartbeat_event = create_system_event(
                    component="engine",
                    status="HEALTHY",
                    message=f"Engine heartbeat - uptime: {self.get_uptime_seconds():.1f}s",
                    event_type=EventType.HEARTBEAT,
                    details={
                        "uptime_seconds": self.get_uptime_seconds(),
                        "events_processed": self.metrics.events_processed,
                        "queue_depth": self.event_queue.qsize(),
                        "state": self.state.value
                    }
                )
                
                await self.publish_event(heartbeat_event)
                self.metrics.last_heartbeat = datetime.utcnow()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                system_logger.error(
                    "Error in heartbeat task",
                    error=str(e),
                    exc_info=True
                )
        
        system_logger.info("Heartbeat task stopped")
    
    async def _metrics_task(self) -> None:
        """Update performance metrics periodically."""
        system_logger.info("Metrics task started")
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Update metrics every minute
                
                if self.shutdown_event.is_set():
                    break
                
                await self._update_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                system_logger.error(
                    "Error in metrics task",
                    error=str(e),
                    exc_info=True
                )
        
        system_logger.info("Metrics task stopped")
    
    async def _update_metrics(self) -> None:
        """Update engine performance metrics."""
        now = datetime.utcnow()
        time_delta = (now - self._last_metrics_update).total_seconds()
        
        if time_delta > 0:
            # Calculate events per second
            events_since_last = self.metrics.events_processed
            self.metrics.events_per_second = events_since_last / time_delta
            
        # Update current metrics
        self.metrics.queue_depth = self.event_queue.qsize()
        self.metrics.active_tasks = len(self.running_tasks)
        self.metrics.uptime_seconds = self.get_uptime_seconds()
        
        self._last_metrics_update = now
        
        system_logger.debug(
            "Metrics updated",
            events_per_second=self.metrics.events_per_second,
            queue_depth=self.metrics.queue_depth,
            active_tasks=self.metrics.active_tasks,
            uptime_seconds=self.metrics.uptime_seconds
        )
    
    async def _attempt_recovery(self, error: Exception) -> None:
        """Attempt to recover from an error.
        
        Args:
            error: The error that triggered recovery
        """
        if self._recovery_attempts >= self._max_recovery_attempts:
            system_logger.critical(
                "Maximum recovery attempts exceeded - engine entering error state",
                recovery_attempts=self._recovery_attempts,
                max_attempts=self._max_recovery_attempts,
                error=str(error)
            )
            self.state = EngineState.ERROR
            return
        
        self._recovery_attempts += 1
        self.state = EngineState.RECOVERY
        
        system_logger.warning(
            "Attempting recovery from error",
            recovery_attempt=self._recovery_attempts,
            max_attempts=self._max_recovery_attempts,
            error=str(error)
        )
        
        try:
            # Wait for recovery timeout
            await asyncio.sleep(self.recovery_timeout)
            
            # Clear recovery event and return to running state
            self.recovery_event.set()
            self.state = EngineState.RUNNING
            self.metrics.recoveries_count += 1
            
            # Publish recovery event
            recovery_event = create_system_event(
                component="engine",
                status="RECOVERED",
                message=f"Engine recovered from error (attempt {self._recovery_attempts})",
                event_type=EventType.RECOVERY_COMPLETED,
                details={
                    "recovery_attempt": self._recovery_attempts,
                    "original_error": str(error)
                }
            )
            await self.publish_event(recovery_event)
            
            system_logger.info(
                "Engine recovery successful",
                recovery_attempt=self._recovery_attempts,
                total_recoveries=self.metrics.recoveries_count
            )
            
        except Exception as recovery_error:
            system_logger.error(
                "Recovery attempt failed",
                recovery_attempt=self._recovery_attempts,
                original_error=str(error),
                recovery_error=str(recovery_error),
                exc_info=True
            )
    
    # Event handlers
    async def _handle_startup(self, event: SystemEvent) -> None:
        """Handle system startup events."""
        system_logger.info("System startup event received", component=event.component)
    
    async def _handle_shutdown(self, event: SystemEvent) -> None:
        """Handle system shutdown events."""
        system_logger.info("System shutdown event received", component=event.component)
    
    async def _handle_heartbeat(self, event: SystemEvent) -> None:
        """Handle heartbeat events."""
        system_logger.debug("Heartbeat event received", component=event.component)
    
    async def _handle_error(self, event: SystemEvent) -> None:
        """Handle error events."""
        system_logger.error(
            "Error event received",
            component=event.component,
            message=event.message,
            error_code=event.error_code
        )
    
    async def _handle_recovery(self, event: SystemEvent) -> None:
        """Handle recovery completion events."""
        system_logger.info(
            "Recovery event received",
            component=event.component,
            message=event.message
        )
        
        # Reset recovery attempts on successful recovery
        self._recovery_attempts = 0
    
    # Utility methods
    def get_uptime_seconds(self) -> float:
        """Get engine uptime in seconds."""
        if self.start_time is None:
            return 0.0
        return (datetime.utcnow() - self.start_time).total_seconds()
    
    def get_metrics(self) -> EngineMetrics:
        """Get current engine metrics."""
        return self.metrics
    
    def is_healthy(self) -> bool:
        """Check if engine is in a healthy state."""
        return self.state == EngineState.RUNNING and self.metrics.errors_count < 10
    
    async def wait_for_shutdown(self) -> None:
        """Wait for engine shutdown."""
        await self.shutdown_event.wait()
