"""Self-healing sentinel system for Krator trading platform.

This module implements comprehensive system monitoring, fault detection,
and automated recovery mechanisms with health checks and alerting.
"""

import asyncio
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import redis
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from loguru import logger
from config.settings import get_settings
from infra.logging_config import system_logger
from infra.alerts import send_system_alert, send_risk_alert


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"


class RecoveryAction(Enum):
    """Recovery action enumeration."""
    RESTART_COMPONENT = "RESTART_COMPONENT"
    RECONNECT = "RECONNECT"
    CLEAR_CACHE = "CLEAR_CACHE"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    ALERT_ONLY = "ALERT_ONLY"
    FAILOVER = "FAILOVER"


@dataclass
class HealthCheck:
    """Health check configuration."""
    name: str
    check_func: Callable[[], Awaitable[Dict[str, Any]]]
    interval_seconds: int = 30
    timeout_seconds: int = 10
    failure_threshold: int = 3
    recovery_threshold: int = 2
    enabled: bool = True
    critical: bool = False
    failure_count: int = field(default=0, init=False)
    success_count: int = field(default=0, init=False)
    last_check: Optional[datetime] = field(default=None, init=False)
    last_status: HealthStatus = field(default=HealthStatus.UNKNOWN, init=False)


@dataclass
class RecoveryRule:
    """Recovery rule configuration."""
    component: str
    condition: str  # Health check name or condition
    action: RecoveryAction
    max_attempts: int = 3
    cooldown_seconds: int = 300
    enabled: bool = True
    attempts_count: int = field(default=0, init=False)
    last_attempt: Optional[datetime] = field(default=None, init=False)


class SystemSentinel:
    """Self-healing system sentinel with monitoring and recovery."""
    
    def __init__(self):
        """Initialize the system sentinel."""
        self.settings = get_settings()
        self.running = False
        self.health_checks: Dict[str, HealthCheck] = {}
        self.recovery_rules: List[RecoveryRule] = []
        self.last_heartbeat = datetime.utcnow()
        self.system_metrics: Dict[str, Any] = {}
        
        # Component status tracking
        self.component_status: Dict[str, HealthStatus] = {}
        self.failure_counts: Dict[str, int] = {}
        
        # Recovery state
        self.circuit_breakers: Dict[str, datetime] = {}  # component -> trip_time
        self.recovery_in_progress: Dict[str, datetime] = {}
        
        # Setup health checks and recovery rules
        self._setup_health_checks()
        self._setup_recovery_rules()
        
        system_logger.info("System sentinel initialized", 
                          health_checks=len(self.health_checks),
                          recovery_rules=len(self.recovery_rules))
    
    def _setup_health_checks(self) -> None:
        """Setup default health checks."""
        self.health_checks = {
            'database': HealthCheck(
                name='database',
                check_func=self._check_database_health,
                interval_seconds=30,
                timeout_seconds=5,
                failure_threshold=3,
                critical=True
            ),
            'redis': HealthCheck(
                name='redis',
                check_func=self._check_redis_health,
                interval_seconds=30,
                timeout_seconds=5,
                failure_threshold=3,
                critical=True
            ),
            'market_data_stream': HealthCheck(
                name='market_data_stream',
                check_func=self._check_stream_health,
                interval_seconds=60,
                timeout_seconds=10,
                failure_threshold=2,
                critical=True
            ),
            'alpaca_api': HealthCheck(
                name='alpaca_api',
                check_func=self._check_alpaca_health,
                interval_seconds=60,
                timeout_seconds=10,
                failure_threshold=3,
                critical=False
            ),
            'system_resources': HealthCheck(
                name='system_resources',
                check_func=self._check_system_resources,
                interval_seconds=30,
                timeout_seconds=5,
                failure_threshold=5,
                critical=False
            ),
            'event_queue': HealthCheck(
                name='event_queue',
                check_func=self._check_event_queue_health,
                interval_seconds=30,
                timeout_seconds=5,
                failure_threshold=3,
                critical=True
            )
        }
    
    def _setup_recovery_rules(self) -> None:
        """Setup automated recovery rules."""
        self.recovery_rules = [
            RecoveryRule(
                component='database',
                condition='database',
                action=RecoveryAction.RECONNECT,
                max_attempts=3,
                cooldown_seconds=60
            ),
            RecoveryRule(
                component='redis',
                condition='redis',
                action=RecoveryAction.RECONNECT,
                max_attempts=3,
                cooldown_seconds=60
            ),
            RecoveryRule(
                component='market_data_stream',
                condition='market_data_stream',
                action=RecoveryAction.RESTART_COMPONENT,
                max_attempts=5,
                cooldown_seconds=30
            ),
            RecoveryRule(
                component='event_queue',
                condition='event_queue',
                action=RecoveryAction.CLEAR_CACHE,
                max_attempts=3,
                cooldown_seconds=120
            ),
            RecoveryRule(
                component='system',
                condition='system_resources',
                action=RecoveryAction.ALERT_ONLY,
                max_attempts=1,
                cooldown_seconds=300
            )
        ]
    
    async def start(self) -> None:
        """Start the sentinel monitoring system."""
        if self.running:
            return
        
        self.running = True
        system_logger.info("Starting system sentinel")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._monitoring_loop()),
            asyncio.create_task(self._recovery_loop()),
            asyncio.create_task(self._metrics_collection_loop())
        ]
        
        await send_system_alert(
            component="sentinel",
            status="STARTED",
            message="System sentinel started and monitoring system health"
        )
        
        # Wait for all tasks to complete (they run indefinitely)
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop(self) -> None:
        """Stop the sentinel monitoring system."""
        self.running = False
        system_logger.info("Stopping system sentinel")
        
        await send_system_alert(
            component="sentinel",
            status="STOPPED",
            message="System sentinel stopped"
        )
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop that runs health checks."""
        system_logger.info("Sentinel monitoring loop started")
        
        while self.running:
            try:
                # Run all enabled health checks
                check_tasks = []
                for check in self.health_checks.values():
                    if check.enabled and self._should_run_check(check):
                        check_tasks.append(self._run_health_check(check))
                
                if check_tasks:
                    await asyncio.gather(*check_tasks, return_exceptions=True)
                
                # Update heartbeat
                self.last_heartbeat = datetime.utcnow()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                system_logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(30)  # Wait longer on error
    
    async def _recovery_loop(self) -> None:
        """Recovery loop that handles automated recovery actions."""
        system_logger.info("Sentinel recovery loop started")
        
        while self.running:
            try:
                # Check for components that need recovery
                for rule in self.recovery_rules:
                    if rule.enabled and self._should_attempt_recovery(rule):
                        await self._execute_recovery_action(rule)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                system_logger.error(f"Error in recovery loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _metrics_collection_loop(self) -> None:
        """Collect and store system metrics."""
        system_logger.info("Sentinel metrics collection started")
        
        while self.running:
            try:
                # Collect system metrics
                metrics = await self._collect_system_metrics()
                self.system_metrics.update(metrics)
                
                # Check for anomalies in metrics
                await self._check_metric_anomalies(metrics)
                
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                system_logger.error(f"Error in metrics collection: {e}", exc_info=True)
                await asyncio.sleep(120)  # Wait longer on error
    
    def _should_run_check(self, check: HealthCheck) -> bool:
        """Determine if a health check should be run."""
        if check.last_check is None:
            return True
        
        time_since_last = (datetime.utcnow() - check.last_check).total_seconds()
        return time_since_last >= check.interval_seconds
    
    async def _run_health_check(self, check: HealthCheck) -> None:
        """Run a single health check."""
        start_time = time.time()
        
        try:
            # Run the check with timeout
            result = await asyncio.wait_for(
                check.check_func(),
                timeout=check.timeout_seconds
            )
            
            response_time = (time.time() - start_time) * 1000  # ms
            
            # Determine status from result
            status = HealthStatus.HEALTHY
            if not result.get('healthy', True):
                if result.get('critical', False):
                    status = HealthStatus.UNHEALTHY
                else:
                    status = HealthStatus.DEGRADED
            
            # Update check state
            check.last_check = datetime.utcnow()
            previous_status = check.last_status
            check.last_status = status
            
            if status == HealthStatus.HEALTHY:
                check.failure_count = 0
                check.success_count += 1
            else:
                check.failure_count += 1
                check.success_count = 0
            
            # Update component status
            self.component_status[check.name] = status
            
            # Log health check result
            system_logger.debug(
                "Health check completed",
                component=check.name,
                status=status.value,
                response_time_ms=response_time,
                failure_count=check.failure_count,
                details=result.get('details', {})
            )
            
            # Send alert on status change
            if previous_status != status and status != HealthStatus.HEALTHY:
                await self._send_health_alert(check, status, result)
            
        except asyncio.TimeoutError:
            await self._handle_check_timeout(check)
        except Exception as e:
            await self._handle_check_error(check, e)
    
    async def _handle_check_timeout(self, check: HealthCheck) -> None:
        """Handle health check timeout."""
        check.last_check = datetime.utcnow()
        check.last_status = HealthStatus.UNHEALTHY
        check.failure_count += 1
        check.success_count = 0
        
        self.component_status[check.name] = HealthStatus.UNHEALTHY
        
        system_logger.error(
            "Health check timeout",
            component=check.name,
            timeout_seconds=check.timeout_seconds,
            failure_count=check.failure_count
        )
        
        await send_system_alert(
            component=check.name,
            status="TIMEOUT",
            message=f"Health check for {check.name} timed out after {check.timeout_seconds}s",
            timeout_seconds=check.timeout_seconds,
            failure_count=check.failure_count
        )
    
    async def _handle_check_error(self, check: HealthCheck, error: Exception) -> None:
        """Handle health check error."""
        check.last_check = datetime.utcnow()
        check.last_status = HealthStatus.UNHEALTHY
        check.failure_count += 1
        check.success_count = 0
        
        self.component_status[check.name] = HealthStatus.UNHEALTHY
        
        system_logger.error(
            "Health check error",
            component=check.name,
            error=str(error),
            failure_count=check.failure_count,
            exc_info=True
        )
        
        await send_system_alert(
            component=check.name,
            status="ERROR",
            message=f"Health check for {check.name} failed: {str(error)}",
            error=str(error),
            failure_count=check.failure_count
        )
    
    async def _send_health_alert(self, check: HealthCheck, status: HealthStatus, result: Dict[str, Any]) -> None:
        """Send health status alert."""
        severity = "CRITICAL" if check.critical and status == HealthStatus.UNHEALTHY else "HIGH"
        
        await send_system_alert(
            component=check.name,
            status=status.value,
            message=f"Component {check.name} health changed to {status.value}",
            severity=severity,
            failure_count=check.failure_count,
            details=result.get('details', {})
        )
    
    def _should_attempt_recovery(self, rule: RecoveryRule) -> bool:
        """Determine if recovery should be attempted for a rule."""
        # Check if component is unhealthy
        component_status = self.component_status.get(rule.condition, HealthStatus.UNKNOWN)
        if component_status == HealthStatus.HEALTHY:
            return False
        
        # Check max attempts
        if rule.attempts_count >= rule.max_attempts:
            return False
        
        # Check cooldown period
        if rule.last_attempt:
            time_since_last = (datetime.utcnow() - rule.last_attempt).total_seconds()
            if time_since_last < rule.cooldown_seconds:
                return False
        
        return True
    
    async def _execute_recovery_action(self, rule: RecoveryRule) -> None:
        """Execute automated recovery action."""
        rule.attempts_count += 1
        rule.last_attempt = datetime.utcnow()
        
        system_logger.info(
            "Executing recovery action",
            component=rule.component,
            action=rule.action.value,
            attempt=rule.attempts_count,
            max_attempts=rule.max_attempts
        )
        
        try:
            success = False
            
            if rule.action == RecoveryAction.RECONNECT:
                success = await self._recovery_reconnect(rule.component)
            elif rule.action == RecoveryAction.RESTART_COMPONENT:
                success = await self._recovery_restart_component(rule.component)
            elif rule.action == RecoveryAction.CLEAR_CACHE:
                success = await self._recovery_clear_cache(rule.component)
            elif rule.action == RecoveryAction.CIRCUIT_BREAKER:
                success = await self._recovery_circuit_breaker(rule.component)
            elif rule.action == RecoveryAction.ALERT_ONLY:
                success = await self._recovery_alert_only(rule.component)
            
            # Send recovery result alert
            status = "RECOVERED" if success else "RECOVERY_FAILED"
            await send_system_alert(
                component=rule.component,
                status=status,
                message=f"Recovery action {rule.action.value} {'succeeded' if success else 'failed'}",
                action=rule.action.value,
                attempt=rule.attempts_count,
                success=success
            )
            
            if success:
                # Reset failure counts on successful recovery
                rule.attempts_count = 0
                if rule.component in self.failure_counts:
                    self.failure_counts[rule.component] = 0
            
        except Exception as e:
            system_logger.error(
                "Recovery action failed",
                component=rule.component,
                action=rule.action.value,
                error=str(e),
                exc_info=True
            )
    
    # Health check implementations
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            engine = create_engine(self.settings.database.url)
            start_time = time.time()
            
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                'healthy': response_time < 1000,  # Less than 1 second
                'critical': response_time > 5000,  # More than 5 seconds
                'details': {
                    'response_time_ms': response_time,
                    'connection_pool_size': self.settings.database.pool_size
                }
            }
            
        except SQLAlchemyError as e:
            return {
                'healthy': False,
                'critical': True,
                'details': {'error': str(e)}
            }
    
    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance."""
        try:
            r = redis.from_url(self.settings.redis.url)
            start_time = time.time()
            
            # Test basic operations
            r.ping()
            info = r.info()
            
            response_time = (time.time() - start_time) * 1000
            memory_usage_mb = info.get('used_memory', 0) / 1024 / 1024
            
            return {
                'healthy': response_time < 500,  # Less than 500ms
                'critical': response_time > 2000,  # More than 2 seconds
                'details': {
                    'response_time_ms': response_time,
                    'memory_usage_mb': memory_usage_mb,
                    'connected_clients': info.get('connected_clients', 0)
                }
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'critical': True,
                'details': {'error': str(e)}
            }
    
    async def _check_stream_health(self) -> Dict[str, Any]:
        """Check market data stream health."""
        # This would check the last received market data timestamp
        # For now, return a basic implementation
        last_data_time = self.system_metrics.get('last_market_data_time')
        
        if not last_data_time:
            return {
                'healthy': False,
                'critical': True,
                'details': {'error': 'No market data received'}
            }
        
        time_since_data = (datetime.utcnow() - last_data_time).total_seconds()
        
        return {
            'healthy': time_since_data < 60,  # Data within last minute
            'critical': time_since_data > 300,  # No data for 5 minutes
            'details': {
                'time_since_last_data_seconds': time_since_data,
                'last_data_time': last_data_time.isoformat()
            }
        }
    
    async def _check_alpaca_health(self) -> Dict[str, Any]:
        """Check Alpaca API connectivity."""
        try:
            # This would make an actual API call to Alpaca
            # For now, return a mock implementation
            return {
                'healthy': True,
                'critical': False,
                'details': {
                    'api_status': 'operational',
                    'rate_limit_remaining': 100
                }
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'critical': False,
                'details': {'error': str(e)}
            }
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            healthy = (
                cpu_percent < 80 and 
                memory_percent < 85 and 
                disk_percent < 90
            )
            
            critical = (
                cpu_percent > 95 or 
                memory_percent > 95 or 
                disk_percent > 95
            )
            
            return {
                'healthy': healthy,
                'critical': critical,
                'details': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'disk_percent': disk_percent,
                    'memory_available_gb': memory.available / 1024**3,
                    'disk_free_gb': disk.free / 1024**3,
                    'network_bytes_sent': net_io.bytes_sent,
                    'network_bytes_recv': net_io.bytes_recv
                }
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'critical': False,
                'details': {'error': str(e)}
            }
    
    async def _check_event_queue_health(self) -> Dict[str, Any]:
        """Check event queue health and depth."""
        # This would check the actual event queue depth
        # For now, return a mock implementation
        queue_depth = self.system_metrics.get('event_queue_depth', 0)
        max_queue_size = self.system_metrics.get('max_queue_size', 10000)
        
        queue_utilization = queue_depth / max_queue_size if max_queue_size > 0 else 0
        
        return {
            'healthy': queue_utilization < 0.8,  # Less than 80% full
            'critical': queue_utilization > 0.95,  # More than 95% full
            'details': {
                'queue_depth': queue_depth,
                'max_queue_size': max_queue_size,
                'queue_utilization': queue_utilization
            }
        }
    
    # Recovery action implementations
    
    async def _recovery_reconnect(self, component: str) -> bool:
        """Attempt to reconnect a component."""
        system_logger.info(f"Attempting to reconnect {component}")
        # Implementation would depend on the specific component
        await asyncio.sleep(1)  # Simulate reconnection time
        return True
    
    async def _recovery_restart_component(self, component: str) -> bool:
        """Attempt to restart a component."""
        system_logger.info(f"Attempting to restart {component}")
        # Implementation would restart the specific component
        await asyncio.sleep(2)  # Simulate restart time
        return True
    
    async def _recovery_clear_cache(self, component: str) -> bool:
        """Clear component cache."""
        system_logger.info(f"Clearing cache for {component}")
        # Implementation would clear relevant caches
        return True
    
    async def _recovery_circuit_breaker(self, component: str) -> bool:
        """Trip circuit breaker for component."""
        system_logger.info(f"Tripping circuit breaker for {component}")
        self.circuit_breakers[component] = datetime.utcnow()
        
        await send_risk_alert(
            risk_type="circuit_breaker",
            severity="CRITICAL",
            message=f"Circuit breaker tripped for {component}",
            component=component,
            action_taken="Trading halted for component"
        )
        
        return True
    
    async def _recovery_alert_only(self, component: str) -> bool:
        """Send alert without taking recovery action."""
        await send_system_alert(
            component=component,
            status="ALERT",
            message=f"Component {component} requires manual intervention"
        )
        return True
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics."""
        metrics = {
            'timestamp': datetime.utcnow(),
            'sentinel_uptime_seconds': (datetime.utcnow() - self.last_heartbeat).total_seconds(),
            'health_checks_total': len(self.health_checks),
            'health_checks_healthy': sum(
                1 for status in self.component_status.values() 
                if status == HealthStatus.HEALTHY
            ),
            'recovery_rules_active': sum(
                1 for rule in self.recovery_rules if rule.enabled
            ),
            'circuit_breakers_active': len(self.circuit_breakers)
        }
        
        return metrics
    
    async def _check_metric_anomalies(self, metrics: Dict[str, Any]) -> None:
        """Check for anomalies in collected metrics."""
        # Check if too many components are unhealthy
        unhealthy_count = sum(
            1 for status in self.component_status.values()
            if status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]
        )
        
        if unhealthy_count >= len(self.component_status) * 0.5:  # More than 50% unhealthy
            await send_risk_alert(
                risk_type="system_degradation",
                severity="CRITICAL",
                message=f"System degradation detected: {unhealthy_count}/{len(self.component_status)} components unhealthy",
                unhealthy_components=unhealthy_count,
                total_components=len(self.component_status)
            )
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.component_status:
            return HealthStatus.UNKNOWN
        
        statuses = list(self.component_status.values())
        
        # If any critical component is unhealthy, system is unhealthy
        for name, check in self.health_checks.items():
            if check.critical and self.component_status.get(name) == HealthStatus.UNHEALTHY:
                return HealthStatus.UNHEALTHY
        
        # If any component is unhealthy, system is degraded
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.DEGRADED
        
        # If any component is degraded, system is degraded
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        
        # All components healthy
        return HealthStatus.HEALTHY
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        return {
            'overall_status': self.get_overall_health().value,
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'components': {
                name: {
                    'status': status.value,
                    'failure_count': self.health_checks[name].failure_count,
                    'last_check': self.health_checks[name].last_check.isoformat() if self.health_checks[name].last_check else None
                }
                for name, status in self.component_status.items()
            },
            'circuit_breakers': {
                component: trip_time.isoformat()
                for component, trip_time in self.circuit_breakers.items()
            },
            'metrics': self.system_metrics
        }


# Test functions for development

def trigger_test_fault(component: str) -> None:
    """Trigger a test fault for RCA testing."""
    system_logger.warning(f"Test fault triggered for {component}")
    # This would simulate a fault in the specified component


# Export public interface
__all__ = [
    'SystemSentinel',
    'HealthStatus',
    'RecoveryAction',
    'trigger_test_fault'
]
