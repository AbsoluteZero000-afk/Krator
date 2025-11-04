"""Slack alerts system for Krator trading platform.

This module handles all Slack notifications including trade alerts,
risk notifications, system events, and daily reports with proper
error handling and formatting.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from decimal import Decimal

import aiohttp
from loguru import logger

from config.settings import get_settings
from infra.logging_config import system_logger


class SlackAlertManager:
    """Manager for Slack alerts with async webhook support."""
    
    def __init__(self):
        """Initialize Slack alert manager."""
        self.settings = get_settings()
        self.webhook_url = self.settings.slack.webhook_url
        self.enabled = self.settings.slack.enabled and bool(self.webhook_url)
        self.timeout = self.settings.slack.timeout_seconds
        self.mention_users = self.settings.slack.mention_user_list
        self.channel = self.settings.slack.channel
        
        if not self.enabled:
            system_logger.warning(
                "Slack alerts disabled",
                webhook_configured=bool(self.webhook_url),
                enabled=self.settings.slack.enabled
            )
    
    async def send_alert(
        self,
        message: str,
        alert_type: str = "info",
        title: Optional[str] = None,
        fields: Optional[List[Dict[str, str]]] = None,
        color: Optional[str] = None,
        mention_users: bool = False,
        **kwargs
    ) -> bool:
        """Send a Slack alert message.
        
        Args:
            message: Alert message text
            alert_type: Type of alert (info, warning, error, critical)
            title: Optional alert title
            fields: List of field dictionaries for structured data
            color: Hex color code for message border
            mention_users: Whether to mention configured users
            **kwargs: Additional alert data
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            system_logger.debug("Slack alerts disabled, skipping alert")
            return False
        
        try:
            # Build the payload
            payload = self._build_payload(
                message=message,
                alert_type=alert_type,
                title=title,
                fields=fields,
                color=color,
                mention_users=mention_users,
                **kwargs
            )
            
            # Send the webhook
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        system_logger.debug(
                            "Slack alert sent successfully",
                            alert_type=alert_type,
                            title=title,
                            message_length=len(message)
                        )
                        return True
                    else:
                        error_text = await response.text()
                        system_logger.error(
                            "Slack alert failed",
                            status=response.status,
                            error=error_text,
                            alert_type=alert_type
                        )
                        return False
                        
        except asyncio.TimeoutError:
            system_logger.error(
                "Slack alert timeout",
                timeout=self.timeout,
                alert_type=alert_type
            )
            return False
        except Exception as e:
            system_logger.error(
                "Slack alert error",
                error=str(e),
                alert_type=alert_type,
                exc_info=True
            )
            return False
    
    def _build_payload(
        self,
        message: str,
        alert_type: str,
        title: Optional[str] = None,
        fields: Optional[List[Dict[str, str]]] = None,
        color: Optional[str] = None,
        mention_users: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Build Slack webhook payload."""
        # Color mapping for different alert types
        color_map = {
            'info': '#36a64f',      # Green
            'warning': '#ffcc00',   # Yellow
            'error': '#ff6b6b',     # Red
            'critical': '#d63031',  # Dark red
            'trade': '#74b9ff',     # Blue
            'profit': '#00b894',    # Teal
            'loss': '#e17055'       # Orange
        }
        
        if not color:
            color = color_map.get(alert_type, '#cccccc')
        
        # Determine emoji for alert type
        emoji_map = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🔴',
            'trade': '📊',
            'profit': '📈',
            'loss': '📉',
            'system': '🔧',
            'heartbeat': '❤️'
        }
        
        emoji = emoji_map.get(alert_type, '📋')
        
        # Build mention string
        mentions = ""
        if mention_users and self.mention_users:
            mentions = " " + " ".join([f"<@{user}>" for user in self.mention_users])
        
        # Build the attachment
        attachment = {
            "color": color,
            "ts": int(datetime.utcnow().timestamp()),
            "footer": "Krator Trading System",
            "footer_icon": "https://raw.githubusercontent.com/AbsoluteZero000-afk/Krator/main/assets/krator-icon.png"
        }
        
        if title:
            attachment["title"] = f"{emoji} {title}"
        
        attachment["text"] = message
        
        if fields:
            attachment["fields"] = fields
        
        # Add additional context fields from kwargs
        context_fields = []
        for key, value in kwargs.items():
            if key not in ['mention_users', 'color', 'title', 'fields']:
                context_fields.append({
                    "title": key.replace('_', ' ').title(),
                    "value": str(value),
                    "short": True
                })
        
        if context_fields:
            attachment.setdefault("fields", []).extend(context_fields)
        
        # Build final payload
        payload = {
            "channel": self.channel,
            "username": "Krator Bot",
            "icon_emoji": ":robot_face:",
            "attachments": [attachment]
        }
        
        if mentions:
            payload["text"] = mentions
        
        return payload


# Global instance
_alert_manager = SlackAlertManager()


# Convenience functions

async def send_trade_alert(
    symbol: str,
    side: str,
    quantity: int,
    price: float,
    strategy: str,
    pnl: Optional[float] = None,
    **kwargs
) -> bool:
    """Send trade execution alert.
    
    Args:
        symbol: Trading symbol
        side: BUY or SELL
        quantity: Number of shares
        price: Execution price
        strategy: Strategy name
        pnl: Profit/loss if available
        **kwargs: Additional trade data
        
    Returns:
        True if sent successfully
    """
    alert_type = "trade"
    if pnl is not None:
        alert_type = "profit" if pnl > 0 else "loss"
    
    title = f"Trade Executed: {symbol}"
    message = f"{side} {quantity} shares of {symbol} at ${price:.2f}"
    
    fields = [
        {"title": "Symbol", "value": symbol, "short": True},
        {"title": "Side", "value": side, "short": True},
        {"title": "Quantity", "value": f"{quantity:,}", "short": True},
        {"title": "Price", "value": f"${price:.2f}", "short": True},
        {"title": "Strategy", "value": strategy, "short": True}
    ]
    
    if pnl is not None:
        fields.append({
            "title": "P&L", 
            "value": f"${pnl:+.2f}", 
            "short": True
        })
    
    return await _alert_manager.send_alert(
        message=message,
        alert_type=alert_type,
        title=title,
        fields=fields,
        **kwargs
    )


async def send_risk_alert(
    risk_type: str,
    severity: str,
    message: str,
    current_value: Optional[float] = None,
    threshold: Optional[float] = None,
    action_taken: Optional[str] = None,
    **kwargs
) -> bool:
    """Send risk management alert.
    
    Args:
        risk_type: Type of risk (drawdown, exposure, etc.)
        severity: Risk severity (LOW, MEDIUM, HIGH, CRITICAL)
        message: Risk message
        current_value: Current risk metric value
        threshold: Risk threshold
        action_taken: Action taken in response
        **kwargs: Additional risk data
        
    Returns:
        True if sent successfully
    """
    alert_type = "critical" if severity == "CRITICAL" else "warning"
    title = f"Risk Alert: {risk_type.title()}"
    
    fields = [
        {"title": "Risk Type", "value": risk_type, "short": True},
        {"title": "Severity", "value": severity, "short": True}
    ]
    
    if current_value is not None:
        fields.append({
            "title": "Current Value", 
            "value": f"{current_value:.2%}" if current_value < 1 else f"{current_value:.2f}", 
            "short": True
        })
    
    if threshold is not None:
        fields.append({
            "title": "Threshold", 
            "value": f"{threshold:.2%}" if threshold < 1 else f"{threshold:.2f}", 
            "short": True
        })
    
    if action_taken:
        fields.append({
            "title": "Action Taken", 
            "value": action_taken, 
            "short": False
        })
    
    return await _alert_manager.send_alert(
        message=message,
        alert_type=alert_type,
        title=title,
        fields=fields,
        mention_users=severity in ["HIGH", "CRITICAL"],
        **kwargs
    )


async def send_system_alert(
    component: str,
    status: str,
    message: str,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    **kwargs
) -> bool:
    """Send system status alert.
    
    Args:
        component: System component name
        status: Component status
        message: Status message
        error_code: Error code if applicable
        details: Additional details
        **kwargs: Additional system data
        
    Returns:
        True if sent successfully
    """
    alert_type = "error" if status in ["ERROR", "FAILED"] else "info"
    title = f"System Alert: {component}"
    
    fields = [
        {"title": "Component", "value": component, "short": True},
        {"title": "Status", "value": status, "short": True}
    ]
    
    if error_code:
        fields.append({
            "title": "Error Code", 
            "value": error_code, 
            "short": True
        })
    
    if details:
        for key, value in details.items():
            fields.append({
                "title": key.replace('_', ' ').title(),
                "value": str(value),
                "short": True
            })
    
    return await _alert_manager.send_alert(
        message=message,
        alert_type=alert_type,
        title=title,
        fields=fields,
        mention_users=alert_type == "error",
        **kwargs
    )


async def send_heartbeat_alert(
    uptime_seconds: float,
    events_processed: int,
    system_status: str = "HEALTHY",
    **kwargs
) -> bool:
    """Send system heartbeat alert.
    
    Args:
        uptime_seconds: System uptime in seconds
        events_processed: Number of events processed
        system_status: Overall system status
        **kwargs: Additional heartbeat data
        
    Returns:
        True if sent successfully
    """
    uptime_hours = uptime_seconds / 3600
    
    message = f"System is {system_status.lower()} and running for {uptime_hours:.1f} hours"
    
    fields = [
        {"title": "Status", "value": system_status, "short": True},
        {"title": "Uptime", "value": f"{uptime_hours:.1f}h", "short": True},
        {"title": "Events Processed", "value": f"{events_processed:,}", "short": True}
    ]
    
    return await _alert_manager.send_alert(
        message=message,
        alert_type="heartbeat",
        title="System Heartbeat",
        fields=fields,
        **kwargs
    )


async def send_daily_report(
    trading_date: datetime,
    total_trades: int,
    net_pnl: float,
    win_rate: float,
    sharpe_ratio: Optional[float] = None,
    max_drawdown: Optional[float] = None,
    **kwargs
) -> bool:
    """Send daily trading report.
    
    Args:
        trading_date: Date of the report
        total_trades: Number of trades executed
        net_pnl: Net profit/loss for the day
        win_rate: Win rate percentage
        sharpe_ratio: Sharpe ratio if available
        max_drawdown: Maximum drawdown
        **kwargs: Additional report data
        
    Returns:
        True if sent successfully
    """
    alert_type = "profit" if net_pnl > 0 else "loss" if net_pnl < 0 else "info"
    title = f"Daily Report - {trading_date.strftime('%Y-%m-%d')}"
    
    pnl_emoji = "📈" if net_pnl > 0 else "📉" if net_pnl < 0 else "📊"
    message = f"{pnl_emoji} Net P&L: ${net_pnl:+.2f} | Trades: {total_trades} | Win Rate: {win_rate:.1f}%"
    
    fields = [
        {"title": "Total Trades", "value": str(total_trades), "short": True},
        {"title": "Net P&L", "value": f"${net_pnl:+.2f}", "short": True},
        {"title": "Win Rate", "value": f"{win_rate:.1f}%", "short": True}
    ]
    
    if sharpe_ratio is not None:
        fields.append({
            "title": "Sharpe Ratio", 
            "value": f"{sharpe_ratio:.2f}", 
            "short": True
        })
    
    if max_drawdown is not None:
        fields.append({
            "title": "Max Drawdown", 
            "value": f"{max_drawdown:.2%}", 
            "short": True
        })
    
    return await _alert_manager.send_alert(
        message=message,
        alert_type=alert_type,
        title=title,
        fields=fields,
        **kwargs
    )


# Synchronous wrapper functions for backward compatibility

def alert_trade(symbol: str, side: str, quantity: int, price: float, strategy: str, **kwargs) -> bool:
    """Synchronous wrapper for trade alerts."""
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            send_trade_alert(symbol, side, quantity, price, strategy, **kwargs)
        )
    except RuntimeError:
        # No event loop running, create one
        return asyncio.run(
            send_trade_alert(symbol, side, quantity, price, strategy, **kwargs)
        )


def alert_risk(risk_type: str, severity: str, message: str, **kwargs) -> bool:
    """Synchronous wrapper for risk alerts."""
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            send_risk_alert(risk_type, severity, message, **kwargs)
        )
    except RuntimeError:
        return asyncio.run(
            send_risk_alert(risk_type, severity, message, **kwargs)
        )


def alert_system(component: str, status: str, message: str, **kwargs) -> bool:
    """Synchronous wrapper for system alerts."""
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            send_system_alert(component, status, message, **kwargs)
        )
    except RuntimeError:
        return asyncio.run(
            send_system_alert(component, status, message, **kwargs)
        )


def alert_heartbeat(message: str, **kwargs) -> bool:
    """Synchronous wrapper for heartbeat alerts."""
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            _alert_manager.send_alert(
                message=message,
                alert_type="heartbeat",
                title="System Heartbeat",
                **kwargs
            )
        )
    except RuntimeError:
        return asyncio.run(
            _alert_manager.send_alert(
                message=message,
                alert_type="heartbeat",
                title="System Heartbeat",
                **kwargs
            )
        )


# Export public interface
__all__ = [
    'SlackAlertManager',
    'send_trade_alert',
    'send_risk_alert', 
    'send_system_alert',
    'send_heartbeat_alert',
    'send_daily_report',
    'alert_trade',
    'alert_risk',
    'alert_system',
    'alert_heartbeat'
]
