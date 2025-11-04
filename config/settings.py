"""Configuration settings for Krator trading system.

This module uses Pydantic BaseSettings to handle environment variable loading
with proper type coercion and validation.
"""

import os
from decimal import Decimal
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field, validator, root_validator
from pydantic.env_settings import SettingsSourceCallable
from functools import lru_cache


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(
        default="postgresql://krator_user:krator_pass@localhost:5432/krator_db",
        env="DATABASE_URL",
        description="Database connection URL"
    )
    pool_size: int = Field(
        default=20,
        env="DATABASE_POOL_SIZE",
        description="Database connection pool size"
    )
    max_overflow: int = Field(
        default=30,
        env="DATABASE_MAX_OVERFLOW",
        description="Maximum overflow connections"
    )
    echo: bool = Field(
        default=False,
        env="DATABASE_ECHO",
        description="Enable SQLAlchemy query logging"
    )


class RedisSettings(BaseSettings):
    """Redis configuration settings."""
    
    url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL",
        description="Redis connection URL"
    )
    max_connections: int = Field(
        default=100,
        env="REDIS_MAX_CONNECTIONS",
        description="Maximum Redis connections"
    )
    socket_timeout: float = Field(
        default=5.0,
        env="REDIS_SOCKET_TIMEOUT",
        description="Redis socket timeout in seconds"
    )
    socket_keepalive: bool = Field(
        default=True,
        env="REDIS_SOCKET_KEEPALIVE",
        description="Enable Redis socket keepalive"
    )


class CelerySettings(BaseSettings):
    """Celery configuration settings with proper string coercion."""
    
    broker_url: str = Field(
        default="redis://localhost:6379/1",
        env="CELERY_BROKER_URL",
        description="Celery broker URL"
    )
    result_backend: str = Field(
        default="redis://localhost:6379/2",
        env="CELERY_RESULT_BACKEND",
        description="Celery result backend URL"
    )
    accept_content: str = Field(
        default="json",
        env="CELERY_ACCEPT_CONTENT",
        description="Celery accepted content types"
    )
    task_serializer: str = Field(
        default="json",
        env="CELERY_TASK_SERIALIZER",
        description="Celery task serializer"
    )
    result_serializer: str = Field(
        default="json",
        env="CELERY_RESULT_SERIALIZER",
        description="Celery result serializer"
    )
    timezone: str = Field(
        default="UTC",
        env="CELERY_TIMEZONE",
        description="Celery timezone"
    )
    enable_utc: bool = Field(
        default=True,
        env="CELERY_ENABLE_UTC",
        description="Enable UTC for Celery"
    )
    task_track_started: bool = Field(
        default=True,
        env="CELERY_TASK_TRACK_STARTED",
        description="Track task start events"
    )
    worker_prefetch_multiplier: int = Field(
        default=1,
        env="CELERY_WORKER_PREFETCH_MULTIPLIER",
        description="Worker prefetch multiplier for fairness"
    )


class TradingSettings(BaseSettings):
    """Trading configuration settings."""
    
    initial_capital: Decimal = Field(
        default=Decimal('100000.0'),
        env="INITIAL_CAPITAL",
        description="Initial trading capital"
    )
    max_portfolio_risk: float = Field(
        default=0.02,
        env="MAX_PORTFOLIO_RISK",
        description="Maximum portfolio risk as percentage"
    )
    max_daily_drawdown: float = Field(
        default=0.05,
        env="MAX_DAILY_DRAWDOWN",
        description="Maximum daily drawdown as percentage"
    )
    risk_free_rate: float = Field(
        default=0.02,
        env="RISK_FREE_RATE",
        description="Risk-free rate for calculations"
    )
    max_concurrent_orders: int = Field(
        default=10,
        env="MAX_CONCURRENT_ORDERS",
        description="Maximum concurrent orders"
    )
    order_timeout_seconds: int = Field(
        default=30,
        env="ORDER_TIMEOUT_SECONDS",
        description="Order timeout in seconds"
    )
    
    @validator('initial_capital', pre=True)
    def parse_initial_capital(cls, v):
        """Parse initial capital from string or number."""
        if isinstance(v, str):
            return Decimal(v)
        return Decimal(str(v))


class AlpacaSettings(BaseSettings):
    """Alpaca API configuration settings."""
    
    api_key: str = Field(
        env="ALPACA_API_KEY",
        description="Alpaca API key"
    )
    secret_key: str = Field(
        env="ALPACA_SECRET_KEY",
        description="Alpaca secret key"
    )
    base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        env="ALPACA_BASE_URL",
        description="Alpaca base URL"
    )
    data_url: str = Field(
        default="https://data.alpaca.markets",
        env="ALPACA_DATA_URL",
        description="Alpaca data URL"
    )
    stream_url: str = Field(
        default="wss://stream.data.alpaca.markets",
        env="ALPACA_STREAM_URL",
        description="Alpaca streaming URL"
    )
    paper_trading: bool = Field(
        default=True,
        env="ALPACA_PAPER_TRADING",
        description="Enable paper trading"
    )


class SlackSettings(BaseSettings):
    """Slack integration settings with proper string coercion."""
    
    webhook_url: Optional[str] = Field(
        default=None,
        env="SLACK_WEBHOOK_URL",
        description="Slack webhook URL for notifications"
    )
    mention_users: str = Field(
        default="",
        env="SLACK_MENTION_USERS",
        description="Comma-separated list of users to mention"
    )
    channel: str = Field(
        default="#trading-alerts",
        env="SLACK_CHANNEL",
        description="Default Slack channel for alerts"
    )
    enabled: bool = Field(
        default=True,
        env="SLACK_ENABLED",
        description="Enable Slack notifications"
    )
    timeout_seconds: int = Field(
        default=10,
        env="SLACK_TIMEOUT_SECONDS",
        description="Slack request timeout"
    )
    
    @property
    def mention_user_list(self) -> List[str]:
        """Get list of users to mention from comma-separated string."""
        if not self.mention_users or self.mention_users.strip() == "":
            return []
        return [user.strip() for user in self.mention_users.split(',') if user.strip()]


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level"
    )
    format: str = Field(
        default="json",
        env="LOG_FORMAT",
        description="Log format (json or text)"
    )
    rotation: str = Field(
        default="100 MB",
        env="LOG_ROTATION",
        description="Log rotation configuration"
    )
    retention: str = Field(
        default="30 days",
        env="LOG_RETENTION",
        description="Log retention period"
    )
    file_path: str = Field(
        default="logs/krator.log",
        env="LOG_FILE_PATH",
        description="Log file path"
    )
    enable_json: bool = Field(
        default=True,
        env="LOG_ENABLE_JSON",
        description="Enable JSON structured logging"
    )


class AIAssistSettings(BaseSettings):
    """AI-assisted RCA configuration settings."""
    
    enabled: bool = Field(
        default=False,
        env="AI_ASSIST_ENABLED",
        description="Enable AI-assisted RCA"
    )
    provider: str = Field(
        default="none",
        env="AI_ASSIST_PROVIDER",
        description="AI provider (openai, anthropic, none)"
    )
    api_key: Optional[str] = Field(
        default=None,
        env="AI_ASSIST_API_KEY",
        description="AI provider API key"
    )
    model: str = Field(
        default="gpt-4o-mini",
        env="AI_ASSIST_MODEL",
        description="AI model to use"
    )
    timeout_secs: int = Field(
        default=2,
        env="AI_ASSIST_TIMEOUT_SECS",
        description="AI request timeout in seconds"
    )
    cache_ttl_secs: int = Field(
        default=600,
        env="AI_ASSIST_CACHE_TTL_SECS",
        description="RCA cache TTL in seconds"
    )
    max_tokens: int = Field(
        default=500,
        env="AI_ASSIST_MAX_TOKENS",
        description="Maximum tokens for AI response"
    )


class MonitoringSettings(BaseSettings):
    """System monitoring configuration settings."""
    
    health_check_interval: int = Field(
        default=30,
        env="HEALTH_CHECK_INTERVAL",
        description="Health check interval in seconds"
    )
    heartbeat_interval: int = Field(
        default=3600,
        env="HEARTBEAT_INTERVAL",
        description="Heartbeat interval in seconds"
    )
    metrics_port: int = Field(
        default=9090,
        env="METRICS_PORT",
        description="Metrics server port"
    )
    stream_reconnect_attempts: int = Field(
        default=5,
        env="STREAM_RECONNECT_ATTEMPTS",
        description="Stream reconnection attempts"
    )
    stream_reconnect_delay: int = Field(
        default=5,
        env="STREAM_RECONNECT_DELAY",
        description="Stream reconnection delay in seconds"
    )
    market_data_buffer_size: int = Field(
        default=10000,
        env="MARKET_DATA_BUFFER_SIZE",
        description="Market data buffer size"
    )


class KratorSettings(BaseSettings):
    """Main Krator settings class combining all configuration."""
    
    # Environment
    environment: str = Field(
        default="development",
        env="ENVIRONMENT",
        description="Environment name"
    )
    debug: bool = Field(
        default=False,
        env="DEBUG",
        description="Enable debug mode"
    )
    test_mode: bool = Field(
        default=False,
        env="TEST_MODE",
        description="Enable test mode"
    )
    
    # Security
    secret_key: str = Field(
        default="change_this_in_production_to_a_secure_random_key",
        env="SECRET_KEY",
        description="Secret key for encryption"
    )
    encryption_key: Optional[str] = Field(
        default=None,
        env="ENCRYPTION_KEY",
        description="Base64 encoded encryption key"
    )
    
    # Nested settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    celery: CelerySettings = CelerySettings()
    trading: TradingSettings = TradingSettings()
    alpaca: AlpacaSettings = AlpacaSettings()
    slack: SlackSettings = SlackSettings()
    logging: LoggingSettings = LoggingSettings()
    ai_assist: AIAssistSettings = AIAssistSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = False
        validate_assignment = True
        
    @root_validator
    def validate_production_settings(cls, values):
        """Validate production-specific settings."""
        environment = values.get('environment', 'development')
        
        if environment == 'production':
            secret_key = values.get('secret_key')
            if secret_key == 'change_this_in_production_to_a_secure_random_key':
                raise ValueError(
                    "SECRET_KEY must be changed in production environment"
                )
                
            # Ensure Alpaca credentials are set
            alpaca = values.get('alpaca')
            if alpaca and (not alpaca.api_key or not alpaca.secret_key):
                raise ValueError(
                    "Alpaca API credentials must be set in production"
                )
                
        return values
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == 'production'
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == 'development'
    
    @property
    def is_testing(self) -> bool:
        """Check if running in test mode."""
        return self.test_mode


@lru_cache()
def get_settings() -> KratorSettings:
    """Get cached settings instance."""
    return KratorSettings()


# Export settings instance for easy access
settings = get_settings()
