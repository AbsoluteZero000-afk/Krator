"""AI-assisted Root Cause Analysis for Krator trading system.

This module provides AI-powered incident analysis with strict privacy controls,
data redaction, timeout handling, and fallback mechanisms.
"""

import asyncio
import json
import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

import aiohttp
from loguru import logger

from config.settings import get_settings
from infra.logging_config import system_logger
from infra.alerts import send_system_alert


class AIProvider(Enum):
    """AI provider enumeration."""
    NONE = "none"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class RCAResult:
    """Root Cause Analysis result."""
    title: str
    confidence: float  # 0.0 to 1.0
    primary_cause: str
    next_steps: List[str]
    analysis_time_ms: int
    data_hash: str
    provider: str
    model: str


class DataRedactor:
    """Data redaction utility for privacy protection."""
    
    def __init__(self):
        """Initialize data redactor with redaction patterns."""
        self.patterns = {
            # API keys and secrets
            'api_key': re.compile(r'(sk-|pk-|secret_)[a-zA-Z0-9_-]{20,}', re.IGNORECASE),
            'bearer_token': re.compile(r'Bearer [a-zA-Z0-9_-]{20,}', re.IGNORECASE),
            'auth_header': re.compile(r'Authorization: [^\n\r]+', re.IGNORECASE),
            
            # Account identifiers
            'account_id': re.compile(r'(account[_-]?id|acct)[_\s]*[:=][_\s]*[a-zA-Z0-9-]{10,}', re.IGNORECASE),
            'user_id': re.compile(r'(user[_-]?id|uid)[_\s]*[:=][_\s]*[a-zA-Z0-9-]{8,}', re.IGNORECASE),
            
            # Email addresses
            'email': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            
            # URLs with query parameters (potential sensitive data)
            'url_params': re.compile(r'https?://[^\s]*\?[^\s]*'),
            
            # IP addresses (internal networks)
            'private_ip': re.compile(r'\b(?:10\.|172\.(?:1[6-9]|2[0-9]|3[01])\.|192\.168\.)\d{1,3}\.\d{1,3}\b'),
            
            # Database connection strings
            'db_connection': re.compile(r'(postgresql|mysql|mongodb)://[^\s]+', re.IGNORECASE),
            
            # Redis URLs
            'redis_url': re.compile(r'redis://[^\s]+', re.IGNORECASE),
            
            # JWT tokens
            'jwt': re.compile(r'eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*'),
        }
        
        self.replacements = {
            'api_key': 'sk_***',
            'bearer_token': 'Bearer ***',
            'auth_header': 'Authorization: ***',
            'account_id': 'acct_***',
            'user_id': 'user_***',
            'email': '***@redacted',
            'url_params': 'https://redacted.com/***',
            'private_ip': '***.***.***.***.***',
            'db_connection': 'postgresql://***:***@redacted/***',
            'redis_url': 'redis://redacted:***',
            'jwt': 'jwt_***'
        }
    
    def redact_data(self, data: Union[str, Dict[str, Any], List[Any]]) -> Union[str, Dict[str, Any], List[Any]]:
        """Redact sensitive data from input.
        
        Args:
            data: Data to redact (string, dict, or list)
            
        Returns:
            Redacted data with same structure
        """
        if isinstance(data, str):
            return self._redact_string(data)
        elif isinstance(data, dict):
            return {k: self.redact_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.redact_data(item) for item in data]
        else:
            return data
    
    def _redact_string(self, text: str) -> str:
        """Redact sensitive patterns from string.
        
        Args:
            text: Text to redact
            
        Returns:
            Redacted text
        """
        for pattern_name, pattern in self.patterns.items():
            replacement = self.replacements.get(pattern_name, '***')
            text = pattern.sub(replacement, text)
        
        return text


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def analyze_incident(
        self, 
        incident_data: Dict[str, Any], 
        timeout_seconds: int = 2
    ) -> RCAResult:
        """Analyze incident and return RCA result.
        
        Args:
            incident_data: Incident information
            timeout_seconds: Analysis timeout
            
        Returns:
            RCA analysis result
        """
        pass


class OpenAIClient(LLMClient):
    """OpenAI client for RCA analysis."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for analysis
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    async def analyze_incident(
        self, 
        incident_data: Dict[str, Any], 
        timeout_seconds: int = 2
    ) -> RCAResult:
        """Analyze incident using OpenAI API."""
        start_time = datetime.utcnow()
        
        try:
            # Build prompt
            prompt = self._build_prompt(incident_data)
            
            # Make API call
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout_seconds)
            ) as session:
                async with session.post(
                    self.base_url,
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"OpenAI API error: {response.status} - {error_text}")
                    
                    result = await response.json()
                    content = result['choices'][0]['message']['content']
                    
                    # Parse JSON response
                    analysis = json.loads(content)
                    
                    analysis_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                    
                    return RCAResult(
                        title=analysis.get('title', 'Incident Analysis'),
                        confidence=float(analysis.get('confidence', 0.5)),
                        primary_cause=analysis.get('primary_cause', 'Unknown'),
                        next_steps=analysis.get('next_steps', []),
                        analysis_time_ms=analysis_time,
                        data_hash=self._hash_incident_data(incident_data),
                        provider="openai",
                        model=self.model
                    )
        
        except asyncio.TimeoutError:
            system_logger.warning("OpenAI RCA timeout", timeout_seconds=timeout_seconds)
            raise
        except Exception as e:
            system_logger.error(f"OpenAI RCA error: {e}", exc_info=True)
            raise
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for RCA analysis."""
        return '''
You are a financial trading system analyst specializing in root cause analysis.
Analyze trading system incidents and provide structured RCA.

Constraints:
- Focus on trading-specific issues: market data delays, order routing failures, risk limit breaches, connectivity problems
- Provide confidence level between 0.0 and 1.0
- Suggest 2-4 actionable next steps
- Be concise and technical
- Never suggest autonomous actions or code changes
- Only analyze the provided data, do not make assumptions

Response format (JSON only):
{
    "title": "Brief incident summary",
    "confidence": 0.75,
    "primary_cause": "Most likely root cause",
    "next_steps": [
        "Specific action 1",
        "Specific action 2"
    ]
}
'''
    
    def _build_prompt(self, incident_data: Dict[str, Any]) -> str:
        """Build analysis prompt from incident data."""
        prompt_parts = ["Trading system incident analysis:"]
        
        # Add incident metadata
        if 'timestamp' in incident_data:
            prompt_parts.append(f"Timestamp: {incident_data['timestamp']}")
        if 'component' in incident_data:
            prompt_parts.append(f"Component: {incident_data['component']}")
        if 'severity' in incident_data:
            prompt_parts.append(f"Severity: {incident_data['severity']}")
        
        # Add error information
        if 'error_message' in incident_data:
            prompt_parts.append(f"Error: {incident_data['error_message']}")
        if 'error_count' in incident_data:
            prompt_parts.append(f"Error count: {incident_data['error_count']}")
        
        # Add system metrics
        if 'metrics' in incident_data:
            prompt_parts.append("System metrics:")
            for key, value in incident_data['metrics'].items():
                prompt_parts.append(f"- {key}: {value}")
        
        # Add recent events
        if 'recent_events' in incident_data:
            prompt_parts.append("Recent events:")
            for event in incident_data['recent_events'][-5:]:  # Last 5 events
                prompt_parts.append(f"- {event}")
        
        return "\n".join(prompt_parts)
    
    def _hash_incident_data(self, data: Dict[str, Any]) -> str:
        """Generate hash of incident data for caching."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()[:12]


class AnthropicClient(LLMClient):
    """Anthropic Claude client for RCA analysis."""
    
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        """Initialize Anthropic client."""
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"
    
    async def analyze_incident(
        self, 
        incident_data: Dict[str, Any], 
        timeout_seconds: int = 2
    ) -> RCAResult:
        """Analyze incident using Anthropic API."""
        start_time = datetime.utcnow()
        
        try:
            prompt = self._build_prompt(incident_data)
            
            payload = {
                "model": self.model,
                "max_tokens": 500,
                "temperature": 0.1,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout_seconds)
            ) as session:
                async with session.post(
                    self.base_url,
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Anthropic API error: {response.status} - {error_text}")
                    
                    result = await response.json()
                    content = result['content'][0]['text']
                    
                    # Extract JSON from response
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if not json_match:
                        raise Exception("No JSON found in Anthropic response")
                    
                    analysis = json.loads(json_match.group())
                    
                    analysis_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                    
                    return RCAResult(
                        title=analysis.get('title', 'Incident Analysis'),
                        confidence=float(analysis.get('confidence', 0.5)),
                        primary_cause=analysis.get('primary_cause', 'Unknown'),
                        next_steps=analysis.get('next_steps', []),
                        analysis_time_ms=analysis_time,
                        data_hash=self._hash_incident_data(incident_data),
                        provider="anthropic",
                        model=self.model
                    )
        
        except asyncio.TimeoutError:
            system_logger.warning("Anthropic RCA timeout", timeout_seconds=timeout_seconds)
            raise
        except Exception as e:
            system_logger.error(f"Anthropic RCA error: {e}", exc_info=True)
            raise
    
    def _build_prompt(self, incident_data: Dict[str, Any]) -> str:
        """Build analysis prompt for Anthropic."""
        system_context = '''
You are a financial trading system analyst. Analyze this trading system incident and provide root cause analysis.

Focus on trading-specific issues:
- Market data delays or gaps
- Order routing failures
- Risk limit breaches
- Database connectivity issues
- API rate limiting
- Memory/CPU resource problems

Provide response as JSON:
{
    "title": "Brief incident summary",
    "confidence": 0.75,
    "primary_cause": "Most likely root cause",
    "next_steps": ["Action 1", "Action 2"]
}
'''
        
        prompt = f"{system_context}\n\nIncident data:\n{json.dumps(incident_data, indent=2)}"
        return prompt
    
    def _hash_incident_data(self, data: Dict[str, Any]) -> str:
        """Generate hash of incident data for caching."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()[:12]


class AIAssistedRCA:
    """AI-assisted root cause analysis system."""
    
    def __init__(self):
        """Initialize AI-assisted RCA system."""
        self.settings = get_settings()
        self.enabled = self.settings.ai_assist.enabled
        self.provider = AIProvider(self.settings.ai_assist.provider)
        self.timeout_seconds = self.settings.ai_assist.timeout_secs
        self.cache_ttl_seconds = self.settings.ai_assist.cache_ttl_secs
        
        # Initialize components
        self.redactor = DataRedactor()
        self.llm_client: Optional[LLMClient] = None
        self.cache: Dict[str, tuple[RCAResult, datetime]] = {}
        
        # Initialize LLM client
        if self.enabled and self.provider != AIProvider.NONE:
            self._initialize_llm_client()
        
        system_logger.info(
            "AI-assisted RCA initialized",
            enabled=self.enabled,
            provider=self.provider.value,
            timeout_seconds=self.timeout_seconds
        )
    
    def _initialize_llm_client(self) -> None:
        """Initialize the appropriate LLM client."""
        api_key = self.settings.ai_assist.api_key
        model = self.settings.ai_assist.model
        
        if not api_key:
            system_logger.warning("AI assist enabled but no API key provided")
            self.enabled = False
            return
        
        try:
            if self.provider == AIProvider.OPENAI:
                self.llm_client = OpenAIClient(api_key, model)
            elif self.provider == AIProvider.ANTHROPIC:
                self.llm_client = AnthropicClient(api_key, model)
            else:
                system_logger.warning(f"Unsupported AI provider: {self.provider.value}")
                self.enabled = False
        
        except Exception as e:
            system_logger.error(f"Failed to initialize LLM client: {e}")
            self.enabled = False
    
    async def analyze_incident(
        self,
        incident_data: Dict[str, Any],
        use_cache: bool = True
    ) -> Optional[RCAResult]:
        """Analyze incident with AI assistance.
        
        Args:
            incident_data: Incident information
            use_cache: Whether to use cached results
            
        Returns:
            RCA result or None if analysis fails/disabled
        """
        if not self.enabled or not self.llm_client:
            system_logger.debug("AI-assisted RCA disabled or not configured")
            return None
        
        try:
            # Redact sensitive data
            clean_data = self.redactor.redact_data(incident_data)
            
            # Check cache
            if use_cache:
                cached_result = self._get_cached_result(clean_data)
                if cached_result:
                    system_logger.debug("Using cached RCA result")
                    return cached_result
            
            # Perform AI analysis
            result = await self.llm_client.analyze_incident(
                clean_data, 
                self.timeout_seconds
            )
            
            # Cache result
            if use_cache:
                self._cache_result(clean_data, result)
            
            # Log successful analysis
            system_logger.info(
                "AI RCA analysis completed",
                provider=result.provider,
                confidence=result.confidence,
                analysis_time_ms=result.analysis_time_ms,
                data_hash=result.data_hash
            )
            
            return result
            
        except asyncio.TimeoutError:
            system_logger.warning(
                "AI RCA analysis timed out",
                timeout_seconds=self.timeout_seconds,
                provider=self.provider.value
            )
            return None
        
        except Exception as e:
            system_logger.error(
                "AI RCA analysis failed",
                provider=self.provider.value,
                error=str(e),
                exc_info=True
            )
            return None
    
    def _get_cached_result(self, incident_data: Dict[str, Any]) -> Optional[RCAResult]:
        """Get cached RCA result if available and not expired."""
        data_hash = self._hash_data(incident_data)
        
        if data_hash in self.cache:
            result, timestamp = self.cache[data_hash]
            
            # Check if cache entry is still valid
            age_seconds = (datetime.utcnow() - timestamp).total_seconds()
            if age_seconds < self.cache_ttl_seconds:
                return result
            else:
                # Remove expired entry
                del self.cache[data_hash]
        
        return None
    
    def _cache_result(self, incident_data: Dict[str, Any], result: RCAResult) -> None:
        """Cache RCA result with timestamp."""
        data_hash = self._hash_data(incident_data)
        self.cache[data_hash] = (result, datetime.utcnow())
        
        # Clean up old cache entries (keep last 100)
        if len(self.cache) > 100:
            # Remove oldest entries
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            
            # Keep only the 50 most recent entries
            self.cache = dict(sorted_items[-50:])
    
    def _hash_data(self, data: Dict[str, Any]) -> str:
        """Generate hash for data caching."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def clear_cache(self) -> None:
        """Clear the RCA result cache."""
        self.cache.clear()
        system_logger.info("AI RCA cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = datetime.utcnow()
        valid_entries = 0
        expired_entries = 0
        
        for result, timestamp in self.cache.values():
            age_seconds = (now - timestamp).total_seconds()
            if age_seconds < self.cache_ttl_seconds:
                valid_entries += 1
            else:
                expired_entries += 1
        
        return {
            'total_entries': len(self.cache),
            'valid_entries': valid_entries,
            'expired_entries': expired_entries,
            'cache_ttl_seconds': self.cache_ttl_seconds
        }


# Global instance
_ai_rca = AIAssistedRCA()


# Convenience functions

async def analyze_system_incident(
    component: str,
    error_message: str,
    severity: str = "HIGH",
    metrics: Optional[Dict[str, Any]] = None,
    recent_events: Optional[List[str]] = None
) -> Optional[RCAResult]:
    """Analyze a system incident with AI assistance.
    
    Args:
        component: System component with issue
        error_message: Error message or description
        severity: Incident severity level
        metrics: System metrics at time of incident
        recent_events: Recent system events
        
    Returns:
        RCA result or None if analysis fails
    """
    incident_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'component': component,
        'error_message': error_message,
        'severity': severity,
        'metrics': metrics or {},
        'recent_events': recent_events or []
    }
    
    return await _ai_rca.analyze_incident(incident_data)


async def analyze_trading_incident(
    symbol: str,
    incident_type: str,
    description: str,
    order_data: Optional[Dict[str, Any]] = None,
    market_conditions: Optional[Dict[str, Any]] = None
) -> Optional[RCAResult]:
    """Analyze a trading-specific incident.
    
    Args:
        symbol: Trading symbol affected
        incident_type: Type of incident (order_failure, data_delay, etc.)
        description: Incident description
        order_data: Related order information
        market_conditions: Market conditions at time of incident
        
    Returns:
        RCA result or None if analysis fails
    """
    incident_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'incident_type': incident_type,
        'symbol': symbol,
        'description': description,
        'order_data': order_data or {},
        'market_conditions': market_conditions or {},
        'component': 'trading_system'
    }
    
    return await _ai_rca.analyze_incident(incident_data)


def test_data_redaction() -> Dict[str, str]:
    """Test data redaction functionality.
    
    Returns:
        Dictionary showing original and redacted data
    """
    test_data = {
        'api_key': 'sk-1234567890abcdef1234567890abcdef',
        'email': 'trader@example.com',
        'account_id': 'acct_1234567890',
        'database_url': 'postgresql://user:pass@localhost/db',
        'normal_text': 'This is normal log data'
    }
    
    redactor = DataRedactor()
    redacted = redactor.redact_data(test_data)
    
    return {
        'original': str(test_data),
        'redacted': str(redacted)
    }


# Export public interface
__all__ = [
    'AIAssistedRCA',
    'RCAResult',
    'DataRedactor',
    'analyze_system_incident',
    'analyze_trading_incident',
    'test_data_redaction'
]
