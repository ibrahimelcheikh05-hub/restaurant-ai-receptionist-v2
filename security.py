"""
Security
========
Security layer for authentication and authorization.

Responsibilities:
- API key validation
- Tenant authentication
- Request validation
- Rate limiting
- Security headers
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone, timedelta
import hashlib
import hmac
import secrets

logger = logging.getLogger(__name__)


class SecurityConfig:
    """Security configuration."""
    
    def __init__(
        self,
        require_api_key: bool = True,
        require_tenant_auth: bool = True,
        enable_rate_limiting: bool = True,
        max_requests_per_minute: int = 60
    ):
        """
        Initialize security config.
        
        Args:
            require_api_key: Require API key authentication
            require_tenant_auth: Require tenant authentication
            enable_rate_limiting: Enable rate limiting
            max_requests_per_minute: Max requests per minute
        """
        self.require_api_key = require_api_key
        self.require_tenant_auth = require_tenant_auth
        self.enable_rate_limiting = enable_rate_limiting
        self.max_requests_per_minute = max_requests_per_minute


class RateLimiter:
    """
    Simple rate limiter.
    
    Uses sliding window algorithm.
    """
    
    def __init__(self, max_requests: int, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests in window
            window_seconds: Window size in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        
        # Track requests per key
        self._requests: Dict[str, list] = {}
    
    def is_allowed(self, key: str) -> bool:
        """
        Check if request is allowed.
        
        Args:
            key: Rate limit key (e.g., API key, IP)
            
        Returns:
            True if allowed
        """
        now = datetime.now(timezone.utc)
        
        # Initialize key if needed
        if key not in self._requests:
            self._requests[key] = []
        
        # Remove old requests outside window
        cutoff = now - timedelta(seconds=self.window_seconds)
        self._requests[key] = [
            ts for ts in self._requests[key]
            if ts > cutoff
        ]
        
        # Check if under limit
        if len(self._requests[key]) >= self.max_requests:
            logger.warning(
                f"Rate limit exceeded for {key}",
                extra={"key": key, "requests": len(self._requests[key])}
            )
            return False
        
        # Record this request
        self._requests[key].append(now)
        return True
    
    def get_remaining(self, key: str) -> int:
        """
        Get remaining requests in window.
        
        Args:
            key: Rate limit key
            
        Returns:
            Remaining requests
        """
        if key not in self._requests:
            return self.max_requests
        
        # Clean old requests
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self.window_seconds)
        self._requests[key] = [
            ts for ts in self._requests[key]
            if ts > cutoff
        ]
        
        return max(0, self.max_requests - len(self._requests[key]))


class APIKeyValidator:
    """
    API key validation.
    
    Validates API keys against stored keys.
    """
    
    def __init__(self):
        """Initialize validator."""
        # In production, load from database
        self._valid_keys: Dict[str, Dict[str, Any]] = {}
    
    def register_key(
        self,
        api_key: str,
        tenant_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register API key.
        
        Args:
            api_key: API key
            tenant_id: Associated tenant
            metadata: Optional metadata
        """
        self._valid_keys[api_key] = {
            "tenant_id": tenant_id,
            "created_at": datetime.now(timezone.utc),
            "metadata": metadata or {}
        }
    
    def validate(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            Key info if valid, None otherwise
        """
        if api_key in self._valid_keys:
            return self._valid_keys[api_key].copy()
        return None
    
    def get_tenant_id(self, api_key: str) -> Optional[str]:
        """
        Get tenant ID for API key.
        
        Args:
            api_key: API key
            
        Returns:
            Tenant ID or None
        """
        key_info = self.validate(api_key)
        return key_info["tenant_id"] if key_info else None
    
    @staticmethod
    def generate_key() -> str:
        """
        Generate new API key.
        
        Returns:
            Generated API key
        """
        return secrets.token_urlsafe(32)


class RequestValidator:
    """
    Request validation.
    
    Validates incoming requests.
    """
    
    @staticmethod
    def validate_phone_number(phone: str) -> bool:
        """
        Validate phone number.
        
        Args:
            phone: Phone number
            
        Returns:
            True if valid
        """
        if not phone:
            return False
        
        # Remove non-digits
        import re
        digits = re.sub(r'\D', '', phone)
        
        # Should be 10-15 digits
        return 10 <= len(digits) <= 15
    
    @staticmethod
    def validate_tenant_id(tenant_id: str) -> bool:
        """
        Validate tenant ID.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            True if valid
        """
        if not tenant_id:
            return False
        
        # Should be alphanumeric
        return tenant_id.replace('_', '').replace('-', '').isalnum()
    
    @staticmethod
    def sanitize_input(text: str, max_length: int = 1000) -> str:
        """
        Sanitize user input.
        
        Args:
            text: Input text
            max_length: Maximum length
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Truncate
        text = text[:max_length]
        
        # Remove control characters
        import re
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        
        return text.strip()


class SecurityManager:
    """
    Central security manager.
    
    Manages authentication, authorization, and rate limiting.
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """
        Initialize security manager.
        
        Args:
            config: Security configuration
        """
        self.config = config or SecurityConfig()
        
        # Components
        self.api_key_validator = APIKeyValidator()
        self.rate_limiter = RateLimiter(
            max_requests=self.config.max_requests_per_minute
        )
        self.request_validator = RequestValidator()
        
        logger.info(
            "SecurityManager initialized",
            extra={
                "require_api_key": self.config.require_api_key,
                "rate_limiting": self.config.enable_rate_limiting
            }
        )
    
    def validate_request(
        self,
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate incoming request.
        
        Args:
            api_key: API key
            tenant_id: Tenant ID
            
        Returns:
            Validation result
        """
        result = {
            "valid": True,
            "errors": [],
            "tenant_id": None
        }
        
        # Validate API key if required
        if self.config.require_api_key:
            if not api_key:
                result["valid"] = False
                result["errors"].append("Missing API key")
                return result
            
            key_info = self.api_key_validator.validate(api_key)
            if not key_info:
                result["valid"] = False
                result["errors"].append("Invalid API key")
                return result
            
            result["tenant_id"] = key_info["tenant_id"]
        
        # Validate tenant ID if required
        if self.config.require_tenant_auth:
            if not tenant_id and not result["tenant_id"]:
                result["valid"] = False
                result["errors"].append("Missing tenant ID")
                return result
            
            tenant_to_check = tenant_id or result["tenant_id"]
            if not self.request_validator.validate_tenant_id(tenant_to_check):
                result["valid"] = False
                result["errors"].append("Invalid tenant ID")
                return result
        
        # Rate limiting
        if self.config.enable_rate_limiting and api_key:
            if not self.rate_limiter.is_allowed(api_key):
                result["valid"] = False
                result["errors"].append("Rate limit exceeded")
                return result
        
        return result
    
    def get_security_headers(self) -> Dict[str, str]:
        """
        Get security headers for responses.
        
        Returns:
            Headers dict
        """
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
        }


# Default instance
_default_manager: Optional[SecurityManager] = None


def get_default_manager() -> SecurityManager:
    """
    Get or create default security manager.
    
    Returns:
        Default manager instance
    """
    global _default_manager
    
    if _default_manager is None:
        _default_manager = SecurityManager()
    
    return _default_manager


def validate_request(
    api_key: Optional[str] = None,
    tenant_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick request validation.
    
    Args:
        api_key: API key
        tenant_id: Tenant ID
        
    Returns:
        Validation result
    """
    manager = get_default_manager()
    return manager.validate_request(api_key, tenant_id)
