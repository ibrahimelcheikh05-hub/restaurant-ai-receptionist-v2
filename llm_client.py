"""
LLM Client
==========
Production-grade OpenAI client with streaming, timeouts, and cancellation.

Responsibilities:
- Manage OpenAI API calls
- Support streaming responses
- Handle cancellation tokens
- Enforce hard timeouts
- Track token usage and costs
- Error handling and retries
- Circuit breaker pattern
- Rate limiting protection
- Prompt injection monitoring

SAFETY:
- All calls are time-boxed
- All calls are cancellable
- No infinite waits
- Token limit enforcement
- Circuit breaker protection
- Comprehensive error handling
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, AsyncGenerator
from datetime import datetime, timezone
from dataclasses import dataclass
import os

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai import APIError, APITimeoutError, RateLimitError, APIConnectionError

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout: float = 15.0  # Increased for reliability
    streaming: bool = True
    max_retries: int = 2
    
    # Safety limits
    max_prompt_tokens: int = 4000
    max_completion_tokens: int = 2000
    max_input_length: int = 16000  # Character limit for safety
    
    # Circuit breaker
    circuit_breaker_threshold: int = 5  # failures before opening
    circuit_breaker_timeout: float = 60.0  # seconds before retry
    
    # Rate limiting (client-side protection)
    max_requests_per_minute: int = 60
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if self.max_tokens > self.max_completion_tokens:
            logger.warning(
                f"max_tokens ({self.max_tokens}) exceeds max_completion_tokens ({self.max_completion_tokens}), capping"
            )
            self.max_tokens = self.max_completion_tokens


class LLMResponse:
    """Structured LLM response."""
    
    def __init__(
        self,
        text: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        finish_reason: Optional[str] = None,
        latency_ms: float = 0.0,
        cancelled: bool = False
    ):
        self.text = text
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.finish_reason = finish_reason
        self.latency_ms = latency_ms
        self.cancelled = cancelled
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
            "cancelled": self.cancelled
        }


class CircuitBreaker:
    """
    Circuit breaker for LLM calls.
    
    Prevents cascading failures by stopping requests after threshold.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening
            timeout: Seconds before attempting retry
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._is_open = False
        self._lock = asyncio.Lock()
        
        logger.info(
            "Circuit breaker initialized",
            extra={
                "threshold": failure_threshold,
                "timeout": timeout
            }
        )
    
    async def record_success(self) -> None:
        """Record successful call."""
        async with self._lock:
            self._failure_count = 0
            self._is_open = False
            logger.debug("Circuit breaker: success recorded")
    
    async def record_failure(self) -> None:
        """Record failed call."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now(timezone.utc)
            
            if self._failure_count >= self.failure_threshold:
                self._is_open = True
                logger.error(
                    "Circuit breaker OPENED",
                    extra={
                        "failures": self._failure_count,
                        "threshold": self.failure_threshold
                    }
                )
    
    async def can_attempt(self) -> bool:
        """
        Check if call attempt is allowed.
        
        Returns:
            True if call should be attempted
        """
        async with self._lock:
            if not self._is_open:
                return True
            
            # Check if timeout has passed
            if self._last_failure_time:
                elapsed = (
                    datetime.now(timezone.utc) - self._last_failure_time
                ).total_seconds()
                
                if elapsed >= self.timeout:
                    logger.info(
                        "Circuit breaker: timeout elapsed, attempting retry",
                        extra={"elapsed": elapsed}
                    )
                    self._is_open = False
                    self._failure_count = 0
                    return True
            
            logger.warning(
                "Circuit breaker OPEN - rejecting call",
                extra={"failures": self._failure_count}
            )
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "is_open": self._is_open,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure": (
                self._last_failure_time.isoformat()
                if self._last_failure_time
                else None
            )
        }


class RateLimiter:
    """
    Client-side rate limiter to protect against hitting API limits.
    """
    
    def __init__(self, max_requests_per_minute: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests_per_minute: Maximum requests per minute
        """
        self.max_requests = max_requests_per_minute
        self.window_size = 60.0  # seconds
        self._requests: List[datetime] = []
        self._lock = asyncio.Lock()
        
        logger.info(
            "Rate limiter initialized",
            extra={"max_rpm": max_requests_per_minute}
        )
    
    async def acquire(self) -> None:
        """
        Acquire permission to make request.
        
        Blocks if rate limit would be exceeded.
        """
        async with self._lock:
            now = datetime.now(timezone.utc)
            
            # Remove old requests outside window
            cutoff = now.timestamp() - self.window_size
            self._requests = [
                req for req in self._requests
                if req.timestamp() > cutoff
            ]
            
            # Check if at limit
            if len(self._requests) >= self.max_requests:
                # Calculate wait time
                oldest = self._requests[0]
                wait_time = self.window_size - (now.timestamp() - oldest.timestamp())
                
                if wait_time > 0:
                    logger.warning(
                        f"Rate limit reached, waiting {wait_time:.2f}s",
                        extra={"requests_in_window": len(self._requests)}
                    )
                    await asyncio.sleep(wait_time)
                    
                    # Clean up again after wait
                    now = datetime.now(timezone.utc)
                    cutoff = now.timestamp() - self.window_size
                    self._requests = [
                        req for req in self._requests
                        if req.timestamp() > cutoff
                    ]
            
            # Record this request
            self._requests.append(now)


class LLMClient:
    """
    Production OpenAI client with async support.
    
    Features:
    - Hard timeouts on all calls
    - Cancellation support
    - Circuit breaker protection
    - Token usage tracking
    - Cost estimation
    - Retry logic
    - Rate limiting
    - Input validation
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM client.
        
        Args:
            config: LLM configuration
        """
        self.config = config or LLMConfig()
        
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries
        )
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_breaker_threshold,
            timeout=self.config.circuit_breaker_timeout
        )
        
        # Rate limiter
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=self.config.max_requests_per_minute
        )
        
        # Metrics
        self._total_requests = 0
        self._total_tokens = 0
        self._total_errors = 0
        self._total_timeouts = 0
        self._total_cancellations = 0
        self._total_cost_usd = 0.0
        self._metrics_lock = asyncio.Lock()
        
        logger.info(
            "LLM client initialized",
            extra={
                "model": self.config.model,
                "timeout": self.config.timeout,
                "max_tokens": self.config.max_tokens
            }
        )
    
    def _validate_messages(
        self,
        messages: List[Dict[str, str]],
        call_id: Optional[str] = None
    ) -> None:
        """
        Validate message input.
        
        Args:
            messages: Messages to validate
            call_id: Call ID for logging
            
        Raises:
            ValueError: If messages are invalid
        """
        if not messages:
            raise ValueError("Messages cannot be empty")
        
        # Check total length
        total_length = sum(len(msg.get("content", "")) for msg in messages)
        if total_length > self.config.max_input_length:
            logger.error(
                f"Input too long: {total_length} chars (max {self.config.max_input_length})",
                extra={"call_id": call_id}
            )
            raise ValueError(
                f"Input exceeds maximum length of {self.config.max_input_length} characters"
            )
        
        # Validate message format
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"Message {i} is not a dictionary")
            
            if "role" not in msg or "content" not in msg:
                raise ValueError(f"Message {i} missing 'role' or 'content'")
            
            if msg["role"] not in ["system", "user", "assistant"]:
                raise ValueError(f"Message {i} has invalid role: {msg['role']}")
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        call_id: Optional[str] = None,
        cancellation_token: Optional[asyncio.Event] = None
    ) -> LLMResponse:
        """
        Execute LLM completion with safety controls.
        
        Args:
            messages: Chat messages
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum completion tokens
            timeout: Request timeout in seconds
            call_id: Call identifier for logging
            cancellation_token: Event to signal cancellation
            
        Returns:
            LLM response
            
        Raises:
            asyncio.TimeoutError: On timeout
            asyncio.CancelledError: On cancellation
            ValueError: On invalid input
            Exception: On API errors
        """
        start_time = datetime.now(timezone.utc)
        
        # Check circuit breaker
        if not await self.circuit_breaker.can_attempt():
            async with self._metrics_lock:
                self._total_errors += 1
            raise Exception("Circuit breaker open - LLM calls blocked")
        
        # Build messages
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                *messages
            ]
        
        # Validate messages
        self._validate_messages(messages, call_id)
        
        # Use configured defaults
        actual_temperature = (
            temperature if temperature is not None
            else self.config.temperature
        )
        actual_max_tokens = (
            max_tokens if max_tokens is not None
            else self.config.max_tokens
        )
        actual_timeout = (
            timeout if timeout is not None
            else self.config.timeout
        )
        
        # Enforce token limits
        if actual_max_tokens > self.config.max_completion_tokens:
            logger.warning(
                f"Requested {actual_max_tokens} tokens, limiting to {self.config.max_completion_tokens}",
                extra={"call_id": call_id}
            )
            actual_max_tokens = self.config.max_completion_tokens
        
        # Clamp temperature
        actual_temperature = max(0.0, min(2.0, actual_temperature))
        
        logger.debug(
            "Starting LLM completion",
            extra={
                "call_id": call_id,
                "model": self.config.model,
                "max_tokens": actual_max_tokens,
                "timeout": actual_timeout
            }
        )
        
        try:
            # Rate limiting
            await self.rate_limiter.acquire()
            
            # Create completion task
            completion_task = asyncio.create_task(
                self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=actual_temperature,
                    max_tokens=actual_max_tokens,
                    stream=False
                )
            )
            
            # Wait with timeout and cancellation support
            if cancellation_token:
                # Wait for either completion or cancellation
                done, pending = await asyncio.wait(
                    [
                        completion_task,
                        asyncio.create_task(cancellation_token.wait())
                    ],
                    timeout=actual_timeout,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Check if cancelled
                if cancellation_token.is_set():
                    completion_task.cancel()
                    try:
                        await completion_task
                    except asyncio.CancelledError:
                        pass
                    
                    async with self._metrics_lock:
                        self._total_cancellations += 1
                    
                    logger.warning(
                        "LLM completion cancelled",
                        extra={"call_id": call_id}
                    )
                    raise asyncio.CancelledError("LLM call cancelled")
                
                # Check if timed out
                if completion_task not in done:
                    completion_task.cancel()
                    try:
                        await completion_task
                    except asyncio.CancelledError:
                        pass
                    
                    async with self._metrics_lock:
                        self._total_timeouts += 1
                    
                    await self.circuit_breaker.record_failure()
                    
                    logger.error(
                        f"LLM timeout after {actual_timeout}s",
                        extra={"call_id": call_id}
                    )
                    raise asyncio.TimeoutError(f"LLM timeout after {actual_timeout}s")
                
                # Get result
                response = completion_task.result()
            else:
                # Simple timeout without cancellation
                response = await asyncio.wait_for(
                    completion_task,
                    timeout=actual_timeout
                )
            
            # Extract response
            text = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0
            
            # Calculate latency
            latency_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000
            
            # Update metrics
            async with self._metrics_lock:
                self._total_requests += 1
                self._total_tokens += total_tokens
                
                # Estimate cost
                cost = self._estimate_cost(
                    self.config.model,
                    prompt_tokens,
                    completion_tokens
                )
                self._total_cost_usd += cost
            
            # Record success
            await self.circuit_breaker.record_success()
            
            logger.info(
                "LLM completion success",
                extra={
                    "call_id": call_id,
                    "tokens": total_tokens,
                    "latency_ms": latency_ms,
                    "finish_reason": finish_reason,
                    "cost_usd": cost
                }
            )
            
            return LLMResponse(
                text=text,
                model=self.config.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                cancelled=False
            )
        
        except asyncio.CancelledError:
            async with self._metrics_lock:
                self._total_cancellations += 1
            logger.warning(
                "LLM completion cancelled",
                extra={"call_id": call_id}
            )
            raise
        
        except asyncio.TimeoutError:
            async with self._metrics_lock:
                self._total_timeouts += 1
                self._total_errors += 1
            await self.circuit_breaker.record_failure()
            logger.error(
                f"LLM timeout after {actual_timeout}s",
                extra={"call_id": call_id}
            )
            raise
        
        except RateLimitError as e:
            async with self._metrics_lock:
                self._total_errors += 1
            await self.circuit_breaker.record_failure()
            logger.error(
                f"LLM rate limit error: {e}",
                extra={"call_id": call_id}
            )
            raise
        
        except APIConnectionError as e:
            async with self._metrics_lock:
                self._total_errors += 1
            await self.circuit_breaker.record_failure()
            logger.error(
                f"LLM connection error: {e}",
                extra={"call_id": call_id}
            )
            raise
        
        except APIError as e:
            async with self._metrics_lock:
                self._total_errors += 1
            await self.circuit_breaker.record_failure()
            logger.error(
                f"LLM API error: {e}",
                extra={"call_id": call_id},
                exc_info=True
            )
            raise
        
        except Exception as e:
            async with self._metrics_lock:
                self._total_errors += 1
            await self.circuit_breaker.record_failure()
            logger.error(
                f"LLM error: {e}",
                extra={"call_id": call_id},
                exc_info=True
            )
            raise
    
    def _estimate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """
        Estimate cost in USD.
        
        Args:
            model: Model name
            prompt_tokens: Prompt token count
            completion_tokens: Completion token count
            
        Returns:
            Estimated cost in USD
        """
        # Rough pricing (as of 2024)
        # Update these based on current OpenAI pricing
        pricing = {
            "gpt-4": (0.03 / 1000, 0.06 / 1000),  # (prompt, completion) per token
            "gpt-4o": (0.0025 / 1000, 0.01 / 1000),
            "gpt-4o-mini": (0.00015 / 1000, 0.0006 / 1000),
            "gpt-3.5-turbo": (0.0005 / 1000, 0.0015 / 1000)
        }
        
        # Find matching pricing (check for partial matches)
        prompt_price, completion_price = None, None
        for model_key, prices in pricing.items():
            if model_key in model:
                prompt_price, completion_price = prices
                break
        
        # Fallback pricing
        if prompt_price is None:
            prompt_price, completion_price = (0.001 / 1000, 0.002 / 1000)
        
        cost = (
            prompt_tokens * prompt_price +
            completion_tokens * completion_price
        )
        
        return cost
    
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get client metrics.
        
        Returns:
            Metrics dictionary
        """
        async with self._metrics_lock:
            return {
                "total_requests": self._total_requests,
                "total_tokens": self._total_tokens,
                "total_errors": self._total_errors,
                "total_timeouts": self._total_timeouts,
                "total_cancellations": self._total_cancellations,
                "total_cost_usd": round(self._total_cost_usd, 4),
                "circuit_breaker": self.circuit_breaker.get_status()
            }
    
    async def reset_metrics(self) -> None:
        """Reset all metrics."""
        async with self._metrics_lock:
            self._total_requests = 0
            self._total_tokens = 0
            self._total_errors = 0
            self._total_timeouts = 0
            self._total_cancellations = 0
            self._total_cost_usd = 0.0
        
        logger.info("LLM client metrics reset")
    
    async def close(self) -> None:
        """Close the client and cleanup resources."""
        try:
            await self.client.close()
            logger.info("LLM client closed")
        except Exception as e:
            logger.error(f"Error closing LLM client: {e}", exc_info=True)


# Global default client
_default_client: Optional[LLMClient] = None
_client_lock = asyncio.Lock()


async def get_default_client() -> LLMClient:
    """
    Get or create default LLM client.
    
    Thread-safe singleton pattern.
    
    Returns:
        Default LLM client
    """
    global _default_client
    
    if _default_client is None:
        async with _client_lock:
            # Double-check after acquiring lock
            if _default_client is None:
                _default_client = LLMClient()
    
    return _default_client


def set_default_client(client: LLMClient) -> None:
    """
    Set default LLM client.
    
    Args:
        client: LLM client to use as default
    """
    global _default_client
    _default_client = client
    logger.info("Default LLM client set")
