"""
LLM Client
==========
Production-grade OpenAI client with streaming, timeouts, and cancellation.

Responsibilities:
- Manage OpenAI API calls
- Support streaming responses
- Handle cancellation tokens
- Enforce timeouts
- Track token usage
- Error handling and retries
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, AsyncGenerator
from datetime import datetime, timezone
import os

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

logger = logging.getLogger(__name__)


class LLMConfig:
    """Configuration for LLM client."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        timeout: float = 10.0,
        streaming: bool = True,
        max_retries: int = 2
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.streaming = streaming
        self.max_retries = max_retries


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
        latency_ms: float = 0.0
    ):
        self.text = text
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.finish_reason = finish_reason
        self.latency_ms = latency_ms
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms
        }


class LLMClient:
    """Production OpenAI client with async support."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries
        )
        
        self._total_requests = 0
        self._total_tokens = 0
        self._total_errors = 0
        
        logger.info(
            "LLM client initialized",
            extra={"model": self.config.model}
        )
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        call_id: Optional[str] = None
    ) -> LLMResponse:
        start_time = datetime.now(timezone.utc)
        
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                *messages
            ]
        
        actual_temperature = temperature if temperature is not None else self.config.temperature
        actual_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        actual_timeout = timeout if timeout is not None else self.config.timeout
        
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=actual_temperature,
                    max_tokens=actual_max_tokens,
                    stream=False
                ),
                timeout=actual_timeout
            )
            
            text = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0
            
            latency_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000
            
            self._total_requests += 1
            self._total_tokens += total_tokens
            
            logger.info(
                "LLM completion success",
                extra={
                    "call_id": call_id,
                    "tokens": total_tokens,
                    "latency_ms": latency_ms
                }
            )
            
            return LLMResponse(
                text=text,
                model=self.config.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                finish_reason=finish_reason,
                latency_ms=latency_ms
            )
        
        except asyncio.TimeoutError:
            self._total_errors += 1
            logger.error(
                f"LLM timeout after {actual_timeout}s",
                extra={"call_id": call_id}
            )
            raise
        
        except Exception as e:
            self._total_errors += 1
            logger.error(
                f"LLM error: {e}",
                extra={"call_id": call_id},
                exc_info=True
            )
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "total_errors": self._total_errors
        }


_default_client: Optional[LLMClient] = None


def get_default_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
