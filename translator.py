"""
Translator
==========
Production translation service with timeout and fallback.

Responsibilities:
- Translate text between languages
- Cache translations
- Enforce timeouts
- Graceful degradation
- Never block call flow
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

# Try to import Google Translate
try:
    from google.cloud import translate_v2 as translate
    from google.api_core.exceptions import GoogleAPIError
    GOOGLE_TRANSLATE_AVAILABLE = True
except ImportError:
    GOOGLE_TRANSLATE_AVAILABLE = False
    logger.warning("Google Translate not available - translation disabled")


class TranslationResult(Enum):
    """Translation operation result."""
    SUCCESS = "success"
    CACHED = "cached"
    BYPASSED = "bypassed"
    TIMEOUT = "timeout"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class TranslationResponse:
    """
    Translation response.
    
    Always returns a result - never raises exceptions.
    """
    text: str
    result: TranslationResult
    source_language: str
    target_language: str
    cached: bool = False
    latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "result": self.result.value,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "cached": self.cached,
            "latency_ms": self.latency_ms
        }


class TranslationCache:
    """
    Simple in-memory translation cache.
    
    Caches translations to reduce API calls and latency.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600
    ):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum cache entries
            ttl_seconds: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
        # Cache: key -> (translation, timestamp)
        self._cache: Dict[str, tuple[str, datetime]] = {}
    
    def _make_key(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """
        Create cache key.
        
        Args:
            text: Text to translate
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Cache key
        """
        # Hash for shorter keys
        content = f"{source_lang}:{target_lang}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Optional[str]:
        """
        Get cached translation.
        
        Args:
            text: Text to translate
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Cached translation or None
        """
        key = self._make_key(text, source_lang, target_lang)
        
        if key not in self._cache:
            return None
        
        translation, timestamp = self._cache[key]
        
        # Check if expired
        age = (datetime.now(timezone.utc) - timestamp).total_seconds()
        if age > self.ttl_seconds:
            del self._cache[key]
            return None
        
        return translation
    
    def set(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        translation: str
    ) -> None:
        """
        Store translation in cache.
        
        Args:
            text: Original text
            source_lang: Source language
            target_lang: Target language
            translation: Translated text
        """
        # Evict oldest if at max size
        if len(self._cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k][1]
            )
            del self._cache[oldest_key]
        
        key = self._make_key(text, source_lang, target_lang)
        self._cache[key] = (translation, datetime.now(timezone.utc))
    
    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self._cache)


class Translator:
    """
    Production translation service.
    
    Features:
    - Google Translate API integration
    - Caching for performance
    - Strict timeouts
    - Graceful fallback
    - Never blocks call flow
    """
    
    def __init__(
        self,
        timeout: float = 2.0,
        enable_cache: bool = True,
        cache_size: int = 1000,
        cache_ttl: int = 3600
    ):
        """
        Initialize translator.
        
        Args:
            timeout: Translation timeout in seconds
            enable_cache: Enable caching
            cache_size: Cache max size
            cache_ttl: Cache TTL in seconds
        """
        self.timeout = timeout
        self.enable_cache = enable_cache
        
        # Initialize cache
        self.cache = TranslationCache(
            max_size=cache_size,
            ttl_seconds=cache_ttl
        ) if enable_cache else None
        
        # Initialize Google Translate client
        self.client = None
        if GOOGLE_TRANSLATE_AVAILABLE:
            try:
                self.client = translate.Client()
                logger.info("Google Translate client initialized")
            except Exception as e:
                logger.error(
                    f"Failed to initialize Google Translate: {e}",
                    exc_info=True
                )
        
        # Metrics
        self._total_translations = 0
        self._cache_hits = 0
        self._timeouts = 0
        self._errors = 0
        
        logger.info(
            "Translator initialized",
            extra={
                "timeout": timeout,
                "cache_enabled": enable_cache,
                "google_available": GOOGLE_TRANSLATE_AVAILABLE
            }
        )
    
    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str
    ) -> TranslationResponse:
        """
        Translate text.
        
        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Translation response (never raises exceptions)
        """
        start_time = datetime.now(timezone.utc)
        
        # Validate inputs
        if not text or not text.strip():
            return TranslationResponse(
                text="",
                result=TranslationResult.BYPASSED,
                source_language=source_language,
                target_language=target_language
            )
        
        # Same language - no translation needed
        if source_language == target_language:
            return TranslationResponse(
                text=text,
                result=TranslationResult.BYPASSED,
                source_language=source_language,
                target_language=target_language
            )
        
        # Check if translation is available
        if not GOOGLE_TRANSLATE_AVAILABLE or not self.client:
            logger.warning("Translation not available - returning original")
            return TranslationResponse(
                text=text,
                result=TranslationResult.DISABLED,
                source_language=source_language,
                target_language=target_language
            )
        
        # Check cache
        if self.cache:
            cached = self.cache.get(text, source_language, target_language)
            if cached:
                self._cache_hits += 1
                latency_ms = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds() * 1000
                
                logger.debug(
                    "Cache hit",
                    extra={
                        "source": source_language,
                        "target": target_language
                    }
                )
                
                return TranslationResponse(
                    text=cached,
                    result=TranslationResult.CACHED,
                    source_language=source_language,
                    target_language=target_language,
                    cached=True,
                    latency_ms=latency_ms
                )
        
        # Perform translation with timeout
        try:
            translation = await asyncio.wait_for(
                self._do_translation(
                    text,
                    source_language,
                    target_language
                ),
                timeout=self.timeout
            )
            
            # Store in cache
            if self.cache:
                self.cache.set(
                    text,
                    source_language,
                    target_language,
                    translation
                )
            
            self._total_translations += 1
            
            latency_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000
            
            logger.debug(
                f"Translation success: {source_language} -> {target_language}",
                extra={"latency_ms": latency_ms}
            )
            
            return TranslationResponse(
                text=translation,
                result=TranslationResult.SUCCESS,
                source_language=source_language,
                target_language=target_language,
                cached=False,
                latency_ms=latency_ms
            )
        
        except asyncio.TimeoutError:
            self._timeouts += 1
            logger.warning(
                f"Translation timeout after {self.timeout}s",
                extra={
                    "source": source_language,
                    "target": target_language
                }
            )
            
            # Return original text on timeout
            return TranslationResponse(
                text=text,
                result=TranslationResult.TIMEOUT,
                source_language=source_language,
                target_language=target_language
            )
        
        except Exception as e:
            self._errors += 1
            logger.error(
                f"Translation error: {e}",
                extra={
                    "source": source_language,
                    "target": target_language
                },
                exc_info=True
            )
            
            # Return original text on error
            return TranslationResponse(
                text=text,
                result=TranslationResult.ERROR,
                source_language=source_language,
                target_language=target_language
            )
    
    async def _do_translation(
        self,
        text: str,
        source_language: str,
        target_language: str
    ) -> str:
        """
        Perform actual translation (async wrapper).
        
        Args:
            text: Text to translate
            source_language: Source language
            target_language: Target language
            
        Returns:
            Translated text
        """
        # Google Translate API is sync, run in executor
        loop = asyncio.get_event_loop()
        
        def _translate_sync():
            result = self.client.translate(
                text,
                source_language=source_language,
                target_language=target_language
            )
            return result['translatedText']
        
        return await loop.run_in_executor(None, _translate_sync)
    
    async def translate_to_english(
        self,
        text: str,
        source_language: str
    ) -> TranslationResponse:
        """
        Translate to English.
        
        Args:
            text: Text to translate
            source_language: Source language
            
        Returns:
            Translation response
        """
        return await self.translate(text, source_language, "en")
    
    async def translate_from_english(
        self,
        text: str,
        target_language: str
    ) -> TranslationResponse:
        """
        Translate from English.
        
        Args:
            text: Text to translate
            target_language: Target language
            
        Returns:
            Translation response
        """
        return await self.translate(text, "en", target_language)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get translator statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_translations": self._total_translations,
            "cache_hits": self._cache_hits,
            "timeouts": self._timeouts,
            "errors": self._errors,
            "cache_size": self.cache.size() if self.cache else 0
        }
        
        if self._total_translations > 0:
            stats["cache_hit_rate"] = (
                self._cache_hits / (self._total_translations + self._cache_hits)
            )
        
        return stats


# Default instance
_default_translator: Optional[Translator] = None


def get_default_translator() -> Translator:
    """
    Get or create default translator.
    
    Returns:
        Default translator instance
    """
    global _default_translator
    
    if _default_translator is None:
        _default_translator = Translator()
    
    return _default_translator


async def translate_text(
    text: str,
    source_language: str,
    target_language: str
) -> TranslationResponse:
    """
    Quick translation using default translator.
    
    Args:
        text: Text to translate
        source_language: Source language
        target_language: Target language
        
    Returns:
        Translation response
    """
    translator = get_default_translator()
    return await translator.translate(text, source_language, target_language)
