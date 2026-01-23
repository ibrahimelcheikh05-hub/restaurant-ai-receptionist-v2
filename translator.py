"""
Translator
==========
FALLBACK-ONLY translation utility.

ARCHITECTURE NOTICE:
This module is NO LONGER in the real-time call path.
The system now uses TRUE MULTILINGUAL design:
- STT → LLM → TTS (no translation)
- LLM speaks user's language natively
- Business logic is language-agnostic

This translator is now ONLY for:
- Emergency fallback scenarios
- Development/testing utilities
- Offline translation of static content
- Administrative tools

DO NOT use this in the hot path of live calls.

Legacy Responsibilities (now deprecated for real-time):
- Translate text between languages
- Cache translations
- Enforce timeouts
- Graceful degradation
- Never block call flow
- PROTECT canonical keys from translation
- SANITIZE all text before translation
- VALIDATE translation results

CRITICAL SAFETY:
- Menu item canonical keys are NEVER translated
- Prices/numbers are preserved
- Special tokens are protected
- Maximum text length enforced
- Unsafe content is rejected
"""

import asyncio
import logging
import re
from typing import Optional, Dict, Any, Set, List
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
    REJECTED = "rejected"
    NOT_REQUIRED = "not_required"  # Added for true multilingual mode


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
    protected_tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "result": self.result.value,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "cached": self.cached,
            "latency_ms": self.latency_ms,
            "protected_tokens": self.protected_tokens
        }


class TextSanitizer:
    """
    Sanitizes text before translation.
    
    SAFETY:
    - Removes control characters
    - Enforces length limits
    - Detects suspicious patterns
    """
    
    MAX_TEXT_LENGTH = 500
    CONTROL_CHARS = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
    
    # Suspicious patterns that shouldn't be translated
    SUSPICIOUS_PATTERNS = [
        r'<script',
        r'javascript:',
        r'onerror=',
        r'onclick=',
        r'eval\(',
        r'<iframe',
    ]
    
    @classmethod
    def sanitize(cls, text: str) -> tuple[str, bool]:
        """
        Sanitize text for translation.
        
        Args:
            text: Raw text
            
        Returns:
            Tuple of (sanitized_text, is_safe)
        """
        if not text:
            return "", True
        
        # Check for suspicious patterns
        text_lower = text.lower()
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                logger.warning(
                    f"Suspicious pattern detected: {pattern}",
                    extra={"text_preview": text[:50]}
                )
                return text, False
        
        # Remove control characters
        text = cls.CONTROL_CHARS.sub('', text)
        
        # Enforce length
        if len(text) > cls.MAX_TEXT_LENGTH:
            logger.warning(
                f"Text truncated from {len(text)} to {cls.MAX_TEXT_LENGTH}"
            )
            text = text[:cls.MAX_TEXT_LENGTH]
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        return text.strip(), True


class CanonicalKeyProtector:
    """
    Protects canonical keys from translation.
    
    CRITICAL: Menu item keys, IDs, and special tokens must NEVER be translated.
    This would break the entire ordering system.
    
    Example:
    - "I want BURGER_CLASSIC" should translate to Spanish as:
      "Quiero BURGER_CLASSIC" (NOT "Quiero HAMBURGUESA_CLASICA")
    """
    
    # Pattern for canonical keys (uppercase with underscores)
    CANONICAL_KEY_PATTERN = re.compile(r'\b[A-Z][A-Z0-9_]{2,}\b')
    
    # Pattern for prices
    PRICE_PATTERN = re.compile(r'\$\d+(?:\.\d{2})?')
    
    # Pattern for quantities
    QUANTITY_PATTERN = re.compile(r'\b\d+\s*(?:x|×)\s*\b', re.IGNORECASE)
    
    # Special tokens to protect
    PROTECTED_TOKENS = {
        'MENU_ITEM', 'ORDER_ID', 'CALL_ID', 'SESSION',
        'CONFIRM', 'CANCEL', 'TRANSFER', 'END'
    }
    
    @classmethod
    def protect_text(cls, text: str) -> tuple[str, Dict[str, str], int]:
        """
        Replace protected tokens with placeholders.
        
        Args:
            text: Original text
            
        Returns:
            Tuple of (protected_text, replacement_map, protected_count)
        """
        replacements = {}
        protected_count = 0
        
        # Protect canonical keys
        for match in cls.CANONICAL_KEY_PATTERN.finditer(text):
            key = match.group(0)
            if len(key) > 50:  # Sanity check
                continue
            
            placeholder = f"__CANON_{protected_count}__"
            replacements[placeholder] = key
            text = text.replace(key, placeholder, 1)
            protected_count += 1
        
        # Protect prices
        for match in cls.PRICE_PATTERN.finditer(text):
            price = match.group(0)
            placeholder = f"__PRICE_{protected_count}__"
            replacements[placeholder] = price
            text = text.replace(price, placeholder, 1)
            protected_count += 1
        
        # Protect quantities
        for match in cls.QUANTITY_PATTERN.finditer(text):
            quantity = match.group(0)
            placeholder = f"__QTY_{protected_count}__"
            replacements[placeholder] = quantity
            text = text.replace(quantity, placeholder, 1)
            protected_count += 1
        
        return text, replacements, protected_count
    
    @classmethod
    def restore_text(cls, text: str, replacements: Dict[str, str]) -> str:
        """
        Restore protected tokens.
        
        Args:
            text: Text with placeholders
            replacements: Replacement map
            
        Returns:
            Text with restored tokens
        """
        for placeholder, original in replacements.items():
            text = text.replace(placeholder, original)
        
        return text


class TranslationCache:
    """
    Simple in-memory translation cache.
    
    Caches recent translations to avoid redundant API calls.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum cache entries
            ttl_seconds: Time-to-live for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, tuple[str, datetime]] = {}
    
    def _make_key(
        self,
        text: str,
        source: str,
        target: str
    ) -> str:
        """Create cache key."""
        data = f"{source}:{target}:{text}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def get(
        self,
        text: str,
        source: str,
        target: str
    ) -> Optional[str]:
        """
        Get cached translation.
        
        Args:
            text: Source text
            source: Source language
            target: Target language
            
        Returns:
            Cached translation or None
        """
        key = self._make_key(text, source, target)
        
        if key not in self._cache:
            return None
        
        cached_text, timestamp = self._cache[key]
        
        # Check if expired
        age = (datetime.now(timezone.utc) - timestamp).total_seconds()
        if age > self.ttl_seconds:
            del self._cache[key]
            return None
        
        return cached_text
    
    def set(
        self,
        text: str,
        source: str,
        target: str,
        translation: str
    ) -> None:
        """
        Cache translation.
        
        Args:
            text: Source text
            source: Source language
            target: Target language
            translation: Translated text
        """
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k][1]
            )
            del self._cache[oldest_key]
        
        key = self._make_key(text, source, target)
        self._cache[key] = (translation, datetime.now(timezone.utc))
    
    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }


class Translator:
    """
    FALLBACK-ONLY translation service.
    
    WARNING: This is NO LONGER used in real-time call paths.
    The system now uses true multilingual LLM responses.
    
    This translator is ONLY for:
    - Emergency fallback scenarios
    - Development utilities
    - Offline content translation
    - Administrative tools
    
    Features:
    - Async translation with timeout
    - Automatic caching
    - Canonical key protection
    - Text sanitization
    - Graceful degradation
    """
    
    DEFAULT_TIMEOUT = 3.0  # Timeout for translation API
    
    def __init__(
        self,
        default_source: str = "auto",
        default_target: str = "en",
        timeout: float = DEFAULT_TIMEOUT,
        enable_cache: bool = True,
        cache_size: int = 1000,
        cache_ttl: int = 3600
    ):
        """
        Initialize translator.
        
        Args:
            default_source: Default source language
            default_target: Default target language
            timeout: API timeout in seconds
            enable_cache: Enable translation caching
            cache_size: Maximum cache entries
            cache_ttl: Cache TTL in seconds
        """
        self.default_source = default_source
        self.default_target = default_target
        self.timeout = timeout
        self.enable_cache = enable_cache
        
        # Initialize cache
        self.cache = TranslationCache(
            max_size=cache_size,
            ttl_seconds=cache_ttl
        ) if enable_cache else None
        
        # Initialize Google Translate client if available
        self.client = None
        if GOOGLE_TRANSLATE_AVAILABLE:
            try:
                self.client = translate.Client()
                logger.info("Google Translate client initialized (FALLBACK ONLY)")
            except Exception as e:
                logger.error(f"Failed to initialize Google Translate: {e}")
                self.client = None
        
        # Statistics
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "errors": 0,
            "bypassed": 0
        }
        
        logger.warning(
            "Translator initialized as FALLBACK ONLY utility - "
            "NOT for use in real-time call paths"
        )
    
    async def translate(
        self,
        text: str,
        source: Optional[str] = None,
        target: Optional[str] = None
    ) -> TranslationResponse:
        """
        Translate text (FALLBACK ONLY - not for real-time use).
        
        WARNING: Do not use in real-time call path.
        Use true multilingual LLM instead.
        
        Args:
            text: Text to translate
            source: Source language (auto-detect if None)
            target: Target language
            
        Returns:
            Translation response
        """
        logger.warning(
            "translate() called - this should NOT be in real-time path"
        )
        
        self._stats["total_requests"] += 1
        
        # Use defaults
        source = source or self.default_source
        target = target or self.default_target
        
        # Bypass if same language
        if source == target and source != "auto":
            self._stats["bypassed"] += 1
            return TranslationResponse(
                text=text,
                result=TranslationResult.BYPASSED,
                source_language=source,
                target_language=target
            )
        
        # Sanitize input
        sanitized_text, is_safe = TextSanitizer.sanitize(text)
        
        if not is_safe:
            self._stats["errors"] += 1
            logger.error("Unsafe text rejected from translation")
            return TranslationResponse(
                text=text,
                result=TranslationResult.REJECTED,
                source_language=source,
                target_language=target
            )
        
        if not sanitized_text:
            return TranslationResponse(
                text="",
                result=TranslationResult.BYPASSED,
                source_language=source,
                target_language=target
            )
        
        # Check cache
        if self.cache:
            cached = self.cache.get(sanitized_text, source, target)
            if cached:
                self._stats["cache_hits"] += 1
                return TranslationResponse(
                    text=cached,
                    result=TranslationResult.CACHED,
                    source_language=source,
                    target_language=target,
                    cached=True
                )
        
        # Protect canonical keys
        protected_text, replacements, protected_count = (
            CanonicalKeyProtector.protect_text(sanitized_text)
        )
        
        # Translate
        start_time = datetime.now(timezone.utc)
        
        try:
            if not self.client:
                self._stats["errors"] += 1
                return TranslationResponse(
                    text=text,
                    result=TranslationResult.DISABLED,
                    source_language=source,
                    target_language=target
                )
            
            # Call Google Translate with timeout
            translation_result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.translate(
                        protected_text,
                        target_language=target,
                        source_language=source if source != "auto" else None
                    )
                ),
                timeout=self.timeout
            )
            
            translated_text = translation_result["translatedText"]
            detected_source = translation_result.get("detectedSourceLanguage", source)
            
            # Restore protected tokens
            restored_text = CanonicalKeyProtector.restore_text(
                translated_text,
                replacements
            )
            
            # Calculate latency
            latency = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            # Cache result
            if self.cache:
                self.cache.set(sanitized_text, source, target, restored_text)
            
            self._stats["api_calls"] += 1
            
            return TranslationResponse(
                text=restored_text,
                result=TranslationResult.SUCCESS,
                source_language=detected_source,
                target_language=target,
                latency_ms=latency,
                protected_tokens=protected_count
            )
        
        except asyncio.TimeoutError:
            self._stats["errors"] += 1
            logger.error(f"Translation timeout after {self.timeout}s")
            return TranslationResponse(
                text=text,
                result=TranslationResult.TIMEOUT,
                source_language=source,
                target_language=target
            )
        
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Translation error: {e}", exc_info=True)
            return TranslationResponse(
                text=text,
                result=TranslationResult.ERROR,
                source_language=source,
                target_language=target
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get translator statistics.
        
        Returns:
            Statistics dict
        """
        stats = self._stats.copy()
        
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        
        stats["cache_hit_rate"] = (
            self._stats["cache_hits"] / max(1, self._stats["total_requests"])
        )
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear translation cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Translation cache cleared")


# Default instance (FALLBACK ONLY)
_default_translator: Optional[Translator] = None


def get_default_translator() -> Translator:
    """
    Get or create default translator (FALLBACK ONLY).
    
    WARNING: Not for use in real-time call paths.
    
    Returns:
        Default translator instance
    """
    global _default_translator
    
    if _default_translator is None:
        _default_translator = Translator()
    
    return _default_translator


async def translate_text(
    text: str,
    source: Optional[str] = None,
    target: Optional[str] = None
) -> TranslationResponse:
    """
    Quick translation utility (FALLBACK ONLY).
    
    WARNING: Do not use in real-time call path.
    
    Args:
        text: Text to translate
        source: Source language
        target: Target language
        
    Returns:
        Translation response
    """
    translator = get_default_translator()
    return await translator.translate(text, source, target)
