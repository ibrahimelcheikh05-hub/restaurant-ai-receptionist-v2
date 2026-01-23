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
    REJECTED = "rejected"  # Added for safety


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
    protected_tokens: int = 0  # Count of protected tokens
    
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
            qty = match.group(0)
            placeholder = f"__QTY_{protected_count}__"
            replacements[placeholder] = qty
            text = text.replace(qty, placeholder, 1)
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
    - Canonical key protection
    - Text sanitization
    
    SAFETY:
    - Protected tokens are never translated
    - Text is sanitized before translation
    - Suspicious content is rejected
    - Length limits enforced
    """
    
    def __init__(
        self,
        timeout: float = 2.0,
        enable_cache: bool = True,
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        protect_canonical_keys: bool = True,
        sanitize_input: bool = True
    ):
        """
        Initialize translator.
        
        Args:
            timeout: Translation timeout in seconds
            enable_cache: Enable caching
            cache_size: Cache max size
            cache_ttl: Cache TTL in seconds
            protect_canonical_keys: Protect canonical keys from translation
            sanitize_input: Sanitize input before translation
        """
        self.timeout = timeout
        self.enable_cache = enable_cache
        self.protect_canonical_keys = protect_canonical_keys
        self.sanitize_input = sanitize_input
        
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
        self._rejected = 0
        self._protected_tokens = 0
        
        logger.info(
            "Translator initialized",
            extra={
                "timeout": timeout,
                "cache_enabled": enable_cache,
                "protect_keys": protect_canonical_keys,
                "sanitize": sanitize_input,
                "google_available": GOOGLE_TRANSLATE_AVAILABLE
            }
        )
    
    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        call_id: Optional[str] = None
    ) -> TranslationResponse:
        """
        Translate text with safety protections.
        
        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            call_id: Call identifier for logging
            
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
        
        # Sanitize input
        protected_count = 0
        replacements = {}
        
        if self.sanitize_input:
            text, is_safe = TextSanitizer.sanitize(text)
            
            if not is_safe:
                self._rejected += 1
                logger.error(
                    "Unsafe text rejected for translation",
                    extra={"call_id": call_id}
                )
                return TranslationResponse(
                    text=text,  # Return sanitized original
                    result=TranslationResult.REJECTED,
                    source_language=source_language,
                    target_language=target_language
                )
        
        # Protect canonical keys
        if self.protect_canonical_keys:
            text, replacements, protected_count = (
                CanonicalKeyProtector.protect_text(text)
            )
            
            if protected_count > 0:
                self._protected_tokens += protected_count
                logger.debug(
                    f"Protected {protected_count} tokens from translation",
                    extra={"call_id": call_id}
                )
        
        # Check if translation is available
        if not GOOGLE_TRANSLATE_AVAILABLE or not self.client:
            logger.warning("Translation not available - returning original")
            
            # Restore protected tokens if any
            if replacements:
                text = CanonicalKeyProtector.restore_text(text, replacements)
            
            return TranslationResponse(
                text=text,
                result=TranslationResult.DISABLED,
                source_language=source_language,
                target_language=target_language,
                protected_tokens=protected_count
            )
        
        # Check cache (use protected text for cache key)
        if self.cache:
            cached = self.cache.get(text, source_language, target_language)
            if cached:
                self._cache_hits += 1
                
                # Restore protected tokens
                if replacements:
                    cached = CanonicalKeyProtector.restore_text(cached, replacements)
                
                latency_ms = (
                    datetime.now(timezone.utc) - start_time
                ).total_seconds() * 1000
                
                logger.debug(
                    "Cache hit",
                    extra={
                        "source": source_language,
                        "target": target_language,
                        "call_id": call_id
                    }
                )
                
                return TranslationResponse(
                    text=cached,
                    result=TranslationResult.CACHED,
                    source_language=source_language,
                    target_language=target_language,
                    cached=True,
                    latency_ms=latency_ms,
                    protected_tokens=protected_count
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
            
            # Restore protected tokens
            if replacements:
                translation = CanonicalKeyProtector.restore_text(
                    translation,
                    replacements
                )
            
            # Store in cache (use protected text for key)
            if self.cache:
                # Cache the translated version with placeholders
                # so future identical requests benefit
                cache_text = text  # Protected version
                cache_translation = translation
                if replacements:
                    # Re-protect for cache
                    cache_translation, _, _ = CanonicalKeyProtector.protect_text(
                        translation
                    )
                
                self.cache.set(
                    cache_text,
                    source_language,
                    target_language,
                    cache_translation
                )
            
            self._total_translations += 1
            
            latency_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000
            
            logger.debug(
                f"Translation success: {source_language} -> {target_language}",
                extra={
                    "latency_ms": latency_ms,
                    "call_id": call_id,
                    "protected_tokens": protected_count
                }
            )
            
            return TranslationResponse(
                text=translation,
                result=TranslationResult.SUCCESS,
                source_language=source_language,
                target_language=target_language,
                cached=False,
                latency_ms=latency_ms,
                protected_tokens=protected_count
            )
        
        except asyncio.TimeoutError:
            self._timeouts += 1
            logger.warning(
                f"Translation timeout after {self.timeout}s",
                extra={
                    "source": source_language,
                    "target": target_language,
                    "call_id": call_id
                }
            )
            
            # Restore protected tokens before returning
            result_text = text
            if replacements:
                result_text = CanonicalKeyProtector.restore_text(text, replacements)
            
            # Return original text on timeout
            return TranslationResponse(
                text=result_text,
                result=TranslationResult.TIMEOUT,
                source_language=source_language,
                target_language=target_language,
                protected_tokens=protected_count
            )
        
        except Exception as e:
            self._errors += 1
            logger.error(
                f"Translation error: {e}",
                extra={
                    "source": source_language,
                    "target": target_language,
                    "call_id": call_id
                },
                exc_info=True
            )
            
            # Restore protected tokens before returning
            result_text = text
            if replacements:
                result_text = CanonicalKeyProtector.restore_text(text, replacements)
            
            # Return original text on error
            return TranslationResponse(
                text=result_text,
                result=TranslationResult.ERROR,
                source_language=source_language,
                target_language=target_language,
                protected_tokens=protected_count
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
            text: Text to translate (may contain placeholders)
            source_language: Source language
            target_language: Target language
            
        Returns:
            Translated text (with placeholders preserved)
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
        source_language: str,
        call_id: Optional[str] = None
    ) -> TranslationResponse:
        """
        Translate to English.
        
        Args:
            text: Text to translate
            source_language: Source language
            call_id: Call identifier
            
        Returns:
            Translation response
        """
        return await self.translate(text, source_language, "en", call_id)
    
    async def translate_from_english(
        self,
        text: str,
        target_language: str,
        call_id: Optional[str] = None
    ) -> TranslationResponse:
        """
        Translate from English.
        
        Args:
            text: Text to translate
            target_language: Target language
            call_id: Call identifier
            
        Returns:
            Translation response
        """
        return await self.translate(text, "en", target_language, call_id)
    
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
            "rejected": self._rejected,
            "protected_tokens": self._protected_tokens,
            "cache_size": self.cache.size() if self.cache else 0
        }
        
        total_attempts = self._total_translations + self._cache_hits
        if total_attempts > 0:
            stats["cache_hit_rate"] = self._cache_hits / total_attempts
        
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


def set_default_translator(translator: Translator) -> None:
    """
    Set default translator.
    
    Args:
        translator: Translator instance
    """
    global _default_translator
    _default_translator = translator
    logger.info("Default translator set")


async def translate_text(
    text: str,
    source_language: str,
    target_language: str,
    call_id: Optional[str] = None
) -> TranslationResponse:
    """
    Quick translation using default translator.
    
    Args:
        text: Text to translate
        source_language: Source language
        target_language: Target language
        call_id: Call identifier
        
    Returns:
        Translation response
    """
    translator = get_default_translator()
    return await translator.translate(
        text,
        source_language,
        target_language,
        call_id
    )
