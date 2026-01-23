"""
Language Detector
=================
Production language detection for voice calls.

Responsibilities:
- Detect language from speech
- Lock language after first detection
- Enforce confidence thresholds
- Prevent mid-call language switching
- Fallback to default language
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import langdetect
try:
    from langdetect import detect_langs, LangDetectException, DetectorFactory
    DetectorFactory.seed = 0  # Consistent results
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logger.warning("langdetect not available - language detection disabled")


class DetectionStatus(Enum):
    """Language detection status."""
    SUCCESS = "success"
    FALLBACK = "fallback"
    LOCKED = "locked"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass
class LanguageLock:
    """
    Immutable language lock for call session.
    
    Once language is detected and locked, it cannot change.
    """
    call_id: str
    language: str
    confidence: float
    locked_at: datetime
    detection_method: str  # 'detected' or 'fallback'


@dataclass
class DetectionResult:
    """Result of language detection."""
    
    status: DetectionStatus
    language: str
    confidence: float
    is_locked: bool
    fallback_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "language": self.language,
            "confidence": self.confidence,
            "is_locked": self.is_locked,
            "fallback_reason": self.fallback_reason
        }


class LanguageDetector:
    """
    Language detector with locking mechanism.
    
    Features:
    - Detect language from text
    - Lock language after first detection
    - Confidence threshold enforcement
    - Fallback to default language
    - Support for multiple languages
    - Timeout enforcement
    - Max detection attempts
    
    SAFETY:
    - Language can only be locked ONCE per call
    - Detection attempts are limited
    - Timeouts prevent infinite detection loops
    - Automatic fallback on repeated failures
    """
    
    MAX_DETECTION_ATTEMPTS = 3
    DETECTION_TIMEOUT_SECONDS = 10.0
    
    def __init__(
        self,
        supported_languages: Optional[List[str]] = None,
        default_language: str = "en",
        min_confidence: float = 0.7,
        min_text_length: int = 20,
        max_attempts: int = MAX_DETECTION_ATTEMPTS,
        timeout_seconds: float = DETECTION_TIMEOUT_SECONDS
    ):
        """
        Initialize language detector.
        
        Args:
            supported_languages: List of supported language codes
            default_language: Default/fallback language
            min_confidence: Minimum confidence threshold
            min_text_length: Minimum text length for detection
            max_attempts: Maximum detection attempts per call
            timeout_seconds: Maximum time allowed for detection
        """
        self.supported_languages = supported_languages or ["en", "es", "ar"]
        self.default_language = default_language
        self.min_confidence = min_confidence
        self.min_text_length = min_text_length
        self.max_attempts = max_attempts
        self.timeout_seconds = timeout_seconds
        
        # Language locks per call
        self._locks: Dict[str, LanguageLock] = {}
        
        # Detection attempt tracking
        self._attempt_counts: Dict[str, int] = {}
        self._first_attempt_time: Dict[str, datetime] = {}
        
        logger.info(
            "LanguageDetector initialized",
            extra={
                "supported": self.supported_languages,
                "default": default_language,
                "min_confidence": min_confidence,
                "max_attempts": max_attempts,
                "timeout": timeout_seconds,
                "langdetect_available": LANGDETECT_AVAILABLE
            }
        )
    
    def detect(
        self,
        call_id: str,
        text: str,
        is_final: bool = True
    ) -> DetectionResult:
        """
        Detect language from text with attempt tracking and timeout.
        
        Args:
            call_id: Call identifier
            text: Text to detect from
            is_final: Whether this is final transcript
            
        Returns:
            Detection result
        """
        # Check if already locked
        if call_id in self._locks:
            lock = self._locks[call_id]
            logger.debug(
                f"Language already locked: {lock.language}",
                extra={"call_id": call_id}
            )
            return DetectionResult(
                status=DetectionStatus.LOCKED,
                language=lock.language,
                confidence=lock.confidence,
                is_locked=True
            )
        
        # Track attempt
        now = datetime.now(timezone.utc)
        if call_id not in self._attempt_counts:
            self._attempt_counts[call_id] = 0
            self._first_attempt_time[call_id] = now
        
        self._attempt_counts[call_id] += 1
        
        # Check if exceeded max attempts
        if self._attempt_counts[call_id] > self.max_attempts:
            logger.warning(
                f"Max detection attempts exceeded: {self._attempt_counts[call_id]}",
                extra={"call_id": call_id}
            )
            return self._fallback_result(
                call_id,
                "max_attempts_exceeded"
            )
        
        # Check if exceeded timeout
        first_attempt = self._first_attempt_time[call_id]
        elapsed = (now - first_attempt).total_seconds()
        
        if elapsed > self.timeout_seconds:
            logger.warning(
                f"Detection timeout exceeded: {elapsed:.1f}s",
                extra={"call_id": call_id}
            )
            return self._fallback_result(
                call_id,
                "timeout_exceeded"
            )
        
        # Reject partial transcripts
        if not is_final:
            logger.debug(
                "Rejecting partial transcript",
                extra={"call_id": call_id}
            )
            return DetectionResult(
                status=DetectionStatus.REJECTED,
                language=self.default_language,
                confidence=0.0,
                is_locked=False,
                fallback_reason="partial_transcript"
            )
        
        # Check text length
        if len(text) < self.min_text_length:
            logger.debug(
                f"Text too short for detection: {len(text)} chars",
                extra={"call_id": call_id}
            )
            # Don't fallback immediately on short text - allow retry
            return DetectionResult(
                status=DetectionStatus.REJECTED,
                language=self.default_language,
                confidence=0.0,
                is_locked=False,
                fallback_reason="text_too_short"
            )
        
        # Attempt detection
        if not LANGDETECT_AVAILABLE:
            return self._fallback_result(
                call_id,
                "library_unavailable"
            )
        
        try:
            # Detect language
            detections = detect_langs(text)
            
            if not detections:
                logger.warning(
                    "No language detected",
                    extra={"call_id": call_id}
                )
                # Allow retry if under attempt limit
                return DetectionResult(
                    status=DetectionStatus.REJECTED,
                    language=self.default_language,
                    confidence=0.0,
                    is_locked=False,
                    fallback_reason="no_detection"
                )
            
            # Get top detection
            top = detections[0]
            detected_lang = top.lang
            confidence = top.prob
            
            logger.info(
                f"Language detected: {detected_lang} ({confidence:.2f})",
                extra={
                    "call_id": call_id,
                    "text_length": len(text),
                    "attempt": self._attempt_counts[call_id]
                }
            )
            
            # Check if supported
            if detected_lang not in self.supported_languages:
                logger.warning(
                    f"Unsupported language: {detected_lang}",
                    extra={"call_id": call_id}
                )
                return self._fallback_result(
                    call_id,
                    "unsupported_language"
                )
            
            # Check confidence
            if confidence < self.min_confidence:
                logger.warning(
                    f"Low confidence: {confidence:.2f}",
                    extra={"call_id": call_id}
                )
                # Allow retry if under attempt limit
                return DetectionResult(
                    status=DetectionStatus.REJECTED,
                    language=self.default_language,
                    confidence=confidence,
                    is_locked=False,
                    fallback_reason="low_confidence"
                )
            
            # Lock language (SUCCESS!)
            self._lock_language(
                call_id=call_id,
                language=detected_lang,
                confidence=confidence,
                method="detected"
            )
            
            # Clean up tracking
            self._cleanup_tracking(call_id)
            
            return DetectionResult(
                status=DetectionStatus.SUCCESS,
                language=detected_lang,
                confidence=confidence,
                is_locked=True
            )
        
        except LangDetectException as e:
            logger.error(
                f"Language detection error: {e}",
                extra={"call_id": call_id},
                exc_info=True
            )
            return self._fallback_result(
                call_id,
                "detection_error"
            )
        
        except Exception as e:
            logger.error(
                f"Unexpected detection error: {e}",
                extra={"call_id": call_id},
                exc_info=True
            )
            return self._fallback_result(
                call_id,
                "error"
            )
    
    def _fallback_result(
        self,
        call_id: str,
        reason: str
    ) -> DetectionResult:
        """
        Create fallback result and lock to default language.
        
        Args:
            call_id: Call identifier
            reason: Fallback reason
            
        Returns:
            Fallback detection result
        """
        # Lock to default language
        self._lock_language(
            call_id=call_id,
            language=self.default_language,
            confidence=1.0,
            method="fallback"
        )
        
        # Clean up tracking
        self._cleanup_tracking(call_id)
        
        logger.info(
            f"Fallback to default language: {self.default_language}",
            extra={"call_id": call_id, "reason": reason}
        )
        
        return DetectionResult(
            status=DetectionStatus.FALLBACK,
            language=self.default_language,
            confidence=1.0,
            is_locked=True,
            fallback_reason=reason
        )
    
    def _cleanup_tracking(self, call_id: str) -> None:
        """
        Clean up attempt tracking for call.
        
        Args:
            call_id: Call identifier
        """
        if call_id in self._attempt_counts:
            del self._attempt_counts[call_id]
        if call_id in self._first_attempt_time:
            del self._first_attempt_time[call_id]
    
    def _lock_language(
        self,
        call_id: str,
        language: str,
        confidence: float,
        method: str
    ) -> None:
        """
        Lock language for call.
        
        Args:
            call_id: Call identifier
            language: Language code
            confidence: Detection confidence
            method: Detection method
        """
        lock = LanguageLock(
            call_id=call_id,
            language=language,
            confidence=confidence,
            locked_at=datetime.now(timezone.utc),
            detection_method=method
        )
        
        self._locks[call_id] = lock
        
        logger.info(
            f"Language locked: {language}",
            extra={
                "call_id": call_id,
                "confidence": confidence,
                "method": method
            }
        )
    
    def get_locked_language(
        self,
        call_id: str
    ) -> Optional[str]:
        """
        Get locked language for call.
        
        Args:
            call_id: Call identifier
            
        Returns:
            Language code or None
        """
        lock = self._locks.get(call_id)
        return lock.language if lock else None
    
    def is_locked(self, call_id: str) -> bool:
        """
        Check if language is locked for call.
        
        Args:
            call_id: Call identifier
            
        Returns:
            True if locked
        """
        return call_id in self._locks
    
    def get_lock_info(
        self,
        call_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get lock information.
        
        Args:
            call_id: Call identifier
            
        Returns:
            Lock info dict or None
        """
        lock = self._locks.get(call_id)
        if not lock:
            return None
        
        return {
            "language": lock.language,
            "confidence": lock.confidence,
            "locked_at": lock.locked_at.isoformat(),
            "detection_method": lock.detection_method
        }
    
    def unlock(self, call_id: str) -> bool:
        """
        Unlock language for call and clean up tracking.
        
        Only use for cleanup after call ends.
        
        Args:
            call_id: Call identifier
            
        Returns:
            True if unlocked
        """
        unlocked = False
        
        if call_id in self._locks:
            del self._locks[call_id]
            unlocked = True
            logger.info(
                "Language unlocked",
                extra={"call_id": call_id}
            )
        
        # Clean up tracking regardless
        self._cleanup_tracking(call_id)
        
        return unlocked
    
    def get_active_locks(self) -> Dict[str, LanguageLock]:
        """Get all active language locks."""
        return self._locks.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get detector statistics.
        
        Returns:
            Statistics dictionary
        """
        locks_by_language: Dict[str, int] = {}
        locks_by_method: Dict[str, int] = {}
        
        for lock in self._locks.values():
            locks_by_language[lock.language] = (
                locks_by_language.get(lock.language, 0) + 1
            )
            locks_by_method[lock.detection_method] = (
                locks_by_method.get(lock.detection_method, 0) + 1
            )
        
        return {
            "active_locks": len(self._locks),
            "locks_by_language": locks_by_language,
            "locks_by_method": locks_by_method,
            "supported_languages": self.supported_languages,
            "default_language": self.default_language,
            "pending_detections": len(self._attempt_counts),
            "max_attempts": self.max_attempts,
            "timeout_seconds": self.timeout_seconds
        }


# Default instance
_default_detector: Optional[LanguageDetector] = None


def get_default_detector() -> LanguageDetector:
    """
    Get or create default detector.
    
    Returns:
        Default detector instance
    """
    global _default_detector
    
    if _default_detector is None:
        _default_detector = LanguageDetector()
    
    return _default_detector


def detect_language(
    call_id: str,
    text: str,
    is_final: bool = True
) -> DetectionResult:
    """
    Quick language detection.
    
    Args:
        call_id: Call identifier
        text: Text to detect from
        is_final: Whether this is final transcript
        
    Returns:
        Detection result
    """
    detector = get_default_detector()
    return detector.detect(call_id, text, is_final)
