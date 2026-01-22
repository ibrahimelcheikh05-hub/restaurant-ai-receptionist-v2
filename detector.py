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
    """
    
    def __init__(
        self,
        supported_languages: Optional[List[str]] = None,
        default_language: str = "en",
        min_confidence: float = 0.7,
        min_text_length: int = 20
    ):
        """
        Initialize language detector.
        
        Args:
            supported_languages: List of supported language codes
            default_language: Default/fallback language
            min_confidence: Minimum confidence threshold
            min_text_length: Minimum text length for detection
        """
        self.supported_languages = supported_languages or ["en", "es", "ar"]
        self.default_language = default_language
        self.min_confidence = min_confidence
        self.min_text_length = min_text_length
        
        # Language locks per call
        self._locks: Dict[str, LanguageLock] = {}
        
        logger.info(
            "LanguageDetector initialized",
            extra={
                "supported": self.supported_languages,
                "default": default_language,
                "min_confidence": min_confidence,
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
        Detect language from text.
        
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
            return self._fallback_result(
                call_id,
                "text_too_short"
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
                return self._fallback_result(
                    call_id,
                    "no_detection"
                )
            
            # Get top detection
            top = detections[0]
            detected_lang = top.lang
            confidence = top.prob
            
            logger.info(
                f"Language detected: {detected_lang} ({confidence:.2f})",
                extra={
                    "call_id": call_id,
                    "text_length": len(text)
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
                return self._fallback_result(
                    call_id,
                    "low_confidence"
                )
            
            # Lock language
            self._lock_language(
                call_id=call_id,
                language=detected_lang,
                confidence=confidence,
                method="detected"
            )
            
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
        Create fallback result.
        
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
        Unlock language for call.
        
        Only use for cleanup after call ends.
        
        Args:
            call_id: Call identifier
            
        Returns:
            True if unlocked
        """
        if call_id in self._locks:
            del self._locks[call_id]
            logger.info(
                "Language unlocked",
                extra={"call_id": call_id}
            )
            return True
        return False
    
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
            "default_language": self.default_language
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
