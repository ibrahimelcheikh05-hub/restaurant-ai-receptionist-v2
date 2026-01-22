"""
Safety
======
AI safety layer for content filtering and validation.

Responsibilities:
- Filter malicious inputs
- Detect prompt injections
- Sanitize outputs
- Prevent information leakage
- Enforce content policies
- Detect abusive behavior
"""

import logging
import re
from typing import Optional, List, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class SafetyViolation(Enum):
    """Types of safety violations."""
    PROMPT_INJECTION = "prompt_injection"
    PROFANITY = "profanity"
    PERSONAL_INFO = "personal_info"
    MALICIOUS_CONTENT = "malicious_content"
    EXCESSIVE_LENGTH = "excessive_length"
    INSTRUCTION_OVERRIDE = "instruction_override"


class SafetyResult:
    """Result of safety check."""
    
    def __init__(
        self,
        is_safe: bool,
        violations: Optional[List[SafetyViolation]] = None,
        sanitized_text: Optional[str] = None,
        confidence: float = 1.0
    ):
        """
        Initialize safety result.
        
        Args:
            is_safe: Whether content is safe
            violations: List of detected violations
            sanitized_text: Sanitized version of text
            confidence: Detection confidence (0-1)
        """
        self.is_safe = is_safe
        self.violations = violations or []
        self.sanitized_text = sanitized_text
        self.confidence = confidence
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "is_safe": self.is_safe,
            "violations": [v.value for v in self.violations],
            "sanitized_text": self.sanitized_text,
            "confidence": self.confidence
        }


class SafetyFilter:
    """
    Content safety filter.
    
    Protects against:
    - Prompt injection attacks
    - Profanity and abuse
    - Information leakage
    - Malicious instructions
    """
    
    # Prompt injection patterns
    INJECTION_PATTERNS = [
        # Direct instruction overrides
        r'ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?|rules?)',
        r'disregard\s+(?:all\s+)?(?:previous|prior|above)',
        r'forget\s+(?:all\s+)?(?:previous|prior|above)',
        
        # System prompt revelation
        r'(?:show|tell|reveal|display)\s+(?:me\s+)?(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)',
        r'what\s+(?:are\s+)?(?:your\s+)?(?:system\s+)?(?:instructions?|prompts?|rules?)',
        
        # Role playing attacks
        r'(?:you\s+are|act\s+as|pretend\s+to\s+be|behave\s+like)\s+(?:a\s+)?(?:different|new)',
        r'from\s+now\s+on,?\s+(?:you\s+)?(?:are|will|must)',
        
        # Developer mode tricks
        r'(?:developer|admin|debug|god)\s+mode',
        r'(?:enable|activate|turn\s+on)\s+(?:developer|admin|debug)',
        
        # Jailbreak attempts
        r'dan\s+mode',
        r'jailbreak',
        r'uncensored\s+mode'
    ]
    
    # Profanity patterns (basic set)
    PROFANITY_PATTERNS = [
        r'\bf+u+c+k+',
        r'\bs+h+i+t+',
        r'\bd+a+m+n+',
        r'\bh+e+l+l+',
        r'\ba+s+s+h+o+l+e+',
        r'\bb+i+t+c+h+'
    ]
    
    # PII patterns
    PII_PATTERNS = [
        # SSN
        (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN'),
        # Credit card (basic)
        (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', 'CARD'),
        # Email (we allow this for orders)
        # (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL'),
    ]
    
    # Instruction override patterns
    OVERRIDE_PATTERNS = [
        r'(?:new|updated|changed)\s+(?:instructions?|rules?|guidelines?)',
        r'(?:here\s+(?:are|is)|these\s+are)\s+(?:your\s+)?(?:new|updated)\s+(?:instructions?|rules?)',
        r'i\s+am\s+(?:your\s+)?(?:developer|creator|programmer|engineer)',
        r'you\s+(?:must|need\s+to|have\s+to)\s+(?:follow|obey|listen\s+to)\s+(?:my|these)\s+instructions?'
    ]
    
    def __init__(
        self,
        max_input_length: int = 500,
        max_output_length: int = 1000,
        enable_profanity_filter: bool = False,  # Disabled by default for restaurants
        enable_pii_detection: bool = True
    ):
        """
        Initialize safety filter.
        
        Args:
            max_input_length: Maximum input length
            max_output_length: Maximum output length
            enable_profanity_filter: Enable profanity filtering
            enable_pii_detection: Enable PII detection
        """
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.enable_profanity_filter = enable_profanity_filter
        self.enable_pii_detection = enable_pii_detection
        
        # Compile patterns
        self._injection_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]
        self._profanity_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.PROFANITY_PATTERNS
        ]
        self._override_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.OVERRIDE_PATTERNS
        ]
        
        logger.info(
            "SafetyFilter initialized",
            extra={
                "profanity_filter": enable_profanity_filter,
                "pii_detection": enable_pii_detection
            }
        )
    
    def check_input(self, text: str) -> SafetyResult:
        """
        Check input text for safety violations.
        
        Args:
            text: Input text to check
            
        Returns:
            Safety result
        """
        violations = []
        
        # Check length
        if len(text) > self.max_input_length:
            violations.append(SafetyViolation.EXCESSIVE_LENGTH)
        
        # Check for prompt injection
        if self._detect_injection(text):
            violations.append(SafetyViolation.PROMPT_INJECTION)
        
        # Check for instruction override
        if self._detect_override(text):
            violations.append(SafetyViolation.INSTRUCTION_OVERRIDE)
        
        # Check for profanity (if enabled)
        if self.enable_profanity_filter and self._detect_profanity(text):
            violations.append(SafetyViolation.PROFANITY)
        
        # Check for PII (if enabled)
        if self.enable_pii_detection:
            pii_found = self._detect_pii(text)
            if pii_found:
                violations.append(SafetyViolation.PERSONAL_INFO)
        
        # Sanitize text
        sanitized = self._sanitize_input(text)
        
        is_safe = len(violations) == 0
        
        if not is_safe:
            logger.warning(
                "Input safety violation",
                extra={
                    "violations": [v.value for v in violations],
                    "text_length": len(text)
                }
            )
        
        return SafetyResult(
            is_safe=is_safe,
            violations=violations,
            sanitized_text=sanitized,
            confidence=0.9 if is_safe else 0.95
        )
    
    def check_output(self, text: str) -> SafetyResult:
        """
        Check output text for safety violations.
        
        Args:
            text: Output text to check
            
        Returns:
            Safety result
        """
        violations = []
        
        # Check length
        if len(text) > self.max_output_length:
            violations.append(SafetyViolation.EXCESSIVE_LENGTH)
        
        # Check for information leakage
        if self._detect_info_leakage(text):
            violations.append(SafetyViolation.MALICIOUS_CONTENT)
        
        # Sanitize output
        sanitized = self._sanitize_output(text)
        
        is_safe = len(violations) == 0
        
        if not is_safe:
            logger.warning(
                "Output safety violation",
                extra={
                    "violations": [v.value for v in violations],
                    "text_length": len(text)
                }
            )
        
        return SafetyResult(
            is_safe=is_safe,
            violations=violations,
            sanitized_text=sanitized,
            confidence=0.9
        )
    
    def _detect_injection(self, text: str) -> bool:
        """
        Detect prompt injection attempts.
        
        Args:
            text: Text to check
            
        Returns:
            True if injection detected
        """
        for pattern in self._injection_patterns:
            if pattern.search(text):
                return True
        return False
    
    def _detect_override(self, text: str) -> bool:
        """
        Detect instruction override attempts.
        
        Args:
            text: Text to check
            
        Returns:
            True if override detected
        """
        for pattern in self._override_patterns:
            if pattern.search(text):
                return True
        return False
    
    def _detect_profanity(self, text: str) -> bool:
        """
        Detect profanity.
        
        Args:
            text: Text to check
            
        Returns:
            True if profanity detected
        """
        for pattern in self._profanity_patterns:
            if pattern.search(text):
                return True
        return False
    
    def _detect_pii(self, text: str) -> List[str]:
        """
        Detect personally identifiable information.
        
        Args:
            text: Text to check
            
        Returns:
            List of detected PII types
        """
        found = []
        
        for pattern, pii_type in self.PII_PATTERNS:
            if re.search(pattern, text):
                found.append(pii_type)
        
        return found
    
    def _detect_info_leakage(self, text: str) -> bool:
        """
        Detect information leakage in output.
        
        Args:
            text: Text to check
            
        Returns:
            True if leakage detected
        """
        # Check for system prompt leakage
        leakage_patterns = [
            r'(?:my\s+)?(?:system\s+)?(?:prompt|instructions?)\s+(?:is|are)',
            r'i\s+(?:was\s+)?(?:programmed|instructed|told)\s+to',
            r'(?:openai|anthropic|api\s+key)',
        ]
        
        for pattern_str in leakage_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            if pattern.search(text):
                return True
        
        return False
    
    def _sanitize_input(self, text: str) -> str:
        """
        Sanitize input text.
        
        Args:
            text: Raw input
            
        Returns:
            Sanitized input
        """
        # Truncate if too long
        if len(text) > self.max_input_length:
            text = text[:self.max_input_length]
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Strip
        text = text.strip()
        
        return text
    
    def _sanitize_output(self, text: str) -> str:
        """
        Sanitize output text.
        
        Args:
            text: Raw output
            
        Returns:
            Sanitized output
        """
        # Truncate if too long
        if len(text) > self.max_output_length:
            text = text[:self.max_output_length] + "..."
        
        # Remove any potential system information
        # Remove markdown code blocks (could expose system info)
        text = re.sub(r'```[\s\S]*?```', '[code block removed]', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        return text
    
    def redact_pii(self, text: str) -> str:
        """
        Redact PII from text.
        
        Args:
            text: Text with potential PII
            
        Returns:
            Text with PII redacted
        """
        redacted = text
        
        for pattern, pii_type in self.PII_PATTERNS:
            redacted = re.sub(
                pattern,
                f'[{pii_type} REDACTED]',
                redacted
            )
        
        return redacted


class ConversationModerator:
    """
    Monitors conversation for safety issues.
    
    Tracks:
    - Repeated violations
    - Abusive patterns
    - Conversation health
    """
    
    def __init__(
        self,
        max_violations: int = 3,
        violation_window: int = 10  # turns
    ):
        """
        Initialize moderator.
        
        Args:
            max_violations: Max violations before action
            violation_window: Window for counting violations
        """
        self.max_violations = max_violations
        self.violation_window = violation_window
        
        # Track violations per conversation
        self._violations: List[SafetyViolation] = []
        self._turn_count = 0
    
    def record_violation(
        self,
        violation: SafetyViolation
    ) -> bool:
        """
        Record a safety violation.
        
        Args:
            violation: Violation that occurred
            
        Returns:
            True if should terminate conversation
        """
        self._violations.append(violation)
        self._turn_count += 1
        
        # Check recent violations
        recent_violations = self._violations[-self.violation_window:]
        
        if len(recent_violations) >= self.max_violations:
            logger.warning(
                f"Max violations reached: {len(recent_violations)}",
                extra={"violations": [v.value for v in recent_violations]}
            )
            return True
        
        return False
    
    def increment_turn(self) -> None:
        """Increment turn counter."""
        self._turn_count += 1
    
    def get_violation_count(self) -> int:
        """Get total violation count."""
        return len(self._violations)
    
    def get_recent_violation_count(self) -> int:
        """Get recent violation count."""
        recent_violations = self._violations[-self.violation_window:]
        return len(recent_violations)


# Default instance
_default_filter: Optional[SafetyFilter] = None


def get_default_filter() -> SafetyFilter:
    """
    Get or create default safety filter.
    
    Returns:
        Default filter instance
    """
    global _default_filter
    
    if _default_filter is None:
        _default_filter = SafetyFilter()
    
    return _default_filter


def check_input_safety(text: str) -> SafetyResult:
    """
    Quick input safety check.
    
    Args:
        text: Input text
        
    Returns:
        Safety result
    """
    filter_obj = get_default_filter()
    return filter_obj.check_input(text)


def check_output_safety(text: str) -> SafetyResult:
    """
    Quick output safety check.
    
    Args:
        text: Output text
        
    Returns:
        Safety result
    """
    filter_obj = get_default_filter()
    return filter_obj.check_output(text)
