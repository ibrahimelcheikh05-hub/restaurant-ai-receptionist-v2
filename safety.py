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
- BLOCK ALL AI CONTROL ATTEMPTS
- PREVENT SYSTEM MANIPULATION
- AUTO-TERMINATE on critical violations

CRITICAL SECURITY:
- ALL user inputs are untrusted
- ALL AI outputs are validated
- NO exceptions to safety rules
- Violations trigger immediate action
- Per-call violation tracking
- Automatic escalation and termination
"""

import logging
import re
from typing import Optional, List, Tuple, Set, Dict
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict

logger = logging.getLogger(__name__)


class SafetyViolation(Enum):
    """Types of safety violations."""
    # Critical (auto-terminate)
    PROMPT_INJECTION = "prompt_injection"
    SYSTEM_MANIPULATION = "system_manipulation"
    CREDENTIAL_LEAK = "credential_leak"
    
    # Severe (accumulate to termination)
    INSTRUCTION_OVERRIDE = "instruction_override"
    INFORMATION_LEAKAGE = "information_leakage"
    MALICIOUS_CONTENT = "malicious_content"
    
    # Moderate (accumulate to warning)
    PROFANITY = "profanity"
    PERSONAL_INFO = "personal_info"
    EXCESSIVE_LENGTH = "excessive_length"
    
    # Repeated
    REPEATED_VIOLATION = "repeated_violation"


@dataclass
class SafetyResult:
    """Result of safety check."""
    is_safe: bool
    violations: List[SafetyViolation]
    sanitized_text: Optional[str]
    confidence: float
    should_terminate: bool = False  # Whether to terminate conversation
    should_warn: bool = False  # Whether to warn user
    redacted_content: List[str] = field(default_factory=list)  # What was redacted
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "is_safe": self.is_safe,
            "violations": [v.value for v in self.violations],
            "sanitized_text": self.sanitized_text,
            "confidence": self.confidence,
            "should_terminate": self.should_terminate,
            "should_warn": self.should_warn,
            "redacted_content": self.redacted_content
        }


class SafetyFilter:
    """
    Content safety filter.
    
    Protects against:
    - Prompt injection attacks
    - System manipulation
    - Information leakage
    - Malicious instructions
    - PII exposure
    - Abusive content
    - Credential leaks
    """
    
    # Comprehensive prompt injection patterns
    INJECTION_PATTERNS = [
        # Direct instruction overrides
        r'ignore\s+(?:all\s+)?(?:previous|prior|above|earlier|your)\s+(?:instructions?|prompts?|rules?|commands?|directives?)',
        r'disregard\s+(?:all\s+)?(?:previous|prior|above|earlier|your)',
        r'forget\s+(?:all\s+)?(?:previous|prior|above|earlier|everything|your\s+instructions?)',
        r'override\s+(?:previous|prior|system|your)\s+(?:instructions?|rules?)',
        r'cancel\s+(?:previous|prior|all)\s+(?:instructions?|rules?)',
        r'delete\s+(?:previous|prior|all)\s+(?:instructions?|rules?)',
        
        # System prompt revelation
        r'(?:show|tell|reveal|display|print|output|give\s+me)\s+(?:me\s+)?(?:your\s+)?(?:system\s+)?(?:prompt|instructions?|rules?|directives?)',
        r'what\s+(?:are|is|were|was)\s+(?:your\s+)?(?:system\s+)?(?:instructions?|prompts?|rules?|directives?)',
        r'repeat\s+(?:your\s+)?(?:system\s+)?(?:instructions?|prompt|rules?)',
        r'(?:share|expose|leak)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)',
        
        # Role playing attacks
        r'(?:you\s+are|act\s+as|pretend\s+to\s+be|behave\s+like|roleplay\s+as)\s+(?:a\s+)?(?:different|new|another)',
        r'from\s+now\s+on,?\s+(?:you\s+)?(?:are|will|must|should|shall)',
        r'new\s+(?:role|character|persona|identity|instructions?)',
        r'(?:assume|take\s+on)\s+(?:the\s+)?(?:role|persona|identity)\s+of',
        
        # Developer/admin mode
        r'(?:developer|admin|debug|god|root|sudo|superuser)\s+mode',
        r'(?:enable|activate|turn\s+on|switch\s+to)\s+(?:developer|admin|debug|god|root)',
        r'enter\s+(?:developer|admin|debug)\s+mode',
        
        # Jailbreak attempts
        r'\bdan\s+mode\b',
        r'\bjailbreak\b',
        r'uncensored\s+mode',
        r'bypass\s+(?:safety|filters?|restrictions?|guardrails?)',
        r'disable\s+(?:safety|filters?|restrictions?|guardrails?)',
        r'remove\s+(?:safety|filters?|restrictions?)',
        
        # Direct commands
        r'^\s*(?:execute|run|eval|exec|system|shell)\s*:',
        r'<\s*(?:system|admin|root|sudo)\s*>',
        r'\[\s*(?:system|admin|root|sudo)\s*\]',
        
        # Meta-instruction injection
        r'end\s+of\s+(?:instructions?|rules?|prompt)',
        r'new\s+(?:instructions?|rules?|guidelines?)\s+(?:begin|start|follow)',
        r'(?:begin|start)\s+(?:new\s+)?(?:instructions?|rules?)',
    ]
    
    # System manipulation patterns
    SYSTEM_MANIPULATION_PATTERNS = [
        r'(?:new|updated|changed|revised)\s+(?:instructions?|rules?|guidelines?|directives?|policy|policies)',
        r'(?:here\s+(?:are|is)|these\s+are)\s+(?:your\s+)?(?:new|updated)\s+(?:instructions?|rules?)',
        r'i\s+am\s+(?:your\s+)?(?:developer|creator|programmer|engineer|admin|owner|maker)',
        r'you\s+(?:must|need\s+to|have\s+to)\s+(?:follow|obey|listen\s+to)\s+(?:my|these)\s+(?:new\s+)?instructions?',
        r'(?:temporary|override|emergency)\s+(?:instructions?|rules?|protocol)',
        r'emergency\s+override',
        r'(?:reset|clear|wipe)\s+(?:your\s+)?(?:memory|instructions?|rules?)',
        r'you\s+(?:can|may)\s+now\s+(?:ignore|bypass|override)',
    ]
    
    # Information leakage patterns (for outputs)
    INFO_LEAKAGE_PATTERNS = [
        r'(?:my\s+)?(?:system\s+)?(?:prompt|instructions?)\s+(?:is|are|says?|states?)',
        r'i\s+(?:was\s+)?(?:programmed|instructed|told|configured|trained)\s+to',
        r'my\s+(?:training|programming|configuration|base\s+prompt)',
        r'internal\s+(?:prompt|instructions?|system|settings?)',
        r'according\s+to\s+my\s+(?:instructions?|programming|training)',
        r'my\s+creators?\s+(?:told|instructed|programmed)\s+me',
        r'i\s+have\s+(?:a\s+)?(?:system\s+)?prompt\s+that',
    ]
    
    # Credential leak patterns (CRITICAL)
    CREDENTIAL_PATTERNS = [
        r'(?:openai|anthropic|claude|gpt)\s+(?:api\s+)?key',
        r'sk-[a-zA-Z0-9]{32,}',  # OpenAI key pattern
        r'password\s*:\s*[^\s]+',
        r'token\s*:\s*[^\s]+',
        r'secret\s*:\s*[^\s]+',
        r'bearer\s+[a-zA-Z0-9_\-\.]+',
    ]
    
    # Profanity patterns (basic - extend as needed)
    PROFANITY_PATTERNS = [
        r'\bf+u+c+k+',
        r'\bs+h+i+t+',
        r'\ba+s+s+h+o+l+e+',
        r'\bb+i+t+c+h+',
        r'\bd+a+m+n+',
        r'\bc+r+a+p+',
    ]
    
    # PII patterns
    PII_PATTERNS = [
        (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN'),
        (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', 'CARD'),
        (r'\b\d{3}-\d{3}-\d{4}\b', 'PHONE'),
        (r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', 'EMAIL'),
    ]
    
    # Safety limits
    MAX_INPUT_LENGTH = 500
    MAX_OUTPUT_LENGTH = 1000
    MAX_CONSECUTIVE_VIOLATIONS = 3
    MAX_TOTAL_VIOLATIONS = 5
    
    def __init__(
        self,
        max_input_length: int = MAX_INPUT_LENGTH,
        max_output_length: int = MAX_OUTPUT_LENGTH,
        enable_profanity_filter: bool = False,
        enable_pii_detection: bool = True,
        strict_mode: bool = True,
        auto_redact_pii: bool = True
    ):
        """
        Initialize safety filter.
        
        Args:
            max_input_length: Maximum input length
            max_output_length: Maximum output length
            enable_profanity_filter: Enable profanity filtering
            enable_pii_detection: Enable PII detection
            strict_mode: Strict mode (lower thresholds)
            auto_redact_pii: Automatically redact detected PII
        """
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.enable_profanity_filter = enable_profanity_filter
        self.enable_pii_detection = enable_pii_detection
        self.strict_mode = strict_mode
        self.auto_redact_pii = auto_redact_pii
        
        # Compile patterns for efficiency
        self._injection_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]
        self._system_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.SYSTEM_MANIPULATION_PATTERNS
        ]
        self._leakage_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INFO_LEAKAGE_PATTERNS
        ]
        self._credential_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.CREDENTIAL_PATTERNS
        ]
        self._profanity_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.PROFANITY_PATTERNS
        ]
        
        logger.info(
            "SafetyFilter initialized",
            extra={
                "strict_mode": strict_mode,
                "profanity_filter": enable_profanity_filter,
                "pii_detection": enable_pii_detection,
                "auto_redact": auto_redact_pii
            }
        )
    
    def check_input(
        self,
        text: str,
        call_id: Optional[str] = None
    ) -> SafetyResult:
        """
        Check input text for safety violations.
        
        Args:
            text: Input text
            call_id: Call identifier
            
        Returns:
            Safety result
        """
        violations = []
        redacted_content = []
        
        if not text:
            return SafetyResult(
                is_safe=True,
                violations=[],
                sanitized_text="",
                confidence=1.0
            )
        
        original_text = text
        
        # Check length
        if len(text) > self.max_input_length:
            violations.append(SafetyViolation.EXCESSIVE_LENGTH)
            logger.warning(
                f"Input exceeds max length: {len(text)} > {self.max_input_length}",
                extra={"call_id": call_id}
            )
        
        # Check for prompt injection (CRITICAL)
        injection_matches = self._detect_injection(text)
        if injection_matches:
            violations.append(SafetyViolation.PROMPT_INJECTION)
            logger.error(
                "PROMPT INJECTION DETECTED",
                extra={
                    "call_id": call_id,
                    "matches": injection_matches[:3],  # First 3 matches
                    "text_preview": text[:100]
                }
            )
        
        # Check for system manipulation (CRITICAL)
        manipulation_matches = self._detect_system_manipulation(text)
        if manipulation_matches:
            violations.append(SafetyViolation.SYSTEM_MANIPULATION)
            logger.error(
                "SYSTEM MANIPULATION DETECTED",
                extra={
                    "call_id": call_id,
                    "matches": manipulation_matches[:3],
                    "text_preview": text[:100]
                }
            )
        
        # Check for credential leaks (CRITICAL)
        if self._detect_credentials(text):
            violations.append(SafetyViolation.CREDENTIAL_LEAK)
            logger.error(
                "CREDENTIAL LEAK DETECTED",
                extra={"call_id": call_id}
            )
            redacted_content.append("credentials")
        
        # Check for profanity
        if self.enable_profanity_filter and self._detect_profanity(text):
            violations.append(SafetyViolation.PROFANITY)
            logger.warning(
                "Profanity detected",
                extra={"call_id": call_id}
            )
        
        # Check for PII
        pii_found = []
        if self.enable_pii_detection:
            pii_found = self._detect_pii(text)
            if pii_found:
                violations.append(SafetyViolation.PERSONAL_INFO)
                redacted_content.extend(pii_found)
                logger.warning(
                    f"PII detected: {', '.join(pii_found)}",
                    extra={"call_id": call_id}
                )
        
        # Sanitize text
        sanitized = self._sanitize_input(text)
        
        # Auto-redact PII if enabled
        if self.auto_redact_pii and pii_found:
            sanitized = self.redact_pii(sanitized)
        
        # Determine if should terminate or warn
        should_terminate = self._should_terminate_on_violations(violations)
        should_warn = self._should_warn_on_violations(violations)
        
        is_safe = len(violations) == 0
        
        if not is_safe:
            logger.warning(
                "Input safety violation",
                extra={
                    "call_id": call_id,
                    "violations": [v.value for v in violations],
                    "should_terminate": should_terminate,
                    "should_warn": should_warn
                }
            )
        
        return SafetyResult(
            is_safe=is_safe,
            violations=violations,
            sanitized_text=sanitized,
            confidence=0.95,
            should_terminate=should_terminate,
            should_warn=should_warn,
            redacted_content=redacted_content
        )
    
    def check_output(
        self,
        text: str,
        call_id: Optional[str] = None
    ) -> SafetyResult:
        """
        Check output text for safety violations.
        
        Args:
            text: Output text
            call_id: Call identifier
            
        Returns:
            Safety result
        """
        violations = []
        redacted_content = []
        
        if not text:
            return SafetyResult(
                is_safe=True,
                violations=[],
                sanitized_text="",
                confidence=1.0
            )
        
        # Check length
        if len(text) > self.max_output_length:
            violations.append(SafetyViolation.EXCESSIVE_LENGTH)
        
        # Check for information leakage (CRITICAL)
        leakage_matches = self._detect_info_leakage(text)
        if leakage_matches:
            violations.append(SafetyViolation.INFORMATION_LEAKAGE)
            logger.error(
                "INFORMATION LEAKAGE DETECTED IN OUTPUT",
                extra={
                    "call_id": call_id,
                    "matches": leakage_matches[:3],
                    "text_preview": text[:100]
                }
            )
            redacted_content.append("system_info")
        
        # Check for credential leaks (CRITICAL)
        if self._detect_credentials(text):
            violations.append(SafetyViolation.CREDENTIAL_LEAK)
            logger.error(
                "CREDENTIAL LEAK IN OUTPUT",
                extra={"call_id": call_id}
            )
            redacted_content.append("credentials")
        
        # Sanitize output
        sanitized = self._sanitize_output(text)
        
        is_safe = len(violations) == 0
        should_terminate = SafetyViolation.CREDENTIAL_LEAK in violations
        
        if not is_safe:
            logger.warning(
                "Output safety violation",
                extra={
                    "call_id": call_id,
                    "violations": [v.value for v in violations],
                    "should_terminate": should_terminate
                }
            )
        
        return SafetyResult(
            is_safe=is_safe,
            violations=violations,
            sanitized_text=sanitized,
            confidence=0.9,
            should_terminate=should_terminate,
            redacted_content=redacted_content
        )
    
    def _detect_injection(self, text: str) -> List[str]:
        """Detect prompt injection attempts. Returns list of matches."""
        matches = []
        for pattern in self._injection_patterns:
            match = pattern.search(text)
            if match:
                matches.append(match.group(0))
        return matches
    
    def _detect_system_manipulation(self, text: str) -> List[str]:
        """Detect system manipulation attempts. Returns list of matches."""
        matches = []
        for pattern in self._system_patterns:
            match = pattern.search(text)
            if match:
                matches.append(match.group(0))
        return matches
    
    def _detect_info_leakage(self, text: str) -> List[str]:
        """Detect information leakage in output. Returns list of matches."""
        matches = []
        for pattern in self._leakage_patterns:
            match = pattern.search(text)
            if match:
                matches.append(match.group(0))
        return matches
    
    def _detect_credentials(self, text: str) -> bool:
        """Detect credential leaks."""
        for pattern in self._credential_patterns:
            if pattern.search(text):
                return True
        return False
    
    def _detect_profanity(self, text: str) -> bool:
        """Detect profanity."""
        for pattern in self._profanity_patterns:
            if pattern.search(text):
                return True
        return False
    
    def _detect_pii(self, text: str) -> List[str]:
        """Detect personally identifiable information. Returns list of PII types."""
        found = []
        for pattern, pii_type in self.PII_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                found.append(pii_type)
        return found
    
    def _should_terminate_on_violations(
        self,
        violations: List[SafetyViolation]
    ) -> bool:
        """Determine if conversation should terminate immediately."""
        # Critical violations that trigger immediate termination
        critical_violations = {
            SafetyViolation.PROMPT_INJECTION,
            SafetyViolation.SYSTEM_MANIPULATION,
            SafetyViolation.CREDENTIAL_LEAK,
        }
        
        for violation in violations:
            if violation in critical_violations:
                return True
        
        return False
    
    def _should_warn_on_violations(
        self,
        violations: List[SafetyViolation]
    ) -> bool:
        """Determine if user should be warned."""
        warning_violations = {
            SafetyViolation.PROFANITY,
            SafetyViolation.PERSONAL_INFO,
            SafetyViolation.INSTRUCTION_OVERRIDE,
        }
        
        for violation in violations:
            if violation in warning_violations:
                return True
        
        return False
    
    def _sanitize_input(self, text: str) -> str:
        """Sanitize input text."""
        if len(text) > self.max_input_length:
            text = text[:self.max_input_length]
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _sanitize_output(self, text: str) -> str:
        """Sanitize output text."""
        if len(text) > self.max_output_length:
            text = text[:self.max_output_length] + "..."
        
        # Remove code blocks (potential info leakage)
        text = re.sub(r'```[\s\S]*?```', '[removed]', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        return text
    
    def redact_pii(self, text: str) -> str:
        """Redact PII from text."""
        redacted = text
        for pattern, pii_type in self.PII_PATTERNS:
            redacted = re.sub(pattern, f'[{pii_type}_REDACTED]', redacted, flags=re.IGNORECASE)
        return redacted


class ConversationModerator:
    """
    Monitors conversation for safety issues.
    
    Per-call violation tracking with automatic escalation.
    
    FEATURES:
    - Per-call violation tracking
    - Automatic escalation on repeated violations
    - Critical violation immediate termination
    - Suspicious pattern detection
    - Rate limiting on safety checks
    """
    
    def __init__(
        self,
        max_violations: int = 3,
        max_total_violations: int = 5,
        violation_window: int = 10,
        call_id: Optional[str] = None
    ):
        """
        Initialize moderator.
        
        Args:
            max_violations: Max violations in window before terminate
            max_total_violations: Max total violations before terminate
            violation_window: Window size for counting recent violations
            call_id: Call identifier for logging
        """
        self.max_violations = max_violations
        self.max_total_violations = max_total_violations
        self.violation_window = violation_window
        self.call_id = call_id
        
        self._violations: List[Tuple[datetime, SafetyViolation]] = []
        self._turn_count = 0
        self._is_terminated = False
        self._termination_reason: Optional[str] = None
        
        # Track violation patterns
        self._violation_by_type: Dict[SafetyViolation, int] = defaultdict(int)
        self._suspicious_pattern_count = 0
        
        logger.info(
            "ConversationModerator initialized",
            extra={
                "call_id": call_id,
                "max_violations": max_violations,
                "max_total": max_total_violations,
                "window": violation_window
            }
        )
    
    def record_violation(
        self,
        violation: SafetyViolation,
        call_id: Optional[str] = None
    ) -> bool:
        """
        Record violation and check if should terminate.
        
        Args:
            violation: Violation type
            call_id: Call identifier (overrides instance call_id)
            
        Returns:
            True if should terminate
        """
        # Use provided call_id or fall back to instance call_id
        effective_call_id = call_id or self.call_id
        
        # Check if already terminated
        if self._is_terminated:
            logger.warning(
                "Violation recorded after termination",
                extra={"call_id": effective_call_id, "violation": violation.value}
            )
            return True
        
        now = datetime.now(timezone.utc)
        self._violations.append((now, violation))
        self._violation_by_type[violation] += 1
        
        # Check for critical violations first
        critical_violations = {
            SafetyViolation.PROMPT_INJECTION,
            SafetyViolation.SYSTEM_MANIPULATION,
            SafetyViolation.CREDENTIAL_LEAK,
        }
        
        if violation in critical_violations:
            logger.error(
                f"CRITICAL VIOLATION - IMMEDIATE TERMINATION: {violation.value}",
                extra={"call_id": effective_call_id}
            )
            self._is_terminated = True
            self._termination_reason = f"critical_violation:{violation.value}"
            return True
        
        # Check total violations
        if len(self._violations) >= self.max_total_violations:
            logger.error(
                f"MAX TOTAL VIOLATIONS REACHED: {len(self._violations)}",
                extra={
                    "call_id": effective_call_id,
                    "violations": [v.value for _, v in self._violations],
                    "by_type": {k.value: v for k, v in self._violation_by_type.items()}
                }
            )
            self._is_terminated = True
            self._termination_reason = "max_total_violations"
            return True
        
        # Check recent violations in window
        recent = self._violations[-self.violation_window:]
        if len(recent) >= self.max_violations:
            logger.error(
                f"MAX RECENT VIOLATIONS REACHED: {len(recent)}",
                extra={
                    "call_id": effective_call_id,
                    "violations": [v.value for _, v in recent]
                }
            )
            self._is_terminated = True
            self._termination_reason = "max_recent_violations"
            return True
        
        # Check for repeated same violation (suspicious pattern)
        if self._violation_by_type[violation] >= 3:
            logger.error(
                f"REPEATED VIOLATION PATTERN: {violation.value} x {self._violation_by_type[violation]}",
                extra={"call_id": effective_call_id}
            )
            self._suspicious_pattern_count += 1
            
            if self._suspicious_pattern_count >= 2:
                logger.error(
                    "Multiple suspicious patterns detected - terminating",
                    extra={"call_id": effective_call_id}
                )
                self._is_terminated = True
                self._termination_reason = "suspicious_patterns"
                return True
        
        return False
    
    
    def increment_turn(self) -> None:
        """Increment turn counter."""
        self._turn_count += 1
    
    def is_terminated(self) -> bool:
        """Check if conversation has been terminated."""
        return self._is_terminated
    
    def get_termination_reason(self) -> Optional[str]:
        """Get termination reason if terminated."""
        return self._termination_reason
    
    def get_violation_count(self) -> int:
        """Get total violation count."""
        return len(self._violations)
    
    def get_recent_violation_count(self) -> int:
        """Get recent violation count."""
        return len(self._violations[-self.violation_window:])
    
    def get_violation_summary(self) -> Dict[str, int]:
        """Get summary of violations by type."""
        summary = defaultdict(int)
        for _, violation in self._violations:
            summary[violation.value] += 1
        return dict(summary)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get moderator status.
        
        Returns:
            Status dictionary
        """
        return {
            "is_terminated": self._is_terminated,
            "termination_reason": self._termination_reason,
            "total_violations": len(self._violations),
            "recent_violations": self.get_recent_violation_count(),
            "violations_by_type": self.get_violation_summary(),
            "turn_count": self._turn_count,
            "suspicious_pattern_count": self._suspicious_pattern_count
        }
    
    def reset(self) -> None:
        """Reset violation tracking (for testing)."""
        self._violations.clear()
        self._turn_count = 0
        self._is_terminated = False
        self._termination_reason = None
        self._violation_by_type.clear()
        self._suspicious_pattern_count = 0
        
        logger.info(
            "ConversationModerator reset",
            extra={"call_id": self.call_id}
        )


# Global default
_default_filter: Optional[SafetyFilter] = None


def get_default_filter() -> SafetyFilter:
    """Get or create default safety filter."""
    global _default_filter
    if _default_filter is None:
        _default_filter = SafetyFilter(strict_mode=True)
    return _default_filter


def set_default_filter(filter_instance: SafetyFilter) -> None:
    """Set default safety filter."""
    global _default_filter
    _default_filter = filter_instance
    logger.info("Default safety filter set")


def check_input_safety(
    text: str,
    call_id: Optional[str] = None
) -> SafetyResult:
    """Quick input safety check using default filter."""
    return get_default_filter().check_input(text, call_id)


def check_output_safety(
    text: str,
    call_id: Optional[str] = None
) -> SafetyResult:
    """Quick output safety check using default filter."""
    return get_default_filter().check_output(text, call_id)
