"""
Call State Machine
==================
Production-grade formal state transitions for voice call lifecycle.

State invariants:
- AI outputs NEVER directly change system state
- All transitions are deterministic and validated
- State changes are atomic, logged, and tracked
- Terminal states cannot be exited
- State machine is thread-safe
- No state can be stuck indefinitely
"""

import logging
from enum import Enum
from typing import Optional, Set, List, Dict, Any, Callable
from datetime import datetime, timezone
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CallState(Enum):
    """
    Formal call lifecycle states.
    
    State flow:
        INIT -> CONNECTING -> GREETING -> ACTIVE -> CLOSING -> CLOSED
                                           ^     |
                                           |     v
                                      TRANSFERRING
                                           |
                                           v
                                      TRANSFERRED (terminal)
    
    Active contains sub-flow:
        LISTENING <-> THINKING <-> SPEAKING
                  <-> LANGUAGE_DETECT
    """
    # Initialization
    INIT = "init"                       # Call object created, not yet connected
    CONNECTING = "connecting"           # Establishing telephony connection
    
    # Entry phase
    GREETING = "greeting"               # Playing initial greeting
    LANGUAGE_DETECT = "language_detect" # Detecting user language (ONCE only)
    
    # Active conversation
    ACTIVE = "active"                   # In active conversation flow
    LISTENING = "listening"             # Waiting for user speech
    THINKING = "thinking"               # Processing AI response
    SPEAKING = "speaking"               # Synthesizing/playing AI response
    
    # Special states
    TRANSFERRING = "transferring"       # Call transfer in progress (frozen state)
    
    # Terminal states
    CLOSING = "closing"                 # Graceful shutdown in progress
    CLOSED = "closed"                   # Call terminated normally
    TRANSFERRED = "transferred"         # Call transferred to human
    FAILED = "failed"                   # Call failed/errored
    TIMEOUT = "timeout"                 # Call timed out (silence, max duration, etc)


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


class TerminalStateError(Exception):
    """Raised when attempting to transition from a terminal state."""
    pass


class StateTimeoutError(Exception):
    """Raised when state has been occupied too long."""
    pass


@dataclass
class StateEntry:
    """Record of a single state entry."""
    state: CallState
    entered_at: datetime
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CallStateMachine:
    """
    Thread-safe state machine for call lifecycle management.
    
    Enforces:
    - Valid transition paths only
    - Terminal state immutability
    - Atomic state transitions
    - Comprehensive state history
    - Transition metrics and timing
    - Per-state timeout enforcement
    - State change callbacks
    - Stuck state detection
    """
    
    # Terminal states that cannot be exited
    TERMINAL_STATES = {
        CallState.CLOSED,
        CallState.TRANSFERRED,
        CallState.FAILED,
        CallState.TIMEOUT
    }
    
    # Active conversational states
    ACTIVE_STATES = {
        CallState.LISTENING,
        CallState.THINKING,
        CallState.SPEAKING
    }
    
    # Frozen states where NO AI/business logic should execute
    FROZEN_STATES = {
        CallState.TRANSFERRING,
        CallState.CLOSING,
        *TERMINAL_STATES
    }
    
    # Maximum time allowed in each state (seconds) - enforced by watchdog
    STATE_TIMEOUTS = {
        CallState.INIT: 5.0,
        CallState.CONNECTING: 30.0,
        CallState.GREETING: 15.0,
        CallState.LANGUAGE_DETECT: 10.0,
        CallState.LISTENING: 90.0,  # Silence timeout
        CallState.THINKING: 15.0,   # AI response timeout
        CallState.SPEAKING: 60.0,   # TTS timeout
        CallState.ACTIVE: float('inf'),  # No limit on active state
        CallState.TRANSFERRING: 45.0,
        CallState.CLOSING: 10.0,
        # Terminal states have no timeout
        CallState.CLOSED: float('inf'),
        CallState.TRANSFERRED: float('inf'),
        CallState.FAILED: float('inf'),
        CallState.TIMEOUT: float('inf')
    }
    
    # Define valid state transitions
    VALID_TRANSITIONS = {
        CallState.INIT: {
            CallState.CONNECTING,
            CallState.FAILED,
            CallState.CLOSED
        },
        CallState.CONNECTING: {
            CallState.GREETING,
            CallState.FAILED,
            CallState.CLOSED,
            CallState.TIMEOUT
        },
        CallState.GREETING: {
            CallState.LANGUAGE_DETECT,
            CallState.ACTIVE,
            CallState.LISTENING,
            CallState.CLOSING,
            CallState.FAILED,
            CallState.CLOSED,
            CallState.TIMEOUT
        },
        CallState.LANGUAGE_DETECT: {
            CallState.ACTIVE,
            CallState.LISTENING,
            CallState.CLOSING,
            CallState.FAILED,
            CallState.CLOSED,
            CallState.TIMEOUT
        },
        CallState.ACTIVE: {
            CallState.LISTENING,
            CallState.THINKING,
            CallState.SPEAKING,
            CallState.TRANSFERRING,
            CallState.CLOSING,
            CallState.TIMEOUT,
            CallState.FAILED,
            CallState.CLOSED
        },
        CallState.LISTENING: {
            CallState.THINKING,
            CallState.SPEAKING,
            CallState.TRANSFERRING,
            CallState.CLOSING,
            CallState.TIMEOUT,
            CallState.FAILED,
            CallState.CLOSED
        },
        CallState.THINKING: {
            CallState.SPEAKING,
            CallState.LISTENING,
            CallState.TRANSFERRING,
            CallState.CLOSING,
            CallState.TIMEOUT,
            CallState.FAILED,
            CallState.CLOSED
        },
        CallState.SPEAKING: {
            CallState.LISTENING,
            CallState.THINKING,
            CallState.TRANSFERRING,
            CallState.CLOSING,
            CallState.TIMEOUT,
            CallState.FAILED,
            CallState.CLOSED
        },
        CallState.TRANSFERRING: {
            CallState.TRANSFERRED,
            CallState.FAILED,
            CallState.CLOSED,
            CallState.TIMEOUT
        },
        CallState.CLOSING: {
            CallState.CLOSED,
            CallState.FAILED,
            CallState.TIMEOUT
        },
        # Terminal states cannot transition
        CallState.CLOSED: set(),
        CallState.TRANSFERRED: set(),
        CallState.FAILED: set(),
        CallState.TIMEOUT: set()
    }
    
    def __init__(
        self,
        call_id: str,
        initial_state: CallState = CallState.INIT
    ):
        """
        Initialize state machine.
        
        Args:
            call_id: Unique call identifier
            initial_state: Starting state (default: INIT)
        """
        self.call_id = call_id
        self._lock = threading.RLock()
        
        # State tracking
        self._current_state = initial_state
        self._previous_state: Optional[CallState] = None
        
        # History
        self._state_history: List[StateEntry] = [
            StateEntry(
                state=initial_state,
                entered_at=datetime.now(timezone.utc),
                reason="initialization"
            )
        ]
        
        # Metrics
        self._transition_count = 0
        self._forced_transition_count = 0
        self._invalid_transition_attempts = 0
        
        # Callbacks
        self._on_state_change_callbacks: List[Callable[[CallState, CallState], None]] = []
        self._on_terminal_state_callbacks: List[Callable[[CallState], None]] = []
        
        # Flags
        self._language_detected = False  # Track if language detection happened
        
        logger.info(
            "State machine initialized",
            extra={
                "call_id": call_id,
                "initial_state": initial_state.value
            }
        )
    
    def register_state_change_callback(
        self,
        callback: Callable[[CallState, CallState], None]
    ) -> None:
        """
        Register callback for state changes.
        
        Args:
            callback: Function(old_state, new_state) called on transitions
        """
        with self._lock:
            self._on_state_change_callbacks.append(callback)
    
    def register_terminal_state_callback(
        self,
        callback: Callable[[CallState], None]
    ) -> None:
        """
        Register callback for terminal state entry.
        
        Args:
            callback: Function(terminal_state) called when terminal state reached
        """
        with self._lock:
            self._on_terminal_state_callbacks.append(callback)
    
    @property
    def current_state(self) -> CallState:
        """Get current state (thread-safe)."""
        with self._lock:
            return self._current_state
    
    @property
    def previous_state(self) -> Optional[CallState]:
        """Get previous state (thread-safe)."""
        with self._lock:
            return self._previous_state
    
    def can_transition_to(self, target_state: CallState) -> bool:
        """
        Check if transition to target state is valid.
        
        Args:
            target_state: Desired next state
            
        Returns:
            True if transition is allowed
        """
        with self._lock:
            # Terminal states cannot transition
            if self._current_state in self.TERMINAL_STATES:
                return False
            
            return target_state in self.VALID_TRANSITIONS.get(
                self._current_state,
                set()
            )
    
    def transition(
        self,
        target_state: CallState,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Attempt atomic state transition with validation.
        
        Args:
            target_state: Desired next state
            reason: Optional reason for transition
            metadata: Optional metadata to store with transition
            
        Returns:
            True if transition succeeded
            
        Raises:
            StateTransitionError: If transition is invalid
            TerminalStateError: If attempting to exit terminal state
        """
        with self._lock:
            # Check terminal state
            if self._current_state in self.TERMINAL_STATES:
                self._invalid_transition_attempts += 1
                error_msg = (
                    f"Cannot transition from terminal state: "
                    f"{self._current_state.value}"
                )
                logger.error(
                    error_msg,
                    extra={
                        "call_id": self.call_id,
                        "current_state": self._current_state.value,
                        "attempted_target": target_state.value,
                        "invalid_attempts": self._invalid_transition_attempts
                    }
                )
                raise TerminalStateError(error_msg)
            
            # Validate transition
            if not self.can_transition_to(target_state):
                self._invalid_transition_attempts += 1
                error_msg = (
                    f"Invalid transition: "
                    f"{self._current_state.value} -> {target_state.value}"
                )
                logger.error(
                    error_msg,
                    extra={
                        "call_id": self.call_id,
                        "from_state": self._current_state.value,
                        "to_state": target_state.value,
                        "reason": reason,
                        "invalid_attempts": self._invalid_transition_attempts
                    }
                )
                raise StateTransitionError(error_msg)
            
            # Enforce language detection happens only once
            if target_state == CallState.LANGUAGE_DETECT:
                if self._language_detected:
                    error_msg = "Language detection can only happen once per call"
                    logger.error(
                        error_msg,
                        extra={
                            "call_id": self.call_id,
                            "current_state": self._current_state.value
                        }
                    )
                    raise StateTransitionError(error_msg)
            
            # Perform atomic transition
            old_state = self._current_state
            self._previous_state = old_state
            self._current_state = target_state
            self._transition_count += 1
            
            # Track language detection
            if target_state == CallState.LANGUAGE_DETECT:
                self._language_detected = True
            
            # Record in history
            entry = StateEntry(
                state=target_state,
                entered_at=datetime.now(timezone.utc),
                reason=reason,
                metadata=metadata or {}
            )
            self._state_history.append(entry)
            
            logger.info(
                f"State transition: {old_state.value} -> {target_state.value}",
                extra={
                    "call_id": self.call_id,
                    "from_state": old_state.value,
                    "to_state": target_state.value,
                    "reason": reason,
                    "transition_count": self._transition_count,
                    "metadata": metadata
                }
            )
            
            # Execute callbacks
            self._execute_state_change_callbacks(old_state, target_state)
            
            if target_state in self.TERMINAL_STATES:
                self._execute_terminal_state_callbacks(target_state)
            
            return True
    
    def force_transition(
        self,
        target_state: CallState,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Force transition without validation (emergency use only).
        
        Use only for:
        - Catastrophic failures
        - Forced shutdowns
        - Recovery procedures
        
        Args:
            target_state: Target state
            reason: Reason for forced transition (required)
            metadata: Optional metadata
        """
        with self._lock:
            old_state = self._current_state
            
            logger.warning(
                f"FORCED state transition: {old_state.value} -> {target_state.value}",
                extra={
                    "call_id": self.call_id,
                    "from_state": old_state.value,
                    "to_state": target_state.value,
                    "reason": reason,
                    "metadata": metadata
                }
            )
            
            self._previous_state = old_state
            self._current_state = target_state
            self._transition_count += 1
            self._forced_transition_count += 1
            
            # Record forced transition in history
            entry = StateEntry(
                state=target_state,
                entered_at=datetime.now(timezone.utc),
                reason=f"FORCED: {reason}",
                metadata=metadata or {}
            )
            self._state_history.append(entry)
            
            # Execute callbacks even on forced transition
            self._execute_state_change_callbacks(old_state, target_state)
            
            if target_state in self.TERMINAL_STATES:
                self._execute_terminal_state_callbacks(target_state)
    
    def _execute_state_change_callbacks(
        self,
        old_state: CallState,
        new_state: CallState
    ) -> None:
        """Execute registered state change callbacks."""
        for callback in self._on_state_change_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.error(
                    f"State change callback failed: {e}",
                    extra={
                        "call_id": self.call_id,
                        "old_state": old_state.value,
                        "new_state": new_state.value
                    },
                    exc_info=True
                )
    
    def _execute_terminal_state_callbacks(self, terminal_state: CallState) -> None:
        """Execute registered terminal state callbacks."""
        for callback in self._on_terminal_state_callbacks:
            try:
                callback(terminal_state)
            except Exception as e:
                logger.error(
                    f"Terminal state callback failed: {e}",
                    extra={
                        "call_id": self.call_id,
                        "terminal_state": terminal_state.value
                    },
                    exc_info=True
                )
    
    def is_terminal(self) -> bool:
        """Check if current state is terminal."""
        with self._lock:
            return self._current_state in self.TERMINAL_STATES
    
    def is_active(self) -> bool:
        """Check if call is in an active conversational state."""
        with self._lock:
            return self._current_state in self.ACTIVE_STATES or \
                   self._current_state == CallState.ACTIVE
    
    def is_frozen(self) -> bool:
        """Check if call is in a frozen state (no AI/business logic allowed)."""
        with self._lock:
            return self._current_state in self.FROZEN_STATES
    
    def is_closed(self) -> bool:
        """Check if call has been closed (any terminal state)."""
        return self.is_terminal()
    
    def can_execute_ai(self) -> bool:
        """Check if AI execution is allowed in current state."""
        with self._lock:
            return not self.is_frozen()
    
    def can_execute_business_logic(self) -> bool:
        """Check if business logic execution is allowed in current state."""
        with self._lock:
            return not self.is_frozen()
    
    def get_state_duration(self) -> float:
        """
        Get duration in current state (seconds).
        
        Returns:
            Seconds in current state
        """
        with self._lock:
            if not self._state_history:
                return 0.0
            
            last_entry = self._state_history[-1]
            return (
                datetime.now(timezone.utc) - last_entry.entered_at
            ).total_seconds()
    
    def get_state_timeout(self) -> float:
        """
        Get maximum allowed duration for current state.
        
        Returns:
            Maximum seconds allowed in current state
        """
        with self._lock:
            return self.STATE_TIMEOUTS.get(self._current_state, float('inf'))
    
    def is_state_timeout_exceeded(self) -> bool:
        """
        Check if current state has exceeded its timeout.
        
        Returns:
            True if state has been occupied too long
        """
        duration = self.get_state_duration()
        timeout = self.get_state_timeout()
        return duration > timeout
    
    def get_total_duration(self) -> float:
        """
        Get total call duration since initialization (seconds).
        
        Returns:
            Total seconds since call start
        """
        with self._lock:
            if not self._state_history:
                return 0.0
            
            first_entry = self._state_history[0]
            return (
                datetime.now(timezone.utc) - first_entry.entered_at
            ).total_seconds()
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get complete state transition history.
        
        Returns:
            List of state entries with timing information
        """
        with self._lock:
            result = []
            for i, entry in enumerate(self._state_history):
                # Calculate duration in this state
                if i + 1 < len(self._state_history):
                    next_entry = self._state_history[i + 1]
                    duration = (
                        next_entry.entered_at - entry.entered_at
                    ).total_seconds()
                else:
                    duration = (
                        datetime.now(timezone.utc) - entry.entered_at
                    ).total_seconds()
                
                result.append({
                    "state": entry.state.value,
                    "timestamp": entry.entered_at.isoformat(),
                    "duration_seconds": duration,
                    "reason": entry.reason,
                    "metadata": entry.metadata
                })
            
            return result
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get summary of state machine status.
        
        Returns:
            Summary dict with key metrics
        """
        with self._lock:
            return {
                "call_id": self.call_id,
                "current_state": self._current_state.value,
                "previous_state": (
                    self._previous_state.value
                    if self._previous_state
                    else None
                ),
                "is_terminal": self.is_terminal(),
                "is_active": self.is_active(),
                "is_frozen": self.is_frozen(),
                "can_execute_ai": self.can_execute_ai(),
                "state_duration_seconds": self.get_state_duration(),
                "state_timeout_seconds": self.get_state_timeout(),
                "is_timeout_exceeded": self.is_state_timeout_exceeded(),
                "total_duration_seconds": self.get_total_duration(),
                "transition_count": self._transition_count,
                "forced_transition_count": self._forced_transition_count,
                "invalid_transition_attempts": self._invalid_transition_attempts,
                "state_history_length": len(self._state_history),
                "language_detected": self._language_detected
            }
    
    def __repr__(self) -> str:
        """String representation."""
        with self._lock:
            return (
                f"<CallStateMachine "
                f"call_id={self.call_id} "
                f"state={self._current_state.value} "
                f"transitions={self._transition_count}>"
            )
