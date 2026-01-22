"""
Watchdog System
===============
Production-grade timeout enforcement and safety monitoring.

Responsibilities:
- Enforce hard timeouts on operations
- Monitor call health metrics
- Detect anomalies and trigger safety actions
- Prevent runaway processes
- Kill switch functionality
"""

import asyncio
import logging
from typing import Optional, Callable, Dict, Any, Set
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class WatchdogViolation(Enum):
    """Types of watchdog violations."""
    MAX_CALL_DURATION = "max_call_duration"
    SILENCE_TIMEOUT = "silence_timeout"
    AI_TIMEOUT = "ai_timeout"
    TRANSFER_TIMEOUT = "transfer_timeout"
    EXCESSIVE_ERRORS = "excessive_errors"
    MEMORY_LIMIT = "memory_limit"
    TURN_LIMIT = "turn_limit"
    GREETING_TIMEOUT = "greeting_timeout"
    STUCK_STATE = "stuck_state"


@dataclass
class WatchdogLimits:
    """Configurable limits for watchdog monitoring."""
    
    # Time limits (seconds)
    max_call_duration: float = 1800.0  # 30 minutes
    max_silence_duration: float = 30.0
    max_ai_response_time: float = 10.0
    max_greeting_time: float = 5.0
    max_transfer_time: float = 30.0
    max_state_duration: float = 120.0  # Max time in single state
    
    # Count limits
    max_consecutive_errors: int = 5
    max_total_errors: int = 20
    max_turns: int = 1000
    
    # Resource limits
    max_memory_mb: int = 512
    
    # Health check
    health_check_interval: float = 5.0


class WatchdogAction(Enum):
    """Actions watchdog can take on violation."""
    LOG_WARNING = "log_warning"
    TRANSFER_TO_HUMAN = "transfer_to_human"
    END_CALL = "end_call"
    KILL_CALL = "kill_call"


class Watchdog:
    """
    Monitors call health and enforces safety limits.
    
    Features:
    - Async timeout monitoring
    - Configurable violation handlers
    - Automatic recovery actions
    - Health metrics tracking
    """
    
    def __init__(
        self,
        call_id: str,
        limits: Optional[WatchdogLimits] = None,
        violation_handler: Optional[Callable] = None
    ):
        """
        Initialize watchdog.
        
        Args:
            call_id: Call identifier
            limits: Watchdog limits configuration
            violation_handler: Callback for violations
        """
        self.call_id = call_id
        self.limits = limits or WatchdogLimits()
        self.violation_handler = violation_handler
        
        # Monitoring state
        self._active = False
        self._shutdown_event = asyncio.Event()
        
        # Timers
        self._call_start_time: Optional[datetime] = None
        self._last_user_activity: Optional[datetime] = None
        self._last_ai_activity: Optional[datetime] = None
        self._greeting_start_time: Optional[datetime] = None
        self._transfer_start_time: Optional[datetime] = None
        self._state_enter_time: Optional[datetime] = None
        
        # Counters
        self._consecutive_errors = 0
        self._total_errors = 0
        self._turn_count = 0
        
        # Violations
        self._violations: Set[WatchdogViolation] = set()
        
        # Background task
        self._monitor_task: Optional[asyncio.Task] = None
        
        logger.info(
            "Watchdog initialized",
            extra={"call_id": call_id}
        )
    
    def start(self) -> None:
        """Start watchdog monitoring."""
        if self._active:
            return
        
        self._active = True
        self._call_start_time = datetime.now(timezone.utc)
        self._last_user_activity = self._call_start_time
        self._last_ai_activity = self._call_start_time
        
        # Start background monitoring
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(),
            name=f"watchdog_{self.call_id}"
        )
        
        logger.info(
            "Watchdog started",
            extra={"call_id": self.call_id}
        )
    
    async def stop(self) -> None:
        """Stop watchdog monitoring."""
        if not self._active:
            return
        
        self._active = False
        self._shutdown_event.set()
        
        # Wait for monitor task to finish
        if self._monitor_task and not self._monitor_task.done():
            try:
                await asyncio.wait_for(self._monitor_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
        
        logger.info(
            "Watchdog stopped",
            extra={
                "call_id": self.call_id,
                "violations": [v.value for v in self._violations]
            }
        )
    
    def mark_user_activity(self) -> None:
        """Mark that user spoke/interacted."""
        self._last_user_activity = datetime.now(timezone.utc)
        logger.debug(
            "User activity marked",
            extra={"call_id": self.call_id}
        )
    
    def mark_ai_activity(self) -> None:
        """Mark that AI responded."""
        self._last_ai_activity = datetime.now(timezone.utc)
        # Reset consecutive errors on successful AI response
        self._consecutive_errors = 0
        logger.debug(
            "AI activity marked",
            extra={"call_id": self.call_id}
        )
    
    def mark_greeting_start(self) -> None:
        """Mark start of greeting."""
        self._greeting_start_time = datetime.now(timezone.utc)
    
    def mark_greeting_end(self) -> None:
        """Mark end of greeting."""
        self._greeting_start_time = None
    
    def mark_transfer_start(self) -> None:
        """Mark start of call transfer."""
        self._transfer_start_time = datetime.now(timezone.utc)
    
    def mark_transfer_end(self) -> None:
        """Mark end of call transfer."""
        self._transfer_start_time = None
    
    def mark_state_change(self) -> None:
        """Mark that state changed."""
        self._state_enter_time = datetime.now(timezone.utc)
    
    def mark_turn(self) -> None:
        """Mark conversation turn."""
        self._turn_count += 1
    
    def mark_error(self) -> None:
        """Mark that an error occurred."""
        self._consecutive_errors += 1
        self._total_errors += 1
        logger.debug(
            "Error marked",
            extra={
                "call_id": self.call_id,
                "consecutive": self._consecutive_errors,
                "total": self._total_errors
            }
        )
    
    def reset_error_count(self) -> None:
        """Reset consecutive error counter."""
        self._consecutive_errors = 0
    
    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        try:
            while self._active and not self._shutdown_event.is_set():
                # Check all timeouts and limits
                await self._check_violations()
                
                # Wait before next check
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.limits.health_check_interval
                    )
                    break  # Shutdown signaled
                except asyncio.TimeoutError:
                    continue  # Normal timeout, continue monitoring
        
        except Exception as e:
            logger.error(
                f"Watchdog monitor loop error: {e}",
                extra={"call_id": self.call_id},
                exc_info=True
            )
    
    async def _check_violations(self) -> None:
        """Check for all types of violations."""
        now = datetime.now(timezone.utc)
        
        # Check call duration
        if self._call_start_time:
            duration = (now - self._call_start_time).total_seconds()
            if duration > self.limits.max_call_duration:
                await self._handle_violation(
                    WatchdogViolation.MAX_CALL_DURATION,
                    WatchdogAction.END_CALL,
                    {"duration": duration}
                )
        
        # Check silence timeout
        if self._last_user_activity:
            silence = (now - self._last_user_activity).total_seconds()
            if silence > self.limits.max_silence_duration:
                await self._handle_violation(
                    WatchdogViolation.SILENCE_TIMEOUT,
                    WatchdogAction.END_CALL,
                    {"silence_duration": silence}
                )
        
        # Check AI timeout
        if self._last_ai_activity:
            ai_delay = (now - self._last_ai_activity).total_seconds()
            if ai_delay > self.limits.max_ai_response_time:
                await self._handle_violation(
                    WatchdogViolation.AI_TIMEOUT,
                    WatchdogAction.TRANSFER_TO_HUMAN,
                    {"ai_delay": ai_delay}
                )
        
        # Check greeting timeout
        if self._greeting_start_time:
            greeting_duration = (now - self._greeting_start_time).total_seconds()
            if greeting_duration > self.limits.max_greeting_time:
                await self._handle_violation(
                    WatchdogViolation.GREETING_TIMEOUT,
                    WatchdogAction.KILL_CALL,
                    {"greeting_duration": greeting_duration}
                )
        
        # Check transfer timeout
        if self._transfer_start_time:
            transfer_duration = (now - self._transfer_start_time).total_seconds()
            if transfer_duration > self.limits.max_transfer_time:
                await self._handle_violation(
                    WatchdogViolation.TRANSFER_TIMEOUT,
                    WatchdogAction.KILL_CALL,
                    {"transfer_duration": transfer_duration}
                )
        
        # Check stuck state
        if self._state_enter_time:
            state_duration = (now - self._state_enter_time).total_seconds()
            if state_duration > self.limits.max_state_duration:
                await self._handle_violation(
                    WatchdogViolation.STUCK_STATE,
                    WatchdogAction.TRANSFER_TO_HUMAN,
                    {"state_duration": state_duration}
                )
        
        # Check error counts
        if self._consecutive_errors >= self.limits.max_consecutive_errors:
            await self._handle_violation(
                WatchdogViolation.EXCESSIVE_ERRORS,
                WatchdogAction.TRANSFER_TO_HUMAN,
                {
                    "consecutive_errors": self._consecutive_errors,
                    "total_errors": self._total_errors
                }
            )
        
        if self._total_errors >= self.limits.max_total_errors:
            await self._handle_violation(
                WatchdogViolation.EXCESSIVE_ERRORS,
                WatchdogAction.KILL_CALL,
                {
                    "consecutive_errors": self._consecutive_errors,
                    "total_errors": self._total_errors
                }
            )
        
        # Check turn limit
        if self._turn_count >= self.limits.max_turns:
            await self._handle_violation(
                WatchdogViolation.TURN_LIMIT,
                WatchdogAction.END_CALL,
                {"turn_count": self._turn_count}
            )
    
    async def _handle_violation(
        self,
        violation: WatchdogViolation,
        action: WatchdogAction,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Handle a watchdog violation.
        
        Args:
            violation: Type of violation
            action: Recommended action
            metadata: Additional violation context
        """
        # Only handle each violation once
        if violation in self._violations:
            return
        
        self._violations.add(violation)
        
        logger.warning(
            f"Watchdog violation: {violation.value}",
            extra={
                "call_id": self.call_id,
                "violation": violation.value,
                "action": action.value,
                "metadata": metadata
            }
        )
        
        # Call violation handler if provided
        if self.violation_handler:
            try:
                if asyncio.iscoroutinefunction(self.violation_handler):
                    await self.violation_handler(
                        violation=violation,
                        action=action,
                        metadata=metadata
                    )
                else:
                    self.violation_handler(
                        violation=violation,
                        action=action,
                        metadata=metadata
                    )
            except Exception as e:
                logger.error(
                    f"Error in violation handler: {e}",
                    extra={"call_id": self.call_id},
                    exc_info=True
                )
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current health status.
        
        Returns:
            Health status dict
        """
        now = datetime.now(timezone.utc)
        
        result = {
            "call_id": self.call_id,
            "active": self._active,
            "violations": [v.value for v in self._violations],
            "metrics": {}
        }
        
        if self._call_start_time:
            result["metrics"]["call_duration"] = (
                now - self._call_start_time
            ).total_seconds()
        
        if self._last_user_activity:
            result["metrics"]["silence_duration"] = (
                now - self._last_user_activity
            ).total_seconds()
        
        if self._last_ai_activity:
            result["metrics"]["ai_idle_duration"] = (
                now - self._last_ai_activity
            ).total_seconds()
        
        if self._state_enter_time:
            result["metrics"]["state_duration"] = (
                now - self._state_enter_time
            ).total_seconds()
        
        result["metrics"]["consecutive_errors"] = self._consecutive_errors
        result["metrics"]["total_errors"] = self._total_errors
        result["metrics"]["turn_count"] = self._turn_count
        
        return result
    
    def has_violations(self) -> bool:
        """Check if any violations have occurred."""
        return len(self._violations) > 0
    
    def get_violations(self) -> Set[WatchdogViolation]:
        """Get set of all violations."""
        return self._violations.copy()
    
    async def __aenter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.stop()
        return False
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<Watchdog "
            f"call_id={self.call_id} "
            f"active={self._active} "
            f"violations={len(self._violations)}>"
        )


class TimeoutContext:
    """
    Async context manager for enforcing operation timeouts.
    
    Usage:
        async with TimeoutContext(10.0, "AI response"):
            result = await ai_call()
    """
    
    def __init__(
        self,
        timeout: float,
        operation_name: str,
        call_id: Optional[str] = None,
        on_timeout: Optional[Callable] = None
    ):
        """
        Initialize timeout context.
        
        Args:
            timeout: Timeout in seconds
            operation_name: Name of operation for logging
            call_id: Optional call ID for logging
            on_timeout: Optional callback on timeout
        """
        self.timeout = timeout
        self.operation_name = operation_name
        self.call_id = call_id
        self.on_timeout = on_timeout
        self._start_time: Optional[datetime] = None
    
    async def __aenter__(self):
        """Enter timeout context."""
        self._start_time = datetime.now(timezone.utc)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit timeout context."""
        if exc_type is asyncio.TimeoutError:
            duration = (
                datetime.now(timezone.utc) - self._start_time
            ).total_seconds()
            
            logger.warning(
                f"Operation timeout: {self.operation_name}",
                extra={
                    "operation": self.operation_name,
                    "timeout": self.timeout,
                    "duration": duration,
                    "call_id": self.call_id
                }
            )
            
            # Call timeout handler
            if self.on_timeout:
                try:
                    if asyncio.iscoroutinefunction(self.on_timeout):
                        await self.on_timeout()
                    else:
                        self.on_timeout()
                except Exception as e:
                    logger.error(
                        f"Error in timeout handler: {e}",
                        exc_info=True
                    )
        
        return False  # Don't suppress exceptions


async def with_timeout(
    coro,
    timeout: float,
    operation_name: str,
    call_id: Optional[str] = None,
    on_timeout: Optional[Callable] = None
) -> Any:
    """
    Execute coroutine with timeout.
    
    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        operation_name: Operation name for logging
        call_id: Optional call ID
        on_timeout: Optional timeout callback
        
    Returns:
        Coroutine result
        
    Raises:
        asyncio.TimeoutError: On timeout
    """
    async with TimeoutContext(timeout, operation_name, call_id, on_timeout):
        return await asyncio.wait_for(coro, timeout=timeout)
