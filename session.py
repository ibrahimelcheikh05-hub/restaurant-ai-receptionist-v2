"""
Call Session
============
Per-call isolation container with resource management.

Responsibilities:
- Own all resources for a single call
- Provide isolated execution context
- Manage lifecycle of call-scoped objects
- Track call metadata and telemetry
- Ensure proper cleanup on termination
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Set
from datetime import datetime, timezone
from dataclasses import dataclass, field
import uuid

logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """Configuration for call session."""
    
    # Timeouts (seconds)
    max_call_duration: float = 1800.0  # 30 minutes
    max_silence_duration: float = 30.0
    max_ai_response_time: float = 10.0
    greeting_timeout: float = 5.0
    transfer_timeout: float = 30.0
    
    # Resource limits
    max_memory_mb: int = 512
    max_conversation_turns: int = 1000
    max_order_items: int = 50
    
    # Feature flags
    enable_barge_in: bool = True
    enable_language_detection: bool = True
    enable_transfer: bool = True
    enable_upsell: bool = True
    enable_memory: bool = True
    
    # Observability
    enable_latency_tracking: bool = True
    enable_detailed_logging: bool = True
    log_transcripts: bool = True


@dataclass
class SessionMetadata:
    """Metadata tracked throughout call session."""
    
    # Identity
    call_id: str
    session_id: str
    tenant_id: str
    customer_phone: Optional[str] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    
    # Language
    detected_language: Optional[str] = None
    language_confidence: Optional[float] = None
    
    # Telephony
    twilio_call_sid: Optional[str] = None
    from_number: Optional[str] = None
    to_number: Optional[str] = None
    
    # Business context
    restaurant_id: Optional[str] = None
    order_id: Optional[str] = None
    
    # Metrics
    turn_count: int = 0
    transcript_word_count: int = 0
    ai_token_count: int = 0
    
    # Outcomes
    transfer_occurred: bool = False
    transfer_reason: Optional[str] = None
    order_created: bool = False
    sms_sent: bool = False
    
    # Error tracking
    error_count: int = 0
    last_error: Optional[str] = None


class CallSession:
    """
    Isolated execution context for a single call.
    
    Owns:
    - Configuration
    - Metadata
    - Async tasks
    - Cancellation events
    - Resource cleanup
    
    Guarantees:
    - All resources are properly cleaned up
    - No resource leaks across calls
    - Isolated state per call
    """
    
    def __init__(
        self,
        call_id: str,
        tenant_id: str,
        config: Optional[SessionConfig] = None,
        **metadata_kwargs
    ):
        """
        Initialize call session.
        
        Args:
            call_id: Unique call identifier
            tenant_id: Tenant/restaurant identifier
            config: Session configuration
            **metadata_kwargs: Additional metadata fields
        """
        self.call_id = call_id
        self.tenant_id = tenant_id
        self.session_id = str(uuid.uuid4())
        
        # Configuration
        self.config = config or SessionConfig()
        
        # Metadata
        self.metadata = SessionMetadata(
            call_id=call_id,
            session_id=self.session_id,
            tenant_id=tenant_id,
            **metadata_kwargs
        )
        
        # Lifecycle tracking
        self._started = False
        self._ended = False
        self._cleanup_done = False
        
        # Async task tracking
        self._tasks: Set[asyncio.Task] = set()
        self._task_lock = asyncio.Lock()
        
        # Cancellation
        self._shutdown_event = asyncio.Event()
        self._shutdown_reason: Optional[str] = None
        
        # Resource tracking
        self._resources: Dict[str, Any] = {}
        
        logger.info(
            "Call session created",
            extra={
                "call_id": call_id,
                "session_id": self.session_id,
                "tenant_id": tenant_id
            }
        )
    
    def mark_started(self) -> None:
        """Mark session as started."""
        if not self._started:
            self._started = True
            self.metadata.started_at = datetime.now(timezone.utc)
            logger.info(
                "Call session started",
                extra={"call_id": self.call_id, "session_id": self.session_id}
            )
    
    def mark_ended(self, reason: Optional[str] = None) -> None:
        """Mark session as ended."""
        if not self._ended:
            self._ended = True
            self.metadata.ended_at = datetime.now(timezone.utc)
            logger.info(
                "Call session ended",
                extra={
                    "call_id": self.call_id,
                    "session_id": self.session_id,
                    "reason": reason,
                    "duration_seconds": self.get_duration()
                }
            )
    
    def is_started(self) -> bool:
        """Check if session has started."""
        return self._started
    
    def is_ended(self) -> bool:
        """Check if session has ended."""
        return self._ended
    
    def is_active(self) -> bool:
        """Check if session is currently active."""
        return self._started and not self._ended
    
    def get_duration(self) -> Optional[float]:
        """
        Get session duration in seconds.
        
        Returns:
            Duration in seconds, or None if not started
        """
        if not self.metadata.started_at:
            return None
        
        end_time = self.metadata.ended_at or datetime.now(timezone.utc)
        return (end_time - self.metadata.started_at).total_seconds()
    
    def should_timeout(self) -> bool:
        """
        Check if session should timeout due to max duration.
        
        Returns:
            True if session has exceeded max duration
        """
        duration = self.get_duration()
        if duration is None:
            return False
        
        return duration >= self.config.max_call_duration
    
    def register_task(self, task: asyncio.Task) -> None:
        """
        Register async task for tracking and cleanup.
        
        Args:
            task: Async task to track
        """
        self._tasks.add(task)
        
        # Auto-remove when done
        def remove_task(t):
            self._tasks.discard(t)
        
        task.add_done_callback(remove_task)
        
        logger.debug(
            "Task registered",
            extra={
                "call_id": self.call_id,
                "task_count": len(self._tasks)
            }
        )
    
    def create_task(
        self,
        coro,
        name: Optional[str] = None
    ) -> asyncio.Task:
        """
        Create and register an async task.
        
        Args:
            coro: Coroutine to run
            name: Optional task name
            
        Returns:
            Created task
        """
        task = asyncio.create_task(coro, name=name)
        self.register_task(task)
        return task
    
    async def cancel_all_tasks(
        self,
        reason: str = "session_shutdown"
    ) -> None:
        """
        Cancel all registered tasks.
        
        Args:
            reason: Reason for cancellation
        """
        logger.info(
            f"Cancelling all tasks: {reason}",
            extra={
                "call_id": self.call_id,
                "task_count": len(self._tasks)
            }
        )
        
        # Cancel all tasks
        for task in list(self._tasks):
            if not task.done():
                task.cancel()
        
        # Wait for cancellation with timeout
        if self._tasks:
            try:
                await asyncio.wait(
                    self._tasks,
                    timeout=5.0,
                    return_when=asyncio.ALL_COMPLETED
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Some tasks did not cancel within timeout",
                    extra={"call_id": self.call_id}
                )
    
    def register_resource(
        self,
        name: str,
        resource: Any,
        cleanup_callback: Optional[callable] = None
    ) -> None:
        """
        Register a resource for lifecycle management.
        
        Args:
            name: Resource name/key
            resource: Resource object
            cleanup_callback: Optional cleanup function
        """
        self._resources[name] = {
            "resource": resource,
            "cleanup": cleanup_callback,
            "registered_at": datetime.now(timezone.utc)
        }
        
        logger.debug(
            f"Resource registered: {name}",
            extra={"call_id": self.call_id}
        )
    
    def get_resource(self, name: str) -> Optional[Any]:
        """
        Get registered resource by name.
        
        Args:
            name: Resource name
            
        Returns:
            Resource object or None
        """
        entry = self._resources.get(name)
        return entry["resource"] if entry else None
    
    def signal_shutdown(self, reason: str) -> None:
        """
        Signal that session should shut down.
        
        Args:
            reason: Reason for shutdown
        """
        if not self._shutdown_event.is_set():
            self._shutdown_reason = reason
            self._shutdown_event.set()
            logger.info(
                f"Shutdown signaled: {reason}",
                extra={"call_id": self.call_id}
            )
    
    async def wait_for_shutdown(self) -> str:
        """
        Wait for shutdown signal.
        
        Returns:
            Shutdown reason
        """
        await self._shutdown_event.wait()
        return self._shutdown_reason or "unknown"
    
    def is_shutdown_signaled(self) -> bool:
        """Check if shutdown has been signaled."""
        return self._shutdown_event.is_set()
    
    async def cleanup(self) -> None:
        """
        Cleanup all session resources.
        
        Idempotent - safe to call multiple times.
        """
        if self._cleanup_done:
            return
        
        logger.info(
            "Starting session cleanup",
            extra={
                "call_id": self.call_id,
                "resource_count": len(self._resources)
            }
        )
        
        # Cancel tasks
        await self.cancel_all_tasks(reason="cleanup")
        
        # Cleanup registered resources
        for name, entry in list(self._resources.items()):
            try:
                cleanup_fn = entry.get("cleanup")
                if cleanup_fn:
                    if asyncio.iscoroutinefunction(cleanup_fn):
                        await cleanup_fn(entry["resource"])
                    else:
                        cleanup_fn(entry["resource"])
                
                logger.debug(
                    f"Resource cleaned up: {name}",
                    extra={"call_id": self.call_id}
                )
            except Exception as e:
                logger.error(
                    f"Error cleaning up resource {name}: {e}",
                    extra={"call_id": self.call_id},
                    exc_info=True
                )
        
        self._resources.clear()
        self._cleanup_done = True
        
        logger.info(
            "Session cleanup completed",
            extra={"call_id": self.call_id}
        )
    
    def increment_turn(self) -> None:
        """Increment conversation turn counter."""
        self.metadata.turn_count += 1
    
    def increment_error(self, error_message: str) -> None:
        """
        Increment error counter.
        
        Args:
            error_message: Error message
        """
        self.metadata.error_count += 1
        self.metadata.last_error = error_message
    
    def set_language(
        self,
        language_code: str,
        confidence: Optional[float] = None
    ) -> None:
        """
        Set detected language.
        
        Args:
            language_code: ISO language code (e.g., 'en', 'es')
            confidence: Detection confidence (0-1)
        """
        self.metadata.detected_language = language_code
        self.metadata.language_confidence = confidence
        logger.info(
            f"Language set: {language_code}",
            extra={
                "call_id": self.call_id,
                "confidence": confidence
            }
        )
    
    def mark_order_created(self, order_id: str) -> None:
        """
        Mark that order was created.
        
        Args:
            order_id: Created order ID
        """
        self.metadata.order_created = True
        self.metadata.order_id = order_id
    
    def mark_transfer(self, reason: str) -> None:
        """
        Mark that call was transferred.
        
        Args:
            reason: Transfer reason
        """
        self.metadata.transfer_occurred = True
        self.metadata.transfer_reason = reason
    
    def mark_sms_sent(self) -> None:
        """Mark that SMS was sent."""
        self.metadata.sms_sent = True
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get session summary.
        
        Returns:
            Summary dict with key information
        """
        return {
            "call_id": self.call_id,
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "started": self._started,
            "ended": self._ended,
            "duration_seconds": self.get_duration(),
            "active_tasks": len(self._tasks),
            "resources": len(self._resources),
            "shutdown_signaled": self.is_shutdown_signaled(),
            "shutdown_reason": self._shutdown_reason,
            "metadata": {
                "detected_language": self.metadata.detected_language,
                "turn_count": self.metadata.turn_count,
                "order_created": self.metadata.order_created,
                "transfer_occurred": self.metadata.transfer_occurred,
                "error_count": self.metadata.error_count
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.mark_started()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        if exc_type:
            logger.error(
                f"Session exiting with exception: {exc_type.__name__}",
                extra={"call_id": self.call_id},
                exc_info=(exc_type, exc_val, exc_tb)
            )
        
        self.mark_ended(
            reason=f"exception: {exc_type.__name__}" if exc_type else "normal"
        )
        await self.cleanup()
        return False  # Don't suppress exceptions
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<CallSession "
            f"call_id={self.call_id} "
            f"session_id={self.session_id} "
            f"active={self.is_active()} "
            f"tasks={len(self._tasks)}>"
        )
