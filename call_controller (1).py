"""
Call Controller
===============
Central orchestrator for single call lifecycle.

This is the BRAIN of each call.

Responsibilities:
- Own exactly one Vocode StreamingConversation
- Manage CallState machine
- Supervise watchdog
- Route all events to business/AI layers
- Control language translation flow
- Approve all transfers
- Enforce all safety boundaries
- Own all cancellation tokens

This does NOT:
- Embed business logic
- Call databases directly
- Implement voice pipeline internals
- Make AI prompt decisions
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime, timezone
from enum import Enum
import uuid

from core.call_state import CallState, CallStateMachine, StateTransitionError
from core.session import CallSession, SessionConfig
from core.watchdog import Watchdog, WatchdogLimits, WatchdogViolation, WatchdogAction

logger = logging.getLogger(__name__)


class CallEvent(Enum):
    """Types of call events."""
    STARTED = "started"
    USER_SPOKE = "user_spoke"
    AI_RESPONDING = "ai_responding"
    AI_SPOKE = "ai_spoke"
    TRANSFER_REQUESTED = "transfer_requested"
    TRANSFER_APPROVED = "transfer_approved"
    TRANSFER_DENIED = "transfer_denied"
    LANGUAGE_DETECTED = "language_detected"
    ERROR = "error"
    CLOSING = "closing"
    CLOSED = "closed"


class CallController:
    """
    Per-call controller - orchestrates single call from start to finish.
    
    Architecture:
    - Wraps ONE Vocode StreamingConversation (voice engine)
    - Owns ONE CallStateMachine (formal state)
    - Owns ONE CallSession (resource isolation)
    - Owns ONE Watchdog (safety monitoring)
    - Routes events to injected handlers (business logic)
    
    Control flow:
    1. Vocode handles: audio, telephony, turn-taking, barge-in
    2. Controller handles: state, AI routing, business rules, safety
    3. Handlers handle: AI calls, menu logic, orders, database
    
    AI NEVER controls system flow.
    Controller makes all state decisions.
    """
    
    def __init__(
        self,
        call_id: str,
        tenant_id: str,
        session_config: Optional[SessionConfig] = None,
        watchdog_limits: Optional[WatchdogLimits] = None,
        handlers: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize call controller.
        
        Args:
            call_id: Unique call identifier
            tenant_id: Tenant/restaurant identifier
            session_config: Session configuration
            watchdog_limits: Watchdog limits
            handlers: Event handler callbacks
        """
        self.call_id = call_id
        self.tenant_id = tenant_id
        self.request_id = str(uuid.uuid4())
        
        # Core components
        self.session = CallSession(
            call_id=call_id,
            tenant_id=tenant_id,
            config=session_config
        )
        
        self.state_machine = CallStateMachine(call_id=call_id)
        
        self.watchdog = Watchdog(
            call_id=call_id,
            limits=watchdog_limits,
            violation_handler=self._handle_watchdog_violation
        )
        
        # Event handlers (injected dependencies)
        self.handlers = handlers or {}
        
        # Vocode conversation (will be set by vocode_bridge)
        self.vocode_conversation = None
        
        # Language state
        self._language_locked = False
        self._detected_language: Optional[str] = None
        self._language_confidence: Optional[float] = None
        
        # Transfer state
        self._transfer_approved = False
        self._transfer_reason: Optional[str] = None
        
        # Event tracking
        self._event_history: List[Dict[str, Any]] = []
        
        logger.info(
            "CallController created",
            extra={
                "call_id": call_id,
                "tenant_id": tenant_id,
                "request_id": self.request_id
            }
        )
    
    async def start(self) -> Dict[str, Any]:
        """
        Start call session.
        
        This is called AFTER Vocode connection is established.
        
        Returns:
            Start result with greeting info
        """
        if self.session.is_started():
            logger.warning(
                "Call already started",
                extra={"call_id": self.call_id}
            )
            return {"status": "already_started"}
        
        logger.info(
            "Starting call",
            extra={
                "call_id": self.call_id,
                "tenant_id": self.tenant_id
            }
        )
        
        try:
            # Mark session started
            self.session.mark_started()
            
            # Start watchdog
            self.watchdog.start()
            self.watchdog.mark_greeting_start()
            
            # Transition to GREETING
            self.state_machine.transition(
                CallState.GREETING,
                reason="call_started"
            )
            
            # Emit event
            await self._emit_event(CallEvent.STARTED)
            
            # Get greeting from handler
            greeting_result = await self._call_handler(
                "on_greeting",
                call_id=self.call_id,
                tenant_id=self.tenant_id
            )
            
            # Mark greeting done
            self.watchdog.mark_greeting_end()
            
            # Decide next state based on config
            if self.session.config.enable_language_detection:
                # Go to language detection
                self.state_machine.transition(
                    CallState.LANGUAGE_DETECT,
                    reason="greeting_complete"
                )
            else:
                # Go straight to active conversation
                self.state_machine.transition(
                    CallState.ACTIVE,
                    reason="greeting_complete"
                )
                self.state_machine.transition(
                    CallState.LISTENING,
                    reason="ready_for_input"
                )
            
            logger.info(
                "Call started successfully",
                extra={
                    "call_id": self.call_id,
                    "state": self.state_machine.current_state.value
                }
            )
            
            return {
                "status": "started",
                "call_id": self.call_id,
                "request_id": self.request_id,
                "greeting": greeting_result
            }
        
        except Exception as e:
            logger.error(
                f"Call start failed: {e}",
                extra={"call_id": self.call_id},
                exc_info=True
            )
            
            # Force close on startup failure
            await self._force_close("start_failure", str(e))
            raise
    
    async def handle_user_speech(
        self,
        transcript: str,
        is_final: bool = True,
        confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Handle user speech transcript from Vocode.
        
        This is the main entry point for user input.
        
        Args:
            transcript: Transcribed text
            is_final: Whether this is final transcript
            confidence: Transcription confidence
            
        Returns:
            Processing result
        """
        if not is_final:
            # Ignore partial transcripts for now
            return {"status": "partial_ignored"}
        
        if self.session.is_ended():
            logger.warning(
                "Speech on ended call",
                extra={"call_id": self.call_id}
            )
            return {"status": "call_ended"}
        
        logger.info(
            "User speech received",
            extra={
                "call_id": self.call_id,
                "state": self.state_machine.current_state.value,
                "text_length": len(transcript),
                "confidence": confidence
            }
        )
        
        # Update watchdog
        self.watchdog.mark_user_activity()
        
        # Increment turn
        self.session.increment_turn()
        self.watchdog.mark_turn()
        
        # Emit event
        await self._emit_event(
            CallEvent.USER_SPOKE,
            {"transcript": transcript, "confidence": confidence}
        )
        
        try:
            # Route based on current state
            current_state = self.state_machine.current_state
            
            if current_state == CallState.LANGUAGE_DETECT:
                return await self._handle_language_detection(transcript)
            
            elif current_state in {CallState.ACTIVE, CallState.LISTENING}:
                return await self._handle_conversation_input(transcript)
            
            elif current_state == CallState.GREETING:
                # User interrupted greeting - go to conversation
                self.watchdog.mark_greeting_end()
                self.state_machine.transition(
                    CallState.ACTIVE,
                    reason="user_interrupted_greeting"
                )
                self.state_machine.transition(
                    CallState.LISTENING,
                    reason="ready_for_input"
                )
                return await self._handle_conversation_input(transcript)
            
            else:
                logger.warning(
                    f"Speech in unexpected state: {current_state.value}",
                    extra={"call_id": self.call_id}
                )
                return {"status": "ignored", "reason": "invalid_state"}
        
        except Exception as e:
            logger.error(
                f"Error handling user speech: {e}",
                extra={"call_id": self.call_id},
                exc_info=True
            )
            
            self.session.increment_error(str(e))
            self.watchdog.mark_error()
            
            await self._emit_event(CallEvent.ERROR, {"error": str(e)})
            
            # Return fallback response
            return await self._get_fallback_response("error")
    
    async def _handle_language_detection(
        self,
        transcript: str
    ) -> Dict[str, Any]:
        """
        Handle language detection phase.
        
        Args:
            transcript: User speech
            
        Returns:
            Detection result
        """
        if self._language_locked:
            # Already detected, move to conversation
            self.state_machine.transition(
                CallState.ACTIVE,
                reason="language_already_detected"
            )
            return await self._handle_conversation_input(transcript)
        
        # Call language detection handler
        detection_result = await self._call_handler(
            "on_language_detect",
            call_id=self.call_id,
            transcript=transcript
        )
        
        language_code = detection_result.get("language")
        confidence = detection_result.get("confidence", 0.0)
        
        # Lock language if confidence is high enough
        if confidence >= 0.7:  # Configurable threshold
            self._detected_language = language_code
            self._language_confidence = confidence
            self._language_locked = True
            
            self.session.set_language(language_code, confidence)
            
            await self._emit_event(
                CallEvent.LANGUAGE_DETECTED,
                {"language": language_code, "confidence": confidence}
            )
            
            logger.info(
                f"Language detected and locked: {language_code}",
                extra={
                    "call_id": self.call_id,
                    "confidence": confidence
                }
            )
            
            # Transition to active conversation
            self.state_machine.transition(
                CallState.ACTIVE,
                reason="language_detected"
            )
            
            # Process the transcript as first conversation turn
            return await self._handle_conversation_input(transcript)
        
        else:
            # Not confident - ask again
            return {
                "status": "language_uncertain",
                "response": detection_result.get("clarification_prompt")
            }
    
    async def _handle_conversation_input(
        self,
        transcript: str
    ) -> Dict[str, Any]:
        """
        Handle normal conversation input.
        
        Args:
            transcript: User input
            
        Returns:
            AI response
        """
        # Transition to THINKING
        self.state_machine.transition(
            CallState.THINKING,
            reason="processing_user_input"
        )
        
        # Emit event
        await self._emit_event(CallEvent.AI_RESPONDING)
        
        # Translate if needed
        if self._detected_language and self._detected_language != 'en':
            translation_result = await self._call_handler(
                "on_translate_to_english",
                call_id=self.call_id,
                text=transcript,
                source_language=self._detected_language
            )
            english_text = translation_result.get("translated_text", transcript)
        else:
            english_text = transcript
        
        # Call AI handler with timeout
        try:
            ai_result = await asyncio.wait_for(
                self._call_handler(
                    "on_ai_request",
                    call_id=self.call_id,
                    user_text=english_text,
                    original_text=transcript,
                    language=self._detected_language or 'en',
                    turn_count=self.session.metadata.turn_count
                ),
                timeout=self.session.config.max_ai_response_time
            )
            
            # Mark AI activity
            self.watchdog.mark_ai_activity()
            self.watchdog.reset_error_count()
        
        except asyncio.TimeoutError:
            logger.error(
                "AI timeout",
                extra={
                    "call_id": self.call_id,
                    "timeout": self.session.config.max_ai_response_time
                }
            )
            
            self.watchdog.mark_error()
            return await self._get_fallback_response("ai_timeout")
        
        # Parse AI response
        response_text = ai_result.get("response_text", "")
        suggested_action = ai_result.get("suggested_action")
        
        # AI NEVER controls state - we interpret suggestions
        if suggested_action == "transfer":
            return await self._handle_transfer_request(
                ai_result.get("transfer_reason", "user_requested")
            )
        
        elif suggested_action == "end_call":
            await self.close(reason="ai_suggested_end")
            return {
                **ai_result,
                "actions": {"end_call": True}
            }
        
        # Translate response back if needed
        if self._detected_language and self._detected_language != 'en':
            translation_result = await self._call_handler(
                "on_translate_from_english",
                call_id=self.call_id,
                text=response_text,
                target_language=self._detected_language
            )
            response_text = translation_result.get("translated_text", response_text)
        
        # Transition to SPEAKING
        self.state_machine.transition(
            CallState.SPEAKING,
            reason="ai_response_ready"
        )
        
        # Emit event
        await self._emit_event(
            CallEvent.AI_SPOKE,
            {"response_text": response_text}
        )
        
        return {
            "status": "success",
            "response_text": response_text,
            "original_response": ai_result,
            "turn": self.session.metadata.turn_count
        }
    
    async def _handle_transfer_request(
        self,
        reason: str
    ) -> Dict[str, Any]:
        """
        Handle request to transfer call to human.
        
        Args:
            reason: Reason for transfer
            
        Returns:
            Transfer decision
        """
        await self._emit_event(
            CallEvent.TRANSFER_REQUESTED,
            {"reason": reason}
        )
        
        # Check if transfer is enabled
        if not self.session.config.enable_transfer:
            logger.warning(
                "Transfer requested but disabled",
                extra={"call_id": self.call_id}
            )
            
            await self._emit_event(CallEvent.TRANSFER_DENIED)
            
            return {
                "status": "transfer_denied",
                "response_text": "I apologize, but I cannot transfer you at this time."
            }
        
        # Call transfer approval handler
        approval_result = await self._call_handler(
            "on_transfer_approval",
            call_id=self.call_id,
            reason=reason,
            tenant_id=self.tenant_id
        )
        
        approved = approval_result.get("approved", False)
        transfer_number = approval_result.get("transfer_number")
        
        if approved and transfer_number:
            self._transfer_approved = True
            self._transfer_reason = reason
            self.session.mark_transfer(reason)
            
            await self._emit_event(
                CallEvent.TRANSFER_APPROVED,
                {"reason": reason, "number": transfer_number}
            )
            
            # Transition to TRANSFERRING
            self.state_machine.transition(
                CallState.TRANSFERRING,
                reason=reason
            )
            
            self.watchdog.mark_transfer_start()
            
            logger.info(
                f"Transfer approved: {reason}",
                extra={
                    "call_id": self.call_id,
                    "transfer_number": transfer_number
                }
            )
            
            return {
                "status": "transfer_approved",
                "transfer_number": transfer_number,
                "response_text": approval_result.get(
                    "transfer_message",
                    "Transferring you now."
                )
            }
        
        else:
            await self._emit_event(CallEvent.TRANSFER_DENIED)
            
            return {
                "status": "transfer_denied",
                "response_text": approval_result.get(
                    "denial_message",
                    "I cannot transfer you right now."
                )
            }
    
    async def handle_speech_complete(self) -> None:
        """
        Handle completion of AI speech playback.
        
        Called by Vocode when TTS finishes.
        """
        if self.state_machine.current_state == CallState.SPEAKING:
            # Return to listening
            self.state_machine.transition(
                CallState.LISTENING,
                reason="speech_complete"
            )
            
            logger.debug(
                "Speech complete, returned to listening",
                extra={"call_id": self.call_id}
            )
    
    async def close(
        self,
        reason: str = "normal"
    ) -> None:
        """
        Gracefully close the call.
        
        Args:
            reason: Reason for closing
        """
        if self.session.is_ended():
            logger.debug(
                "Call already closed",
                extra={"call_id": self.call_id}
            )
            return
        
        logger.info(
            f"Closing call: {reason}",
            extra={"call_id": self.call_id}
        )
        
        await self._emit_event(CallEvent.CLOSING, {"reason": reason})
        
        try:
            # Transition to CLOSING
            if not self.state_machine.is_terminal():
                self.state_machine.transition(
                    CallState.CLOSING,
                    reason=reason
                )
            
            # Stop watchdog
            await self.watchdog.stop()
            
            # Call closing handler
            await self._call_handler(
                "on_closing",
                call_id=self.call_id,
                reason=reason,
                session_summary=self.session.get_summary()
            )
            
            # Transition to CLOSED
            self.state_machine.transition(
                CallState.CLOSED,
                reason=reason
            )
            
            # Mark session ended
            self.session.mark_ended(reason)
            
            await self._emit_event(CallEvent.CLOSED, {"reason": reason})
            
            logger.info(
                "Call closed successfully",
                extra={
                    "call_id": self.call_id,
                    "duration": self.session.get_duration()
                }
            )
        
        except Exception as e:
            logger.error(
                f"Error during close: {e}",
                extra={"call_id": self.call_id},
                exc_info=True
            )
            
            await self._force_close(reason, str(e))
    
    async def _force_close(
        self,
        reason: str,
        error: Optional[str] = None
    ) -> None:
        """
        Force close call (emergency).
        
        Args:
            reason: Reason for force close
            error: Optional error message
        """
        logger.warning(
            f"FORCE CLOSE: {reason}",
            extra={
                "call_id": self.call_id,
                "error": error
            }
        )
        
        try:
            # Force state transition
            self.state_machine.force_transition(
                CallState.FAILED,
                reason=f"force_close: {reason}",
                metadata={"error": error}
            )
            
            # Stop watchdog
            await self.watchdog.stop()
            
            # Mark session ended
            self.session.mark_ended(f"forced: {reason}")
        
        except Exception as e:
            logger.error(
                f"Error in force close: {e}",
                extra={"call_id": self.call_id},
                exc_info=True
            )
    
    async def _handle_watchdog_violation(
        self,
        violation: WatchdogViolation,
        action: WatchdogAction,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Handle watchdog violation.
        
        Args:
            violation: Type of violation
            action: Recommended action
            metadata: Violation context
        """
        logger.warning(
            f"Watchdog violation: {violation.value} -> {action.value}",
            extra={
                "call_id": self.call_id,
                "metadata": metadata
            }
        )
        
        # Execute action
        if action == WatchdogAction.END_CALL:
            await self.close(reason=f"watchdog_{violation.value}")
        
        elif action == WatchdogAction.KILL_CALL:
            await self._force_close(
                f"watchdog_{violation.value}",
                str(metadata)
            )
        
        elif action == WatchdogAction.TRANSFER_TO_HUMAN:
            await self._handle_transfer_request(
                f"watchdog_{violation.value}"
            )
    
    async def _call_handler(
        self,
        handler_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call registered event handler.
        
        Args:
            handler_name: Handler name
            **kwargs: Handler arguments
            
        Returns:
            Handler result
        """
        handler = self.handlers.get(handler_name)
        
        if not handler:
            logger.warning(
                f"No handler registered: {handler_name}",
                extra={"call_id": self.call_id}
            )
            return {}
        
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**kwargs)
            else:
                result = handler(**kwargs)
            
            return result or {}
        
        except Exception as e:
            logger.error(
                f"Handler error: {handler_name}: {e}",
                extra={"call_id": self.call_id},
                exc_info=True
            )
            return {}
    
    async def _emit_event(
        self,
        event_type: CallEvent,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Emit call event.
        
        Args:
            event_type: Event type
            data: Event data
        """
        event = {
            "type": event_type.value,
            "call_id": self.call_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data or {}
        }
        
        self._event_history.append(event)
        
        # Call event handler if registered
        await self._call_handler(
            "on_event",
            event=event
        )
    
    async def _get_fallback_response(
        self,
        error_type: str
    ) -> Dict[str, Any]:
        """
        Get fallback response on error.
        
        Args:
            error_type: Type of error
            
        Returns:
            Fallback response
        """
        return await self._call_handler(
            "on_fallback",
            call_id=self.call_id,
            error_type=error_type
        )
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current call status.
        
        Returns:
            Status dict
        """
        return {
            "call_id": self.call_id,
            "tenant_id": self.tenant_id,
            "state": self.state_machine.current_state.value,
            "session": self.session.get_summary(),
            "watchdog": self.watchdog.get_health_status(),
            "language": {
                "detected": self._detected_language,
                "confidence": self._language_confidence,
                "locked": self._language_locked
            },
            "transfer": {
                "approved": self._transfer_approved,
                "reason": self._transfer_reason
            }
        }
    
    async def cleanup(self) -> None:
        """Cleanup all resources."""
        logger.info(
            "Cleaning up call controller",
            extra={"call_id": self.call_id}
        )
        
        await self.watchdog.stop()
        await self.session.cleanup()
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<CallController "
            f"call_id={self.call_id} "
            f"state={self.state_machine.current_state.value} "
            f"active={self.session.is_active()}>"
        )
