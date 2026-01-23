"""
Call Controller
===============
Central orchestrator for single call lifecycle.

This is the BRAIN and SINGLE AUTHORITY of each call.

Responsibilities:
- Own exactly one Vocode StreamingConversation
- Manage CallState machine (ONLY module allowed to change state)
- Supervise watchdog
- Route all events to business/AI layers
- Control language translation flow (detect ONCE, lock forever)
- Approve all transfers (strict sequence enforcement)
- Enforce all safety boundaries
- Own all cancellation tokens
- Own THE ONLY terminate_call() routine that ALL shutdowns go through
- Kill STT, TTS, LLM, Vocode on termination
- Enforce max call duration, silence timeout, AI timeout, stuck-state kill

This does NOT:
- Embed business logic
- Call databases directly
- Implement voice pipeline internals
- Make AI prompt decisions
- Let AI control system flow
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime, timezone
from enum import Enum
import uuid

from call_state import CallState, CallStateMachine, StateTransitionError, TerminalStateError
from watchdog import Watchdog, WatchdogLimits, WatchdogViolation, WatchdogAction

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
    TRANSFER_EXECUTING = "transfer_executing"
    LANGUAGE_DETECTED = "language_detected"
    ERROR = "error"
    CLOSING = "closing"
    CLOSED = "closed"
    TERMINATED = "terminated"


class CallController:
    """
    Per-call controller - orchestrates single call from start to finish.
    
    THIS IS THE SINGLE AUTHORITY FOR THE CALL.
    
    Architecture:
    - Wraps ONE Vocode StreamingConversation (voice engine)
    - Owns ONE CallStateMachine (formal state) - ONLY this module changes state
    - Owns ONE Watchdog (safety monitoring)
    - Routes events to injected handlers (business logic)
    - Owns ALL cancellation tokens
    
    Control flow:
    1. Vocode handles: audio, telephony, turn-taking, barge-in
    2. Controller handles: state, AI routing, business rules, safety, termination
    3. Handlers handle: AI calls, menu logic, orders, database
    
    AI NEVER controls system flow.
    Controller makes all state decisions.
    ALL terminations go through terminate_call().
    """
    
    def __init__(
        self,
        call_id: str,
        tenant_id: str,
        watchdog_limits: Optional[WatchdogLimits] = None,
        handlers: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize call controller.
        
        Args:
            call_id: Unique call identifier
            tenant_id: Tenant/restaurant identifier
            watchdog_limits: Watchdog limits
            handlers: Event handler callbacks
        """
        self.call_id = call_id
        self.tenant_id = tenant_id
        self.request_id = str(uuid.uuid4())
        
        # Core components - THIS CONTROLLER OWNS THESE
        self.state_machine = CallStateMachine(call_id=call_id)
        
        # Register state change callbacks
        self.state_machine.register_state_change_callback(self._on_state_change)
        self.state_machine.register_terminal_state_callback(self._on_terminal_state)
        
        self.watchdog = Watchdog(
            call_id=call_id,
            limits=watchdog_limits,
            violation_handler=self._handle_watchdog_violation
        )
        
        # Event handlers (injected dependencies)
        self.handlers = handlers or {}
        
        # Vocode conversation (will be set by vocode_bridge)
        # Controller owns this reference and can kill it
        self.vocode_conversation = None
        
        # Session state
        self._started_at: Optional[datetime] = None
        self._ended_at: Optional[datetime] = None
        self._end_reason: Optional[str] = None
        
        # Language state - LOCKED AFTER FIRST DETECTION
        self._language_locked = False
        self._detected_language: Optional[str] = None
        self._language_confidence: Optional[float] = None
        self._language_detection_attempted = False
        
        # Transfer state - STRICT APPROVAL REQUIRED
        self._transfer_approved = False
        self._transfer_reason: Optional[str] = None
        self._transfer_number: Optional[str] = None
        self._transfer_in_progress = False
        
        # Cancellation control - CONTROLLER OWNS ALL TOKENS
        self._main_task: Optional[asyncio.Task] = None
        self._ai_task: Optional[asyncio.Task] = None
        self._stt_task: Optional[asyncio.Task] = None
        self._tts_task: Optional[asyncio.Task] = None
        self._active_tasks: List[asyncio.Task] = []
        
        # Event tracking
        self._event_history: List[Dict[str, Any]] = []
        
        # Termination control
        self._termination_lock = asyncio.Lock()
        self._termination_in_progress = False
        self._termination_complete = False
        
        # Metrics
        self._turn_count = 0
        self._error_count = 0
        
        logger.info(
            "CallController created",
            extra={
                "call_id": call_id,
                "tenant_id": tenant_id,
                "request_id": self.request_id
            }
        )
    
    def _on_state_change(self, old_state: CallState, new_state: CallState) -> None:
        """
        Callback when state changes.
        
        Args:
            old_state: Previous state
            new_state: New state
        """
        logger.info(
            f"State changed: {old_state.value} -> {new_state.value}",
            extra={
                "call_id": self.call_id,
                "from_state": old_state.value,
                "to_state": new_state.value
            }
        )
        
        # Update watchdog on state change
        self.watchdog.mark_state_change()
    
    def _on_terminal_state(self, terminal_state: CallState) -> None:
        """
        Callback when terminal state is reached.
        
        Args:
            terminal_state: The terminal state reached
        """
        logger.warning(
            f"Terminal state reached: {terminal_state.value}",
            extra={
                "call_id": self.call_id,
                "terminal_state": terminal_state.value
            }
        )
    
    async def start(self) -> Dict[str, Any]:
        """
        Start call session.
        
        This is called AFTER Vocode connection is established.
        
        Returns:
            Start result with greeting info
        """
        if self._started_at:
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
            self._started_at = datetime.now(timezone.utc)
            
            # Start watchdog
            self.watchdog.start()
            self.watchdog.mark_greeting_start()
            self.watchdog.mark_state_change()
            
            # Transition to CONNECTING
            self.state_machine.transition(
                CallState.CONNECTING,
                reason="vocode_connected"
            )
            
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
            
            # Decide next state
            # Language detection is controlled by handler response
            if greeting_result.get("enable_language_detection", False) and not self._language_locked:
                # Go to language detection (ONCE ONLY)
                self.state_machine.transition(
                    CallState.LANGUAGE_DETECT,
                    reason="greeting_complete"
                )
                self.watchdog.mark_language_detect_start()
                self._language_detection_attempted = True
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
            
            # Force termination on startup failure
            await self.terminate_call("start_failure", error=str(e))
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
        # Reject if not final
        if not is_final:
            return {"status": "partial_ignored"}
        
        # Reject if session ended
        if self._ended_at:
            logger.warning(
                "Speech on ended call - rejecting",
                extra={"call_id": self.call_id}
            )
            return {"status": "call_ended"}
        
        # Reject if in frozen state (AI/business logic not allowed)
        if self.state_machine.is_frozen():
            logger.warning(
                "Speech in frozen state - rejecting",
                extra={
                    "call_id": self.call_id,
                    "state": self.state_machine.current_state.value
                }
            )
            return {"status": "frozen_state"}
        
        # Reject if termination in progress
        if self._termination_in_progress:
            logger.warning(
                "Speech during termination - rejecting",
                extra={"call_id": self.call_id}
            )
            return {"status": "terminating"}
        
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
        self.watchdog.mark_pipeline_activity()
        
        # Increment turn
        self._turn_count += 1
        self.watchdog.mark_turn()
        
        # Handle language detection if needed
        if self.state_machine.current_state == CallState.LANGUAGE_DETECT:
            return await self._handle_language_detection(transcript, confidence)
        
        # Transition to THINKING
        try:
            if self.state_machine.current_state == CallState.LISTENING:
                self.state_machine.transition(
                    CallState.THINKING,
                    reason="user_speech_received"
                )
        except (StateTransitionError, TerminalStateError) as e:
            logger.error(
                f"Invalid state transition: {e}",
                extra={"call_id": self.call_id}
            )
            return {"status": "invalid_state"}
        
        # Emit event
        await self._emit_event(
            CallEvent.USER_SPOKE,
            {
                "transcript": transcript,
                "confidence": confidence,
                "turn": self._turn_count
            }
        )
        
        # Call AI handler
        try:
            await self._emit_event(CallEvent.AI_RESPONDING)
            
            # Create AI task for cancellation control
            ai_coro = self._call_handler(
                "on_user_speech",
                call_id=self.call_id,
                transcript=transcript,
                confidence=confidence,
                language=self._detected_language
            )
            
            # Store task for cancellation
            self._ai_task = asyncio.create_task(ai_coro)
            self._active_tasks.append(self._ai_task)
            
            # Wait for AI response
            ai_result = await self._ai_task
            
            # Remove from active tasks
            if self._ai_task in self._active_tasks:
                self._active_tasks.remove(self._ai_task)
            
            # Update watchdog on successful AI response
            self.watchdog.mark_ai_activity()
            self.watchdog.mark_pipeline_activity()
            
            # Check if transfer was requested in AI response
            if ai_result.get("transfer_requested"):
                transfer_reason = ai_result.get("transfer_reason", "user_request")
                transfer_number = ai_result.get("transfer_number")
                await self._handle_transfer_request(transfer_reason, transfer_number)
                return {"status": "transfer_initiated"}
            
            # Transition to SPEAKING (only if not frozen)
            if not self.state_machine.is_frozen():
                self.state_machine.transition(
                    CallState.SPEAKING,
                    reason="ai_response_ready"
                )
            
            await self._emit_event(
                CallEvent.AI_SPOKE,
                {"response": ai_result}
            )
            
            # After AI speaks, go back to LISTENING (if not frozen)
            if not self.state_machine.is_frozen():
                self.state_machine.transition(
                    CallState.LISTENING,
                    reason="ai_response_complete"
                )
            
            return {
                "status": "processed",
                "ai_response": ai_result
            }
        
        except asyncio.CancelledError:
            logger.warning(
                "AI processing cancelled",
                extra={"call_id": self.call_id}
            )
            return {"status": "cancelled"}
        
        except Exception as e:
            logger.error(
                f"AI processing error: {e}",
                extra={"call_id": self.call_id},
                exc_info=True
            )
            
            # Mark error in watchdog
            self._error_count += 1
            self.watchdog.mark_error(str(e))
            
            # Get fallback response
            fallback = await self._get_fallback_response("ai_error")
            
            # Go back to LISTENING (if not frozen)
            if not self.state_machine.is_frozen():
                try:
                    self.state_machine.transition(
                        CallState.LISTENING,
                        reason="error_recovery"
                    )
                except (StateTransitionError, TerminalStateError):
                    pass
            
            return {
                "status": "error",
                "fallback": fallback
            }
    
    async def _handle_language_detection(
        self,
        transcript: str,
        confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Handle language detection - HAPPENS EXACTLY ONCE.
        
        Args:
            transcript: User's first speech
            confidence: Transcript confidence
            
        Returns:
            Detection result
        """
        # Prevent re-detection
        if self._language_locked:
            logger.warning(
                "Language already detected and locked",
                extra={"call_id": self.call_id}
            )
            return {"status": "already_detected"}
        
        logger.info(
            "Detecting language",
            extra={"call_id": self.call_id}
        )
        
        try:
            # Call language detection handler
            detection_result = await self._call_handler(
                "on_language_detect",
                call_id=self.call_id,
                transcript=transcript,
                confidence=confidence
            )
            
            # Extract detected language
            detected_lang = detection_result.get("language", "en")
            lang_confidence = detection_result.get("confidence", 0.0)
            
            # LOCK LANGUAGE - NO FURTHER DETECTION ALLOWED
            self._detected_language = detected_lang
            self._language_confidence = lang_confidence
            self._language_locked = True
            
            # Mark detection complete
            self.watchdog.mark_language_detect_end()
            
            logger.info(
                "Language detected and LOCKED",
                extra={
                    "call_id": self.call_id,
                    "language": detected_lang,
                    "confidence": lang_confidence
                }
            )
            
            # Emit event
            await self._emit_event(
                CallEvent.LANGUAGE_DETECTED,
                {
                    "language": detected_lang,
                    "confidence": lang_confidence
                }
            )
            
            # Transition to ACTIVE conversation
            self.state_machine.transition(
                CallState.ACTIVE,
                reason="language_detected"
            )
            
            self.state_machine.transition(
                CallState.LISTENING,
                reason="ready_for_conversation"
            )
            
            return {
                "status": "detected",
                "language": detected_lang,
                "confidence": lang_confidence
            }
        
        except Exception as e:
            logger.error(
                f"Language detection error: {e}",
                extra={"call_id": self.call_id},
                exc_info=True
            )
            
            # Fallback to English and lock
            self._detected_language = "en"
            self._language_confidence = 0.0
            self._language_locked = True
            
            self.watchdog.mark_language_detect_end()
            
            # Continue to active conversation
            self.state_machine.transition(
                CallState.ACTIVE,
                reason="language_detection_failed"
            )
            
            self.state_machine.transition(
                CallState.LISTENING,
                reason="fallback_to_english"
            )
            
            return {
                "status": "fallback",
                "language": "en"
            }
    
    async def _handle_transfer_request(
        self,
        reason: str,
        transfer_number: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle transfer request - STRICT APPROVAL AND SEQUENCE.
        
        Transfer sequence:
        1. Check if allowed
        2. Approve transfer
        3. FREEZE state (enter TRANSFERRING)
        4. STOP AI generation
        5. STOP TTS
        6. STOP STT
        7. DETACH Vocode
        8. Execute transfer
        9. Mark TRANSFERRED terminal state
        
        Args:
            reason: Reason for transfer
            transfer_number: Number to transfer to
            
        Returns:
            Transfer result
        """
        logger.info(
            f"Transfer requested: {reason}",
            extra={
                "call_id": self.call_id,
                "current_state": self.state_machine.current_state.value
            }
        )
        
        # Check if already in progress
        if self._transfer_in_progress:
            logger.warning(
                "Transfer already in progress",
                extra={"call_id": self.call_id}
            )
            return {"status": "already_in_progress"}
        
        # Check if already transferred
        if self.state_machine.current_state == CallState.TRANSFERRED:
            logger.warning(
                "Call already transferred",
                extra={"call_id": self.call_id}
            )
            return {"status": "already_transferred"}
        
        # Check if call is already terminating
        if self._termination_in_progress:
            logger.warning(
                "Cannot transfer - termination in progress",
                extra={"call_id": self.call_id}
            )
            return {"status": "terminating"}
        
        # Emit transfer request event
        await self._emit_event(
            CallEvent.TRANSFER_REQUESTED,
            {"reason": reason, "number": transfer_number}
        )
        
        # Call transfer approval handler
        approval_result = await self._call_handler(
            "on_transfer_request",
            call_id=self.call_id,
            reason=reason,
            transfer_number=transfer_number
        )
        
        # Check if approved
        approved = approval_result.get("approved", False)
        actual_number = approval_result.get("transfer_number", transfer_number)
        
        if not approved:
            logger.warning(
                "Transfer denied",
                extra={
                    "call_id": self.call_id,
                    "reason": reason
                }
            )
            await self._emit_event(
                CallEvent.TRANSFER_DENIED,
                {"reason": reason}
            )
            return {"status": "denied"}
        
        # Validate transfer number
        if not actual_number:
            logger.error(
                "Transfer approved but no number provided",
                extra={"call_id": self.call_id}
            )
            return {"status": "invalid_number"}
        
        # APPROVE TRANSFER
        self._transfer_approved = True
        self._transfer_reason = reason
        self._transfer_number = actual_number
        self._transfer_in_progress = True
        
        logger.info(
            "Transfer APPROVED - beginning shutdown sequence",
            extra={
                "call_id": self.call_id,
                "reason": reason,
                "number": actual_number
            }
        )
        
        await self._emit_event(
            CallEvent.TRANSFER_APPROVED,
            {"reason": reason, "number": actual_number}
        )
        
        # EXECUTE TRANSFER SEQUENCE
        await self._execute_transfer_sequence(reason, actual_number)
        
        return {
            "status": "approved",
            "number": actual_number
        }
    
    async def _execute_transfer_sequence(
        self,
        reason: str,
        transfer_number: str
    ) -> None:
        """
        Execute strict transfer sequence.
        
        SEQUENCE:
        1. FREEZE - Enter TRANSFERRING state
        2. STOP AI - Cancel all AI generation
        3. STOP TTS - Cancel speech synthesis
        4. STOP STT - Cancel speech recognition
        5. DETACH - Detach Vocode conversation
        6. TRANSFER - Execute telephony transfer
        7. TERMINAL - Mark TRANSFERRED state
        
        Args:
            reason: Transfer reason
            transfer_number: Number to transfer to
        """
        logger.info(
            "Executing transfer sequence",
            extra={
                "call_id": self.call_id,
                "reason": reason,
                "number": transfer_number
            }
        )
        
        try:
            # STEP 1: FREEZE - Enter TRANSFERRING state
            if not self.state_machine.is_terminal():
                self.state_machine.transition(
                    CallState.TRANSFERRING,
                    reason=f"transfer: {reason}"
                )
                self.watchdog.mark_transfer_start()
            
            await self._emit_event(
                CallEvent.TRANSFER_EXECUTING,
                {"reason": reason, "number": transfer_number}
            )
            
            # STEP 2: STOP AI - Cancel AI generation
            await self._cancel_ai_generation()
            
            # STEP 3: STOP TTS - Cancel speech synthesis
            await self._cancel_tts()
            
            # STEP 4: STOP STT - Cancel speech recognition
            await self._cancel_stt()
            
            # STEP 5: Cancel all active tasks
            await self._cancel_all_tasks()
            
            # STEP 6: DETACH - Detach Vocode conversation
            await self._detach_vocode()
            
            # STEP 7: TRANSFER - Execute telephony transfer via handler
            transfer_result = await self._call_handler(
                "on_transfer_execute",
                call_id=self.call_id,
                transfer_number=transfer_number,
                reason=reason
            )
            
            self.watchdog.mark_transfer_end()
            
            # STEP 8: TERMINAL - Mark TRANSFERRED state
            self.state_machine.force_transition(
                CallState.TRANSFERRED,
                reason=f"transfer_complete: {reason}",
                metadata={
                    "number": transfer_number,
                    "result": transfer_result
                }
            )
            
            # Mark session ended
            self._ended_at = datetime.now(timezone.utc)
            self._end_reason = f"transferred: {reason}"
            
            logger.info(
                "Transfer sequence complete",
                extra={
                    "call_id": self.call_id,
                    "final_state": self.state_machine.current_state.value
                }
            )
        
        except Exception as e:
            logger.error(
                f"Transfer sequence failed: {e}",
                extra={"call_id": self.call_id},
                exc_info=True
            )
            
            # Force termination on transfer failure
            await self.terminate_call(
                "transfer_failed",
                error=str(e)
            )
    
    async def _cancel_ai_generation(self) -> None:
        """Cancel any ongoing AI generation."""
        if self._ai_task and not self._ai_task.done():
            logger.info(
                "Cancelling AI generation",
                extra={"call_id": self.call_id}
            )
            self._ai_task.cancel()
            try:
                await asyncio.wait_for(self._ai_task, timeout=1.0)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning(
                    "AI task did not cancel within timeout",
                    extra={"call_id": self.call_id}
                )
            except Exception as e:
                logger.error(
                    f"Error cancelling AI: {e}",
                    extra={"call_id": self.call_id}
                )
    
    async def _cancel_tts(self) -> None:
        """Cancel any ongoing TTS."""
        if self._tts_task and not self._tts_task.done():
            logger.info(
                "Cancelling TTS",
                extra={"call_id": self.call_id}
            )
            self._tts_task.cancel()
            try:
                await asyncio.wait_for(self._tts_task, timeout=1.0)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning(
                    "TTS task did not cancel within timeout",
                    extra={"call_id": self.call_id}
                )
            except Exception as e:
                logger.error(
                    f"Error cancelling TTS: {e}",
                    extra={"call_id": self.call_id}
                )
    
    async def _cancel_stt(self) -> None:
        """Cancel any ongoing STT."""
        if self._stt_task and not self._stt_task.done():
            logger.info(
                "Cancelling STT",
                extra={"call_id": self.call_id}
            )
            self._stt_task.cancel()
            try:
                await asyncio.wait_for(self._stt_task, timeout=1.0)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning(
                    "STT task did not cancel within timeout",
                    extra={"call_id": self.call_id}
                )
            except Exception as e:
                logger.error(
                    f"Error cancelling STT: {e}",
                    extra={"call_id": self.call_id}
                )
    
    async def _cancel_all_tasks(self) -> None:
        """Cancel all active tasks."""
        if not self._active_tasks:
            return
        
        logger.info(
            f"Cancelling {len(self._active_tasks)} active tasks",
            extra={"call_id": self.call_id}
        )
        
        for task in self._active_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for all tasks to cancel with timeout
        if self._active_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._active_tasks, return_exceptions=True),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Some tasks did not cancel within timeout",
                    extra={"call_id": self.call_id}
                )
            except Exception as e:
                logger.error(
                    f"Error cancelling tasks: {e}",
                    extra={"call_id": self.call_id}
                )
        
        self._active_tasks.clear()
    
    async def _detach_vocode(self) -> None:
        """Detach Vocode conversation."""
        if self.vocode_conversation:
            logger.info(
                "Detaching Vocode conversation",
                extra={"call_id": self.call_id}
            )
            try:
                # Call handler to detach Vocode
                await self._call_handler(
                    "on_vocode_detach",
                    call_id=self.call_id
                )
                
                # Clear reference
                self.vocode_conversation = None
            except Exception as e:
                logger.error(
                    f"Error detaching Vocode: {e}",
                    extra={"call_id": self.call_id},
                    exc_info=True
                )
    
    async def terminate_call(
        self,
        reason: str,
        error: Optional[str] = None
    ) -> None:
        """
        THE SINGLE TERMINATION ROUTINE.
        
        ALL call shutdowns MUST go through this method.
        
        This is the ONLY method that:
        - Stops all pipelines (STT, TTS, LLM)
        - Stops Vocode conversation
        - Transitions to terminal state
        - Cleans up resources
        
        Args:
            reason: Termination reason
            error: Optional error message
        """
        # Use lock to prevent concurrent terminations
        async with self._termination_lock:
            if self._termination_in_progress:
                logger.debug(
                    "Termination already in progress",
                    extra={"call_id": self.call_id}
                )
                return
            
            self._termination_in_progress = True
        
        logger.warning(
            f"TERMINATING CALL: {reason}",
            extra={
                "call_id": self.call_id,
                "error": error,
                "current_state": self.state_machine.current_state.value
            }
        )
        
        await self._emit_event(
            CallEvent.TERMINATED,
            {"reason": reason, "error": error}
        )
        
        try:
            # STEP 1: Stop all AI/pipeline components with timeout
            logger.info(
                "Stopping all pipeline components",
                extra={"call_id": self.call_id}
            )
            
            await asyncio.gather(
                self._cancel_ai_generation(),
                self._cancel_tts(),
                self._cancel_stt(),
                self._cancel_all_tasks(),
                return_exceptions=True
            )
            
            # STEP 2: Detach Vocode
            await self._detach_vocode()
            
            # STEP 3: Stop watchdog
            await self.watchdog.stop()
            
            # STEP 4: Transition to terminal state
            if error or self.watchdog.is_killed():
                # Force FAILED state on error or watchdog kill
                self.state_machine.force_transition(
                    CallState.FAILED,
                    reason=f"terminated: {reason}",
                    metadata={"error": error, "killed": self.watchdog.is_killed()}
                )
            else:
                # Normal termination
                if not self.state_machine.is_terminal():
                    try:
                        # Try graceful transition
                        self.state_machine.transition(
                            CallState.CLOSING,
                            reason=reason
                        )
                        self.state_machine.transition(
                            CallState.CLOSED,
                            reason=reason
                        )
                    except (StateTransitionError, TerminalStateError):
                        # Force if normal transition fails
                        self.state_machine.force_transition(
                            CallState.CLOSED,
                            reason=reason
                        )
            
            # STEP 5: Mark session ended
            self._ended_at = datetime.now(timezone.utc)
            self._end_reason = f"terminated: {reason}"
            
            # STEP 6: Call cleanup handler
            await self._call_handler(
                "on_terminated",
                call_id=self.call_id,
                reason=reason,
                error=error,
                session_summary=self.get_summary()
            )
            
            # Mark termination complete
            self._termination_complete = True
            
            logger.warning(
                "Call terminated successfully",
                extra={
                    "call_id": self.call_id,
                    "reason": reason,
                    "final_state": self.state_machine.current_state.value,
                    "duration": self.get_duration()
                }
            )
        
        except Exception as e:
            logger.error(
                f"Error during termination: {e}",
                extra={"call_id": self.call_id},
                exc_info=True
            )
            
            # Force final state
            try:
                self.state_machine.force_transition(
                    CallState.FAILED,
                    reason="termination_error",
                    metadata={"original_reason": reason, "error": str(e)}
                )
            except Exception:
                pass
            
            self._termination_complete = True
    
    async def close(
        self,
        reason: str = "normal"
    ) -> None:
        """
        Gracefully close the call.
        
        This routes through terminate_call() for consistency.
        
        Args:
            reason: Reason for closing
        """
        await self.terminate_call(reason)
    
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
        logger.error(
            f"WATCHDOG VIOLATION: {violation.value} -> {action.value}",
            extra={
                "call_id": self.call_id,
                "violation": violation.value,
                "action": action.value,
                "metadata": metadata
            }
        )
        
        # Execute action through terminate_call()
        if action == WatchdogAction.END_CALL:
            await self.terminate_call(f"watchdog_{violation.value}")
        
        elif action == WatchdogAction.KILL_CALL:
            await self.terminate_call(
                f"watchdog_{violation.value}",
                error=str(metadata)
            )
        
        elif action == WatchdogAction.TRANSFER_TO_HUMAN:
            # Only transfer if not already terminating
            if not self._termination_in_progress:
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
            logger.debug(
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
            
            # Mark error in watchdog
            self._error_count += 1
            self.watchdog.mark_error(f"{handler_name}: {str(e)}")
            
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
            "state": self.state_machine.current_state.value,
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
    
    def get_duration(self) -> float:
        """
        Get call duration in seconds.
        
        Returns:
            Duration in seconds
        """
        if not self._started_at:
            return 0.0
        
        end_time = self._ended_at or datetime.now(timezone.utc)
        return (end_time - self._started_at).total_seconds()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get call session summary.
        
        Returns:
            Summary dict
        """
        return {
            "call_id": self.call_id,
            "tenant_id": self.tenant_id,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "ended_at": self._ended_at.isoformat() if self._ended_at else None,
            "duration": self.get_duration(),
            "end_reason": self._end_reason,
            "turn_count": self._turn_count,
            "error_count": self._error_count
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current call status.
        
        Returns:
            Status dict
        """
        return {
            "call_id": self.call_id,
            "tenant_id": self.tenant_id,
            "request_id": self.request_id,
            "state": self.state_machine.current_state.value,
            "state_summary": self.state_machine.get_state_summary(),
            "session": self.get_summary(),
            "watchdog": self.watchdog.get_health_status(),
            "language": {
                "detected": self._detected_language,
                "confidence": self._language_confidence,
                "locked": self._language_locked,
                "detection_attempted": self._language_detection_attempted
            },
            "transfer": {
                "approved": self._transfer_approved,
                "reason": self._transfer_reason,
                "number": self._transfer_number,
                "in_progress": self._transfer_in_progress
            },
            "termination": {
                "in_progress": self._termination_in_progress,
                "complete": self._termination_complete
            }
        }
    
    async def cleanup(self) -> None:
        """
        Cleanup all resources.
        
        This should be called after terminate_call() completes.
        """
        logger.info(
            "Cleaning up call controller",
            extra={"call_id": self.call_id}
        )
        
        try:
            # Ensure watchdog is stopped
            if self.watchdog._active:
                await self.watchdog.stop()
            
            # Cancel any remaining tasks
            await self._cancel_all_tasks()
            
            # Clear references
            self.vocode_conversation = None
            self.handlers.clear()
            
        except Exception as e:
            logger.error(
                f"Error during cleanup: {e}",
                extra={"call_id": self.call_id},
                exc_info=True
            )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<CallController "
            f"call_id={self.call_id} "
            f"state={self.state_machine.current_state.value} "
            f"terminated={self._termination_complete}>"
        )
