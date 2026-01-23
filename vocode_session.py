"""
Vocode Session
==============
Bridge between Vocode StreamingConversation and CallController.

Responsibilities:
- Create Vocode StreamingConversation
- Wire up STT, TTS, Agent
- Route Vocode events to CallController
- Handle barge-in and interruptions
- Manage WebSocket lifecycle
- ENFORCE hard shutdown of voice pipelines
- Configure ElevenLabs TTS with streaming and language support

This is the GLUE between Vocode's voice engine and our control layer.

CRITICAL: This layer NEVER executes business logic.
It ONLY bridges voice I/O to CallController.

SHUTDOWN RULES:
- Controller can KILL this at any time
- Must stop STT, TTS, Agent immediately
- Must close WebSocket cleanly
- Must not block on shutdown
"""

import asyncio
import logging
import json
from typing import Optional, Dict, Any, AsyncGenerator, Callable
from datetime import datetime, timezone

from fastapi import WebSocket

from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.models.transcriber import TranscriberConfig
from vocode.streaming.models.synthesizer import SynthesizerConfig
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.transcriber.base_transcriber import BaseTranscriber
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer
from vocode.streaming.agent.base_agent import BaseAgent, AgentInput, AgentResponse
from vocode.streaming.models.events import Event, EventType
from vocode.streaming.output_device.twilio_output_device import TwilioOutputDevice
from vocode.streaming.models.telephony import TwilioConfig

# ElevenLabs imports
try:
    from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
    from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False

from settings import get_settings

logger = logging.getLogger(__name__)


def create_elevenlabs_config(
    voice_id: Optional[str] = None,
    language_code: Optional[str] = None,
    api_key: Optional[str] = None
) -> Optional[SynthesizerConfig]:
    """
    Create ElevenLabs synthesizer configuration.
    
    Args:
        voice_id: ElevenLabs voice ID (if None, uses default from settings)
        language_code: Language code for voice mapping (e.g., 'en', 'es', 'ar')
        api_key: ElevenLabs API key (if None, uses settings)
        
    Returns:
        ElevenLabs synthesizer config or None if not available
    """
    if not ELEVENLABS_AVAILABLE:
        logger.error("ElevenLabs synthesizer not available in Vocode installation")
        return None
    
    settings = get_settings()
    
    # Get API key
    if not api_key:
        api_key = settings.speech.elevenlabs_api_key
    
    if not api_key:
        logger.error("ElevenLabs API key not configured")
        return None
    
    # Determine voice ID
    if voice_id:
        selected_voice_id = voice_id
    elif language_code:
        # Use language-specific voice if available
        selected_voice_id = settings.get_voice_id_for_language(language_code)
    else:
        # Use default voice
        selected_voice_id = settings.speech.elevenlabs_default_voice_id
    
    logger.info(
        "Creating ElevenLabs config",
        extra={
            "voice_id": selected_voice_id,
            "language": language_code
        }
    )
    
    try:
        config = ElevenLabsSynthesizerConfig(
            api_key=api_key,
            voice_id=selected_voice_id,
            stability=0.5,  # Voice stability (0-1)
            similarity_boost=0.75,  # Voice similarity (0-1)
            optimize_streaming_latency=4,  # Optimize for low latency (0-4)
            should_encode_as_wav=True,  # Required for Twilio
            sampling_rate=8000,  # Twilio uses 8kHz
            audio_encoding="mulaw"  # Twilio requires mulaw
        )
        
        return config
    
    except Exception as e:
        logger.error(
            f"Failed to create ElevenLabs config: {e}",
            exc_info=True
        )
        return None


def create_elevenlabs_synthesizer(
    config: SynthesizerConfig
) -> Optional[BaseSynthesizer]:
    """
    Create ElevenLabs synthesizer instance.
    
    Args:
        config: ElevenLabs synthesizer configuration
        
    Returns:
        ElevenLabs synthesizer or None on failure
    """
    if not ELEVENLABS_AVAILABLE:
        logger.error("ElevenLabs not available")
        return None
    
    if not isinstance(config, ElevenLabsSynthesizerConfig):
        logger.error("Invalid ElevenLabs config type")
        return None
    
    try:
        synthesizer = ElevenLabsSynthesizer(config)
        logger.info("ElevenLabs synthesizer created successfully")
        return synthesizer
    
    except Exception as e:
        logger.error(
            f"Failed to create ElevenLabs synthesizer: {e}",
            exc_info=True
        )
        return None


class VocodeSession:
    """
    Manages Vocode StreamingConversation lifecycle.
    
    This class:
    - Creates Vocode components (STT, TTS, Agent)
    - Starts StreamingConversation
    - Routes transcripts to CallController
    - Routes AI responses from CallController to Vocode
    - Handles interruptions and barge-in
    - Enforces hard shutdown when controller terminates
    - Configures ElevenLabs TTS with language support
    
    CRITICAL RULES:
    - Controller owns this session
    - Controller can KILL this session at any time
    - ALL transcripts route through controller
    - NO direct business logic execution
    - Shutdown must complete within hard timeout
    """
    
    def __init__(
        self,
        call_id: str,
        websocket: WebSocket,
        twilio_config: TwilioConfig,
        transcriber_config: TranscriberConfig,
        synthesizer_config: SynthesizerConfig,
        agent_config: AgentConfig,
        transcriber_factory,
        synthesizer_factory,
        agent_factory,
        language_code: Optional[str] = None
    ):
        """
        Initialize Vocode session.
        
        Args:
            call_id: Unique call identifier
            websocket: Twilio WebSocket
            twilio_config: Twilio configuration
            transcriber_config: STT configuration
            synthesizer_config: TTS configuration
            agent_config: Agent configuration
            transcriber_factory: Factory for creating transcriber
            synthesizer_factory: Factory for creating synthesizer
            agent_factory: Factory for creating agent
            language_code: Detected language code for voice selection
        """
        self.call_id = call_id
        self.websocket = websocket
        self.twilio_config = twilio_config
        self.language_code = language_code
        
        # Vocode configuration
        self.transcriber_config = transcriber_config
        self.synthesizer_config = synthesizer_config
        self.agent_config = agent_config
        
        # Factories
        self.transcriber_factory = transcriber_factory
        self.synthesizer_factory = synthesizer_factory
        self.agent_factory = agent_factory
        
        # Vocode conversation (created in start)
        self.conversation: Optional[StreamingConversation] = None
        
        # Vocode components (for explicit shutdown)
        self.transcriber: Optional[BaseTranscriber] = None
        self.synthesizer: Optional[BaseSynthesizer] = None
        self.agent: Optional[BaseAgent] = None
        
        # Tasks
        self._conversation_task: Optional[asyncio.Task] = None
        self._active_tasks: list = []
        
        # State
        self._started = False
        self._closed = False
        self._shutdown_in_progress = False
        self._shutdown_complete = False
        self._shutdown_lock = asyncio.Lock()
        
        # Callback for routing transcripts
        self._transcript_callback: Optional[Callable] = None
        
        # Metrics
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        
        logger.info(
            "VocodeSession created",
            extra={
                "call_id": call_id,
                "language": language_code
            }
        )
    
    def set_transcript_callback(self, callback: Callable) -> None:
        """
        Set callback for transcript routing.
        
        Args:
            callback: Async function(transcript, is_final, confidence)
        """
        self._transcript_callback = callback
        logger.debug(
            "Transcript callback registered",
            extra={"call_id": self.call_id}
        )
    
    def update_language(self, language_code: str) -> None:
        """
        Update session language and potentially switch voice.
        
        Args:
            language_code: New language code
        """
        if self.language_code == language_code:
            return
        
        logger.info(
            f"Updating session language: {self.language_code} -> {language_code}",
            extra={"call_id": self.call_id}
        )
        
        self.language_code = language_code
        
        # Note: Voice switching during active call requires stopping
        # and restarting synthesizer - should only be done at call start
    
    async def start(self) -> None:
        """
        Start Vocode session.
        
        Creates and starts StreamingConversation with ElevenLabs TTS.
        """
        if self._started:
            logger.warning(
                "Session already started",
                extra={"call_id": self.call_id}
            )
            return
        
        if self._closed:
            logger.error(
                "Cannot start closed session",
                extra={"call_id": self.call_id}
            )
            raise RuntimeError("Session is closed")
        
        logger.info(
            "Starting Vocode session",
            extra={"call_id": self.call_id}
        )
        
        try:
            self._start_time = datetime.now(timezone.utc)
            
            # Create Vocode components
            self.transcriber = self.transcriber_factory.create_transcriber(
                self.transcriber_config
            )
            logger.debug(
                "Transcriber created",
                extra={"call_id": self.call_id}
            )
            
            # Create ElevenLabs synthesizer
            self.synthesizer = self._create_elevenlabs_synthesizer()
            if not self.synthesizer:
                # Fallback to provided factory if ElevenLabs fails
                logger.warning(
                    "ElevenLabs creation failed, using fallback synthesizer",
                    extra={"call_id": self.call_id}
                )
                self.synthesizer = self.synthesizer_factory.create_synthesizer(
                    self.synthesizer_config
                )
            
            logger.debug(
                "Synthesizer created",
                extra={"call_id": self.call_id}
            )
            
            # Create agent wrapper that routes through callback
            self.agent = self._create_agent_wrapper()
            logger.debug(
                "Agent wrapper created",
                extra={"call_id": self.call_id}
            )
            
            # Create output device
            output_device = TwilioOutputDevice()
            
            # Create StreamingConversation
            self.conversation = StreamingConversation(
                output_device=output_device,
                transcriber=self.transcriber,
                agent=self.agent,
                synthesizer=self.synthesizer,
                conversation_id=self.call_id,
                per_chunk_allowance_seconds=0.01,  # Low latency
                events_manager=None
            )
            
            logger.debug(
                "StreamingConversation created",
                extra={"call_id": self.call_id}
            )
            
            # Wire up event handlers
            self._setup_event_handlers()
            
            # Start Vocode conversation
            # This begins audio processing loop
            self._conversation_task = asyncio.create_task(
                self.conversation.start(),
                name=f"vocode_conversation_{self.call_id}"
            )
            self._active_tasks.append(self._conversation_task)
            
            self._started = True
            
            logger.info(
                "Vocode session started",
                extra={"call_id": self.call_id}
            )
        
        except Exception as e:
            logger.error(
                f"Failed to start Vocode session: {e}",
                extra={"call_id": self.call_id},
                exc_info=True
            )
            
            # Cleanup on failure
            await self._force_shutdown()
            raise
    
    def _create_elevenlabs_synthesizer(self) -> Optional[BaseSynthesizer]:
        """
        Create ElevenLabs synthesizer with proper configuration.
        
        Returns:
            ElevenLabs synthesizer or None on failure
        """
        try:
            # Create ElevenLabs config
            elevenlabs_config = create_elevenlabs_config(
                language_code=self.language_code
            )
            
            if not elevenlabs_config:
                logger.warning(
                    "Could not create ElevenLabs config",
                    extra={"call_id": self.call_id}
                )
                return None
            
            # Create synthesizer
            synthesizer = create_elevenlabs_synthesizer(elevenlabs_config)
            
            if synthesizer:
                logger.info(
                    "ElevenLabs synthesizer created successfully",
                    extra={
                        "call_id": self.call_id,
                        "voice_id": elevenlabs_config.voice_id
                    }
                )
            
            return synthesizer
        
        except Exception as e:
            logger.error(
                f"Error creating ElevenLabs synthesizer: {e}",
                extra={"call_id": self.call_id},
                exc_info=True
            )
            return None
    
    def _create_agent_wrapper(self) -> BaseAgent:
        """
        Create agent that routes through transcript callback.
        
        Returns:
            Agent wrapper
        """
        # Create base agent
        base_agent = self.agent_factory.create_agent(self.agent_config)
        
        # Wrap it to route through callback
        class TranscriptRoutingAgent(BaseAgent):
            """
            Agent wrapper that routes transcripts through callback.
            
            This ensures all user input goes through CallController.
            """
            
            def __init__(
                self,
                base_agent: BaseAgent,
                session: "VocodeSession"
            ):
                super().__init__(base_agent.get_agent_config())
                self.base_agent = base_agent
                self.session = session
            
            async def respond(
                self,
                agent_input: AgentInput,
                conversation_id: str,
                is_interrupt: bool = False
            ) -> AsyncGenerator[AgentResponse, None]:
                """
                Intercept agent input and route through callback.
                
                Args:
                    agent_input: Input from user
                    conversation_id: Conversation ID
                    is_interrupt: Whether this is an interruption
                    
                Yields:
                    Agent responses
                """
                # Check if session is shutting down
                if self.session._shutdown_in_progress or self.session._closed:
                    logger.debug(
                        "Agent received input during shutdown - ignoring",
                        extra={"call_id": self.session.call_id}
                    )
                    return
                
                # Extract transcript
                if hasattr(agent_input, 'transcription'):
                    transcript = agent_input.transcription.message
                    confidence = getattr(
                        agent_input.transcription,
                        'confidence',
                        None
                    )
                    is_final = getattr(
                        agent_input.transcription,
                        'is_final',
                        True
                    )
                    
                    logger.debug(
                        f"Agent received transcript: {transcript[:50]}...",
                        extra={
                            "call_id": self.session.call_id,
                            "is_final": is_final,
                            "is_interrupt": is_interrupt
                        }
                    )
                    
                    # Route through callback if available
                    if self.session._transcript_callback:
                        try:
                            result = await self.session._transcript_callback(
                                transcript=transcript,
                                is_final=is_final,
                                confidence=confidence
                            )
                            
                            # Convert controller response to Vocode format
                            response_text = result.get("response_text", "")
                            
                            if response_text:
                                from vocode.streaming.agent.base_agent import (
                                    AgentResponseMessage
                                )
                                from vocode.streaming.models.message import BaseMessage
                                
                                logger.debug(
                                    f"Yielding response: {response_text[:50]}...",
                                    extra={"call_id": self.session.call_id}
                                )
                                
                                yield AgentResponseMessage(
                                    message=BaseMessage(text=response_text),
                                    is_interruptible=True
                                )
                            
                            # Handle special actions
                            actions = result.get("actions", {})
                            if actions.get("end_call"):
                                from vocode.streaming.agent.base_agent import (
                                    AgentResponseStop
                                )
                                logger.info(
                                    "End call action received",
                                    extra={"call_id": self.session.call_id}
                                )
                                yield AgentResponseStop()
                        
                        except asyncio.CancelledError:
                            logger.debug(
                                "Agent response cancelled",
                                extra={"call_id": self.session.call_id}
                            )
                            raise
                        
                        except Exception as e:
                            logger.error(
                                f"Error in transcript callback: {e}",
                                extra={"call_id": self.session.call_id},
                                exc_info=True
                            )
                            # Don't yield anything on error
                            return
                    else:
                        # Fallback to base agent if no callback
                        logger.warning(
                            "No transcript callback set - using base agent",
                            extra={"call_id": self.session.call_id}
                        )
                        async for response in self.base_agent.respond(
                            agent_input,
                            conversation_id,
                            is_interrupt
                        ):
                            yield response
            
            def get_agent_config(self):
                return self.base_agent.get_agent_config()
        
        return TranscriptRoutingAgent(base_agent, self)
    
    def _setup_event_handlers(self) -> None:
        """Setup Vocode event handlers."""
        if not self.conversation:
            return
        
        # Hook into transcript updates for monitoring
        if hasattr(self.conversation, 'transcript'):
            original_add_human = self.conversation.transcript.add_human_message
            
            def wrapped_add_human(*args, **kwargs):
                # Call original
                result = original_add_human(*args, **kwargs)
                
                # Log for monitoring
                text = kwargs.get('text') or (args[0] if args else '')
                logger.debug(
                    f"Human message added to transcript: {text[:50]}",
                    extra={"call_id": self.call_id}
                )
                
                return result
            
            self.conversation.transcript.add_human_message = wrapped_add_human
    
    async def handle_interrupt(self) -> None:
        """
        Handle user interruption (barge-in).
        
        Called when user speaks while bot is speaking.
        """
        if not self.conversation:
            return
        
        if self._shutdown_in_progress:
            return
        
        logger.info(
            "Handling interrupt",
            extra={"call_id": self.call_id}
        )
        
        # Vocode handles the actual interruption
        # We just log for monitoring
    
    async def stop_transcription(self) -> None:
        """Stop STT component with timeout."""
        if not self.transcriber:
            return
        
        logger.info(
            "Stopping transcription",
            extra={"call_id": self.call_id}
        )
        
        try:
            # Try multiple shutdown methods with timeout
            shutdown_coro = None
            
            if hasattr(self.transcriber, 'terminate'):
                shutdown_coro = self.transcriber.terminate()
            elif hasattr(self.transcriber, 'stop'):
                if asyncio.iscoroutinefunction(self.transcriber.stop):
                    shutdown_coro = self.transcriber.stop()
                else:
                    # Synchronous stop
                    self.transcriber.stop()
                    return
            
            if shutdown_coro:
                await asyncio.wait_for(shutdown_coro, timeout=1.0)
            
        except asyncio.TimeoutError:
            logger.warning(
                "Transcriber stop timed out",
                extra={"call_id": self.call_id}
            )
        except Exception as e:
            logger.error(
                f"Error stopping transcriber: {e}",
                extra={"call_id": self.call_id},
                exc_info=True
            )
    
    async def stop_synthesis(self) -> None:
        """Stop TTS component with timeout."""
        if not self.synthesizer:
            return
        
        logger.info(
            "Stopping synthesis",
            extra={"call_id": self.call_id}
        )
        
        try:
            # Try multiple shutdown methods with timeout
            shutdown_coro = None
            
            if hasattr(self.synthesizer, 'terminate'):
                shutdown_coro = self.synthesizer.terminate()
            elif hasattr(self.synthesizer, 'stop'):
                if asyncio.iscoroutinefunction(self.synthesizer.stop):
                    shutdown_coro = self.synthesizer.stop()
                else:
                    # Synchronous stop
                    self.synthesizer.stop()
                    return
            
            if shutdown_coro:
                await asyncio.wait_for(shutdown_coro, timeout=1.0)
            
        except asyncio.TimeoutError:
            logger.warning(
                "Synthesizer stop timed out",
                extra={"call_id": self.call_id}
            )
        except Exception as e:
            logger.error(
                f"Error stopping synthesizer: {e}",
                extra={"call_id": self.call_id},
                exc_info=True
            )
    
    async def stop_agent(self) -> None:
        """Stop agent component with timeout."""
        if not self.agent:
            return
        
        logger.info(
            "Stopping agent",
            extra={"call_id": self.call_id}
        )
        
        try:
            # Try multiple shutdown methods with timeout
            shutdown_coro = None
            
            if hasattr(self.agent, 'terminate'):
                shutdown_coro = self.agent.terminate()
            elif hasattr(self.agent, 'stop'):
                if asyncio.iscoroutinefunction(self.agent.stop):
                    shutdown_coro = self.agent.stop()
                else:
                    # Synchronous stop
                    self.agent.stop()
                    return
            
            if shutdown_coro:
                await asyncio.wait_for(shutdown_coro, timeout=1.0)
            
        except asyncio.TimeoutError:
            logger.warning(
                "Agent stop timed out",
                extra={"call_id": self.call_id}
            )
        except Exception as e:
            logger.error(
                f"Error stopping agent: {e}",
                extra={"call_id": self.call_id},
                exc_info=True
            )
    
    async def stop_conversation(self) -> None:
        """Stop conversation with timeout."""
        if not self.conversation:
            return
        
        logger.info(
            "Stopping conversation",
            extra={"call_id": self.call_id}
        )
        
        try:
            await asyncio.wait_for(
                self.conversation.terminate(),
                timeout=2.0
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Conversation termination timed out",
                extra={"call_id": self.call_id}
            )
        except Exception as e:
            logger.error(
                f"Error terminating conversation: {e}",
                extra={"call_id": self.call_id},
                exc_info=True
            )
    
    async def cancel_all_tasks(self) -> None:
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
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Some tasks did not cancel within timeout",
                    extra={"call_id": self.call_id}
                )
        
        self._active_tasks.clear()
    
    async def stop(self) -> None:
        """
        Stop Vocode session gracefully.
        
        This is the normal shutdown path.
        """
        async with self._shutdown_lock:
            if self._closed or self._shutdown_complete:
                return
            
            if self._shutdown_in_progress:
                logger.debug(
                    "Shutdown already in progress",
                    extra={"call_id": self.call_id}
                )
                return
            
            self._shutdown_in_progress = True
        
        logger.info(
            "Stopping Vocode session",
            extra={"call_id": self.call_id}
        )
        
        try:
            # Stop components in order with individual timeouts
            await asyncio.gather(
                self.stop_synthesis(),
                self.stop_transcription(),
                self.stop_agent(),
                return_exceptions=True
            )
            
            # Stop conversation
            await self.stop_conversation()
            
            # Cancel all tasks
            await self.cancel_all_tasks()
            
            self._closed = True
            self._shutdown_complete = True
            self._end_time = datetime.now(timezone.utc)
            
            logger.info(
                "Vocode session stopped",
                extra={
                    "call_id": self.call_id,
                    "duration": self.get_duration()
                }
            )
        
        except Exception as e:
            logger.error(
                f"Error stopping Vocode session: {e}",
                extra={"call_id": self.call_id},
                exc_info=True
            )
            
            # Force shutdown on error
            await self._force_shutdown()
    
    async def _force_shutdown(self) -> None:
        """
        Force immediate shutdown.
        
        Used for emergency termination.
        """
        async with self._shutdown_lock:
            if self._shutdown_complete:
                return
            
            self._shutdown_in_progress = True
        
        logger.warning(
            "FORCE SHUTDOWN of Vocode session",
            extra={"call_id": self.call_id}
        )
        
        try:
            # Stop all components concurrently with hard timeout
            tasks = []
            
            if self.synthesizer:
                tasks.append(self.stop_synthesis())
            
            if self.transcriber:
                tasks.append(self.stop_transcription())
            
            if self.agent:
                tasks.append(self.stop_agent())
            
            if self.conversation:
                tasks.append(self.stop_conversation())
            
            # Wait for all with hard timeout
            if tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=2.0
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "Force shutdown timed out - components did not stop",
                        extra={"call_id": self.call_id}
                    )
            
            # Cancel all tasks
            await self.cancel_all_tasks()
            
            self._closed = True
            self._shutdown_complete = True
            self._end_time = datetime.now(timezone.utc)
            
            logger.warning(
                "Vocode session force shutdown complete",
                extra={
                    "call_id": self.call_id,
                    "duration": self.get_duration()
                }
            )
        
        except Exception as e:
            logger.error(
                f"Error in force shutdown: {e}",
                extra={"call_id": self.call_id},
                exc_info=True
            )
            
            # Mark as closed anyway
            self._closed = True
            self._shutdown_complete = True
    
    async def kill(self) -> None:
        """
        KILL session immediately.
        
        This is the nuclear option - used by controller when
        watchdog triggers hard kill.
        """
        logger.error(
            "KILLING Vocode session",
            extra={"call_id": self.call_id}
        )
        
        await self._force_shutdown()
    
    async def send_message(self, text: str) -> None:
        """
        Send bot message to user.
        
        Args:
            text: Message text
        """
        if not self.conversation:
            logger.warning(
                "Cannot send message - no conversation",
                extra={"call_id": self.call_id}
            )
            return
        
        if self._shutdown_in_progress:
            logger.warning(
                "Cannot send message - shutdown in progress",
                extra={"call_id": self.call_id}
            )
            return
        
        logger.debug(
            f"Sending message: {text[:50]}",
            extra={"call_id": self.call_id}
        )
        
        # Message will be handled by Vocode's normal flow
    
    def get_duration(self) -> float:
        """
        Get session duration in seconds.
        
        Returns:
            Duration in seconds
        """
        if not self._start_time:
            return 0.0
        
        end_time = self._end_time or datetime.now(timezone.utc)
        return (end_time - self._start_time).total_seconds()
    
    def is_active(self) -> bool:
        """Check if session is active."""
        return self._started and not self._closed and not self._shutdown_in_progress
    
    def is_closed(self) -> bool:
        """Check if session is closed."""
        return self._closed
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get session status.
        
        Returns:
            Status dict
        """
        return {
            "call_id": self.call_id,
            "started": self._started,
            "closed": self._closed,
            "shutdown_in_progress": self._shutdown_in_progress,
            "shutdown_complete": self._shutdown_complete,
            "duration": self.get_duration(),
            "active_tasks": len(self._active_tasks),
            "language": self.language_code
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<VocodeSession "
            f"call_id={self.call_id} "
            f"started={self._started} "
            f"closed={self._closed}>"
        )


class VocodeSessionManager:
    """
    Manages multiple Vocode sessions.
    
    Tracks active sessions and provides lifecycle management.
    """
    
    def __init__(self, max_concurrent_sessions: int = 100):
        """
        Initialize session manager.
        
        Args:
            max_concurrent_sessions: Maximum concurrent sessions allowed
        """
        self._sessions: Dict[str, VocodeSession] = {}
        self._lock = asyncio.Lock()
        self.max_concurrent_sessions = max_concurrent_sessions
        
        logger.info(
            "VocodeSessionManager initialized",
            extra={"max_concurrent": max_concurrent_sessions}
        )
    
    async def create_session(
        self,
        call_id: str,
        **kwargs
    ) -> VocodeSession:
        """
        Create and register a new session.
        
        Args:
            call_id: Call identifier
            **kwargs: Session arguments
            
        Returns:
            Created session
            
        Raises:
            RuntimeError: If max concurrent sessions exceeded
        """
        async with self._lock:
            # Check concurrent limit
            if len(self._sessions) >= self.max_concurrent_sessions:
                logger.error(
                    "Max concurrent sessions reached",
                    extra={
                        "current": len(self._sessions),
                        "max": self.max_concurrent_sessions
                    }
                )
                raise RuntimeError(
                    f"Max concurrent sessions ({self.max_concurrent_sessions}) exceeded"
                )
            
            # Check if call_id already exists
            if call_id in self._sessions:
                logger.warning(
                    "Session already exists for call_id",
                    extra={"call_id": call_id}
                )
                # Return existing session
                return self._sessions[call_id]
            
            # Create new session
            session = VocodeSession(call_id=call_id, **kwargs)
            self._sessions[call_id] = session
        
        logger.info(
            f"Session created and registered: {call_id}",
            extra={
                "call_id": call_id,
                "total_sessions": len(self._sessions)
            }
        )
        
        return session
    
    async def get_session(self, call_id: str) -> Optional[VocodeSession]:
        """
        Get session by call ID.
        
        Args:
            call_id: Call identifier
            
        Returns:
            Session if found, None otherwise
        """
        async with self._lock:
            return self._sessions.get(call_id)
    
    async def remove_session(self, call_id: str) -> None:
        """
        Remove session from registry.
        
        Args:
            call_id: Call identifier
        """
        async with self._lock:
            if call_id in self._sessions:
                session = self._sessions[call_id]
                
                # Ensure session is stopped
                if not session.is_closed():
                    logger.warning(
                        "Removing active session - stopping first",
                        extra={"call_id": call_id}
                    )
                    try:
                        await asyncio.wait_for(session.stop(), timeout=3.0)
                    except asyncio.TimeoutError:
                        logger.error(
                            "Session stop timed out during removal - force killing",
                            extra={"call_id": call_id}
                        )
                        await session.kill()
                    except Exception as e:
                        logger.error(
                            f"Error stopping session during removal: {e}",
                            extra={"call_id": call_id},
                            exc_info=True
                        )
                        # Try to kill anyway
                        try:
                            await session.kill()
                        except Exception:
                            pass
                
                del self._sessions[call_id]
                logger.info(
                    f"Session removed: {call_id}",
                    extra={"total_sessions": len(self._sessions)}
                )
    
    async def stop_all(self) -> None:
        """Stop all active sessions."""
        logger.warning(
            f"Stopping all {len(self._sessions)} sessions",
            extra={"total": len(self._sessions)}
        )
        
        async with self._lock:
            sessions = list(self._sessions.values())
        
        # Stop all sessions concurrently
        stop_tasks = []
        for session in sessions:
            if not session.is_closed():
                stop_tasks.append(session.stop())
        
        if stop_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*stop_tasks, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.error("Some sessions did not stop within timeout - force killing")
                
                # Force kill remaining sessions
                kill_tasks = []
                for session in sessions:
                    if not session.is_closed():
                        kill_tasks.append(session.kill())
                
                if kill_tasks:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*kill_tasks, return_exceptions=True),
                            timeout=2.0
                        )
                    except asyncio.TimeoutError:
                        logger.error("Some sessions did not kill within timeout")
        
        async with self._lock:
            self._sessions.clear()
        
        logger.warning("All sessions stopped")
    
    def get_session_count(self) -> int:
        """Get count of active sessions."""
        return len(self._sessions)
    
    def get_all_session_ids(self) -> list:
        """Get list of all active session IDs."""
        return list(self._sessions.keys())
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get manager status.
        
        Returns:
            Status dict
        """
        return {
            "active_sessions": len(self._sessions),
            "max_concurrent": self.max_concurrent_sessions,
            "session_ids": list(self._sessions.keys())
        }
