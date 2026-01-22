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

This is the GLUE between Vocode's voice engine and our control layer.
"""

import asyncio
import logging
from typing import Optional, Dict, Any

from fastapi import WebSocket

from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.models.transcriber import TranscriberConfig
from vocode.streaming.models.synthesizer import SynthesizerConfig
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.transcriber.base_transcriber import BaseTranscriber
from vocode.streaming.synthesizer.base_synthesizer import BaseSynthesizer
from vocode.streaming.agent.base_agent import BaseAgent, AgentInput, AgentResponse
from vocode.streaming.models.events import Event, EventType
from vocode.streaming.telephony.conversation.twilio_phone_conversation import (
    TwilioPhoneConversation
)
from vocode.streaming.output_device.twilio_output_device import TwilioOutputDevice
from vocode.streaming.models.telephony import TwilioConfig

from core.call_controller import CallController

logger = logging.getLogger(__name__)


class VocodeSession:
    """
    Manages Vocode StreamingConversation lifecycle.
    
    This class:
    - Creates Vocode components (STT, TTS, Agent)
    - Starts StreamingConversation
    - Routes transcripts to CallController
    - Routes AI responses from CallController to Vocode
    - Handles interruptions and barge-in
    """
    
    def __init__(
        self,
        controller: CallController,
        websocket: WebSocket,
        twilio_config: TwilioConfig,
        transcriber_config: TranscriberConfig,
        synthesizer_config: SynthesizerConfig,
        agent_config: AgentConfig,
        transcriber_factory,
        synthesizer_factory,
        agent_factory
    ):
        """
        Initialize Vocode session.
        
        Args:
            controller: CallController instance
            websocket: Twilio WebSocket
            twilio_config: Twilio configuration
            transcriber_config: STT configuration
            synthesizer_config: TTS configuration
            agent_config: Agent configuration
            transcriber_factory: Factory for creating transcriber
            synthesizer_factory: Factory for creating synthesizer
            agent_factory: Factory for creating agent
        """
        self.controller = controller
        self.websocket = websocket
        self.twilio_config = twilio_config
        
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
        
        # State
        self._started = False
        self._closed = False
        
        logger.info(
            "VocodeSession created",
            extra={"call_id": controller.call_id}
        )
    
    async def start(self) -> None:
        """
        Start Vocode session.
        
        Creates and starts StreamingConversation.
        """
        if self._started:
            logger.warning(
                "Session already started",
                extra={"call_id": self.controller.call_id}
            )
            return
        
        logger.info(
            "Starting Vocode session",
            extra={"call_id": self.controller.call_id}
        )
        
        try:
            # Create Vocode components
            transcriber = self.transcriber_factory.create_transcriber(
                self.transcriber_config
            )
            
            synthesizer = self.synthesizer_factory.create_synthesizer(
                self.synthesizer_config
            )
            
            # Create custom agent wrapper that routes through CallController
            agent = self._create_agent_wrapper()
            
            # Create output device
            output_device = TwilioOutputDevice()
            
            # Create StreamingConversation
            # Note: In production, use TwilioPhoneConversation
            # For now, simplified approach
            self.conversation = StreamingConversation(
                output_device=output_device,
                transcriber=transcriber,
                agent=agent,
                synthesizer=synthesizer,
                conversation_id=self.controller.call_id,
                per_chunk_allowance_seconds=0.01,  # Low latency
                events_manager=None  # Can add custom events manager
            )
            
            # Wire up event handlers
            self._setup_event_handlers()
            
            # Attach to controller
            self.controller.vocode_conversation = self.conversation
            
            # Start controller
            await self.controller.start()
            
            # Start Vocode conversation
            # This begins audio processing loop
            await self.conversation.start()
            
            self._started = True
            
            logger.info(
                "Vocode session started",
                extra={"call_id": self.controller.call_id}
            )
        
        except Exception as e:
            logger.error(
                f"Failed to start Vocode session: {e}",
                extra={"call_id": self.controller.call_id},
                exc_info=True
            )
            raise
    
    def _create_agent_wrapper(self) -> BaseAgent:
        """
        Create agent that routes through CallController.
        
        Returns:
            Agent wrapper
        """
        # Create base agent
        base_agent = self.agent_factory.create_agent(self.agent_config)
        
        # Wrap it to intercept and route through controller
        class ControllerRoutingAgent(BaseAgent):
            """
            Agent wrapper that routes all requests through CallController.
            
            This ensures CallController maintains control over all AI decisions.
            """
            
            def __init__(self, base_agent: BaseAgent, controller: CallController):
                super().__init__(base_agent.get_agent_config())
                self.base_agent = base_agent
                self.controller = controller
            
            async def respond(
                self,
                agent_input: AgentInput,
                conversation_id: str,
                is_interrupt: bool = False
            ) -> AsyncGenerator[AgentResponse, None]:
                """
                Intercept agent response and route through controller.
                
                Args:
                    agent_input: Input from user
                    conversation_id: Conversation ID
                    is_interrupt: Whether this is an interruption
                    
                Yields:
                    Agent responses
                """
                # Extract transcript
                if hasattr(agent_input, 'transcription'):
                    transcript = agent_input.transcription.message
                    confidence = agent_input.transcription.confidence
                    is_final = agent_input.transcription.is_final
                    
                    # Route through controller
                    result = await self.controller.handle_user_speech(
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
                        yield AgentResponseStop()
                else:
                    # Fallback to base agent
                    async for response in self.base_agent.respond(
                        agent_input,
                        conversation_id,
                        is_interrupt
                    ):
                        yield response
            
            def get_agent_config(self):
                return self.base_agent.get_agent_config()
        
        return ControllerRoutingAgent(base_agent, self.controller)
    
    def _setup_event_handlers(self) -> None:
        """Setup Vocode event handlers."""
        if not self.conversation:
            return
        
        # Hook into transcript updates
        original_add_human = self.conversation.transcript.add_human_message
        
        def wrapped_add_human(*args, **kwargs):
            # Call original
            result = original_add_human(*args, **kwargs)
            
            # Notify controller
            text = kwargs.get('text') or (args[0] if args else '')
            logger.debug(
                f"Human message added to transcript: {text}",
                extra={"call_id": self.controller.call_id}
            )
            
            return result
        
        self.conversation.transcript.add_human_message = wrapped_add_human
        
        # Hook into bot message completion
        original_add_bot = self.conversation.transcript.add_bot_message
        
        def wrapped_add_bot(*args, **kwargs):
            # Call original
            result = original_add_bot(*args, **kwargs)
            
            # Check if message is final
            is_final = kwargs.get('is_final', True)
            if is_final:
                # Notify controller that speech is complete
                asyncio.create_task(
                    self.controller.handle_speech_complete()
                )
            
            return result
        
        self.conversation.transcript.add_bot_message = wrapped_add_bot
    
    async def handle_interrupt(self) -> None:
        """
        Handle user interruption (barge-in).
        
        Called when user speaks while bot is speaking.
        """
        if not self.conversation:
            return
        
        logger.info(
            "Handling interrupt",
            extra={"call_id": self.controller.call_id}
        )
        
        # Vocode handles the actual interruption
        # We just need to ensure controller state is consistent
        
        # Mark user activity in watchdog
        self.controller.watchdog.mark_user_activity()
    
    async def stop(self) -> None:
        """Stop Vocode session."""
        if self._closed:
            return
        
        logger.info(
            "Stopping Vocode session",
            extra={"call_id": self.controller.call_id}
        )
        
        try:
            # Stop conversation
            if self.conversation:
                await self.conversation.terminate()
            
            self._closed = True
            
            logger.info(
                "Vocode session stopped",
                extra={"call_id": self.controller.call_id}
            )
        
        except Exception as e:
            logger.error(
                f"Error stopping Vocode session: {e}",
                extra={"call_id": self.controller.call_id},
                exc_info=True
            )
    
    async def send_message(self, text: str) -> None:
        """
        Send bot message to user.
        
        Args:
            text: Message text
        """
        if not self.conversation:
            logger.warning(
                "Cannot send message - no conversation",
                extra={"call_id": self.controller.call_id}
            )
            return
        
        # Add to transcript and synthesize
        # This will be handled by Vocode's normal flow
        logger.debug(
            f"Sending message: {text}",
            extra={"call_id": self.controller.call_id}
        )
    
    def is_active(self) -> bool:
        """Check if session is active."""
        return self._started and not self._closed


async def create_and_run_session(
    controller: CallController,
    websocket: WebSocket,
    twilio_config: TwilioConfig,
    transcriber_config: TranscriberConfig,
    synthesizer_config: SynthesizerConfig,
    agent_config: AgentConfig,
    transcriber_factory,
    synthesizer_factory,
    agent_factory
) -> None:
    """
    Create and run Vocode session.
    
    This is a helper function that manages the full lifecycle.
    
    Args:
        controller: CallController
        websocket: Twilio WebSocket
        twilio_config: Twilio config
        transcriber_config: STT config
        synthesizer_config: TTS config
        agent_config: Agent config
        transcriber_factory: STT factory
        synthesizer_factory: TTS factory
        agent_factory: Agent factory
    """
    session = VocodeSession(
        controller=controller,
        websocket=websocket,
        twilio_config=twilio_config,
        transcriber_config=transcriber_config,
        synthesizer_config=synthesizer_config,
        agent_config=agent_config,
        transcriber_factory=transcriber_factory,
        synthesizer_factory=synthesizer_factory,
        agent_factory=agent_factory
    )
    
    try:
        # Start session
        await session.start()
        
        # Wait for controller to signal shutdown
        shutdown_reason = await controller.session.wait_for_shutdown()
        
        logger.info(
            f"Session shutdown signaled: {shutdown_reason}",
            extra={"call_id": controller.call_id}
        )
    
    except Exception as e:
        logger.error(
            f"Session error: {e}",
            extra={"call_id": controller.call_id},
            exc_info=True
        )
    
    finally:
        # Cleanup
        await session.stop()
        await controller.cleanup()


class VocodeSessionManager:
    """
    Manages multiple Vocode sessions.
    
    Tracks active sessions and provides lifecycle management.
    """
    
    def __init__(self):
        self._sessions: Dict[str, VocodeSession] = {}
        self._lock = asyncio.Lock()
    
    async def create_session(
        self,
        controller: CallController,
        **kwargs
    ) -> VocodeSession:
        """
        Create and register a new session.
        
        Args:
            controller: CallController
            **kwargs: Session arguments
            
        Returns:
            Created session
        """
        session = VocodeSession(controller=controller, **kwargs)
        
        async with self._lock:
            self._sessions[controller.call_id] = session
        
        logger.info(
            f"Session created and registered: {controller.call_id}",
            extra={"total_sessions": len(self._sessions)}
        )
        
        return session
    
    async def get_session(self, call_id: str) -> Optional[VocodeSession]:
        """Get session by call ID."""
        async with self._lock:
            return self._sessions.get(call_id)
    
    async def remove_session(self, call_id: str) -> None:
        """Remove session from registry."""
        async with self._lock:
            if call_id in self._sessions:
                del self._sessions[call_id]
                logger.info(
                    f"Session removed: {call_id}",
                    extra={"total_sessions": len(self._sessions)}
                )
    
    async def stop_all(self) -> None:
        """Stop all active sessions."""
        async with self._lock:
            for session in list(self._sessions.values()):
                try:
                    await session.stop()
                except Exception as e:
                    logger.error(
                        f"Error stopping session: {e}",
                        exc_info=True
                    )
            
            self._sessions.clear()
