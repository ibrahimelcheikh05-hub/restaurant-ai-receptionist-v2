"""
Twilio Server (No Vocode)
==========================
Direct Twilio Media Streams implementation without vocode dependency.

Features:
- Twilio inbound call handling
- WebSocket media streaming
- Direct Deepgram + ElevenLabs integration
- Your custom event handlers
- Low latency
- Barge-in support
"""

import asyncio
import logging
from typing import Dict, Optional, Callable
from fastapi import FastAPI, WebSocket, Request, Form, WebSocketDisconnect
from fastapi.responses import Response
import uvicorn

from audio_pipeline import AudioPipeline
from call_controller import CallController
from watchdog import WatchdogLimits

logger = logging.getLogger(__name__)


class TwilioServer:
    """
    FastAPI server for Twilio integration without vocode.
    
    Handles:
    - Inbound call webhooks
    - WebSocket media streams
    - Audio pipeline management
    - Call lifecycle
    """
    
    def __init__(
        self,
        base_url: str,
        deepgram_api_key: str,
        elevenlabs_api_key: str,
        elevenlabs_voice_id: str,
        openai_api_key: str,
        handler_factory: Callable,
        watchdog_limits: Optional[WatchdogLimits] = None
    ):
        """
        Initialize server.
        
        Args:
            base_url: Public base URL for webhooks
            deepgram_api_key: Deepgram API key
            elevenlabs_api_key: ElevenLabs API key
            elevenlabs_voice_id: ElevenLabs voice ID
            openai_api_key: OpenAI API key
            handler_factory: Factory for creating event handlers
            watchdog_limits: Watchdog limits
        """
        self.base_url = base_url
        self.deepgram_api_key = deepgram_api_key
        self.elevenlabs_api_key = elevenlabs_api_key
        self.elevenlabs_voice_id = elevenlabs_voice_id
        self.openai_api_key = openai_api_key
        self.handler_factory = handler_factory
        self.watchdog_limits = watchdog_limits or WatchdogLimits()
        
        # Active calls
        self.calls: Dict[str, CallController] = {}
        self.audio_pipelines: Dict[str, AudioPipeline] = {}
        
        # Create FastAPI app
        self.app = FastAPI(title="Restaurant AI Receptionist")
        
        self._setup_routes()
        
        logger.info("TwilioServer initialized")
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.post("/inbound/twilio")
        async def handle_inbound_call(
            CallSid: str = Form(...),
            From: str = Form(...),
            To: str = Form(...)
        ):
            """Handle inbound Twilio call."""
            logger.info(f"Inbound call: {CallSid} from {From}")
            
            call_id = CallSid
            tenant_id = self._extract_tenant_id(To)
            
            # Create call controller
            handlers = self.handler_factory(
                call_id=call_id,
                tenant_id=tenant_id,
                call_sid=CallSid,
                from_phone=From,
                to_phone=To
            )
            
            controller = CallController(
                call_id=call_id,
                tenant_id=tenant_id,
                watchdog_limits=self.watchdog_limits,
                handlers=handlers
            )
            
            self.calls[call_id] = controller
            
            logger.info(f"CallController created for {call_id}")
            
            # Return TwiML to connect WebSocket
            ws_url = f"{self.base_url.replace('https://', 'wss://').replace('http://', 'ws://')}/media/{call_id}"
            
            twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}" />
    </Connect>
</Response>'''
            
            logger.info(f"TwiML returned: {ws_url}")
            
            return Response(content=twiml, media_type="application/xml")
        
        @self.app.websocket("/media/{call_id}")
        async def handle_media_stream(websocket: WebSocket, call_id: str):
            """Handle Twilio media stream WebSocket."""
            logger.info(f"WebSocket connection attempt for call {call_id}")
            
            try:
                await websocket.accept()
                logger.info(f"✓ WebSocket accepted for call {call_id}")
            except Exception as e:
                logger.error(f"✗ WebSocket accept failed: {e}", exc_info=True)
                return
            
            controller = self.calls.get(call_id)
            if not controller:
                logger.error(f"✗ No controller found for call {call_id}")
                try:
                    await websocket.close()
                except:
                    pass
                return
            
            logger.info(f"✓ Controller found for call {call_id}")
            
            try:
                # Test: Just log messages for now to see what we're receiving
                logger.info(f"Starting WebSocket message loop for call {call_id}")
                message_count = 0
                
                while True:
                    try:
                        message = await websocket.receive_json()
                        message_count += 1
                        
                        event_type = message.get("event", "unknown")
                        
                        if message_count <= 5:  # Log first 5 messages
                            logger.info(f"Message #{message_count}: event={event_type}, keys={list(message.keys())}")
                        
                        if event_type == "start":
                            logger.info(f"✓ Twilio stream STARTED for call {call_id}")
                            logger.info(f"Stream info: {message.get('start', {})}")
                        
                        elif event_type == "media":
                            if message_count == 6:  # Log only once after first 5
                                logger.info(f"✓ Receiving media packets (will log every 100th)")
                            if message_count % 100 == 0:
                                logger.info(f"Received {message_count} messages so far...")
                        
                        elif event_type == "stop":
                            logger.info(f"✓ Twilio stream STOPPED for call {call_id}")
                            break
                        
                        else:
                            logger.info(f"Unknown event type: {event_type}")
                    
                    except Exception as e:
                        logger.error(f"Error processing message: {e}", exc_info=True)
                        break
                
                logger.info(f"WebSocket loop ended for call {call_id}, total messages: {message_count}")
                logger.info(f"WebSocket loop ended for call {call_id}, total messages: {message_count}")
            
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for call {call_id}")
            
            except Exception as e:
                logger.error(f"Media stream error: {e}", exc_info=True)
            
            finally:
                # Cleanup
                if call_id in self.audio_pipelines:
                    try:
                        await self.audio_pipelines[call_id].stop()
                    except:
                        pass
                    del self.audio_pipelines[call_id]
                
                if call_id in self.calls:
                    try:
                        await self.calls[call_id].stop()
                    except:
                        pass
                    del self.calls[call_id]
                
                logger.info(f"Call {call_id} cleaned up")
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "active_calls": len(self.calls),
                "websocket_endpoint": f"{self.base_url}/media/{{call_id}}"
            }
        
        @self.app.get("/test-ws/{call_id}")
        async def test_websocket(call_id: str):
            """Test if WebSocket endpoint is accessible."""
            ws_url = f"{self.base_url.replace('https://', 'wss://').replace('http://', 'ws://')}/media/{call_id}"
            return {
                "websocket_url": ws_url,
                "message": "WebSocket endpoint ready"
            }
    
    def _extract_tenant_id(self, to_phone: str) -> str:
        """Extract tenant ID from phone number."""
        # Simple implementation - just use phone as tenant
        return to_phone.replace("+", "").replace("-", "")
