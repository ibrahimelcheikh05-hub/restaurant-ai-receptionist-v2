"""
Twilio Server (No Vocode)
==========================
Direct Twilio Media Streams implementation without vocode dependency.
"""

import asyncio
import logging
from typing import Dict, Optional, Callable
from fastapi import FastAPI, WebSocket, Form, WebSocketDisconnect
from fastapi.responses import Response

from audio_pipeline import AudioPipeline
from call_controller import CallController
from watchdog import WatchdogLimits

logger = logging.getLogger(__name__)


class TwilioServer:
    """FastAPI server for Twilio integration without vocode."""
    
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
        self.base_url = base_url
        self.deepgram_api_key = deepgram_api_key
        self.elevenlabs_api_key = elevenlabs_api_key
        self.elevenlabs_voice_id = elevenlabs_voice_id
        self.openai_api_key = openai_api_key
        self.handler_factory = handler_factory
        self.watchdog_limits = watchdog_limits or WatchdogLimits()
        
        self.calls: Dict[str, CallController] = {}
        self.audio_pipelines: Dict[str, AudioPipeline] = {}
        
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
            tenant_id = To.replace("+", "").replace("-", "")
            
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
                logger.error(f"WebSocket accept failed: {e}", exc_info=True)
                return
            
            controller = self.calls.get(call_id)
            if not controller:
                logger.error(f"No controller found for call {call_id}")
                try:
                    await websocket.close()
                except:
                    pass
                return
            
            logger.info(f"✓ Controller found for call {call_id}")
            
            try:
                logger.info(f"Creating audio pipeline for call {call_id}")
                
                async def on_transcript(text: str, is_final: bool):
                    if is_final and text.strip():
                        logger.info(f"🎤 User said: {text}")
                        result = await controller._call_handler(
                            "on_ai_request",
                            user_text=text,
                            turn_count=1
                        )
                        response_text = result.get("response_text", "")
                        if response_text:
                            logger.info(f"🤖 AI responding: {response_text}")
                            await audio_pipeline.speak(response_text)
                
                async def on_speaking_started():
                    logger.info(f"🔊 AI started speaking")
                
                async def on_speaking_finished():
                    logger.info(f"🔇 AI finished speaking")
                
                audio_pipeline = AudioPipeline(
                    call_id=call_id,
                    deepgram_api_key=self.deepgram_api_key,
                    elevenlabs_api_key=self.elevenlabs_api_key,
                    elevenlabs_voice_id=self.elevenlabs_voice_id,
                    on_transcript=on_transcript,
                    on_speaking_started=on_speaking_started,
                    on_speaking_finished=on_speaking_finished
                )
                
                self.audio_pipelines[call_id] = audio_pipeline
                logger.info(f"✓ Audio pipeline created")
                
                await controller.start()
                logger.info(f"✓ Call controller started")
                
                greeting_result = await controller._call_handler("on_greeting")
                greeting_text = greeting_result.get("greeting") if greeting_result else None
                
                if not greeting_text:
                    greeting_text = "Thank you for calling! How can I help you today?"
                    logger.warning(f"No greeting from handler, using default")
                
                logger.info(f"🎙️ Greeting: {greeting_text}")
                
                await audio_pipeline.speak(greeting_text)
                
                logger.info(f"Starting audio pipeline processing...")
                await audio_pipeline.start(websocket)
            
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for call {call_id}")
            except Exception as e:
                logger.error(f"Media stream error: {e}", exc_info=True)
            finally:
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
            return {
                "status": "healthy",
                "active_calls": len(self.calls)
            }
