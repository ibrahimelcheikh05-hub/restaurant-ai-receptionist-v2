"""
Audio Pipeline
===============
Direct Twilio Media Streams integration with Deepgram (STT) and ElevenLabs (TTS).

Features:
- Low latency streaming
- Barge-in support (interrupt AI mid-sentence)
- Direct WebSocket audio processing
- No vocode dependency

Architecture:
Twilio → WebSocket → Deepgram STT → Your Handlers → OpenAI → ElevenLabs TTS → Twilio
"""

import asyncio
import base64
import json
import logging
from typing import Optional, Dict, Callable
from datetime import datetime

import aiohttp
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class AudioPipeline:
    """
    Manages audio streaming between Twilio, Deepgram, and ElevenLabs.
    
    Features:
    - Bidirectional audio streaming
    - Barge-in detection
    - Low latency optimization
    """
    
    def __init__(
        self,
        call_id: str,
        deepgram_api_key: str,
        elevenlabs_api_key: str,
        elevenlabs_voice_id: str,
        on_transcript: Callable,
        on_speaking_started: Callable = None,
        on_speaking_finished: Callable = None
    ):
        """
        Initialize audio pipeline.
        
        Args:
            call_id: Call identifier
            deepgram_api_key: Deepgram API key
            elevenlabs_api_key: ElevenLabs API key
            elevenlabs_voice_id: ElevenLabs voice ID
            on_transcript: Callback when speech is transcribed
            on_speaking_started: Callback when AI starts speaking
            on_speaking_finished: Callback when AI finishes speaking
        """
        self.call_id = call_id
        self.deepgram_api_key = deepgram_api_key
        self.elevenlabs_api_key = elevenlabs_api_key
        self.elevenlabs_voice_id = elevenlabs_voice_id
        
        self.on_transcript = on_transcript
        self.on_speaking_started = on_speaking_started
        self.on_speaking_finished = on_speaking_finished
        
        # State
        self.is_speaking = False
        self.barge_in_detected = False
        self.deepgram_ws = None
        self.twilio_ws = None
        self.stream_sid = None  # Will be set from Twilio start event
        
        # Buffers
        self.audio_queue = asyncio.Queue()
        self.tts_queue = asyncio.Queue()
        
        logger.info(f"AudioPipeline initialized for call {call_id}")
    
    async def start(self, twilio_ws: WebSocket):
        """
        Start the audio pipeline.
        
        Args:
            twilio_ws: Twilio WebSocket connection
        """
        self.twilio_ws = twilio_ws
        
        # Start concurrent tasks
        tasks = [
            asyncio.create_task(self._deepgram_stream()),
            asyncio.create_task(self._process_twilio_messages()),
            asyncio.create_task(self._send_audio_to_twilio())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Audio pipeline error: {e}", exc_info=True)
        finally:
            await self.stop()
    
    async def _deepgram_stream(self):
        """Connect to Deepgram for real-time transcription."""
        url = "wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000&channels=1&punctuate=true&interim_results=true&endpointing=300&utterance_end_ms=1000"
        
        headers = {
            "Authorization": f"Token {self.deepgram_api_key}"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(url, headers=headers) as ws:
                    self.deepgram_ws = ws
                    logger.info(f"Deepgram connected for call {self.call_id}")
                    
                    # Send audio and receive transcripts concurrently
                    send_task = asyncio.create_task(self._send_audio_to_deepgram(ws))
                    receive_task = asyncio.create_task(self._receive_from_deepgram(ws))
                    
                    await asyncio.gather(send_task, receive_task)
        
        except Exception as e:
            logger.error(f"Deepgram error: {e}", exc_info=True)
    
    async def _send_audio_to_deepgram(self, ws):
        """Send audio from Twilio to Deepgram."""
        try:
            while True:
                audio_data = await self.audio_queue.get()
                
                if audio_data is None:  # Stop signal
                    break
                
                await ws.send_bytes(audio_data)
        
        except Exception as e:
            logger.error(f"Error sending to Deepgram: {e}")
    
    async def _receive_from_deepgram(self, ws):
        """Receive transcripts from Deepgram."""
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    
                    # Check if we have a transcript
                    if "channel" in data:
                        alternatives = data["channel"]["alternatives"]
                        if alternatives:
                            transcript = alternatives[0]["transcript"]
                            is_final = data.get("is_final", False)
                            
                            if transcript.strip():
                                # Detect barge-in
                                if self.is_speaking and is_final:
                                    logger.info(f"Barge-in detected: {transcript}")
                                    self.barge_in_detected = True
                                    await self._stop_speaking()
                                
                                # Call transcript callback
                                if is_final:
                                    logger.info(f"Final transcript: {transcript}")
                                    await self.on_transcript(transcript, is_final=True)
                                else:
                                    await self.on_transcript(transcript, is_final=False)
        
        except Exception as e:
            logger.error(f"Error receiving from Deepgram: {e}")
    
    async def _process_twilio_messages(self):
        """Process incoming messages from Twilio."""
        try:
            while True:
                message = await self.twilio_ws.receive_json()
                event_type = message.get("event")
                
                if event_type == "media":
                    # Audio from caller
                    payload = message["media"]["payload"]
                    audio_data = base64.b64decode(payload)
                    
                    # Send to Deepgram
                    await self.audio_queue.put(audio_data)
                
                elif event_type == "start":
                    # Store the stream SID for sending audio back
                    self.stream_sid = message.get("start", {}).get("streamSid")
                    logger.info(f"Twilio stream started for call {self.call_id}, stream_sid: {self.stream_sid}")
                
                elif event_type == "stop":
                    logger.info(f"Twilio stream stopped for call {self.call_id}")
                    break
        
        except Exception as e:
            logger.error(f"Error processing Twilio messages: {e}")
    
    async def _send_audio_to_twilio(self):
        """Send TTS audio back to Twilio."""
        try:
            while True:
                audio_chunk = await self.tts_queue.get()
                
                if audio_chunk is None:  # Stop signal
                    break
                
                # Send to Twilio
                message = {
                    "event": "media",
                    "streamSid": self.stream_sid or self.call_id,  # Use captured stream_sid
                    "media": {
                        "payload": base64.b64encode(audio_chunk).decode()
                    }
                }
                
                await self.twilio_ws.send_json(message)
        
        except Exception as e:
            logger.error(f"Error sending to Twilio: {e}")
    
    async def speak(self, text: str):
        """
        Convert text to speech and send to caller.
        
        Args:
            text: Text to speak
        """
        if not text.strip():
            return
        
        self.is_speaking = True
        self.barge_in_detected = False
        
        if self.on_speaking_started:
            await self.on_speaking_started()
        
        logger.info(f"Speaking: {text}")
        
        try:
            # Call ElevenLabs streaming API
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.elevenlabs_voice_id}/stream"
            
            headers = {
                "Accept": "audio/mpeg",  # ElevenLabs returns MP3
                "Content-Type": "application/json",
                "xi-api-key": self.elevenlabs_api_key
            }
            
            data = {
                "text": text,
                "model_id": "eleven_turbo_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                },
                "output_format": "ulaw_8000"  # Request mulaw format directly!
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        # Stream audio chunks
                        async for chunk in response.content.iter_chunked(1024):
                            if self.barge_in_detected:
                                logger.info("Barge-in detected, stopping speech")
                                break
                            
                            # Convert MP3 to mulaw if needed, or send directly
                            # For simplicity, sending as-is (Twilio handles MP3)
                            await self.tts_queue.put(chunk)
                    
                    else:
                        logger.error(f"ElevenLabs error: {response.status}")
        
        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
        
        finally:
            self.is_speaking = False
            if self.on_speaking_finished:
                await self.on_speaking_finished()
    
    async def _stop_speaking(self):
        """Stop current speech (for barge-in)."""
        self.is_speaking = False
        # Clear TTS queue
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
    
    async def stop(self):
        """Stop the audio pipeline."""
        logger.info(f"Stopping audio pipeline for call {self.call_id}")
        
        # Signal stop
        await self.audio_queue.put(None)
        await self.tts_queue.put(None)
        
        # Close Deepgram connection
        if self.deepgram_ws:
            await self.deepgram_ws.close()
