"""
Vocode Server
=============
FastAPI server for Twilio telephony integration.

Responsibilities:
- Accept inbound Twilio calls
- Create CallController for each call
- Bridge Twilio WebSocket to CallController
- Handle call lifecycle via Vocode
- Maintain call registry

This is the HTTP/WebSocket entry point.
"""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, Request, Response, Form, HTTPException
from fastapi.responses import PlainTextResponse
import uvicorn

from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.telephony.config_manager.base_config_manager import BaseConfigManager
from vocode.streaming.telephony.templater import get_connection_twiml
from vocode.streaming.utils import create_conversation_id

from core.call_controller import CallController
from core.session import SessionConfig
from core.watchdog import WatchdogLimits

logger = logging.getLogger(__name__)


class CallRegistry:
    """Thread-safe registry of active calls."""
    
    def __init__(self):
        self._calls: Dict[str, CallController] = {}
        self._lock = asyncio.Lock()
    
    async def register(self, call_id: str, controller: CallController) -> None:
        """Register a call controller."""
        async with self._lock:
            self._calls[call_id] = controller
            logger.info(
                f"Call registered: {call_id}",
                extra={"total_calls": len(self._calls)}
            )
    
    async def unregister(self, call_id: str) -> None:
        """Unregister a call controller."""
        async with self._lock:
            if call_id in self._calls:
                del self._calls[call_id]
                logger.info(
                    f"Call unregistered: {call_id}",
                    extra={"total_calls": len(self._calls)}
                )
    
    async def get(self, call_id: str) -> Optional[CallController]:
        """Get call controller by ID."""
        async with self._lock:
            return self._calls.get(call_id)
    
    async def get_all_active(self) -> Dict[str, CallController]:
        """Get all active calls."""
        async with self._lock:
            return self._calls.copy()
    
    async def count(self) -> int:
        """Get count of active calls."""
        async with self._lock:
            return len(self._calls)


class VocodeServer:
    """
    Production telephony server.
    
    Handles:
    - Twilio inbound calls
    - WebSocket connections
    - Call lifecycle management
    - CallController creation
    """
    
    def __init__(
        self,
        base_url: str,
        config_manager: BaseConfigManager,
        twilio_config: Optional[TwilioConfig] = None,
        session_config: Optional[SessionConfig] = None,
        watchdog_limits: Optional[WatchdogLimits] = None,
        handler_factory: Optional[callable] = None
    ):
        """
        Initialize server.
        
        Args:
            base_url: Public base URL for webhooks
            config_manager: Vocode config manager
            twilio_config: Twilio configuration
            session_config: Default session config
            watchdog_limits: Default watchdog limits
            handler_factory: Factory function for creating event handlers
        """
        self.base_url = base_url
        self.config_manager = config_manager
        self.twilio_config = twilio_config
        self.session_config = session_config or SessionConfig()
        self.watchdog_limits = watchdog_limits or WatchdogLimits()
        self.handler_factory = handler_factory
        
        # Call registry
        self.registry = CallRegistry()
        
        # FastAPI app
        self.app = FastAPI(title="Vocode Telephony Server")
        
        # Setup routes
        self._setup_routes()
        
        logger.info(
            "VocodeServer initialized",
            extra={"base_url": base_url}
        )
    
    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""
        
        @self.app.post("/inbound/twilio")
        async def handle_inbound_twilio(
            CallSid: str = Form(...),
            From: str = Form(...),
            To: str = Form(...),
            AccountSid: str = Form(...)
        ) -> Response:
            """
            Handle inbound Twilio call.
            
            This is called by Twilio when a call comes in.
            Returns TwiML to connect WebSocket.
            """
            return await self._handle_inbound_call(
                call_sid=CallSid,
                from_phone=From,
                to_phone=To,
                account_sid=AccountSid
            )
        
        @self.app.websocket("/connect/{call_id}")
        async def websocket_endpoint(
            websocket: WebSocket,
            call_id: str
        ):
            """
            WebSocket endpoint for Twilio media stream.
            
            This receives the audio stream from Twilio.
            """
            await self._handle_websocket(websocket, call_id)
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            active_calls = await self.registry.count()
            return {
                "status": "healthy",
                "active_calls": active_calls,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        @self.app.get("/calls")
        async def list_calls():
            """List active calls."""
            calls = await self.registry.get_all_active()
            return {
                "count": len(calls),
                "calls": [
                    {
                        "call_id": call_id,
                        "status": controller.get_status()
                    }
                    for call_id, controller in calls.items()
                ]
            }
        
        @self.app.get("/calls/{call_id}")
        async def get_call_status(call_id: str):
            """Get status of specific call."""
            controller = await self.registry.get(call_id)
            if not controller:
                raise HTTPException(status_code=404, detail="Call not found")
            
            return controller.get_status()
        
        @self.app.post("/calls/{call_id}/hangup")
        async def hangup_call(call_id: str):
            """Manually hang up a call."""
            controller = await self.registry.get(call_id)
            if not controller:
                raise HTTPException(status_code=404, detail="Call not found")
            
            await controller.close(reason="manual_hangup")
            return {"status": "closing"}
    
    async def _handle_inbound_call(
        self,
        call_sid: str,
        from_phone: str,
        to_phone: str,
        account_sid: str
    ) -> Response:
        """
        Handle inbound Twilio call.
        
        Args:
            call_sid: Twilio call SID
            from_phone: Caller phone number
            to_phone: Destination phone number
            account_sid: Twilio account SID
            
        Returns:
            TwiML response
        """
        logger.info(
            "Inbound call received",
            extra={
                "call_sid": call_sid,
                "from": from_phone,
                "to": to_phone
            }
        )
        
        # Create conversation ID
        call_id = create_conversation_id()
        
        # Extract tenant_id from to_phone or use default
        # In production, map phone numbers to tenants
        tenant_id = self._extract_tenant_id(to_phone)
        
        # Create CallController
        controller = await self._create_call_controller(
            call_id=call_id,
            tenant_id=tenant_id,
            call_sid=call_sid,
            from_phone=from_phone,
            to_phone=to_phone
        )
        
        # Register controller
        await self.registry.register(call_id, controller)
        
        # Return TwiML to connect WebSocket
        twiml = get_connection_twiml(
            base_url=self.base_url,
            call_id=call_id
        )
        
        logger.info(
            "TwiML returned for call",
            extra={"call_id": call_id, "call_sid": call_sid}
        )
        
        return PlainTextResponse(
            content=str(twiml),
            media_type="application/xml"
        )
    
    async def _handle_websocket(
        self,
        websocket: WebSocket,
        call_id: str
    ) -> None:
        """
        Handle WebSocket connection from Twilio.
        
        Args:
            websocket: FastAPI WebSocket
            call_id: Call identifier
        """
        logger.info(
            "WebSocket connection received",
            extra={"call_id": call_id}
        )
        
        # Accept WebSocket
        await websocket.accept()
        
        # Get controller
        controller = await self.registry.get(call_id)
        if not controller:
            logger.error(
                "No controller found for WebSocket",
                extra={"call_id": call_id}
            )
            await websocket.close(code=1008, reason="No controller")
            return
        
        try:
            # This will be implemented in vocode_session.py
            # which creates the actual Vocode StreamingConversation
            # and bridges it to the controller
            
            # For now, placeholder
            logger.info(
                "WebSocket connected, awaiting session implementation",
                extra={"call_id": call_id}
            )
            
            # Keep connection alive
            while True:
                try:
                    message = await websocket.receive_json()
                    # Process message (will be handled by vocode_session)
                    logger.debug(
                        "WebSocket message received",
                        extra={"call_id": call_id}
                    )
                except Exception as e:
                    logger.error(
                        f"WebSocket error: {e}",
                        extra={"call_id": call_id},
                        exc_info=True
                    )
                    break
        
        except Exception as e:
            logger.error(
                f"WebSocket handler error: {e}",
                extra={"call_id": call_id},
                exc_info=True
            )
        
        finally:
            # Cleanup
            logger.info(
                "WebSocket disconnected",
                extra={"call_id": call_id}
            )
            
            # Close call
            if controller:
                await controller.close(reason="websocket_closed")
            
            # Unregister
            await self.registry.unregister(call_id)
    
    async def _create_call_controller(
        self,
        call_id: str,
        tenant_id: str,
        call_sid: str,
        from_phone: str,
        to_phone: str
    ) -> CallController:
        """
        Create CallController for new call.
        
        Args:
            call_id: Call identifier
            tenant_id: Tenant identifier
            call_sid: Twilio call SID
            from_phone: Caller phone
            to_phone: Destination phone
            
        Returns:
            Created CallController
        """
        # Create event handlers
        handlers = {}
        if self.handler_factory:
            handlers = self.handler_factory(
                call_id=call_id,
                tenant_id=tenant_id
            )
        
        # Create session config
        session_config = SessionConfig(
            max_call_duration=self.session_config.max_call_duration,
            max_silence_duration=self.session_config.max_silence_duration,
            max_ai_response_time=self.session_config.max_ai_response_time,
            enable_barge_in=self.session_config.enable_barge_in,
            enable_language_detection=self.session_config.enable_language_detection,
            enable_transfer=self.session_config.enable_transfer
        )
        
        # Create controller
        controller = CallController(
            call_id=call_id,
            tenant_id=tenant_id,
            session_config=session_config,
            watchdog_limits=self.watchdog_limits,
            handlers=handlers
        )
        
        # Set metadata
        controller.session.metadata.twilio_call_sid = call_sid
        controller.session.metadata.from_number = from_phone
        controller.session.metadata.to_number = to_phone
        
        logger.info(
            "CallController created",
            extra={
                "call_id": call_id,
                "tenant_id": tenant_id,
                "call_sid": call_sid
            }
        )
        
        return controller
    
    def _extract_tenant_id(self, to_phone: str) -> str:
        """
        Extract tenant ID from destination phone number.
        
        In production, this would lookup a database mapping
        phone numbers to tenants/restaurants.
        
        Args:
            to_phone: Destination phone number
            
        Returns:
            Tenant identifier
        """
        # For now, return default
        # In production, query database:
        # SELECT tenant_id FROM phone_numbers WHERE number = to_phone
        return "default_tenant"
    
    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        **kwargs
    ) -> None:
        """
        Run the server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            **kwargs: Additional uvicorn arguments
        """
        logger.info(
            f"Starting VocodeServer on {host}:{port}",
            extra={"base_url": self.base_url}
        )
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            **kwargs
        )


def create_server(
    base_url: str,
    config_manager: BaseConfigManager,
    **kwargs
) -> VocodeServer:
    """
    Factory function to create VocodeServer.
    
    Args:
        base_url: Public base URL
        config_manager: Vocode config manager
        **kwargs: Additional server arguments
        
    Returns:
        Configured VocodeServer
    """
    return VocodeServer(
        base_url=base_url,
        config_manager=config_manager,
        **kwargs
    )
