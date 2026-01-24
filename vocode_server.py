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
- ENFORCE concurrency limits
- ENFORCE resource limits
- REJECT overload conditions
- Graceful shutdown with forced termination

This is the HTTP/WebSocket entry point.
"""

import asyncio
import logging
from typing import Dict, Optional, Callable, Any
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, Request, Response, Form, HTTPException
from fastapi.responses import PlainTextResponse
import uvicorn

from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.telephony.config_manager.base_config_manager import BaseConfigManager
from vocode.streaming.telephony.templater import get_connection_twiml
from vocode.streaming.utils import create_conversation_id

from call_controller import CallController
from watchdog import WatchdogLimits

logger = logging.getLogger(__name__)


class CallRegistry:
    """
    Thread-safe registry of active calls with resource limits.
    
    Features:
    - Atomic registration/unregistration
    - Concurrent call limits
    - Safe iteration
    - Cleanup enforcement
    - Admission control
    """
    
    def __init__(self, max_concurrent_calls: int = 100):
        """
        Initialize call registry.
        
        Args:
            max_concurrent_calls: Maximum concurrent calls allowed
        """
        self._calls: Dict[str, CallController] = {}
        self._lock = asyncio.Lock()
        self.max_concurrent_calls = max_concurrent_calls
        
        # Metrics
        self._total_registered = 0
        self._total_rejected = 0
        
        logger.info(
            "CallRegistry initialized",
            extra={"max_concurrent": max_concurrent_calls}
        )
    
    async def register(
        self,
        call_id: str,
        controller: CallController
    ) -> bool:
        """
        Register a call controller with admission control.
        
        Args:
            call_id: Call identifier
            controller: Call controller instance
            
        Returns:
            True if registered, False if rejected (over limit)
        """
        async with self._lock:
            # Check limit
            if len(self._calls) >= self.max_concurrent_calls:
                self._total_rejected += 1
                logger.error(
                    "Max concurrent calls reached - rejecting call",
                    extra={
                        "call_id": call_id,
                        "current": len(self._calls),
                        "max": self.max_concurrent_calls,
                        "total_rejected": self._total_rejected
                    }
                )
                return False
            
            # Check for duplicate
            if call_id in self._calls:
                logger.warning(
                    "Call ID already registered",
                    extra={"call_id": call_id}
                )
                return False
            
            # Register
            self._calls[call_id] = controller
            self._total_registered += 1
            logger.info(
                f"Call registered: {call_id}",
                extra={
                    "call_id": call_id,
                    "total_calls": len(self._calls),
                    "total_registered": self._total_registered
                }
            )
            return True
    
    async def unregister(self, call_id: str) -> Optional[CallController]:
        """
        Unregister a call controller.
        
        Args:
            call_id: Call identifier
            
        Returns:
            Controller if it was registered, None otherwise
        """
        async with self._lock:
            controller = self._calls.pop(call_id, None)
            if controller:
                logger.info(
                    f"Call unregistered: {call_id}",
                    extra={
                        "call_id": call_id,
                        "total_calls": len(self._calls)
                    }
                )
            return controller
    
    async def get(self, call_id: str) -> Optional[CallController]:
        """Get call controller by ID."""
        async with self._lock:
            return self._calls.get(call_id)
    
    async def get_all_active(self) -> Dict[str, CallController]:
        """Get all active calls (copy)."""
        async with self._lock:
            return self._calls.copy()
    
    async def count(self) -> int:
        """Get count of active calls."""
        async with self._lock:
            return len(self._calls)
    
    async def is_at_capacity(self) -> bool:
        """Check if at maximum capacity."""
        async with self._lock:
            return len(self._calls) >= self.max_concurrent_calls
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get registry metrics."""
        async with self._lock:
            return {
                "active_calls": len(self._calls),
                "max_concurrent": self.max_concurrent_calls,
                "total_registered": self._total_registered,
                "total_rejected": self._total_rejected,
                "capacity_utilization": len(self._calls) / self.max_concurrent_calls if self.max_concurrent_calls > 0 else 0.0
            }
    
    async def cleanup_all(self) -> None:
        """
        Cleanup all registered calls.
        
        Used during server shutdown.
        """
        logger.warning(
            "Cleaning up all registered calls",
            extra={"count": len(self._calls)}
        )
        
        async with self._lock:
            controllers = list(self._calls.values())
            call_ids = list(self._calls.keys())
            self._calls.clear()
        
        # Terminate all calls concurrently with timeout
        cleanup_tasks = []
        for controller in controllers:
            cleanup_tasks.append(controller.terminate_call("server_shutdown"))
        
        if cleanup_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Some calls did not cleanup within timeout - forcing",
                    extra={"remaining": len([c for c in controllers if not c._termination_complete])}
                )
                
                # Force terminate remaining calls
                force_tasks = []
                for controller in controllers:
                    if not controller._termination_complete:
                        force_tasks.append(
                            controller.terminate_call(
                                "server_shutdown_forced",
                                error="shutdown_timeout"
                            )
                        )
                
                if force_tasks:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*force_tasks, return_exceptions=True),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        logger.error("Force termination also timed out")
        
        logger.warning(
            "All calls cleaned up",
            extra={"cleaned": len(call_ids)}
        )


class VocodeServer:
    """
    Production telephony server.
    
    Handles:
    - Twilio inbound calls
    - WebSocket connections
    - Call lifecycle management
    - CallController creation
    - Resource limits enforcement
    - Graceful shutdown
    - Admission control
    """
    
    def __init__(
        self,
        base_url: str,
        config_manager: BaseConfigManager,
        twilio_config: Optional[TwilioConfig] = None,
        watchdog_limits: Optional[WatchdogLimits] = None,
        handler_factory: Optional[Callable] = None,
        max_concurrent_calls: int = 100
    ):
        """
        Initialize server.
        
        Args:
            base_url: Public base URL for webhooks
            config_manager: Vocode config manager
            twilio_config: Twilio configuration
            watchdog_limits: Default watchdog limits
            handler_factory: Factory function for creating event handlers
            max_concurrent_calls: Maximum concurrent calls
        """
        self.base_url = base_url
        self.config_manager = config_manager
        self.twilio_config = twilio_config
        self.watchdog_limits = watchdog_limits or WatchdogLimits()
        self.handler_factory = handler_factory
        self.max_concurrent_calls = max_concurrent_calls
        
        # Call registry with limits
        self.registry = CallRegistry(max_concurrent_calls=max_concurrent_calls)
        
        # Shutdown management
        self._shutdown_event = asyncio.Event()
        self._is_shutting_down = False
        self._shutdown_complete = False
        
        # FastAPI app
        self.app = FastAPI(
            title="Vocode Telephony Server",
            lifespan=self._lifespan
        )
        
        # Setup routes
        self._setup_routes()
        
        logger.info(
            "VocodeServer initialized",
            extra={
                "base_url": base_url,
                "max_concurrent_calls": max_concurrent_calls
            }
        )
    
    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """
        Lifespan context manager for startup/shutdown.
        
        Args:
            app: FastAPI application
        """
        # Startup
        logger.info("Server starting up")
        yield
        
        # Shutdown
        logger.warning("Server shutting down")
        self._is_shutting_down = True
        self._shutdown_event.set()
        
        # Cleanup all active calls
        await self.registry.cleanup_all()
        
        self._shutdown_complete = True
        logger.warning("Server shutdown complete")
    
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
            # Reject if shutting down
            if self._is_shutting_down:
                logger.warning(
                    "Rejecting call - server shutting down",
                    extra={"call_sid": CallSid}
                )
                return PlainTextResponse(
                    content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>Service temporarily unavailable</Say><Hangup/></Response>',
                    media_type="application/xml"
                )
            
            # Check capacity
            if await self.registry.is_at_capacity():
                logger.error(
                    "Rejecting call - at capacity",
                    extra={
                        "call_sid": CallSid,
                        "active_calls": await self.registry.count(),
                        "max": self.max_concurrent_calls
                    }
                )
                return PlainTextResponse(
                    content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>All agents are busy. Please try again later.</Say><Hangup/></Response>',
                    media_type="application/xml"
                )
            
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
            # Reject if shutting down
            if self._is_shutting_down:
                logger.warning(
                    "Rejecting WebSocket - server shutting down",
                    extra={"call_id": call_id}
                )
                await websocket.close(code=1001, reason="Server shutting down")
                return
            
            await self._handle_websocket(websocket, call_id)
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            active_calls = await self.registry.count()
            at_capacity = await self.registry.is_at_capacity()
            
            return {
                "status": "healthy" if not self._is_shutting_down else "shutting_down",
                "active_calls": active_calls,
                "max_calls": self.max_concurrent_calls,
                "at_capacity": at_capacity,
                "shutdown_complete": self._shutdown_complete,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        @self.app.get("/metrics")
        async def metrics():
            """Metrics endpoint."""
            registry_metrics = await self.registry.get_metrics()
            
            return {
                **registry_metrics,
                "is_shutting_down": self._is_shutting_down,
                "shutdown_complete": self._shutdown_complete,
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
            
            # Don't await - let it happen in background
            asyncio.create_task(controller.close(reason="manual_hangup"))
            return {"status": "closing"}
        
        @self.app.post("/shutdown")
        async def trigger_shutdown():
            """Trigger graceful shutdown (admin endpoint)."""
            if self._is_shutting_down:
                return {"status": "already_shutting_down"}
            
            logger.warning("Shutdown triggered via API")
            self._is_shutting_down = True
            self._shutdown_event.set()
            
            # Cleanup in background
            asyncio.create_task(self.registry.cleanup_all())
            
            return {"status": "shutdown_initiated"}
    
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
        
        try:
            # Create conversation ID
            call_id = create_conversation_id()
            
            # Extract tenant_id from to_phone or use default
            tenant_id = self._extract_tenant_id(to_phone)
            
            # Create CallController
            controller = await self._create_call_controller(
                call_id=call_id,
                tenant_id=tenant_id,
                call_sid=call_sid,
                from_phone=from_phone,
                to_phone=to_phone
            )
            
            # Register controller (with admission control)
            registered = await self.registry.register(call_id, controller)
            
            if not registered:
                logger.error(
                    "Failed to register call - capacity exceeded",
                    extra={"call_id": call_id}
                )
                # Clean up controller since not registered
                try:
                    await controller.cleanup()
                except Exception as e:
                    logger.error(
                        f"Error cleaning up unregistered controller: {e}",
                        extra={"call_id": call_id}
                    )
                
                return PlainTextResponse(
                    content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>All agents are busy. Please try again later.</Say><Hangup/></Response>',
                    media_type="application/xml"
                )
            
            # Return TwiML to connect WebSocket
            # Generate TwiML manually to avoid vocode's broken get_connection_twiml
            
            # Convert BASE_URL to WebSocket URL
            base_url_fixed = self.base_url
            if base_url_fixed.startswith('http://'):
                ws_base = base_url_fixed.replace('http://', 'ws://')
            elif base_url_fixed.startswith('https://'):
                ws_base = base_url_fixed.replace('https://', 'wss://')
            else:
                # BASE_URL doesn't have protocol - add wss://
                ws_base = f'wss://{base_url_fixed}'
            
            ws_url = f"{ws_base}/connect/{call_id}"
            
            # Generate valid TwiML
            twiml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}">
            <Parameter name="call_id" value="{call_id}" />
            <Parameter name="call_sid" value="{call_sid}" />
        </Stream>
    </Connect>
</Response>'''
            
            logger.info(
                "TwiML returned for call",
                extra={
                    "call_id": call_id, 
                    "call_sid": call_sid,
                    "ws_url": ws_url,
                    "base_url": self.base_url
                }
            )
            
            return PlainTextResponse(
                content=twiml_content,
                media_type="application/xml"
            )
        
        except Exception as e:
            logger.error(
                f"Error handling inbound call: {e}",
                extra={"call_sid": call_sid},
                exc_info=True
            )
            
            # Return error TwiML
            return PlainTextResponse(
                content='<?xml version="1.0" encoding="UTF-8"?><Response><Say>An error occurred. Please try again later.</Say><Hangup/></Response>',
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
        
        controller = None
        
        try:
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
            
            # Start controller session
            try:
                await controller.start()
            except Exception as e:
                logger.error(
                    f"Error starting controller: {e}",
                    extra={"call_id": call_id},
                    exc_info=True
                )
                await websocket.close(code=1011, reason="Controller start failed")
                return
            
            # This will be fully implemented when integrating VocodeSession
            # For now, keep connection alive and handle basic lifecycle
            
            logger.info(
                "WebSocket connected and controller started",
                extra={"call_id": call_id}
            )
            
            # Keep connection alive until termination
            while not controller._termination_complete and not self._is_shutting_down:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(
                        websocket.receive_json(),
                        timeout=1.0
                    )
                    
                    # Process message (will be handled by VocodeSession)
                    logger.debug(
                        "WebSocket message received",
                        extra={"call_id": call_id}
                    )
                
                except asyncio.TimeoutError:
                    # Normal timeout, check if should continue
                    continue
                
                except Exception as e:
                    logger.error(
                        f"WebSocket receive error: {e}",
                        extra={"call_id": call_id},
                        exc_info=True
                    )
                    break
            
            logger.info(
                "WebSocket loop ended",
                extra={
                    "call_id": call_id,
                    "termination_complete": controller._termination_complete,
                    "shutting_down": self._is_shutting_down
                }
            )
        
        except Exception as e:
            logger.error(
                f"WebSocket handler error: {e}",
                extra={"call_id": call_id},
                exc_info=True
            )
        
        finally:
            # Cleanup
            logger.info(
                "WebSocket disconnected - cleaning up",
                extra={"call_id": call_id}
            )
            
            # Close call if not already terminated
            if controller and not controller._termination_complete:
                try:
                    await asyncio.wait_for(
                        controller.close(reason="websocket_closed"),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "Controller close timed out - forcing termination",
                        extra={"call_id": call_id}
                    )
                    # Force termination
                    try:
                        await asyncio.wait_for(
                            controller.terminate_call(
                                "websocket_close_timeout",
                                error="close_timeout"
                            ),
                            timeout=3.0
                        )
                    except asyncio.TimeoutError:
                        logger.error(
                            "Forced termination also timed out",
                            extra={"call_id": call_id}
                        )
                except Exception as e:
                    logger.error(
                        f"Error closing controller: {e}",
                        extra={"call_id": call_id},
                        exc_info=True
                    )
            
            # Cleanup controller
            if controller:
                try:
                    await asyncio.wait_for(
                        controller.cleanup(),
                        timeout=2.0
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "Controller cleanup timed out",
                        extra={"call_id": call_id}
                    )
                except Exception as e:
                    logger.error(
                        f"Error cleaning up controller: {e}",
                        extra={"call_id": call_id},
                        exc_info=True
                    )
            
            # Unregister
            await self.registry.unregister(call_id)
            
            # Close WebSocket if not already closed
            try:
                await websocket.close()
            except Exception:
                pass
    
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
            try:
                handlers = self.handler_factory(
                    call_id=call_id,
                    tenant_id=tenant_id,
                    call_sid=call_sid,
                    from_phone=from_phone,
                    to_phone=to_phone
                )
            except Exception as e:
                logger.error(
                    f"Error creating handlers: {e}",
                    extra={"call_id": call_id},
                    exc_info=True
                )
        
        # Create controller
        controller = CallController(
            call_id=call_id,
            tenant_id=tenant_id,
            watchdog_limits=self.watchdog_limits,
            handlers=handlers
        )
        
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
            extra={
                "base_url": self.base_url,
                "max_concurrent_calls": self.max_concurrent_calls
            }
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
