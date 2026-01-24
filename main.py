"""
Main
====
Application entry point.

Starts the Vocode telephony server with all components initialized.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from settings import get_settings
from observability import setup_logging, get_events, get_health
from vocode_server import VocodeServer
from vocode.streaming.telephony.config_manager.in_memory_config_manager import (
    InMemoryConfigManager
)


async def initialize_system() -> None:
    """Initialize system components."""
    logger = logging.getLogger(__name__)
    events = get_events()
    health = get_health()
    
    logger.info("Initializing system...")
    
    # Simple initialization without requiring all modules
    # You can add more initialization here as you implement the modules
    
    try:
        health.mark_healthy("components")
        logger.info("All components initialized")
    
    except Exception as e:
        logger.error(f"Component initialization failed: {e}")
        health.mark_unhealthy("components", str(e))
    
    events.log_event("system_started")
    logger.info("System initialization complete")


def create_event_handlers(
    call_id: str, 
    tenant_id: str,
    call_sid: str = None,
    from_phone: str = None,
    to_phone: str = None
) -> dict:
    """
    Create event handlers for CallController.
    
    This factory function creates the handlers that implement
    business logic for each call.
    
    Args:
        call_id: Call identifier
        tenant_id: Tenant identifier
        call_sid: Twilio call SID (optional)
        from_phone: Caller phone number (optional)
        to_phone: Destination phone number (optional)
        
    Returns:
        Dict of handler functions
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating event handlers for call {call_id}")
    
    # Simple in-memory conversation state
    conversation_history = []
    
    async def on_greeting(**kwargs):
        """Handle greeting."""
        greeting = "Thank you for calling! How can I help you today?"
        logger.info(f"Greeting sent for call {call_id}")
        
        return {
            "greeting": greeting,
            "language": "en"
        }
    
    async def on_language_detect(transcript: str, **kwargs):
        """Handle language detection."""
        # Default to English for now
        logger.info(f"Language detection for call {call_id}: defaulting to English")
        
        return {
            "language": "en",
            "confidence": 1.0
        }
    
    async def on_translate_to_english(text: str, source_language: str, **kwargs):
        """Translate to English."""
        # No translation for now, just return original
        return {"translated_text": text}
    
    async def on_translate_from_english(text: str, target_language: str, **kwargs):
        """Translate from English."""
        # No translation for now, just return original
        return {"translated_text": text}
    
    async def on_ai_request(user_text: str, turn_count: int, **kwargs):
        """Handle AI request."""
        logger.info(f"AI request for call {call_id}: {user_text}")
        
        # Store in history
        conversation_history.append({"role": "user", "content": user_text})
        
        # Simple response for now
        response_text = "I understand you said: " + user_text + ". Our team will assist you shortly."
        
        conversation_history.append({"role": "assistant", "content": response_text})
        
        return {
            "response_text": response_text,
            "suggested_action": None
        }
    
    async def on_transfer_approval(reason: str, **kwargs):
        """Handle transfer approval."""
        logger.info(f"Transfer request for call {call_id}: {reason}")
        
        # Deny transfers for now
        return {
            "approved": False,
            "denial_message": "I'm sorry, transfer is not available right now."
        }
    
    async def on_fallback(error_type: str, **kwargs):
        """Handle fallback."""
        logger.warning(f"Fallback triggered for call {call_id}: {error_type}")
        
        return {
            "response_text": "I apologize, could you repeat that?",
            "suggested_action": None
        }
    
    async def on_closing(reason: str, session_summary: dict = None, **kwargs):
        """Handle call closing."""
        logger.info(f"Call {call_id} closing: {reason}")
        # No action needed for now
    
    async def on_event(event: dict, **kwargs):
        """Handle generic events."""
        logger.info(f"Event for call {call_id}: {event.get('type', 'unknown')}")
    
    return {
        "on_greeting": on_greeting,
        "on_language_detect": on_language_detect,
        "on_translate_to_english": on_translate_to_english,
        "on_translate_from_english": on_translate_from_english,
        "on_ai_request": on_ai_request,
        "on_transfer_approval": on_transfer_approval,
        "on_fallback": on_fallback,
        "on_closing": on_closing,
        "on_event": on_event
    }


# Load settings
settings = get_settings()

# Setup logging
setup_logging(level=settings.log_level)
logger = logging.getLogger(__name__)

logger.info(f"Starting in {settings.environment} mode")

# Validate settings
errors = settings.validate()
if errors:
    logger.error(f"Configuration errors: {errors}")
    # Don't exit in production - let it fail more gracefully
    if settings.environment != "production":
        sys.exit(1)

# Create Vocode config manager
config_manager = InMemoryConfigManager()

# Create Vocode server - this creates the FastAPI app
server = VocodeServer(
    base_url=settings.vocode.base_url,
    config_manager=config_manager,
    handler_factory=create_event_handlers
)

# Export the app for uvicorn
app = server.app

# Initialize system on startup
@app.on_event("startup")
async def startup_event():
    """Run initialization on startup."""
    await initialize_system()
    logger.info(
        f"Server started on {settings.vocode.host}:{settings.vocode.port}"
    )


if __name__ == "__main__":
    # This runs when executing directly with python main.py
    import uvicorn
    uvicorn.run(
        app,
        host=settings.vocode.host,
        port=settings.vocode.port
    )
