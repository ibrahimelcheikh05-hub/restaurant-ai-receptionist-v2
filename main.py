"""
Main Entry Point (No Vocode)
=============================
Restaurant AI Receptionist - Direct Twilio implementation.

Features:
- No vocode dependency
- Direct Twilio Media Streams
- Low latency audio streaming
- Barge-in support
- Your custom handlers intact
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from settings import get_settings
from observability import setup_logging, get_events, get_health
from twilio_server import TwilioServer


async def initialize_system() -> None:
    """Initialize system components."""
    logger = logging.getLogger(__name__)
    events = get_events()
    health = get_health()
    
    logger.info("Initializing system components...")
    
    try:
        # Initialize database
        from db import get_default_db
        db = get_default_db()
        await db.initialize()
        health.mark_healthy("database")
        logger.info("✓ Database initialized")
    
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        health.mark_unhealthy("database", str(e))
    
    try:
        # Initialize AI components
        from llm_client import get_default_client
        from detector import get_default_detector
        from translator import get_default_translator
        from menu_engine import get_default_engine as get_menu_engine
        from order_engine import get_default_engine as get_order_engine
        from upsell_engine import get_default_engine as get_upsell_engine
        from output_parser import get_default_parser
        from safety import get_default_filter
        from prompt_builder import create_operational_prompt_builder
        from sms import get_default_client as get_sms_client
        from tenant_config import get_default_manager as get_tenant_manager
        
        # Instantiate singletons
        await get_default_client()
        get_default_detector()
        get_default_translator()
        get_menu_engine()
        get_order_engine()
        get_upsell_engine()
        get_default_parser()
        get_default_filter()
        create_operational_prompt_builder()
        get_sms_client()
        get_tenant_manager()
        
        health.mark_healthy("components")
        logger.info("✓ All AI components initialized")
    
    except Exception as e:
        logger.error(f"Component initialization failed: {e}", exc_info=True)
        health.mark_unhealthy("components", str(e))
    
    events.log_event("system_started")
    logger.info("System initialization complete ✓")


def create_event_handlers(
    call_id: str, 
    tenant_id: str,
    call_sid: str = None,
    from_phone: str = None,
    to_phone: str = None
) -> dict:
    """Create event handlers for CallController."""
    from menu_engine import get_default_engine as get_menu_engine
    from order_engine import get_default_engine as get_order_engine
    from upsell_engine import get_default_engine as get_upsell_engine
    from llm_client import get_default_client as get_llm_client
    from prompt_builder import create_operational_prompt_builder as get_prompt_builder
    from output_parser import get_default_parser as get_output_parser
    from safety import get_default_filter as get_safety_filter
    from detector import get_default_detector
    from translator import get_default_translator
    from conversation_memory import create_memory
    from sms import send_order_confirmation
    from tenant_config import get_tenant_config
    
    logger = logging.getLogger(__name__)
    logger.info(f"Creating event handlers for call {call_id}")
    
    # Get engines
    menu_engine = get_menu_engine()
    order_engine = get_order_engine()
    upsell_engine = get_upsell_engine()
    prompt_builder = get_prompt_builder()
    output_parser = get_output_parser()
    safety_filter = get_safety_filter()
    detector = get_default_detector()
    translator = get_default_translator()
    
    # Create conversation memory
    memory = create_memory(call_id)
    
    # Create order
    order = order_engine.create_order(call_id, tenant_id)
    
    async def on_greeting(**kwargs):
        """Handle greeting."""
        try:
            config = await get_tenant_config(tenant_id)
            greeting = None
            
            if config and hasattr(config, 'greeting_message'):
                greeting = config.greeting_message
            
            if not greeting:
                greeting = "Thank you for calling! How can I help you today?"
            
            logger.info(f"Greeting handler returning: {greeting}")
            
            return {
                "greeting": greeting,
                "language": "en"
            }
        except Exception as e:
            logger.error(f"Error in on_greeting: {e}", exc_info=True)
            return {
                "greeting": "Thank you for calling! How can I help you today?",
                "language": "en"
            }
    
    async def on_ai_request(user_text: str, turn_count: int, **kwargs):
        """Handle AI request."""
        try:
            # Safety check
            safety_result = safety_filter.check_input(user_text)
            if not safety_result.is_safe:
                logger.warning(f"Unsafe input detected: {safety_result.reason}")
                return {
                    "response_text": "I didn't quite understand that. Could you rephrase?",
                    "suggested_action": None
                }
            
            # Get menu and order
            menu = await menu_engine.get_menu(tenant_id)
            order_summary = order.get_summary() if order else None
            
            # Get upsell suggestions
            upsell_suggestions = upsell_engine.get_suggestions(order, menu, max_suggestions=2)
            upsell_text = [s.reason for s in upsell_suggestions]
            
            # Build prompt
            messages = prompt_builder.build_messages(
                user_input=user_text,
                conversation_history=memory.get_history_as_messages(),
                menu_data={"items": [item.to_dict() for item in menu]},
                order_data=order.to_dict() if order else None,
                upsell_suggestions=upsell_text
            )
            
            # Call LLM
            llm_client = await get_llm_client()
            response = await llm_client.complete(messages, call_id=call_id)
            
            # Parse output
            parsed = output_parser.parse(response.text)
            
            # Add to memory
            memory.add_user_turn(user_text)
            memory.add_assistant_turn(response.text)
            
            return {
                "response_text": parsed.text,
                "suggested_action": parsed.action
            }
        
        except Exception as e:
            logger.error(f"Error in AI request: {e}", exc_info=True)
            return {
                "response_text": "I apologize, could you repeat that?",
                "suggested_action": None
            }
    
    async def on_closing(reason: str, session_summary: dict = None, **kwargs):
        """Handle call closing."""
        try:
            logger.info(f"Call {call_id} closing: {reason}")
            
            # Send SMS if order created
            if order and not order.is_empty() and order.customer_phone:
                try:
                    await send_order_confirmation(
                        to_number=order.customer_phone,
                        order_summary=order_engine.get_summary(call_id)
                    )
                    logger.info(f"Order confirmation SMS sent")
                except Exception as e:
                    logger.error(f"Failed to send SMS: {e}")
        
        except Exception as e:
            logger.error(f"Error in closing handler: {e}")
    
    async def on_event(event: dict, **kwargs):
        """Handle generic events."""
        try:
            events = get_events()
            events.log_event(event.get("type", "unknown"), **event.get("data", {}))
        except Exception as e:
            logger.error(f"Error logging event: {e}")
    
    return {
        "on_greeting": on_greeting,
        "on_ai_request": on_ai_request,
        "on_closing": on_closing,
        "on_event": on_event
    }


# Load settings
settings = get_settings()

# Setup logging
setup_logging(level=settings.log_level)
logger = logging.getLogger(__name__)

logger.info(f"Starting in {settings.environment} mode (NO VOCODE)")

# Validate settings
errors = settings.validate()
if errors:
    logger.error(f"Configuration errors: {errors}")
    if settings.environment != "production":
        sys.exit(1)

# Create Twilio server (NO VOCODE!)
server = TwilioServer(
    base_url=settings.vocode.base_url,
    deepgram_api_key=settings.speech.deepgram_api_key,
    elevenlabs_api_key=settings.speech.elevenlabs_api_key,
    elevenlabs_voice_id=settings.speech.elevenlabs_default_voice_id,
    openai_api_key=settings.ai.openai_api_key,
    handler_factory=create_event_handlers
)

# Export the app for uvicorn
app = server.app

# Initialize system on startup
@app.on_event("startup")
async def startup_event():
    """Run initialization on startup."""
    await initialize_system()
    logger.info("Server ready ✓")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.vocode.host,
        port=settings.vocode.port
    )
