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
    
    try:
        # Initialize database
        from db import get_default_db
        db = get_default_db()
        await db.initialize()
        health.mark_healthy("database")
        
        logger.info("Database initialized")
    
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        health.mark_unhealthy("database", str(e))
    
    try:
        # Initialize other components
        from llm_client import get_default_client
        from detector import get_default_detector
        from translator import get_default_translator
        from menu_engine import get_default_engine as get_menu_engine
        from order_engine import get_default_engine as get_order_engine
        
        # Just instantiate to initialize
        get_default_client()
        get_default_detector()
        get_default_translator()
        get_menu_engine()
        get_order_engine()
        
        health.mark_healthy("components")
        logger.info("All components initialized")
    
    except Exception as e:
        logger.error(f"Component initialization failed: {e}")
        health.mark_unhealthy("components", str(e))
    
    events.log_event("system_started")
    logger.info("System initialization complete")


def create_event_handlers(call_id: str, tenant_id: str) -> dict:
    """
    Create event handlers for CallController.
    
    This factory function creates the handlers that implement
    business logic for each call.
    
    Args:
        call_id: Call identifier
        tenant_id: Tenant identifier
        
    Returns:
        Dict of handler functions
    """
    from menu_engine import get_default_engine as get_menu_engine
    from order_engine import get_default_engine as get_order_engine
    from upsell_engine import get_default_engine as get_upsell_engine
    from llm_client import get_default_client as get_llm_client
    from prompt_builder import get_default_builder as get_prompt_builder
    from output_parser import get_default_parser as get_output_parser
    from safety import get_default_filter as get_safety_filter
    from detector import get_default_detector
    from translator import get_default_translator
    from conversation_memory import create_memory, get_memory
    from sms import send_order_confirmation
    from tenant_config import get_tenant_config
    
    # Get engines
    menu_engine = get_menu_engine()
    order_engine = get_order_engine()
    upsell_engine = get_upsell_engine()
    llm_client = get_llm_client()
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
        # Get tenant config
        config = await get_tenant_config(tenant_id)
        greeting = config.greeting_message if config else "Thank you for calling!"
        
        return {
            "greeting": greeting,
            "language": "en"
        }
    
    async def on_language_detect(transcript: str, **kwargs):
        """Handle language detection."""
        result = detector.detect(call_id, transcript, is_final=True)
        
        return {
            "language": result.language,
            "confidence": result.confidence
        }
    
    async def on_translate_to_english(text: str, source_language: str, **kwargs):
        """Translate to English."""
        result = await translator.translate_to_english(text, source_language)
        return {"translated_text": result.text}
    
    async def on_translate_from_english(text: str, target_language: str, **kwargs):
        """Translate from English."""
        result = await translator.translate_from_english(text, target_language)
        return {"translated_text": result.text}
    
    async def on_ai_request(user_text: str, turn_count: int, **kwargs):
        """Handle AI request."""
        # Safety check
        safety_result = safety_filter.check_input(user_text)
        if not safety_result.is_safe:
            return {
                "response_text": "I didn't quite understand that. Could you rephrase?",
                "suggested_action": None
            }
        
        # Get menu and order
        menu = await menu_engine.get_menu(tenant_id)
        order_summary = order_engine.get_summary(call_id)
        
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
    
    async def on_transfer_approval(reason: str, **kwargs):
        """Handle transfer approval."""
        # Get tenant config
        config = await get_tenant_config(tenant_id)
        
        if config and config.enable_transfer and config.transfer_number:
            return {
                "approved": True,
                "transfer_number": config.transfer_number,
                "transfer_message": "Let me transfer you to someone who can help."
            }
        
        return {
            "approved": False,
            "denial_message": "I'm sorry, transfer is not available right now."
        }
    
    async def on_fallback(error_type: str, **kwargs):
        """Handle fallback."""
        return {
            "response_text": "I apologize, could you repeat that?",
            "suggested_action": None
        }
    
    async def on_closing(reason: str, session_summary: dict, **kwargs):
        """Handle call closing."""
        # Send SMS if order created
        if order and not order.is_empty() and order.customer_phone:
            try:
                await send_order_confirmation(
                    to_number=order.customer_phone,
                    order_summary=order_engine.get_summary(call_id)
                )
            except Exception as e:
                logging.error(f"Failed to send SMS: {e}")
    
    async def on_event(event: dict, **kwargs):
        """Handle generic events."""
        events = get_events()
        events.log_event(event["type"], **event.get("data", {}))
    
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


async def main():
    """Main entry point."""
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
        sys.exit(1)
    
    # Initialize system
    await initialize_system()
    
    # Create Vocode config manager
    config_manager = InMemoryConfigManager()
    
    # Create Vocode server
    server = VocodeServer(
        base_url=settings.vocode.base_url,
        config_manager=config_manager,
        handler_factory=create_event_handlers
    )
    
    logger.info(
        f"Starting server on {settings.vocode.host}:{settings.vocode.port}"
    )
    
    # Run server
    server.run(
        host=settings.vocode.host,
        port=settings.vocode.port
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
