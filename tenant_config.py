"""
Tenant Config
=============
Multi-tenant configuration management.

Responsibilities:
- Load tenant configurations
- Cache tenant settings
- Provide tenant-specific configs
- Support per-tenant customization
- Manage per-tenant voice, language, and model settings
- Support per-tenant greetings and closings
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class TenantConfig:
    """Configuration for a single tenant."""
    
    tenant_id: str
    name: str
    
    # Contact info
    phone_numbers: list = field(default_factory=list)
    email: Optional[str] = None
    
    # Business hours
    business_hours: Dict[str, Any] = field(default_factory=dict)
    timezone: str = "America/New_York"
    
    # Features
    enable_language_detection: bool = True
    enable_translation: bool = True
    enable_transfer: bool = True
    enable_upsell: bool = True
    enable_sms: bool = True
    
    # Limits
    max_call_duration: float = 1800.0  # 30 minutes
    max_order_items: int = 50
    
    # Customization - Greetings & Closings (per language)
    # Format: {"en": "greeting text", "es": "greeting text", ...}
    greetings: Dict[str, str] = field(default_factory=dict)
    closings: Dict[str, str] = field(default_factory=dict)
    
    # Legacy single greeting (for backward compatibility)
    greeting_message: Optional[str] = None
    
    # Transfer
    transfer_number: Optional[str] = None
    
    # Language Configuration
    default_language: str = "en"
    supported_languages: list = field(default_factory=lambda: ["en", "es", "ar"])
    
    # ElevenLabs Voice Configuration
    elevenlabs_voice_id: Optional[str] = None  # Default voice for this tenant
    elevenlabs_voice_map: Dict[str, str] = field(default_factory=dict)  # Language -> voice_id mapping
    
    # AI Model Preferences
    use_primary_model: bool = True  # If True, use gpt-4o; if False, use gpt-3.5-turbo
    primary_model_override: Optional[str] = None  # Override primary model (e.g., "gpt-4o-mini")
    fast_model_override: Optional[str] = None  # Override fast model
    
    # AI Behavior
    response_style: str = "professional"  # professional, casual, friendly
    max_response_length: int = 100  # Max words in AI response
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_greeting(self, language: str = "en") -> str:
        """
        Get greeting for specified language.
        
        Args:
            language: Language code
            
        Returns:
            Greeting text
        """
        # Try language-specific greeting
        if language in self.greetings:
            return self.greetings[language]
        
        # Fall back to English
        if "en" in self.greetings:
            return self.greetings["en"]
        
        # Fall back to legacy greeting_message
        if self.greeting_message:
            return self.greeting_message
        
        # Final fallback
        return "Thank you for calling! How can I help you today?"
    
    def get_closing(self, language: str = "en", reason: str = "normal") -> str:
        """
        Get closing message for specified language.
        
        Args:
            language: Language code
            reason: Closing reason (normal, order_complete, transfer)
            
        Returns:
            Closing text
        """
        # Try to get reason-specific closing
        closing_key = f"{reason}_{language}"
        if closing_key in self.closings:
            return self.closings[closing_key]
        
        # Try language-specific closing
        if language in self.closings:
            return self.closings[language]
        
        # Fall back to English
        if "en" in self.closings:
            return self.closings["en"]
        
        # Final fallback
        if reason == "order_complete":
            return "Thank you for your order! Have a great day!"
        else:
            return "Thank you for calling. Goodbye!"
    
    def get_voice_id(self, language: str = "en") -> Optional[str]:
        """
        Get ElevenLabs voice ID for specified language.
        
        Args:
            language: Language code
            
        Returns:
            Voice ID or None
        """
        # Try language-specific voice
        if language in self.elevenlabs_voice_map:
            return self.elevenlabs_voice_map[language]
        
        # Fall back to default voice
        return self.elevenlabs_voice_id
    
    def get_primary_model(self) -> str:
        """
        Get primary model name.
        
        Returns:
            Model name
        """
        if self.primary_model_override:
            return self.primary_model_override
        
        return "gpt-4o" if self.use_primary_model else "gpt-3.5-turbo"
    
    def get_fast_model(self) -> str:
        """
        Get fast model name.
        
        Returns:
            Model name
        """
        if self.fast_model_override:
            return self.fast_model_override
        
        return "gpt-3.5-turbo"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "phone_numbers": self.phone_numbers,
            "email": self.email,
            "business_hours": self.business_hours,
            "timezone": self.timezone,
            "enable_language_detection": self.enable_language_detection,
            "enable_translation": self.enable_translation,
            "enable_transfer": self.enable_transfer,
            "enable_upsell": self.enable_upsell,
            "enable_sms": self.enable_sms,
            "max_call_duration": self.max_call_duration,
            "max_order_items": self.max_order_items,
            "greetings": self.greetings,
            "closings": self.closings,
            "greeting_message": self.greeting_message,
            "transfer_number": self.transfer_number,
            "default_language": self.default_language,
            "supported_languages": self.supported_languages,
            "elevenlabs_voice_id": self.elevenlabs_voice_id,
            "elevenlabs_voice_map": self.elevenlabs_voice_map,
            "use_primary_model": self.use_primary_model,
            "primary_model_override": self.primary_model_override,
            "fast_model_override": self.fast_model_override,
            "response_style": self.response_style,
            "max_response_length": self.max_response_length,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class TenantConfigManager:
    """
    Tenant configuration manager.
    
    Manages configurations for multiple tenants with full SaaS support.
    """
    
    def __init__(self):
        """Initialize tenant config manager."""
        # Config cache
        self._configs: Dict[str, TenantConfig] = {}
        
        # Load default configs
        self._load_default_configs()
        
        logger.info("TenantConfigManager initialized")
    
    def _load_default_configs(self) -> None:
        """Load default tenant configurations."""
        # Default tenant with comprehensive config
        default_config = TenantConfig(
            tenant_id="default_tenant",
            name="Default Restaurant",
            phone_numbers=["+15555551234"],
            
            # Greetings per language
            greetings={
                "en": "Thank you for calling! How can I help you today?",
                "es": "¡Gracias por llamar! ¿Cómo puedo ayudarte hoy?",
                "ar": "شكرا لك على الاتصال! كيف يمكنني مساعدتك اليوم؟"
            },
            
            # Closings per language
            closings={
                "en": "Thank you for calling. Have a great day!",
                "es": "Gracias por llamar. ¡Que tengas un gran día!",
                "ar": "شكرا لك على الاتصال. أتمنى لك يوما عظيما!",
                "order_complete_en": "Thank you for your order! We'll have it ready soon.",
                "order_complete_es": "¡Gracias por tu orden! Lo tendremos listo pronto.",
                "order_complete_ar": "شكرا لك على طلبك! سنجهزه قريبا."
            },
            
            # Language settings
            default_language="en",
            supported_languages=["en", "es", "ar"],
            
            # Voice settings (using default ElevenLabs voice)
            elevenlabs_voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice
            elevenlabs_voice_map={
                # Can map different voices per language if needed
                # "es": "spanish_voice_id",
                # "ar": "arabic_voice_id"
            },
            
            # Model settings
            use_primary_model=True,  # Use gpt-4o by default
            response_style="professional",
            max_response_length=100
        )
        
        self._configs["default_tenant"] = default_config
    
    async def load_config(self, tenant_id: str) -> Optional[TenantConfig]:
        """
        Load tenant configuration.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Tenant config or None
        """
        # Check cache
        if tenant_id in self._configs:
            return self._configs[tenant_id]
        
        # In production, load from database
        # query = """
        #     SELECT * FROM tenant_configs
        #     WHERE tenant_id = $1
        # """
        # result = await db.fetch_one(query, tenant_id)
        # if result:
        #     config = TenantConfig(**result)
        #     self._configs[tenant_id] = config
        #     return config
        
        # For now, return default
        logger.warning(
            f"Tenant config not found: {tenant_id}, using default",
            extra={"tenant_id": tenant_id}
        )
        
        return self._configs.get("default_tenant")
    
    def get_config(self, tenant_id: str) -> Optional[TenantConfig]:
        """
        Get cached tenant configuration.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Tenant config or None
        """
        return self._configs.get(tenant_id)
    
    def register_tenant(self, config: TenantConfig) -> None:
        """
        Register tenant configuration.
        
        Args:
            config: Tenant configuration
        """
        self._configs[config.tenant_id] = config
        
        logger.info(
            f"Tenant registered: {config.tenant_id}",
            extra={"tenant_id": config.tenant_id, "name": config.name}
        )
    
    def update_config(
        self,
        tenant_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update tenant configuration.
        
        Args:
            tenant_id: Tenant identifier
            updates: Configuration updates
            
        Returns:
            True if updated
        """
        if tenant_id not in self._configs:
            logger.error(f"Tenant not found: {tenant_id}")
            return False
        
        config = self._configs[tenant_id]
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        config.updated_at = datetime.now(timezone.utc)
        
        logger.info(
            f"Tenant config updated: {tenant_id}",
            extra={"tenant_id": tenant_id}
        )
        
        return True
    
    def get_all_tenants(self) -> Dict[str, TenantConfig]:
        """Get all tenant configurations."""
        return self._configs.copy()
    
    def set_voice_for_language(
        self,
        tenant_id: str,
        language: str,
        voice_id: str
    ) -> bool:
        """
        Set voice ID for specific language for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            language: Language code
            voice_id: ElevenLabs voice ID
            
        Returns:
            True if set successfully
        """
        config = self.get_config(tenant_id)
        if not config:
            return False
        
        config.elevenlabs_voice_map[language] = voice_id
        config.updated_at = datetime.now(timezone.utc)
        
        logger.info(
            f"Voice set for {tenant_id}: {language} -> {voice_id}",
            extra={"tenant_id": tenant_id, "language": language}
        )
        
        return True
    
    def set_greeting_for_language(
        self,
        tenant_id: str,
        language: str,
        greeting: str
    ) -> bool:
        """
        Set greeting for specific language for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            language: Language code
            greeting: Greeting text
            
        Returns:
            True if set successfully
        """
        config = self.get_config(tenant_id)
        if not config:
            return False
        
        config.greetings[language] = greeting
        config.updated_at = datetime.now(timezone.utc)
        
        logger.info(
            f"Greeting set for {tenant_id}: {language}",
            extra={"tenant_id": tenant_id, "language": language}
        )
        
        return True
    
    def set_closing_for_language(
        self,
        tenant_id: str,
        language: str,
        closing: str,
        reason: str = "normal"
    ) -> bool:
        """
        Set closing message for specific language and reason for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            language: Language code
            closing: Closing text
            reason: Closing reason
            
        Returns:
            True if set successfully
        """
        config = self.get_config(tenant_id)
        if not config:
            return False
        
        # Store with reason if specified
        if reason != "normal":
            key = f"{reason}_{language}"
        else:
            key = language
        
        config.closings[key] = closing
        config.updated_at = datetime.now(timezone.utc)
        
        logger.info(
            f"Closing set for {tenant_id}: {key}",
            extra={"tenant_id": tenant_id, "language": language, "reason": reason}
        )
        
        return True


# Default instance
_default_manager: Optional[TenantConfigManager] = None


def get_default_manager() -> TenantConfigManager:
    """
    Get or create default tenant config manager.
    
    Returns:
        Default manager instance
    """
    global _default_manager
    
    if _default_manager is None:
        _default_manager = TenantConfigManager()
    
    return _default_manager


async def get_tenant_config(tenant_id: str) -> Optional[TenantConfig]:
    """
    Quick tenant config retrieval.
    
    Args:
        tenant_id: Tenant identifier
        
    Returns:
        Tenant config or None
    """
    manager = get_default_manager()
    return await manager.load_config(tenant_id)
