"""
Tenant Config
=============
Multi-tenant configuration management.

Responsibilities:
- Load tenant configurations
- Cache tenant settings
- Provide tenant-specific configs
- Support per-tenant customization
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
    
    # Customization
    greeting_message: Optional[str] = None
    transfer_number: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
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
            "greeting_message": self.greeting_message,
            "transfer_number": self.transfer_number,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class TenantConfigManager:
    """
    Tenant configuration manager.
    
    Manages configurations for multiple tenants.
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
        # Default tenant
        default_config = TenantConfig(
            tenant_id="default_tenant",
            name="Default Restaurant",
            phone_numbers=["+15555551234"],
            greeting_message="Thank you for calling! How can I help you today?"
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
