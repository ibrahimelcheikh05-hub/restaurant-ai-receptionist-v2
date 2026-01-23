"""
Database
========
Database abstraction layer.

Responsibilities:
- Database connection management
- Query execution
- Connection pooling
- Transaction support
- Error handling
- Business & FAQ data access
- Greeting and closing message retrieval
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import os

logger = logging.getLogger(__name__)

# Try to import database drivers
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    logger.warning("asyncpg not available - PostgreSQL disabled")


class Database:
    """
    Database abstraction layer.
    
    Supports:
    - PostgreSQL (via asyncpg)
    - Connection pooling
    - Async queries
    - Transactions
    """
    
    def __init__(
        self,
        db_url: Optional[str] = None,
        min_pool_size: int = 5,
        max_pool_size: int = 20
    ):
        """
        Initialize database.
        
        Args:
            db_url: Database URL
            min_pool_size: Minimum pool size
            max_pool_size: Maximum pool size
        """
        self.db_url = db_url or os.getenv(
            "DATABASE_URL",
            "postgresql://localhost/restaurant"
        )
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        
        # Connection pool
        self.pool = None
        self._initialized = False
        
        logger.info(
            "Database initialized",
            extra={
                "driver": "asyncpg" if ASYNCPG_AVAILABLE else "none",
                "pool_size": f"{min_pool_size}-{max_pool_size}"
            }
        )
    
    async def initialize(self) -> None:
        """Initialize connection pool."""
        if self._initialized:
            return
        
        if not ASYNCPG_AVAILABLE:
            logger.warning("Database driver not available")
            return
        
        try:
            self.pool = await asyncpg.create_pool(
                self.db_url,
                min_size=self.min_pool_size,
                max_size=self.max_pool_size
            )
            
            self._initialized = True
            logger.info("Database pool created")
        
        except Exception as e:
            logger.error(
                f"Failed to create database pool: {e}",
                exc_info=True
            )
    
    async def close(self) -> None:
        """Close database pool."""
        if self.pool:
            await self.pool.close()
            self._initialized = False
            logger.info("Database pool closed")
    
    async def execute(
        self,
        query: str,
        *args
    ) -> str:
        """
        Execute query (INSERT/UPDATE/DELETE).
        
        Args:
            query: SQL query
            *args: Query parameters
            
        Returns:
            Query result status
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.pool:
            logger.warning("No database pool - skipping query")
            return "SKIPPED"
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(query, *args)
                return result
        
        except Exception as e:
            logger.error(
                f"Query execution error: {e}",
                extra={"query": query[:100]},
                exc_info=True
            )
            raise
    
    async def fetch_one(
        self,
        query: str,
        *args
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch single row.
        
        Args:
            query: SQL query
            *args: Query parameters
            
        Returns:
            Row dict or None
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.pool:
            logger.warning("No database pool - returning None")
            return None
        
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(query, *args)
                return dict(row) if row else None
        
        except Exception as e:
            logger.error(
                f"Query error: {e}",
                extra={"query": query[:100]},
                exc_info=True
            )
            return None
    
    async def fetch_all(
        self,
        query: str,
        *args
    ) -> List[Dict[str, Any]]:
        """
        Fetch multiple rows.
        
        Args:
            query: SQL query
            *args: Query parameters
            
        Returns:
            List of row dicts
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.pool:
            logger.warning("No database pool - returning empty list")
            return []
        
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *args)
                return [dict(row) for row in rows]
        
        except Exception as e:
            logger.error(
                f"Query error: {e}",
                extra={"query": query[:100]},
                exc_info=True
            )
            return []
    
    async def fetch_val(
        self,
        query: str,
        *args
    ) -> Any:
        """
        Fetch single value.
        
        Args:
            query: SQL query
            *args: Query parameters
            
        Returns:
            Single value
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.pool:
            return None
        
        try:
            async with self.pool.acquire() as conn:
                return await conn.fetchval(query, *args)
        
        except Exception as e:
            logger.error(
                f"Query error: {e}",
                exc_info=True
            )
            return None
    
    # =========================================================================
    # High-level operations
    # =========================================================================
    
    async def store_call_log(
        self,
        call_data: Dict[str, Any]
    ) -> bool:
        """
        Store call log.
        
        Args:
            call_data: Call data to store
            
        Returns:
            True if stored
        """
        try:
            query = """
                INSERT INTO call_logs (
                    call_id, tenant_id, from_phone, to_phone,
                    direction, status, duration_seconds,
                    created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """
            
            await self.execute(
                query,
                call_data.get("call_id"),
                call_data.get("tenant_id"),
                call_data.get("from_phone"),
                call_data.get("to_phone"),
                call_data.get("direction", "inbound"),
                call_data.get("status", "completed"),
                call_data.get("duration_seconds", 0),
                datetime.now(timezone.utc)
            )
            
            logger.info(
                "Call log stored",
                extra={"call_id": call_data.get("call_id")}
            )
            
            return True
        
        except Exception as e:
            logger.error(
                f"Failed to store call log: {e}",
                exc_info=True
            )
            return False
    
    async def store_order(
        self,
        order_data: Dict[str, Any]
    ) -> bool:
        """
        Store order.
        
        Args:
            order_data: Order data to store
            
        Returns:
            True if stored
        """
        try:
            query = """
                INSERT INTO orders (
                    order_id, tenant_id, call_id,
                    items, total, customer_phone,
                    created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """
            
            await self.execute(
                query,
                order_data.get("order_id"),
                order_data.get("tenant_id"),
                order_data.get("call_id"),
                order_data.get("items"),  # JSON
                order_data.get("total"),
                order_data.get("customer_phone"),
                datetime.now(timezone.utc)
            )
            
            logger.info(
                "Order stored",
                extra={"order_id": order_data.get("order_id")}
            )
            
            return True
        
        except Exception as e:
            logger.error(
                f"Failed to store order: {e}",
                exc_info=True
            )
            return False
    
    async def get_tenant_config(
        self,
        tenant_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get tenant configuration.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Config dict or None
        """
        query = """
            SELECT * FROM tenants
            WHERE tenant_id = $1
        """
        
        return await self.fetch_one(query, tenant_id)
    
    async def get_menu(
        self,
        tenant_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get menu for tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            List of menu items
        """
        query = """
            SELECT * FROM menu_items
            WHERE tenant_id = $1
            AND available = true
            ORDER BY category, name
        """
        
        return await self.fetch_all(query, tenant_id)
    
    async def get_business_info(
        self,
        tenant_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get business information for tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Business info dict or None
        """
        query = """
            SELECT 
                name,
                address,
                phone,
                hours,
                description
            FROM tenants
            WHERE tenant_id = $1
        """
        
        result = await self.fetch_one(query, tenant_id)
        return result
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False


# Default instance
_default_db: Optional[Database] = None


def get_default_db() -> Database:
    """
    Get or create default database.
    
    Returns:
        Default database instance
    """
    global _default_db
    
    if _default_db is None:
        _default_db = Database()
    
    return _default_db


async def store_call_log(call_data: Dict[str, Any]) -> bool:
    """
    Quick call log storage.
    
    Args:
        call_data: Call data
        
    Returns:
        True if stored
    """
    db = get_default_db()
    return await db.store_call_log(call_data)


async def store_order(order_data: Dict[str, Any]) -> bool:
    """
    Quick order storage.
    
    Args:
        order_data: Order data
        
    Returns:
        True if stored
    """
    db = get_default_db()
    return await db.store_order(order_data)


# =========================================================================
# System-Controlled Message Retrieval Functions
# =========================================================================

# Mock data for greetings (will be replaced with DB queries in production)
_MOCK_GREETINGS = {
    "default_tenant": {
        "en": "Thank you for calling! How can I help you today?",
        "es": "¡Gracias por llamar! ¿Cómo puedo ayudarte hoy?",
        "ar": "شكرا لك على الاتصال! كيف يمكنني مساعدتك اليوم؟"
    }
}

# Mock data for closing messages
_MOCK_CLOSINGS = {
    "default_tenant": {
        "normal": {
            "en": "Thank you for calling. Have a great day!",
            "es": "Gracias por llamar. ¡Que tengas un gran día!",
            "ar": "شكرا لك على الاتصال. أتمنى لك يوما عظيما!"
        },
        "order_complete": {
            "en": "Thank you for your order! We'll have it ready soon. Have a great day!",
            "es": "¡Gracias por tu orden! Lo tendremos listo pronto. ¡Que tengas un gran día!",
            "ar": "شكرا لك على طلبك! سنجهزه قريبا. أتمنى لك يوما عظيما!"
        },
        "transfer": {
            "en": "Transferring you now. Please hold.",
            "es": "Transfiriéndote ahora. Por favor espera.",
            "ar": "نحيلك الآن. يرجى الانتظار."
        }
    }
}

# Mock data for FAQ answers
_MOCK_FAQS = {
    "default_tenant": {
        "hours": {
            "en": "We're open Monday through Friday, 11 AM to 9 PM, and weekends from 10 AM to 10 PM.",
            "es": "Estamos abiertos de lunes a viernes, de 11 AM a 9 PM, y los fines de semana de 10 AM a 10 PM.",
            "ar": "نحن مفتوحون من الاثنين إلى الجمعة، من 11 صباحًا إلى 9 مساءً، وعطلات نهاية الأسبوع من 10 صباحًا إلى 10 مساءً."
        },
        "location": {
            "en": "We're located at 123 Main Street, downtown. There's parking available in the lot behind the building.",
            "es": "Estamos ubicados en 123 Main Street, en el centro. Hay estacionamiento disponible en el lote detrás del edificio.",
            "ar": "نحن موجودون في 123 Main Street، وسط المدينة. يتوفر موقف سيارات في الساحة خلف المبنى."
        },
        "delivery": {
            "en": "We offer delivery within 5 miles. Delivery fee is $3.99 and takes about 30-45 minutes.",
            "es": "Ofrecemos entrega dentro de 5 millas. La tarifa de entrega es de $3.99 y toma alrededor de 30-45 minutos.",
            "ar": "نقدم التوصيل ضمن 5 أميال. رسوم التوصيل 3.99 دولار ويستغرق حوالي 30-45 دقيقة."
        },
        "parking": {
            "en": "We have a parking lot behind the building with free parking for customers.",
            "es": "Tenemos un estacionamiento detrás del edificio con estacionamiento gratuito para clientes.",
            "ar": "لدينا موقف سيارات خلف المبنى مع مواقف مجانية للعملاء."
        }
    }
}


async def get_greeting(
    tenant_id: str,
    language: str = "en"
) -> str:
    """
    Get greeting message for tenant and language.
    
    Args:
        tenant_id: Tenant identifier
        language: Language code (e.g., 'en', 'es', 'ar')
        
    Returns:
        Greeting message text
    """
    try:
        # Try to get from database first
        db = get_default_db()
        
        # In production, query database:
        # query = """
        #     SELECT greeting_text
        #     FROM greetings
        #     WHERE tenant_id = $1 AND language = $2
        # """
        # result = await db.fetch_one(query, tenant_id, language)
        # if result:
        #     return result.get("greeting_text", "")
        
        # For now, use mock data
        tenant_greetings = _MOCK_GREETINGS.get(tenant_id, _MOCK_GREETINGS.get("default_tenant", {}))
        greeting = tenant_greetings.get(language, tenant_greetings.get("en", ""))
        
        if greeting:
            logger.info(
                f"Greeting retrieved: {tenant_id}, {language}",
                extra={"tenant_id": tenant_id, "language": language}
            )
            return greeting
        
        # Fallback
        return "Thank you for calling. How can I help you?"
    
    except Exception as e:
        logger.error(
            f"Error getting greeting: {e}",
            extra={"tenant_id": tenant_id, "language": language},
            exc_info=True
        )
        return "Thank you for calling. How can I help you?"


async def get_closing_message(
    tenant_id: str,
    reason: str = "normal",
    language: str = "en"
) -> str:
    """
    Get closing message for tenant, reason, and language.
    
    Args:
        tenant_id: Tenant identifier
        reason: Closing reason (e.g., 'normal', 'order_complete', 'transfer')
        language: Language code (e.g., 'en', 'es', 'ar')
        
    Returns:
        Closing message text
    """
    try:
        # Try to get from database first
        db = get_default_db()
        
        # In production, query database:
        # query = """
        #     SELECT closing_text
        #     FROM closing_messages
        #     WHERE tenant_id = $1 AND reason = $2 AND language = $3
        # """
        # result = await db.fetch_one(query, tenant_id, reason, language)
        # if result:
        #     return result.get("closing_text", "")
        
        # For now, use mock data
        tenant_closings = _MOCK_CLOSINGS.get(tenant_id, _MOCK_CLOSINGS.get("default_tenant", {}))
        reason_closings = tenant_closings.get(reason, tenant_closings.get("normal", {}))
        closing = reason_closings.get(language, reason_closings.get("en", ""))
        
        if closing:
            logger.info(
                f"Closing message retrieved: {tenant_id}, {reason}, {language}",
                extra={"tenant_id": tenant_id, "reason": reason, "language": language}
            )
            return closing
        
        # Fallback
        return "Thank you for calling. Goodbye!"
    
    except Exception as e:
        logger.error(
            f"Error getting closing message: {e}",
            extra={"tenant_id": tenant_id, "reason": reason, "language": language},
            exc_info=True
        )
        return "Thank you for calling. Goodbye!"


async def get_faq_answer(
    tenant_id: str,
    intent: str,
    language: str = "en"
) -> Optional[str]:
    """
    Get FAQ answer for tenant, intent, and language.
    
    Args:
        tenant_id: Tenant identifier
        intent: FAQ intent (e.g., 'hours', 'location', 'delivery')
        language: Language code (e.g., 'en', 'es', 'ar')
        
    Returns:
        FAQ answer text or None if not found
    """
    try:
        # Try to get from database first
        db = get_default_db()
        
        # In production, query database:
        # query = """
        #     SELECT answer_text
        #     FROM faq_answers
        #     WHERE tenant_id = $1 AND intent = $2 AND language = $3
        # """
        # result = await db.fetch_one(query, tenant_id, intent, language)
        # if result:
        #     return result.get("answer_text")
        
        # For now, use mock data
        tenant_faqs = _MOCK_FAQS.get(tenant_id, _MOCK_FAQS.get("default_tenant", {}))
        intent_faqs = tenant_faqs.get(intent, {})
        answer = intent_faqs.get(language, intent_faqs.get("en"))
        
        if answer:
            logger.info(
                f"FAQ answer retrieved: {tenant_id}, {intent}, {language}",
                extra={"tenant_id": tenant_id, "intent": intent, "language": language}
            )
            return answer
        
        logger.debug(
            f"No FAQ answer found: {tenant_id}, {intent}, {language}",
            extra={"tenant_id": tenant_id, "intent": intent, "language": language}
        )
        return None
    
    except Exception as e:
        logger.error(
            f"Error getting FAQ answer: {e}",
            extra={"tenant_id": tenant_id, "intent": intent, "language": language},
            exc_info=True
        )
        return None


async def get_business_info(tenant_id: str) -> Optional[Dict[str, Any]]:
    """
    Get business information for tenant.
    
    Args:
        tenant_id: Tenant identifier
        
    Returns:
        Business info dict or None
    """
    db = get_default_db()
    return await db.get_business_info(tenant_id)
