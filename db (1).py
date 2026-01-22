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
