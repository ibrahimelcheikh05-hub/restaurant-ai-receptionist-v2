"""
Menu Engine
===========
Menu data access and item matching.

Responsibilities:
- Load and cache menu data
- Match user requests to menu items
- Validate menu items
- Prevent hallucination (AI cannot invent items)
- Provide menu context to AI
"""

import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class MenuItem:
    """Menu item with validation."""
    
    item_id: str
    name: str
    price: float
    description: str = ""
    category: str = "Other"
    available: bool = True
    synonyms: List[str] = None
    
    def __post_init__(self):
        """Validate after initialization."""
        if self.synonyms is None:
            self.synonyms = []
        
        # Validation
        if not self.item_id or not self.name:
            raise ValueError("item_id and name are required")
        
        if self.price < 0:
            raise ValueError("price cannot be negative")
    
    def matches(self, query: str) -> bool:
        """
        Check if query matches this item.
        
        Args:
            query: Search query
            
        Returns:
            True if matches
        """
        query_lower = query.lower().strip()
        
        # Exact name match
        if query_lower == self.name.lower():
            return True
        
        # Synonym match
        for synonym in self.synonyms:
            if query_lower == synonym.lower():
                return True
        
        # Partial match (contains)
        if query_lower in self.name.lower():
            return True
        
        # Category match
        if query_lower == self.category.lower():
            return False  # Don't match category alone
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "name": self.name,
            "price": self.price,
            "description": self.description,
            "category": self.category,
            "available": self.available,
            "synonyms": self.synonyms
        }


class MenuCache:
    """Simple menu cache."""
    
    def __init__(self, ttl_seconds: int = 300):
        """
        Initialize cache.
        
        Args:
            ttl_seconds: Time-to-live in seconds
        """
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, tuple[List[MenuItem], datetime]] = {}
    
    def get(self, tenant_id: str) -> Optional[List[MenuItem]]:
        """Get cached menu."""
        if tenant_id not in self._cache:
            return None
        
        menu, timestamp = self._cache[tenant_id]
        
        # Check if expired
        age = (datetime.now(timezone.utc) - timestamp).total_seconds()
        if age > self.ttl_seconds:
            del self._cache[tenant_id]
            return None
        
        return menu
    
    def set(self, tenant_id: str, menu: List[MenuItem]) -> None:
        """Cache menu."""
        self._cache[tenant_id] = (menu, datetime.now(timezone.utc))
    
    def invalidate(self, tenant_id: str) -> None:
        """Invalidate cached menu."""
        if tenant_id in self._cache:
            del self._cache[tenant_id]
    
    def clear(self) -> None:
        """Clear all cache."""
        self._cache.clear()


class MenuEngine:
    """
    Menu data access and matching engine.
    
    Features:
    - Load menu from database
    - Cache menus per tenant
    - Match user queries to items
    - Validate items exist
    - Prevent hallucination
    """
    
    def __init__(
        self,
        cache_ttl: int = 300,
        enable_cache: bool = True
    ):
        """
        Initialize menu engine.
        
        Args:
            cache_ttl: Cache TTL in seconds
            enable_cache: Enable caching
        """
        self.enable_cache = enable_cache
        self.cache = MenuCache(ttl_seconds=cache_ttl) if enable_cache else None
        
        # Metrics
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(
            "MenuEngine initialized",
            extra={"cache_enabled": enable_cache, "cache_ttl": cache_ttl}
        )
    
    async def get_menu(
        self,
        tenant_id: str
    ) -> List[MenuItem]:
        """
        Get menu for tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            List of menu items
        """
        # Check cache
        if self.cache:
            cached = self.cache.get(tenant_id)
            if cached:
                self._cache_hits += 1
                logger.debug(
                    "Menu cache hit",
                    extra={"tenant_id": tenant_id}
                )
                return cached
            self._cache_misses += 1
        
        # Load from database
        menu = await self._load_menu_from_db(tenant_id)
        
        # Cache it
        if self.cache:
            self.cache.set(tenant_id, menu)
        
        logger.info(
            f"Menu loaded: {len(menu)} items",
            extra={"tenant_id": tenant_id}
        )
        
        return menu
    
    async def _load_menu_from_db(
        self,
        tenant_id: str
    ) -> List[MenuItem]:
        """
        Load menu from database.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            List of menu items
        """
        # TODO: Replace with actual database query
        # For now, return sample menu
        
        logger.debug(
            "Loading menu from database",
            extra={"tenant_id": tenant_id}
        )
        
        # Sample menu (replace with DB query)
        return [
            MenuItem(
                item_id="fish_combo",
                name="Fish Combo",
                price=12.99,
                description="2 pieces of fish with fries and coleslaw",
                category="Combos",
                synonyms=["fish", "2 piece fish", "fish meal"]
            ),
            MenuItem(
                item_id="chicken_combo",
                name="Chicken Combo",
                price=11.99,
                description="3 pieces of chicken with fries and coleslaw",
                category="Combos",
                synonyms=["chicken", "3 piece chicken", "chicken meal"]
            ),
            MenuItem(
                item_id="shrimp_basket",
                name="Shrimp Basket",
                price=13.99,
                description="10 pieces of shrimp with fries",
                category="Baskets",
                synonyms=["shrimp", "fried shrimp"]
            ),
            MenuItem(
                item_id="wings",
                name="Chicken Wings",
                price=9.99,
                description="6 wings with sauce",
                category="Sides",
                synonyms=["wings", "buffalo wings"]
            ),
            MenuItem(
                item_id="fries",
                name="French Fries",
                price=2.99,
                description="Crispy french fries",
                category="Sides",
                synonyms=["fries", "chips"]
            ),
            MenuItem(
                item_id="coleslaw",
                name="Coleslaw",
                price=1.99,
                description="Fresh coleslaw",
                category="Sides",
                synonyms=["slaw", "cole slaw"]
            ),
            MenuItem(
                item_id="soda",
                name="Soft Drink",
                price=1.99,
                description="Fountain soda",
                category="Drinks",
                synonyms=["soda", "pop", "coke", "drink"]
            )
        ]
    
    async def find_item(
        self,
        tenant_id: str,
        query: str
    ) -> Optional[MenuItem]:
        """
        Find menu item by query.
        
        Args:
            tenant_id: Tenant identifier
            query: Search query
            
        Returns:
            Matched item or None
        """
        menu = await self.get_menu(tenant_id)
        
        # Try exact match first
        for item in menu:
            if item.matches(query):
                logger.debug(
                    f"Item matched: {item.name}",
                    extra={"query": query}
                )
                return item
        
        logger.debug(
            "No item matched",
            extra={"query": query}
        )
        return None
    
    async def find_items(
        self,
        tenant_id: str,
        queries: List[str]
    ) -> List[MenuItem]:
        """
        Find multiple items.
        
        Args:
            tenant_id: Tenant identifier
            queries: List of search queries
            
        Returns:
            List of matched items
        """
        items = []
        for query in queries:
            item = await self.find_item(tenant_id, query)
            if item:
                items.append(item)
        
        return items
    
    async def validate_item_exists(
        self,
        tenant_id: str,
        item_name: str
    ) -> bool:
        """
        Validate that item exists in menu.
        
        This prevents hallucination - AI cannot order items
        that don't exist.
        
        Args:
            tenant_id: Tenant identifier
            item_name: Item name to validate
            
        Returns:
            True if item exists
        """
        item = await self.find_item(tenant_id, item_name)
        return item is not None
    
    async def get_items_by_category(
        self,
        tenant_id: str,
        category: str
    ) -> List[MenuItem]:
        """
        Get items by category.
        
        Args:
            tenant_id: Tenant identifier
            category: Category name
            
        Returns:
            List of items in category
        """
        menu = await self.get_menu(tenant_id)
        
        return [
            item for item in menu
            if item.category.lower() == category.lower()
        ]
    
    async def get_categories(
        self,
        tenant_id: str
    ) -> List[str]:
        """
        Get all categories.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            List of category names
        """
        menu = await self.get_menu(tenant_id)
        
        categories = set(item.category for item in menu)
        return sorted(categories)
    
    def format_menu_for_prompt(
        self,
        menu: List[MenuItem],
        include_descriptions: bool = True
    ) -> str:
        """
        Format menu for AI prompt.
        
        Args:
            menu: Menu items
            include_descriptions: Include item descriptions
            
        Returns:
            Formatted menu text
        """
        # Group by category
        by_category: Dict[str, List[MenuItem]] = {}
        for item in menu:
            if item.category not in by_category:
                by_category[item.category] = []
            by_category[item.category].append(item)
        
        # Format each category
        lines = []
        for category in sorted(by_category.keys()):
            lines.append(f"\n{category}:")
            
            for item in by_category[category]:
                if not item.available:
                    continue
                
                line = f"  - {item.name} (${item.price:.2f})"
                
                if include_descriptions and item.description:
                    line += f": {item.description}"
                
                lines.append(line)
        
        return "\n".join(lines)
    
    def invalidate_cache(self, tenant_id: str) -> None:
        """
        Invalidate cached menu.
        
        Call this when menu is updated.
        
        Args:
            tenant_id: Tenant identifier
        """
        if self.cache:
            self.cache.invalidate(tenant_id)
            logger.info(
                "Menu cache invalidated",
                extra={"tenant_id": tenant_id}
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get engine statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses
        }
        
        total = self._cache_hits + self._cache_misses
        if total > 0:
            stats["cache_hit_rate"] = self._cache_hits / total
        
        return stats


# Default instance
_default_engine: Optional[MenuEngine] = None


def get_default_engine() -> MenuEngine:
    """
    Get or create default menu engine.
    
    Returns:
        Default engine instance
    """
    global _default_engine
    
    if _default_engine is None:
        _default_engine = MenuEngine()
    
    return _default_engine


async def get_menu(tenant_id: str) -> List[MenuItem]:
    """
    Quick menu retrieval.
    
    Args:
        tenant_id: Tenant identifier
        
    Returns:
        Menu items
    """
    engine = get_default_engine()
    return await engine.get_menu(tenant_id)
