"""
Menu Engine
===========
Menu data access and item matching with canonical ID enforcement.

Responsibilities:
- Load and cache menu data
- Match user requests to menu items via canonical IDs
- Validate menu items strictly
- Prevent hallucination (AI cannot invent items)
- Provide menu context to AI
- ENFORCE canonical ID usage
- NEVER allow direct name-based ordering

CRITICAL RULES:
- All items have CANONICAL_IDS (e.g., BURGER_CLASSIC, FRIES_LARGE)
- User requests map to canonical IDs via synonyms
- Orders reference canonical IDs ONLY
- Prices validated against canonical items
- AI cannot create new items
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class ItemAvailability(Enum):
    """Item availability status."""
    AVAILABLE = "available"
    OUT_OF_STOCK = "out_of_stock"
    DISCONTINUED = "discontinued"
    HIDDEN = "hidden"  # Not shown to customers


@dataclass
class MenuItem:
    """
    Menu item with canonical ID.
    
    CRITICAL: canonical_id is the source of truth.
    Name can change, canonical_id never changes.
    """
    
    canonical_id: str  # e.g., "BURGER_CLASSIC", "FRIES_LARGE"
    name: str  # Display name
    price: float
    description: str = ""
    category: str = "Other"
    availability: ItemAvailability = ItemAvailability.AVAILABLE
    synonyms: List[str] = field(default_factory=list)
    
    # Validation
    max_quantity: int = 99
    min_quantity: int = 1
    
    # Metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate after initialization."""
        # Validate canonical ID format
        if not self._is_valid_canonical_id(self.canonical_id):
            raise ValueError(
                f"Invalid canonical_id: {self.canonical_id}. "
                "Must be uppercase alphanumeric with underscores"
            )
        
        if not self.name or not self.name.strip():
            raise ValueError("name is required")
        
        if self.price < 0:
            raise ValueError("price cannot be negative")
        
        if self.price > 999.99:
            raise ValueError("price cannot exceed $999.99")
        
        if self.max_quantity < self.min_quantity:
            raise ValueError("max_quantity must be >= min_quantity")
        
        # Normalize synonyms
        self.synonyms = [s.lower().strip() for s in self.synonyms if s]
        
        # Add name as synonym
        name_lower = self.name.lower().strip()
        if name_lower not in self.synonyms:
            self.synonyms.insert(0, name_lower)
    
    @staticmethod
    def _is_valid_canonical_id(canonical_id: str) -> bool:
        """
        Validate canonical ID format.
        
        Args:
            canonical_id: ID to validate
            
        Returns:
            True if valid
        """
        if not canonical_id:
            return False
        
        # Must be uppercase alphanumeric with underscores
        # Must start with letter
        # 3-50 characters
        pattern = r'^[A-Z][A-Z0-9_]{2,49}$'
        return bool(re.match(pattern, canonical_id))
    
    def is_available(self) -> bool:
        """Check if item is available for ordering."""
        return self.availability == ItemAvailability.AVAILABLE
    
    def matches_synonym(self, query: str) -> bool:
        """
        Check if query matches any synonym.
        
        Args:
            query: Search query
            
        Returns:
            True if matches
        """
        query_lower = query.lower().strip()
        
        if not query_lower:
            return False
        
        # Check synonyms
        for synonym in self.synonyms:
            # Exact match
            if query_lower == synonym:
                return True
            
            # Partial match (query contains synonym or vice versa)
            if len(query_lower) >= 3 and len(synonym) >= 3:
                if query_lower in synonym or synonym in query_lower:
                    return True
        
        return False
    
    def validate_quantity(self, quantity: int) -> Tuple[bool, Optional[str]]:
        """
        Validate quantity.
        
        Args:
            quantity: Requested quantity
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if quantity < self.min_quantity:
            return False, f"Minimum quantity is {self.min_quantity}"
        
        if quantity > self.max_quantity:
            return False, f"Maximum quantity is {self.max_quantity}"
        
        return True, None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "canonical_id": self.canonical_id,
            "name": self.name,
            "price": self.price,
            "description": self.description,
            "category": self.category,
            "availability": self.availability.value,
            "synonyms": self.synonyms,
            "max_quantity": self.max_quantity,
            "min_quantity": self.min_quantity
        }


class MenuCache:
    """Thread-safe menu cache with TTL."""
    
    def __init__(self, ttl_seconds: int = 300):
        """
        Initialize cache.
        
        Args:
            ttl_seconds: Time-to-live in seconds
        """
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[List[MenuItem], datetime]] = {}
        self._id_index: Dict[str, Dict[str, MenuItem]] = {}  # tenant_id -> {canonical_id: item}
        self._synonym_index: Dict[str, Dict[str, str]] = {}  # tenant_id -> {synonym: canonical_id}
    
    def get(self, tenant_id: str) -> Optional[List[MenuItem]]:
        """Get cached menu with validation."""
        if tenant_id not in self._cache:
            return None
        
        menu, timestamp = self._cache[tenant_id]
        
        # Check if expired
        age = (datetime.now(timezone.utc) - timestamp).total_seconds()
        if age > self.ttl_seconds:
            self._invalidate(tenant_id)
            return None
        
        return menu
    
    def set(self, tenant_id: str, menu: List[MenuItem]) -> None:
        """
        Cache menu and build indexes.
        
        Args:
            tenant_id: Tenant identifier
            menu: Menu items
        """
        self._cache[tenant_id] = (menu, datetime.now(timezone.utc))
        
        # Build ID index
        id_index = {}
        synonym_index = {}
        
        for item in menu:
            # Index by canonical ID
            id_index[item.canonical_id] = item
            
            # Index synonyms
            for synonym in item.synonyms:
                if synonym in synonym_index:
                    logger.warning(
                        f"Duplicate synonym '{synonym}' for tenant {tenant_id}",
                        extra={
                            "existing_id": synonym_index[synonym],
                            "new_id": item.canonical_id
                        }
                    )
                synonym_index[synonym] = item.canonical_id
        
        self._id_index[tenant_id] = id_index
        self._synonym_index[tenant_id] = synonym_index
        
        logger.debug(
            "Menu indexes built",
            extra={
                "tenant_id": tenant_id,
                "items": len(id_index),
                "synonyms": len(synonym_index)
            }
        )
    
    def get_by_canonical_id(
        self,
        tenant_id: str,
        canonical_id: str
    ) -> Optional[MenuItem]:
        """Get item by canonical ID (fast lookup)."""
        if tenant_id not in self._id_index:
            return None
        
        return self._id_index[tenant_id].get(canonical_id)
    
    def get_by_synonym(
        self,
        tenant_id: str,
        synonym: str
    ) -> Optional[MenuItem]:
        """Get item by synonym (fast lookup)."""
        synonym_lower = synonym.lower().strip()
        
        if tenant_id not in self._synonym_index:
            return None
        
        canonical_id = self._synonym_index[tenant_id].get(synonym_lower)
        if not canonical_id:
            return None
        
        return self._id_index[tenant_id].get(canonical_id)
    
    def _invalidate(self, tenant_id: str) -> None:
        """Invalidate cache and indexes."""
        if tenant_id in self._cache:
            del self._cache[tenant_id]
        if tenant_id in self._id_index:
            del self._id_index[tenant_id]
        if tenant_id in self._synonym_index:
            del self._synonym_index[tenant_id]
    
    def invalidate(self, tenant_id: str) -> None:
        """Public invalidation method."""
        self._invalidate(tenant_id)
    
    def clear(self) -> None:
        """Clear all cache."""
        self._cache.clear()
        self._id_index.clear()
        self._synonym_index.clear()


class MenuEngine:
    """
    Menu data access and matching engine with canonical ID enforcement.
    
    Features:
    - Canonical ID-based item identification
    - Synonym-based fuzzy matching
    - Strict validation
    - Fast indexed lookups
    - Hallucination prevention
    
    CRITICAL RULES:
    - ALL items have canonical IDs
    - Orders use canonical IDs ONLY
    - Prices validated against canonical items
    - AI cannot invent items
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
        self._validation_failures = 0
        
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
        
        # Validate menu
        menu = self._validate_menu(menu, tenant_id)
        
        # Cache it
        if self.cache:
            self.cache.set(tenant_id, menu)
        
        logger.info(
            f"Menu loaded: {len(menu)} items",
            extra={"tenant_id": tenant_id}
        )
        
        return menu
    
    def _validate_menu(
        self,
        menu: List[MenuItem],
        tenant_id: str
    ) -> List[MenuItem]:
        """
        Validate menu items.
        
        Args:
            menu: Menu items
            tenant_id: Tenant identifier
            
        Returns:
            Validated menu items
        """
        validated = []
        canonical_ids = set()
        
        for item in menu:
            # Check for duplicate canonical IDs
            if item.canonical_id in canonical_ids:
                logger.error(
                    f"Duplicate canonical_id: {item.canonical_id}",
                    extra={"tenant_id": tenant_id}
                )
                self._validation_failures += 1
                continue
            
            canonical_ids.add(item.canonical_id)
            validated.append(item)
        
        return validated
    
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
        # For now, return sample menu with canonical IDs
        
        logger.debug(
            "Loading menu from database",
            extra={"tenant_id": tenant_id}
        )
        
        # Sample menu with canonical IDs
        return [
            MenuItem(
                canonical_id="FISH_COMBO",
                name="Fish Combo",
                price=12.99,
                description="2 pieces of fish with fries and coleslaw",
                category="Combos",
                synonyms=["fish combo", "fish", "2 piece fish", "fish meal", "pescado"]
            ),
            MenuItem(
                canonical_id="CHICKEN_COMBO",
                name="Chicken Combo",
                price=11.99,
                description="3 pieces of chicken with fries and coleslaw",
                category="Combos",
                synonyms=["chicken combo", "chicken", "3 piece chicken", "chicken meal", "pollo"]
            ),
            MenuItem(
                canonical_id="SHRIMP_BASKET",
                name="Shrimp Basket",
                price=13.99,
                description="10 pieces of shrimp with fries",
                category="Baskets",
                synonyms=["shrimp basket", "shrimp", "fried shrimp", "camarones"]
            ),
            MenuItem(
                canonical_id="WINGS",
                name="Chicken Wings",
                price=9.99,
                description="6 wings with sauce",
                category="Sides",
                synonyms=["wings", "chicken wings", "buffalo wings", "alitas"]
            ),
            MenuItem(
                canonical_id="FRIES_LARGE",
                name="French Fries (Large)",
                price=2.99,
                description="Large crispy french fries",
                category="Sides",
                synonyms=["fries", "french fries", "large fries", "chips", "papas fritas"]
            ),
            MenuItem(
                canonical_id="COLESLAW",
                name="Coleslaw",
                price=1.99,
                description="Fresh coleslaw",
                category="Sides",
                synonyms=["coleslaw", "slaw", "cole slaw", "ensalada de col"]
            ),
            MenuItem(
                canonical_id="SODA_MEDIUM",
                name="Soft Drink (Medium)",
                price=1.99,
                description="Medium fountain soda",
                category="Drinks",
                synonyms=["soda", "pop", "coke", "drink", "soft drink", "refresco"]
            )
        ]
    
    async def get_item_by_canonical_id(
        self,
        tenant_id: str,
        canonical_id: str
    ) -> Optional[MenuItem]:
        """
        Get item by canonical ID (PREFERRED METHOD).
        
        Args:
            tenant_id: Tenant identifier
            canonical_id: Canonical item ID
            
        Returns:
            Menu item or None
        """
        # Try cache first (fast)
        if self.cache:
            item = self.cache.get_by_canonical_id(tenant_id, canonical_id)
            if item:
                return item
        
        # Fallback to full menu
        menu = await self.get_menu(tenant_id)
        
        for item in menu:
            if item.canonical_id == canonical_id:
                return item
        
        logger.warning(
            f"Item not found by canonical_id: {canonical_id}",
            extra={"tenant_id": tenant_id}
        )
        return None
    
    async def find_item_by_query(
        self,
        tenant_id: str,
        query: str
    ) -> Optional[MenuItem]:
        """
        Find menu item by user query (maps to canonical ID).
        
        Args:
            tenant_id: Tenant identifier
            query: User search query
            
        Returns:
            Matched item or None
        """
        if not query or not query.strip():
            return None
        
        query_clean = query.lower().strip()
        
        # Try cache synonym index first (fast)
        if self.cache:
            item = self.cache.get_by_synonym(tenant_id, query_clean)
            if item and item.is_available():
                logger.debug(
                    f"Item matched via synonym: {item.canonical_id}",
                    extra={"query": query}
                )
                return item
        
        # Fallback to fuzzy matching
        menu = await self.get_menu(tenant_id)
        
        for item in menu:
            if not item.is_available():
                continue
            
            if item.matches_synonym(query_clean):
                logger.debug(
                    f"Item matched: {item.canonical_id}",
                    extra={"query": query}
                )
                return item
        
        logger.debug(
            "No item matched",
            extra={"query": query, "tenant_id": tenant_id}
        )
        return None
    
    async def find_items_by_queries(
        self,
        tenant_id: str,
        queries: List[str]
    ) -> List[MenuItem]:
        """
        Find multiple items by queries.
        
        Args:
            tenant_id: Tenant identifier
            queries: List of search queries
            
        Returns:
            List of matched items (deduplicated by canonical_id)
        """
        items = []
        seen_ids = set()
        
        for query in queries:
            item = await self.find_item_by_query(tenant_id, query)
            if item and item.canonical_id not in seen_ids:
                items.append(item)
                seen_ids.add(item.canonical_id)
        
        return items
    
    async def validate_item_exists(
        self,
        tenant_id: str,
        canonical_id: str
    ) -> bool:
        """
        Validate that item exists (by canonical ID).
        
        CRITICAL: This prevents hallucination - AI cannot order items
        that don't exist.
        
        Args:
            tenant_id: Tenant identifier
            canonical_id: Canonical item ID
            
        Returns:
            True if item exists and is available
        """
        item = await self.get_item_by_canonical_id(tenant_id, canonical_id)
        
        if not item:
            self._validation_failures += 1
            return False
        
        return item.is_available()
    
    async def validate_price(
        self,
        tenant_id: str,
        canonical_id: str,
        expected_price: float
    ) -> Tuple[bool, Optional[float]]:
        """
        Validate price matches canonical item.
        
        Args:
            tenant_id: Tenant identifier
            canonical_id: Canonical item ID
            expected_price: Expected price
            
        Returns:
            Tuple of (is_valid, actual_price)
        """
        item = await self.get_item_by_canonical_id(tenant_id, canonical_id)
        
        if not item:
            return False, None
        
        # Allow small floating point differences
        if abs(item.price - expected_price) < 0.01:
            return True, item.price
        
        logger.warning(
            f"Price mismatch for {canonical_id}: expected ${expected_price:.2f}, actual ${item.price:.2f}",
            extra={"tenant_id": tenant_id}
        )
        return False, item.price
    
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
            List of available items in category
        """
        menu = await self.get_menu(tenant_id)
        
        return [
            item for item in menu
            if item.category.lower() == category.lower()
            and item.is_available()
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
        
        categories = set(
            item.category for item in menu
            if item.is_available()
        )
        return sorted(categories)
    
    def format_menu_for_prompt(
        self,
        menu: List[MenuItem],
        include_descriptions: bool = True,
        include_canonical_ids: bool = True
    ) -> str:
        """
        Format menu for AI prompt.
        
        IMPORTANT: Include canonical IDs so AI learns to use them.
        
        Args:
            menu: Menu items
            include_descriptions: Include item descriptions
            include_canonical_ids: Include canonical IDs
            
        Returns:
            Formatted menu text
        """
        # Group by category
        by_category: Dict[str, List[MenuItem]] = {}
        for item in menu:
            if not item.is_available():
                continue
            
            if item.category not in by_category:
                by_category[item.category] = []
            by_category[item.category].append(item)
        
        # Format each category
        lines = []
        for category in sorted(by_category.keys()):
            lines.append(f"\n{category}:")
            
            for item in by_category[category]:
                if include_canonical_ids:
                    line = f"  - {item.name} [ID: {item.canonical_id}] (${item.price:.2f})"
                else:
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
            "cache_misses": self._cache_misses,
            "validation_failures": self._validation_failures
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


def set_default_engine(engine: MenuEngine) -> None:
    """
    Set default menu engine.
    
    Args:
        engine: MenuEngine instance
    """
    global _default_engine
    _default_engine = engine
    logger.info("Default menu engine set")


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
