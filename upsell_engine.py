"""
Upsell Engine
=============
Intelligent upsell recommendation system.

Responsibilities:
- Suggest complementary items
- Prevent spam (frequency limiting)
- Track upsell history per call
- Calculate upsell relevance
- Format suggestions for AI
"""

import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timezone, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class UpsellCategory(Enum):
    """Upsell categories."""
    DRINKS = "drinks"
    SIDES = "sides"
    DESSERTS = "desserts"
    UPGRADES = "upgrades"


class UpsellRule:
    """Rule for suggesting upsells."""
    
    def __init__(
        self,
        category: UpsellCategory,
        trigger_items: List[str],
        suggested_items: List[str],
        max_suggestions: int = 1
    ):
        """
        Initialize upsell rule.
        
        Args:
            category: Upsell category
            trigger_items: Items that trigger this rule
            suggested_items: Items to suggest
            max_suggestions: Max items to suggest
        """
        self.category = category
        self.trigger_items = trigger_items
        self.suggested_items = suggested_items
        self.max_suggestions = max_suggestions


class UpsellTracker:
    """
    Tracks upsell history per call.
    
    Prevents:
    - Suggesting same item multiple times
    - Too many upsells
    - Upsells too frequently
    """
    
    def __init__(
        self,
        call_id: str,
        max_upsells: int = 3,
        min_interval_seconds: float = 30.0
    ):
        """
        Initialize tracker.
        
        Args:
            call_id: Call identifier
            max_upsells: Max total upsells per call
            min_interval_seconds: Min seconds between upsells
        """
        self.call_id = call_id
        self.max_upsells = max_upsells
        self.min_interval_seconds = min_interval_seconds
        
        # Tracking
        self.offered_items: Set[str] = set()
        self.accepted_items: Set[str] = set()
        self.rejected_items: Set[str] = set()
        
        # Timestamps
        self.last_offer_time: Optional[datetime] = None
        self.created_at = datetime.now(timezone.utc)
        
        # Counters
        self.total_offers = 0
        self.total_acceptances = 0
        self.total_rejections = 0
    
    def can_offer(self) -> bool:
        """
        Check if can offer upsell now.
        
        Returns:
            True if can offer
        """
        # Check max upsells
        if self.total_offers >= self.max_upsells:
            return False
        
        # Check time interval
        if self.last_offer_time:
            elapsed = (
                datetime.now(timezone.utc) - self.last_offer_time
            ).total_seconds()
            
            if elapsed < self.min_interval_seconds:
                return False
        
        return True
    
    def record_offer(self, item_id: str) -> None:
        """
        Record upsell offer.
        
        Args:
            item_id: Offered item ID
        """
        self.offered_items.add(item_id)
        self.last_offer_time = datetime.now(timezone.utc)
        self.total_offers += 1
        
        logger.debug(
            f"Upsell offered: {item_id}",
            extra={"call_id": self.call_id}
        )
    
    def record_acceptance(self, item_id: str) -> None:
        """
        Record upsell acceptance.
        
        Args:
            item_id: Accepted item ID
        """
        self.accepted_items.add(item_id)
        self.total_acceptances += 1
        
        logger.info(
            f"Upsell accepted: {item_id}",
            extra={"call_id": self.call_id}
        )
    
    def record_rejection(self, item_id: str) -> None:
        """
        Record upsell rejection.
        
        Args:
            item_id: Rejected item ID
        """
        self.rejected_items.add(item_id)
        self.total_rejections += 1
        
        logger.debug(
            f"Upsell rejected: {item_id}",
            extra={"call_id": self.call_id}
        )
    
    def was_offered(self, item_id: str) -> bool:
        """Check if item was already offered."""
        return item_id in self.offered_items
    
    def was_rejected(self, item_id: str) -> bool:
        """Check if item was rejected."""
        return item_id in self.rejected_items
    
    def get_conversion_rate(self) -> float:
        """
        Calculate conversion rate.
        
        Returns:
            Conversion rate (0-1)
        """
        if self.total_offers == 0:
            return 0.0
        
        return self.total_acceptances / self.total_offers


class UpsellEngine:
    """
    Upsell recommendation engine.
    
    Features:
    - Rule-based suggestions
    - Frequency limiting
    - Per-call tracking
    - Contextual relevance
    """
    
    def __init__(
        self,
        max_upsells_per_call: int = 3,
        min_interval_seconds: float = 30.0
    ):
        """
        Initialize upsell engine.
        
        Args:
            max_upsells_per_call: Max upsells per call
            min_interval_seconds: Min seconds between upsells
        """
        self.max_upsells_per_call = max_upsells_per_call
        self.min_interval_seconds = min_interval_seconds
        
        # Trackers per call
        self._trackers: Dict[str, UpsellTracker] = {}
        
        # Upsell rules
        self._rules = self._build_default_rules()
        
        logger.info(
            "UpsellEngine initialized",
            extra={
                "max_per_call": max_upsells_per_call,
                "min_interval": min_interval_seconds
            }
        )
    
    def _build_default_rules(self) -> List[UpsellRule]:
        """
        Build default upsell rules.
        
        Returns:
            List of upsell rules
        """
        return [
            # Drinks with combos
            UpsellRule(
                category=UpsellCategory.DRINKS,
                trigger_items=["fish_combo", "chicken_combo"],
                suggested_items=["soda"],
                max_suggestions=1
            ),
            # Sides with entrees
            UpsellRule(
                category=UpsellCategory.SIDES,
                trigger_items=["fish_combo", "chicken_combo"],
                suggested_items=["fries", "coleslaw"],
                max_suggestions=1
            )
        ]
    
    def _get_tracker(self, call_id: str) -> UpsellTracker:
        """
        Get or create tracker for call.
        
        Args:
            call_id: Call identifier
            
        Returns:
            Upsell tracker
        """
        if call_id not in self._trackers:
            self._trackers[call_id] = UpsellTracker(
                call_id=call_id,
                max_upsells=self.max_upsells_per_call,
                min_interval_seconds=self.min_interval_seconds
            )
        
        return self._trackers[call_id]
    
    def suggest_upsells(
        self,
        call_id: str,
        current_order_items: List[str],
        menu_items: List[Any]
    ) -> List[str]:
        """
        Suggest upsells based on current order.
        
        Args:
            call_id: Call identifier
            current_order_items: List of item IDs in order
            menu_items: Available menu items
            
        Returns:
            List of suggested item names
        """
        tracker = self._get_tracker(call_id)
        
        # Check if can offer
        if not tracker.can_offer():
            return []
        
        # Find matching rules
        suggestions = []
        
        for rule in self._rules:
            # Check if any trigger items in order
            has_trigger = any(
                item in current_order_items
                for item in rule.trigger_items
            )
            
            if not has_trigger:
                continue
            
            # Get suggestions that haven't been offered
            for suggested_item in rule.suggested_items:
                if tracker.was_offered(suggested_item):
                    continue
                
                if tracker.was_rejected(suggested_item):
                    continue
                
                suggestions.append(suggested_item)
                tracker.record_offer(suggested_item)
                
                # Limit suggestions per rule
                if len(suggestions) >= rule.max_suggestions:
                    break
            
            # Stop after first matching rule
            if suggestions:
                break
        
        return suggestions
    
    def cleanup_tracker(self, call_id: str) -> None:
        """
        Remove tracker for call.
        
        Args:
            call_id: Call identifier
        """
        if call_id in self._trackers:
            del self._trackers[call_id]


# Default instance
_default_engine: Optional[UpsellEngine] = None


def get_default_engine() -> UpsellEngine:
    """
    Get or create default upsell engine.
    
    Returns:
        Default engine instance
    """
    global _default_engine
    
    if _default_engine is None:
        _default_engine = UpsellEngine()
    
    return _default_engine
