"""
Upsell Engine
=============
Intelligent upsell recommendation system with strict controls.

Responsibilities:
- Suggest complementary items at appropriate times
- Prevent spam (frequency limiting)
- Track upsell history per call
- Calculate upsell relevance
- Format suggestions for AI
- PHASE-LOCKED upsells (only during specific order states)
- NEVER suggest during finalization/confirmation

CRITICAL RULES:
- Upsells only in BUILDING state
- Never during REVIEWING or CONFIRMED states
- Maximum 2 upsells per call
- Minimum 45 seconds between upsells
- Use canonical IDs from menu
- No duplicate suggestions
"""

import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class UpsellPhase(Enum):
    """When upsells can be triggered."""
    INITIAL = "initial"  # After first item added
    MID_ORDER = "mid_order"  # During building
    NEVER = "never"  # Blocked phases


class UpsellCategory(Enum):
    """Upsell categories."""
    DRINKS = "drinks"
    SIDES = "sides"
    DESSERTS = "desserts"
    UPGRADES = "upgrades"
    COMBO = "combo"


@dataclass
class UpsellRule:
    """
    Rule for suggesting upsells.
    
    Uses canonical IDs from menu_engine.
    """
    
    rule_id: str
    category: UpsellCategory
    trigger_canonical_ids: List[str]  # Canonical IDs that trigger this rule
    suggested_canonical_ids: List[str]  # Canonical IDs to suggest
    max_suggestions: int = 1
    phase: UpsellPhase = UpsellPhase.MID_ORDER
    min_order_items: int = 1  # Min items in order to trigger
    max_order_items: int = 99  # Max items in order to trigger
    
    def __post_init__(self):
        """Validate rule."""
        if not self.rule_id:
            raise ValueError("rule_id required")
        
        if not self.trigger_canonical_ids:
            raise ValueError("trigger_canonical_ids required")
        
        if not self.suggested_canonical_ids:
            raise ValueError("suggested_canonical_ids required")
        
        if self.max_suggestions < 1:
            raise ValueError("max_suggestions must be >= 1")


class UpsellTracker:
    """
    Tracks upsell history per call with strict limits.
    
    Prevents:
    - Suggesting same item multiple times
    - Too many upsells (max 2 per call)
    - Upsells too frequently (min 45s between)
    - Upsells during wrong order states
    """
    
    MAX_UPSELLS_PER_CALL = 2
    MIN_INTERVAL_SECONDS = 45.0
    
    def __init__(
        self,
        call_id: str,
        max_upsells: int = MAX_UPSELLS_PER_CALL,
        min_interval_seconds: float = MIN_INTERVAL_SECONDS
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
        
        # Tracking (use canonical IDs)
        self.offered_canonical_ids: Set[str] = set()
        self.accepted_canonical_ids: Set[str] = set()
        self.rejected_canonical_ids: Set[str] = set()
        
        # Offer history
        self.offer_history: List[Tuple[str, datetime]] = []
        
        # Timestamps
        self.last_offer_time: Optional[datetime] = None
        self.created_at = datetime.now(timezone.utc)
        
        # Counters
        self.total_offers = 0
        self.total_acceptances = 0
        self.total_rejections = 0
        
        # State tracking
        self.blocked_until: Optional[datetime] = None
    
    def can_offer(self) -> Tuple[bool, Optional[str]]:
        """
        Check if can offer upsell now.
        
        Returns:
            Tuple of (can_offer, reason_if_not)
        """
        # Check if blocked
        if self.blocked_until:
            if datetime.now(timezone.utc) < self.blocked_until:
                return False, "temporarily_blocked"
            else:
                self.blocked_until = None
        
        # Check max upsells
        if self.total_offers >= self.max_upsells:
            return False, "max_upsells_reached"
        
        # Check time interval
        if self.last_offer_time:
            elapsed = (
                datetime.now(timezone.utc) - self.last_offer_time
            ).total_seconds()
            
            if elapsed < self.min_interval_seconds:
                return False, f"too_soon ({elapsed:.0f}s < {self.min_interval_seconds}s)"
        
        return True, None
    
    def record_offer(self, canonical_id: str) -> None:
        """
        Record upsell offer.
        
        Args:
            canonical_id: Offered item canonical ID
        """
        now = datetime.now(timezone.utc)
        
        self.offered_canonical_ids.add(canonical_id)
        self.offer_history.append((canonical_id, now))
        self.last_offer_time = now
        self.total_offers += 1
        
        logger.info(
            f"Upsell offered: {canonical_id} (total: {self.total_offers}/{self.max_upsells})",
            extra={"call_id": self.call_id}
        )
    
    def record_acceptance(self, canonical_id: str) -> None:
        """
        Record upsell acceptance.
        
        Args:
            canonical_id: Accepted item canonical ID
        """
        self.accepted_canonical_ids.add(canonical_id)
        self.total_acceptances += 1
        
        logger.info(
            f"Upsell ACCEPTED: {canonical_id}",
            extra={
                "call_id": self.call_id,
                "conversion_rate": self.get_conversion_rate()
            }
        )
    
    def record_rejection(self, canonical_id: str) -> None:
        """
        Record upsell rejection.
        
        Args:
            canonical_id: Rejected item canonical ID
        """
        self.rejected_canonical_ids.add(canonical_id)
        self.total_rejections += 1
        
        logger.debug(
            f"Upsell rejected: {canonical_id}",
            extra={"call_id": self.call_id}
        )
    
    def block_temporarily(self, seconds: float) -> None:
        """
        Block upsells for specified seconds.
        
        Args:
            seconds: Seconds to block
        """
        self.blocked_until = (
            datetime.now(timezone.utc) + timedelta(seconds=seconds)
        )
        
        logger.debug(
            f"Upsells blocked for {seconds}s",
            extra={"call_id": self.call_id}
        )
    
    def was_offered(self, canonical_id: str) -> bool:
        """Check if item was already offered."""
        return canonical_id in self.offered_canonical_ids
    
    def was_rejected(self, canonical_id: str) -> bool:
        """Check if item was rejected."""
        return canonical_id in self.rejected_canonical_ids
    
    def get_conversion_rate(self) -> float:
        """
        Calculate conversion rate.
        
        Returns:
            Conversion rate (0-1)
        """
        if self.total_offers == 0:
            return 0.0
        
        return self.total_acceptances / self.total_offers
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            "total_offers": self.total_offers,
            "total_acceptances": self.total_acceptances,
            "total_rejections": self.total_rejections,
            "conversion_rate": self.get_conversion_rate(),
            "offered_items": list(self.offered_canonical_ids),
            "accepted_items": list(self.accepted_canonical_ids),
            "can_offer_more": self.can_offer()[0]
        }


class UpsellEngine:
    """
    Upsell recommendation engine with phase-locking.
    
    Features:
    - Rule-based suggestions using canonical IDs
    - Phase-locked (only in BUILDING state)
    - Strict frequency limiting
    - Per-call tracking
    - Order state awareness
    
    SAFETY:
    - Never suggests during REVIEWING or CONFIRMED
    - Maximum 2 upsells per call
    - Minimum 45 seconds between upsells
    - No duplicate suggestions
    """
    
    def __init__(
        self,
        max_upsells_per_call: int = 2,
        min_interval_seconds: float = 45.0
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
        
        # Upsell rules (with canonical IDs)
        self._rules = self._build_default_rules()
        
        logger.info(
            "UpsellEngine initialized",
            extra={
                "max_per_call": max_upsells_per_call,
                "min_interval": min_interval_seconds,
                "rules": len(self._rules)
            }
        )
    
    def _build_default_rules(self) -> List[UpsellRule]:
        """
        Build default upsell rules with canonical IDs.
        
        Returns:
            List of upsell rules
        """
        return [
            # Drinks with combos
            UpsellRule(
                rule_id="drinks_with_combos",
                category=UpsellCategory.DRINKS,
                trigger_canonical_ids=["FISH_COMBO", "CHICKEN_COMBO", "SHRIMP_BASKET"],
                suggested_canonical_ids=["SODA_MEDIUM"],
                max_suggestions=1,
                phase=UpsellPhase.INITIAL,
                min_order_items=1,
                max_order_items=3
            ),
            # Sides with meals
            UpsellRule(
                rule_id="sides_with_meals",
                category=UpsellCategory.SIDES,
                trigger_canonical_ids=["FISH_COMBO", "CHICKEN_COMBO"],
                suggested_canonical_ids=["FRIES_LARGE", "COLESLAW"],
                max_suggestions=1,
                phase=UpsellPhase.MID_ORDER,
                min_order_items=1,
                max_order_items=5
            ),
            # Wings as addon
            UpsellRule(
                rule_id="wings_addon",
                category=UpsellCategory.SIDES,
                trigger_canonical_ids=["FISH_COMBO", "CHICKEN_COMBO", "SHRIMP_BASKET"],
                suggested_canonical_ids=["WINGS"],
                max_suggestions=1,
                phase=UpsellPhase.MID_ORDER,
                min_order_items=1,
                max_order_items=4
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
    
    def can_suggest_upsells(
        self,
        call_id: str,
        order_state: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if upsells can be suggested based on order state.
        
        CRITICAL: Upsells only allowed in BUILDING state.
        
        Args:
            call_id: Call identifier
            order_state: Current order state (from OrderState enum)
            
        Returns:
            Tuple of (can_suggest, reason_if_not)
        """
        # Phase lock: only allow in BUILDING state
        if order_state not in {"building", "empty"}:
            return False, f"wrong_order_state:{order_state}"
        
        # Check tracker limits
        tracker = self._get_tracker(call_id)
        return tracker.can_offer()
    
    def suggest_upsells(
        self,
        call_id: str,
        order_canonical_ids: List[str],
        order_state: str,
        order_item_count: int = 0
    ) -> List[str]:
        """
        Suggest upsells based on current order.
        
        Args:
            call_id: Call identifier
            order_canonical_ids: List of canonical IDs in order
            order_state: Current order state
            order_item_count: Number of items in order
            
        Returns:
            List of suggested canonical IDs
        """
        # Check if can suggest
        can_suggest, reason = self.can_suggest_upsells(call_id, order_state)
        
        if not can_suggest:
            logger.debug(
                f"Cannot suggest upsells: {reason}",
                extra={"call_id": call_id}
            )
            return []
        
        tracker = self._get_tracker(call_id)
        
        # Convert order items to set for faster lookup
        order_items_set = set(order_canonical_ids)
        
        # Find matching rules
        suggestions = []
        
        for rule in self._rules:
            # Check item count limits
            if order_item_count < rule.min_order_items:
                continue
            
            if order_item_count > rule.max_order_items:
                continue
            
            # Check if any trigger items in order
            has_trigger = any(
                canonical_id in order_items_set
                for canonical_id in rule.trigger_canonical_ids
            )
            
            if not has_trigger:
                continue
            
            # Get suggestions that haven't been offered/rejected
            for suggested_id in rule.suggested_canonical_ids:
                # Skip if already in order
                if suggested_id in order_items_set:
                    continue
                
                # Skip if already offered
                if tracker.was_offered(suggested_id):
                    continue
                
                # Skip if rejected
                if tracker.was_rejected(suggested_id):
                    continue
                
                suggestions.append(suggested_id)
                tracker.record_offer(suggested_id)
                
                logger.info(
                    f"Upsell suggested: {suggested_id} (rule: {rule.rule_id})",
                    extra={"call_id": call_id}
                )
                
                # Limit suggestions per rule
                if len(suggestions) >= rule.max_suggestions:
                    break
            
            # Stop after first matching rule
            if suggestions:
                break
        
        return suggestions
    
    def record_acceptance(
        self,
        call_id: str,
        canonical_id: str
    ) -> None:
        """
        Record that user accepted an upsell.
        
        Args:
            call_id: Call identifier
            canonical_id: Accepted item canonical ID
        """
        tracker = self._get_tracker(call_id)
        tracker.record_acceptance(canonical_id)
    
    def record_rejection(
        self,
        call_id: str,
        canonical_id: str
    ) -> None:
        """
        Record that user rejected an upsell.
        
        Args:
            call_id: Call identifier
            canonical_id: Rejected item canonical ID
        """
        tracker = self._get_tracker(call_id)
        tracker.record_rejection(canonical_id)
    
    def block_upsells(
        self,
        call_id: str,
        seconds: float = 60.0
    ) -> None:
        """
        Temporarily block upsells for a call.
        
        Use when customer shows frustration with upsells.
        
        Args:
            call_id: Call identifier
            seconds: Seconds to block
        """
        tracker = self._get_tracker(call_id)
        tracker.block_temporarily(seconds)
        
        logger.info(
            f"Upsells blocked for {seconds}s",
            extra={"call_id": call_id}
        )
    
    def get_tracker_stats(self, call_id: str) -> Dict[str, Any]:
        """
        Get statistics for call.
        
        Args:
            call_id: Call identifier
            
        Returns:
            Statistics dictionary
        """
        if call_id not in self._trackers:
            return {
                "total_offers": 0,
                "total_acceptances": 0,
                "total_rejections": 0,
                "conversion_rate": 0.0,
                "can_offer_more": True
            }
        
        tracker = self._trackers[call_id]
        return tracker.get_stats()
    
    def cleanup_tracker(self, call_id: str) -> None:
        """
        Remove tracker for call.
        
        Args:
            call_id: Call identifier
        """
        if call_id in self._trackers:
            tracker = self._trackers[call_id]
            
            logger.info(
                f"Upsell tracker cleanup - conversion: {tracker.get_conversion_rate():.1%}",
                extra={
                    "call_id": call_id,
                    "offers": tracker.total_offers,
                    "acceptances": tracker.total_acceptances
                }
            )
            
            del self._trackers[call_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get engine statistics.
        
        Returns:
            Statistics dictionary
        """
        total_offers = 0
        total_acceptances = 0
        total_rejections = 0
        
        for tracker in self._trackers.values():
            total_offers += tracker.total_offers
            total_acceptances += tracker.total_acceptances
            total_rejections += tracker.total_rejections
        
        conversion_rate = 0.0
        if total_offers > 0:
            conversion_rate = total_acceptances / total_offers
        
        return {
            "active_trackers": len(self._trackers),
            "total_offers": total_offers,
            "total_acceptances": total_acceptances,
            "total_rejections": total_rejections,
            "conversion_rate": conversion_rate,
            "rules_count": len(self._rules)
        }


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


def set_default_engine(engine: UpsellEngine) -> None:
    """
    Set default upsell engine.
    
    Args:
        engine: UpsellEngine instance
    """
    global _default_engine
    _default_engine = engine
    logger.info("Default upsell engine set")
