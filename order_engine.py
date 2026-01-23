"""
Order Engine
============
Transactional order state machine with strict validation.

Responsibilities:
- Manage order lifecycle with formal state machine
- Add/remove items using CANONICAL IDs ONLY
- Calculate totals with validation
- Enforce immutability after confirmation
- Prevent order tampering
- Support safe rollback

CRITICAL RULES:
- All items must use canonical_id from menu_engine
- State transitions are strictly enforced
- CONFIRMED orders are IMMUTABLE
- Price validation against menu required
- Maximum order limits enforced
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy
import uuid
import asyncio

logger = logging.getLogger(__name__)


class OrderState(Enum):
    """
    Order lifecycle states with strict transitions.
    
    State diagram:
    EMPTY -> BUILDING -> REVIEWING -> CONFIRMED -> COMPLETED
                 ↓         ↓           ↓
              CANCELLED CANCELLED  CANCELLED
    """
    EMPTY = "empty"
    BUILDING = "building"
    REVIEWING = "reviewing"  # AI presenting order for confirmation
    CONFIRMED = "confirmed"  # Customer confirmed (IMMUTABLE after this)
    COMPLETED = "completed"  # Order submitted to POS
    CANCELLED = "cancelled"
    
    def is_mutable(self) -> bool:
        """Check if order can be modified in this state."""
        return self in {OrderState.EMPTY, OrderState.BUILDING, OrderState.REVIEWING}
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in {OrderState.COMPLETED, OrderState.CANCELLED}


# Valid state transitions
VALID_TRANSITIONS: Dict[OrderState, Set[OrderState]] = {
    OrderState.EMPTY: {OrderState.BUILDING, OrderState.CANCELLED},
    OrderState.BUILDING: {OrderState.REVIEWING, OrderState.EMPTY, OrderState.CANCELLED},
    OrderState.REVIEWING: {OrderState.CONFIRMED, OrderState.BUILDING, OrderState.CANCELLED},
    OrderState.CONFIRMED: {OrderState.COMPLETED, OrderState.CANCELLED},
    OrderState.COMPLETED: set(),  # Terminal
    OrderState.CANCELLED: set()   # Terminal
}


@dataclass
class OrderItem:
    """
    Single item in an order.
    
    CRITICAL: Uses canonical_id from menu_engine.
    """
    
    canonical_id: str  # e.g., "FISH_COMBO"
    name: str  # Display name (for readback)
    price: float  # Validated against menu
    quantity: int = 1
    modifications: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    
    # Validation metadata
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Validate after initialization."""
        if not self.canonical_id or not self.canonical_id.strip():
            raise ValueError("canonical_id is required")
        
        if self.quantity < 1:
            raise ValueError("Quantity must be at least 1")
        
        if self.quantity > 99:
            raise ValueError("Quantity cannot exceed 99")
        
        if self.price < 0:
            raise ValueError("Price cannot be negative")
        
        if self.price > 999.99:
            raise ValueError("Price exceeds maximum ($999.99)")
    
    @property
    def subtotal(self) -> float:
        """Calculate subtotal."""
        return round(self.price * self.quantity, 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "canonical_id": self.canonical_id,
            "name": self.name,
            "price": self.price,
            "quantity": self.quantity,
            "modifications": self.modifications,
            "notes": self.notes,
            "subtotal": self.subtotal
        }


class StateTransitionError(Exception):
    """Invalid state transition."""
    pass


class Order:
    """
    Transactional order with strict state machine.
    
    States:
    - EMPTY: No items
    - BUILDING: Adding/modifying items
    - REVIEWING: AI presenting for confirmation
    - CONFIRMED: Customer confirmed (IMMUTABLE)
    - COMPLETED: Submitted to POS
    - CANCELLED: Order cancelled
    
    SAFETY:
    - State transitions validated
    - CONFIRMED orders cannot be modified
    - All items validated against menu
    - Maximum item limits enforced
    """
    
    MAX_ITEMS = 20
    MAX_TOTAL = 999.99
    
    def __init__(
        self,
        order_id: str,
        tenant_id: str
    ):
        """
        Initialize order.
        
        Args:
            order_id: Order identifier
            tenant_id: Tenant identifier
        """
        self.order_id = order_id
        self.tenant_id = tenant_id
        
        # State machine
        self.state = OrderState.EMPTY
        self._state_history: List[Tuple[OrderState, datetime]] = [
            (OrderState.EMPTY, datetime.now(timezone.utc))
        ]
        
        # Items
        self.items: List[OrderItem] = []
        
        # Metadata
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
        self.confirmed_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        
        # Customer info
        self.customer_name: Optional[str] = None
        self.customer_phone: Optional[str] = None
        
        # Snapshot for rollback
        self._snapshot: Optional[List[OrderItem]] = None
        
        # Validation
        self._validated_canonical_ids: Set[str] = set()
        
        logger.info(
            "Order created",
            extra={
                "order_id": order_id,
                "tenant_id": tenant_id
            }
        )
    
    def _transition_state(self, new_state: OrderState) -> None:
        """
        Transition to new state with validation.
        
        Args:
            new_state: Target state
            
        Raises:
            StateTransitionError: If transition is invalid
        """
        # Check if transition is valid
        valid_targets = VALID_TRANSITIONS.get(self.state, set())
        
        if new_state not in valid_targets:
            raise StateTransitionError(
                f"Invalid transition: {self.state.value} -> {new_state.value}"
            )
        
        old_state = self.state
        self.state = new_state
        self._state_history.append((new_state, datetime.now(timezone.utc)))
        
        logger.info(
            f"Order state transition: {old_state.value} -> {new_state.value}",
            extra={"order_id": self.order_id}
        )
    
    def _check_mutable(self) -> None:
        """
        Check if order can be modified.
        
        Raises:
            ValueError: If order is immutable
        """
        if not self.state.is_mutable():
            raise ValueError(
                f"Cannot modify order in {self.state.value} state"
            )
    
    def add_item(
        self,
        canonical_id: str,
        name: str,
        price: float,
        quantity: int = 1,
        modifications: Optional[List[str]] = None,
        notes: Optional[str] = None,
        validated: bool = False
    ) -> None:
        """
        Add item to order using canonical ID.
        
        Args:
            canonical_id: Canonical item ID from menu
            name: Display name
            price: Item price (must match menu)
            quantity: Quantity
            modifications: Modifications list
            notes: Optional notes
            validated: Whether item was validated against menu
            
        Raises:
            ValueError: If order cannot be modified or limits exceeded
        """
        self._check_mutable()
        
        # Check item limit
        if len(self.items) >= self.MAX_ITEMS:
            raise ValueError(f"Cannot exceed {self.MAX_ITEMS} items")
        
        # Track validated canonical IDs
        if validated:
            self._validated_canonical_ids.add(canonical_id)
        
        # Check if item already exists
        for existing in self.items:
            if existing.canonical_id == canonical_id:
                # Update quantity
                new_qty = existing.quantity + quantity
                if new_qty > 99:
                    raise ValueError("Item quantity cannot exceed 99")
                
                existing.quantity = new_qty
                self._mark_updated()
                
                logger.info(
                    f"Item quantity updated: {name} x{existing.quantity}",
                    extra={"order_id": self.order_id}
                )
                return
        
        # Add new item
        item = OrderItem(
            canonical_id=canonical_id,
            name=name,
            price=price,
            quantity=quantity,
            modifications=modifications or [],
            notes=notes
        )
        
        self.items.append(item)
        self._mark_updated()
        
        # Transition from EMPTY to BUILDING
        if self.state == OrderState.EMPTY:
            self._transition_state(OrderState.BUILDING)
        
        logger.info(
            f"Item added: {canonical_id} ({name}) x{quantity}",
            extra={"order_id": self.order_id}
        )
    
    def remove_item(self, canonical_id: str) -> bool:
        """
        Remove item by canonical ID.
        
        Args:
            canonical_id: Canonical item ID
            
        Returns:
            True if removed
            
        Raises:
            ValueError: If order cannot be modified
        """
        self._check_mutable()
        
        for i, item in enumerate(self.items):
            if item.canonical_id == canonical_id:
                removed = self.items.pop(i)
                self._mark_updated()
                
                # Transition to EMPTY if no items
                if not self.items and self.state == OrderState.BUILDING:
                    self._transition_state(OrderState.EMPTY)
                
                logger.info(
                    f"Item removed: {canonical_id} ({removed.name})",
                    extra={"order_id": self.order_id}
                )
                return True
        
        return False
    
    def update_quantity(
        self,
        canonical_id: str,
        quantity: int
    ) -> bool:
        """
        Update item quantity.
        
        Args:
            canonical_id: Canonical item ID
            quantity: New quantity (0 to remove)
            
        Returns:
            True if updated
        """
        self._check_mutable()
        
        if quantity < 0 or quantity > 99:
            raise ValueError("Quantity must be 0-99")
        
        if quantity == 0:
            return self.remove_item(canonical_id)
        
        for item in self.items:
            if item.canonical_id == canonical_id:
                item.quantity = quantity
                self._mark_updated()
                
                logger.info(
                    f"Quantity updated: {canonical_id} x{quantity}",
                    extra={"order_id": self.order_id}
                )
                return True
        
        return False
    
    def clear(self) -> None:
        """Clear all items."""
        self._check_mutable()
        
        self.items.clear()
        self._validated_canonical_ids.clear()
        
        if self.state == OrderState.BUILDING:
            self._transition_state(OrderState.EMPTY)
        
        self._mark_updated()
        
        logger.info(
            "Order cleared",
            extra={"order_id": self.order_id}
        )
    
    def create_snapshot(self) -> None:
        """Create snapshot for rollback."""
        self._snapshot = deepcopy(self.items)
        logger.debug(
            f"Snapshot created ({len(self.items)} items)",
            extra={"order_id": self.order_id}
        )
    
    def rollback(self) -> bool:
        """
        Rollback to snapshot.
        
        Returns:
            True if rolled back
        """
        if self._snapshot is None:
            return False
        
        self._check_mutable()
        
        self.items = deepcopy(self._snapshot)
        self._mark_updated()
        
        logger.info(
            "Order rolled back",
            extra={"order_id": self.order_id}
        )
        return True
    
    def start_review(self) -> None:
        """Mark order for customer review."""
        if self.is_empty():
            raise ValueError("Cannot review empty order")
        
        if self.state != OrderState.BUILDING:
            raise StateTransitionError(
                f"Can only start review from BUILDING state, not {self.state.value}"
            )
        
        # Validate total
        if self.get_total() > self.MAX_TOTAL:
            raise ValueError(f"Order total exceeds maximum (${self.MAX_TOTAL})")
        
        self._transition_state(OrderState.REVIEWING)
    
    def return_to_building(self) -> None:
        """Return to building from review."""
        if self.state != OrderState.REVIEWING:
            raise StateTransitionError(
                f"Can only return to building from REVIEWING state"
            )
        
        self._transition_state(OrderState.BUILDING)
    
    def confirm(
        self,
        customer_name: Optional[str] = None,
        customer_phone: Optional[str] = None
    ) -> None:
        """
        Confirm order (makes IMMUTABLE).
        
        Args:
            customer_name: Customer name
            customer_phone: Customer phone
            
        Raises:
            ValueError: If order cannot be confirmed
            StateTransitionError: If not in REVIEWING state
        """
        if self.is_empty():
            raise ValueError("Cannot confirm empty order")
        
        if self.state != OrderState.REVIEWING:
            raise StateTransitionError(
                f"Can only confirm from REVIEWING state, not {self.state.value}"
            )
        
        # Final validation
        if self.get_total() > self.MAX_TOTAL:
            raise ValueError(f"Order total exceeds maximum (${self.MAX_TOTAL})")
        
        self._transition_state(OrderState.CONFIRMED)
        self.confirmed_at = datetime.now(timezone.utc)
        
        # Set customer info
        self.customer_name = customer_name
        self.customer_phone = customer_phone
        
        logger.info(
            f"Order CONFIRMED (IMMUTABLE): {self.get_item_count()} items, ${self.get_total():.2f}",
            extra={
                "order_id": self.order_id,
                "total": self.get_total()
            }
        )
    
    def complete(self) -> None:
        """Mark order as completed (submitted to POS)."""
        if self.state != OrderState.CONFIRMED:
            raise StateTransitionError(
                "Can only complete CONFIRMED orders"
            )
        
        self._transition_state(OrderState.COMPLETED)
        self.completed_at = datetime.now(timezone.utc)
        
        logger.info(
            "Order completed",
            extra={"order_id": self.order_id}
        )
    
    def cancel(self, reason: str = "user_requested") -> None:
        """
        Cancel order.
        
        Args:
            reason: Cancellation reason
        """
        if self.state.is_terminal():
            raise ValueError(f"Cannot cancel {self.state.value} order")
        
        self._transition_state(OrderState.CANCELLED)
        
        logger.info(
            f"Order cancelled: {reason}",
            extra={"order_id": self.order_id}
        )
    
    def get_total(self) -> float:
        """Calculate order total."""
        return round(sum(item.subtotal for item in self.items), 2)
    
    def get_item_count(self) -> int:
        """Get total item count."""
        return sum(item.quantity for item in self.items)
    
    def is_empty(self) -> bool:
        """Check if order is empty."""
        return len(self.items) == 0
    
    def is_confirmed(self) -> bool:
        """Check if order is confirmed (immutable)."""
        return self.state == OrderState.CONFIRMED
    
    def is_terminal(self) -> bool:
        """Check if in terminal state."""
        return self.state.is_terminal()
    
    def get_unvalidated_items(self) -> List[str]:
        """
        Get list of canonical IDs that weren't validated against menu.
        
        Returns:
            List of unvalidated canonical IDs
        """
        return [
            item.canonical_id for item in self.items
            if item.canonical_id not in self._validated_canonical_ids
        ]
    
    def _mark_updated(self) -> None:
        """Mark order as updated."""
        self.updated_at = datetime.now(timezone.utc)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get order summary."""
        return {
            "order_id": self.order_id,
            "tenant_id": self.tenant_id,
            "state": self.state.value,
            "items": [item.to_dict() for item in self.items],
            "item_count": self.get_item_count(),
            "total": self.get_total(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "confirmed_at": (
                self.confirmed_at.isoformat()
                if self.confirmed_at
                else None
            ),
            "customer": {
                "name": self.customer_name,
                "phone": self.customer_phone
            },
            "validation": {
                "unvalidated_items": self.get_unvalidated_items()
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.get_summary()


class OrderEngine:
    """
    Order management engine with validation.
    
    Manages active orders per call with menu validation.
    """
    
    def __init__(self, menu_engine=None):
        """
        Initialize order engine.
        
        Args:
            menu_engine: Optional MenuEngine for validation
        """
        self._orders: Dict[str, Order] = {}
        self.menu_engine = menu_engine
        
        logger.info("OrderEngine initialized")
    
    def create_order(
        self,
        call_id: str,
        tenant_id: str
    ) -> Order:
        """
        Create new order for call.
        
        Args:
            call_id: Call identifier
            tenant_id: Tenant identifier
            
        Returns:
            Created order
        """
        order_id = f"order_{uuid.uuid4().hex[:8]}"
        order = Order(order_id=order_id, tenant_id=tenant_id)
        
        self._orders[call_id] = order
        
        logger.info(
            "Order created for call",
            extra={
                "call_id": call_id,
                "order_id": order_id
            }
        )
        
        return order
    
    def get_order(self, call_id: str) -> Optional[Order]:
        """Get order for call."""
        return self._orders.get(call_id)
    
    def delete_order(self, call_id: str) -> bool:
        """Delete order."""
        if call_id in self._orders:
            del self._orders[call_id]
            logger.info(
                "Order deleted",
                extra={"call_id": call_id}
            )
            return True
        return False
    
    async def validate_order(
        self,
        order: Order
    ) -> Tuple[bool, List[str]]:
        """
        Validate all items in order against menu.
        
        Args:
            order: Order to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        if not self.menu_engine:
            logger.warning("No menu_engine - skipping validation")
            return True, []
        
        errors = []
        
        for item in order.items:
            # Validate item exists
            exists = await self.menu_engine.validate_item_exists(
                order.tenant_id,
                item.canonical_id
            )
            
            if not exists:
                errors.append(
                    f"Item {item.canonical_id} does not exist in menu"
                )
                continue
            
            # Validate price
            is_valid, actual_price = await self.menu_engine.validate_price(
                order.tenant_id,
                item.canonical_id,
                item.price
            )
            
            if not is_valid and actual_price is not None:
                errors.append(
                    f"Price mismatch for {item.canonical_id}: "
                    f"expected ${item.price:.2f}, actual ${actual_price:.2f}"
                )
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.warning(
                f"Order validation failed: {len(errors)} errors",
                extra={
                    "order_id": order.order_id,
                    "errors": errors
                }
            )
        
        return is_valid, errors
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        states = {}
        total_value = 0.0
        
        for order in self._orders.values():
            state = order.state.value
            states[state] = states.get(state, 0) + 1
            total_value += order.get_total()
        
        return {
            "active_orders": len(self._orders),
            "orders_by_state": states,
            "total_value": round(total_value, 2)
        }


# Default instance
_default_engine: Optional[OrderEngine] = None


def get_default_engine() -> OrderEngine:
    """Get or create default order engine."""
    global _default_engine
    
    if _default_engine is None:
        _default_engine = OrderEngine()
    
    return _default_engine


def set_default_engine(engine: OrderEngine) -> None:
    """Set default engine."""
    global _default_engine
    _default_engine = engine
    logger.info("Default order engine set")
