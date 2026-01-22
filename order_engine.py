"""
Order Engine
============
Transactional order state machine.

Responsibilities:
- Manage order lifecycle
- Add/remove items
- Calculate totals
- Validate orders
- Prevent order tampering
- Support rollback
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy
import uuid

logger = logging.getLogger(__name__)


class OrderState(Enum):
    """Order lifecycle states."""
    EMPTY = "empty"
    BUILDING = "building"
    CONFIRMING = "confirming"
    FINALIZED = "finalized"
    CANCELLED = "cancelled"


@dataclass
class OrderItem:
    """Single item in an order."""
    
    item_id: str
    name: str
    price: float
    quantity: int = 1
    notes: Optional[str] = None
    
    def __post_init__(self):
        """Validate after initialization."""
        if self.quantity < 1:
            raise ValueError("Quantity must be at least 1")
        if self.price < 0:
            raise ValueError("Price cannot be negative")
    
    @property
    def subtotal(self) -> float:
        """Calculate subtotal."""
        return round(self.price * self.quantity, 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "name": self.name,
            "price": self.price,
            "quantity": self.quantity,
            "notes": self.notes,
            "subtotal": self.subtotal
        }


class Order:
    """
    Mutable order with state machine.
    
    States:
    - EMPTY: No items
    - BUILDING: Adding/modifying items
    - CONFIRMING: Ready for confirmation
    - FINALIZED: Order confirmed
    - CANCELLED: Order cancelled
    """
    
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
        
        # State
        self.state = OrderState.EMPTY
        self.items: List[OrderItem] = []
        
        # Metadata
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
        self.finalized_at: Optional[datetime] = None
        
        # Customer info (set when finalizing)
        self.customer_name: Optional[str] = None
        self.customer_phone: Optional[str] = None
        self.customer_email: Optional[str] = None
        
        # Snapshot for rollback
        self._snapshot: Optional[List[OrderItem]] = None
        
        logger.info(
            "Order created",
            extra={
                "order_id": order_id,
                "tenant_id": tenant_id
            }
        )
    
    def add_item(
        self,
        item_id: str,
        name: str,
        price: float,
        quantity: int = 1,
        notes: Optional[str] = None
    ) -> None:
        """
        Add item to order.
        
        Args:
            item_id: Item identifier
            name: Item name
            price: Item price
            quantity: Quantity
            notes: Optional notes
        """
        if self.is_finalized():
            raise ValueError("Cannot modify finalized order")
        
        # Check if item already exists
        for existing in self.items:
            if existing.item_id == item_id:
                # Update quantity
                existing.quantity += quantity
                self._mark_updated()
                logger.info(
                    f"Item quantity updated: {name} x{existing.quantity}",
                    extra={"order_id": self.order_id}
                )
                return
        
        # Add new item
        item = OrderItem(
            item_id=item_id,
            name=name,
            price=price,
            quantity=quantity,
            notes=notes
        )
        
        self.items.append(item)
        self._mark_updated()
        
        # Transition from EMPTY to BUILDING
        if self.state == OrderState.EMPTY:
            self.state = OrderState.BUILDING
        
        logger.info(
            f"Item added: {name} x{quantity}",
            extra={"order_id": self.order_id}
        )
    
    def remove_item(self, item_id: str) -> bool:
        """
        Remove item from order.
        
        Args:
            item_id: Item to remove
            
        Returns:
            True if removed
        """
        if self.is_finalized():
            raise ValueError("Cannot modify finalized order")
        
        for i, item in enumerate(self.items):
            if item.item_id == item_id:
                removed = self.items.pop(i)
                self._mark_updated()
                
                # Transition to EMPTY if no items
                if not self.items:
                    self.state = OrderState.EMPTY
                
                logger.info(
                    f"Item removed: {removed.name}",
                    extra={"order_id": self.order_id}
                )
                return True
        
        return False
    
    def update_quantity(
        self,
        item_id: str,
        quantity: int
    ) -> bool:
        """
        Update item quantity.
        
        Args:
            item_id: Item to update
            quantity: New quantity
            
        Returns:
            True if updated
        """
        if self.is_finalized():
            raise ValueError("Cannot modify finalized order")
        
        if quantity < 1:
            # Remove if quantity is 0
            return self.remove_item(item_id)
        
        for item in self.items:
            if item.item_id == item_id:
                item.quantity = quantity
                self._mark_updated()
                
                logger.info(
                    f"Quantity updated: {item.name} x{quantity}",
                    extra={"order_id": self.order_id}
                )
                return True
        
        return False
    
    def clear(self) -> None:
        """Clear all items."""
        if self.is_finalized():
            raise ValueError("Cannot modify finalized order")
        
        self.items.clear()
        self.state = OrderState.EMPTY
        self._mark_updated()
        
        logger.info(
            "Order cleared",
            extra={"order_id": self.order_id}
        )
    
    def create_snapshot(self) -> None:
        """Create snapshot for rollback."""
        self._snapshot = deepcopy(self.items)
        logger.debug(
            "Snapshot created",
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
        
        self.items = deepcopy(self._snapshot)
        self._mark_updated()
        
        logger.info(
            "Order rolled back",
            extra={"order_id": self.order_id}
        )
        return True
    
    def mark_confirming(self) -> None:
        """Mark order as confirming (ready for review)."""
        if self.state == OrderState.EMPTY:
            raise ValueError("Cannot confirm empty order")
        
        if self.is_finalized():
            raise ValueError("Order already finalized")
        
        self.state = OrderState.CONFIRMING
        logger.info(
            "Order marked for confirmation",
            extra={"order_id": self.order_id}
        )
    
    def finalize(
        self,
        customer_name: Optional[str] = None,
        customer_phone: Optional[str] = None,
        customer_email: Optional[str] = None
    ) -> None:
        """
        Finalize order (make immutable).
        
        Args:
            customer_name: Customer name
            customer_phone: Customer phone
            customer_email: Customer email
        """
        if self.state == OrderState.EMPTY:
            raise ValueError("Cannot finalize empty order")
        
        if self.is_finalized():
            raise ValueError("Order already finalized")
        
        self.state = OrderState.FINALIZED
        self.finalized_at = datetime.now(timezone.utc)
        
        # Set customer info
        self.customer_name = customer_name
        self.customer_phone = customer_phone
        self.customer_email = customer_email
        
        logger.info(
            "Order finalized",
            extra={
                "order_id": self.order_id,
                "total": self.get_total()
            }
        )
    
    def cancel(self) -> None:
        """Cancel order."""
        if self.state == OrderState.FINALIZED:
            logger.warning(
                "Cancelling finalized order",
                extra={"order_id": self.order_id}
            )
        
        self.state = OrderState.CANCELLED
        logger.info(
            "Order cancelled",
            extra={"order_id": self.order_id}
        )
    
    def get_total(self) -> float:
        """
        Calculate order total.
        
        Returns:
            Total price
        """
        return round(sum(item.subtotal for item in self.items), 2)
    
    def get_item_count(self) -> int:
        """
        Get total item count.
        
        Returns:
            Number of items
        """
        return sum(item.quantity for item in self.items)
    
    def is_empty(self) -> bool:
        """Check if order is empty."""
        return len(self.items) == 0
    
    def is_finalized(self) -> bool:
        """Check if order is finalized."""
        return self.state == OrderState.FINALIZED
    
    def is_cancelled(self) -> bool:
        """Check if order is cancelled."""
        return self.state == OrderState.CANCELLED
    
    def _mark_updated(self) -> None:
        """Mark order as updated."""
        self.updated_at = datetime.now(timezone.utc)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get order summary.
        
        Returns:
            Summary dictionary
        """
        return {
            "order_id": self.order_id,
            "tenant_id": self.tenant_id,
            "state": self.state.value,
            "items": [item.to_dict() for item in self.items],
            "item_count": self.get_item_count(),
            "total": self.get_total(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "finalized_at": (
                self.finalized_at.isoformat()
                if self.finalized_at
                else None
            ),
            "customer": {
                "name": self.customer_name,
                "phone": self.customer_phone,
                "email": self.customer_email
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.get_summary()


class OrderEngine:
    """
    Order management engine.
    
    Manages active orders per call.
    """
    
    def __init__(self):
        """Initialize order engine."""
        self._orders: Dict[str, Order] = {}
        
        logger.info("OrderEngine initialized")
    
    def create_order(
        self,
        call_id: str,
        tenant_id: str
    ) -> Order:
        """
        Create new order.
        
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
        """
        Get order for call.
        
        Args:
            call_id: Call identifier
            
        Returns:
            Order or None
        """
        return self._orders.get(call_id)
    
    def delete_order(self, call_id: str) -> bool:
        """
        Delete order.
        
        Args:
            call_id: Call identifier
            
        Returns:
            True if deleted
        """
        if call_id in self._orders:
            del self._orders[call_id]
            logger.info(
                "Order deleted",
                extra={"call_id": call_id}
            )
            return True
        return False
    
    def get_active_orders(self) -> Dict[str, Order]:
        """Get all active orders."""
        return self._orders.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get engine statistics.
        
        Returns:
            Statistics dictionary
        """
        states = {}
        total_value = 0.0
        
        for order in self._orders.values():
            state = order.state.value
            states[state] = states.get(state, 0) + 1
            total_value += order.get_total()
        
        return {
            "active_orders": len(self._orders),
            "orders_by_state": states,
            "total_value": total_value
        }


# Default instance
_default_engine: Optional[OrderEngine] = None


def get_default_engine() -> OrderEngine:
    """
    Get or create default order engine.
    
    Returns:
        Default engine instance
    """
    global _default_engine
    
    if _default_engine is None:
        _default_engine = OrderEngine()
    
    return _default_engine


def create_order(call_id: str, tenant_id: str) -> Order:
    """
    Quick order creation.
    
    Args:
        call_id: Call identifier
        tenant_id: Tenant identifier
        
    Returns:
        Created order
    """
    engine = get_default_engine()
    return engine.create_order(call_id, tenant_id)


def get_order(call_id: str) -> Optional[Order]:
    """
    Quick order retrieval.
    
    Args:
        call_id: Call identifier
        
    Returns:
        Order or None
    """
    engine = get_default_engine()
    return engine.get_order(call_id)
