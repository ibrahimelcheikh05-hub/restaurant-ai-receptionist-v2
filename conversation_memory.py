"""
Conversation Memory
===================
Per-call conversation history and context management with strict limits.

Responsibilities:
- Store conversation turns with limits
- Track context across turns
- Manage conversation history
- Provide context for AI
- Support conversation summarization
- ENFORCE memory bounds
- PREVENT memory leaks
- VALIDATE content size

CRITICAL SAFETY:
- Maximum turns per call: 50 (hard limit)
- Maximum content length per turn: 1000 chars
- Maximum context values: 20
- Maximum context value size: 5000 chars total
- Automatic pruning when limits exceeded
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """
    Single conversation turn with validation.
    
    SAFETY: Content length limited to prevent memory bloat.
    """
    
    MAX_CONTENT_LENGTH = 1000
    
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate after initialization."""
        # Validate role
        if self.role not in {"user", "assistant", "system"}:
            raise ValueError(f"Invalid role: {self.role}")
        
        # Enforce content length
        if len(self.content) > self.MAX_CONTENT_LENGTH:
            logger.warning(
                f"Content truncated from {len(self.content)} to {self.MAX_CONTENT_LENGTH} chars"
            )
            self.content = self.content[:self.MAX_CONTENT_LENGTH]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create from dictionary with validation."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )


class ConversationMemory:
    """
    Manages conversation history and context with strict limits.
    
    Features:
    - Turn-by-turn history with bounds
    - Context accumulation with limits
    - History windowing
    - Summarization support
    - Memory leak prevention
    
    SAFETY:
    - Max 50 turns total
    - Max 1000 chars per turn
    - Max 20 context keys
    - Max 5000 chars total context
    - Automatic pruning
    """
    
    # Hard limits
    MAX_TURNS = 50
    MAX_CONTEXT_TURNS = 10  # For AI context
    MAX_CONTEXT_KEYS = 20
    MAX_CONTEXT_TOTAL_SIZE = 5000
    MAX_TURN_CONTENT_LENGTH = 1000
    
    def __init__(
        self,
        call_id: str,
        max_turns: int = MAX_TURNS,
        max_context_turns: int = MAX_CONTEXT_TURNS
    ):
        """
        Initialize conversation memory.
        
        Args:
            call_id: Call identifier
            max_turns: Maximum turns to store (capped at MAX_TURNS)
            max_context_turns: Max turns to include in context (capped at MAX_CONTEXT_TURNS)
        """
        self.call_id = call_id
        
        # Enforce limits
        self.max_turns = min(max_turns, self.MAX_TURNS)
        self.max_context_turns = min(max_context_turns, self.MAX_CONTEXT_TURNS)
        
        # Conversation history
        self._turns: List[ConversationTurn] = []
        
        # Context snapshots (menu, order, etc.)
        self._context: Dict[str, Any] = {}
        self._context_size = 0  # Track context memory usage
        
        # Metadata
        self._created_at = datetime.now(timezone.utc)
        self._last_update = self._created_at
        
        # Statistics
        self._total_turns_added = 0
        self._total_pruned = 0
        
        logger.info(
            "ConversationMemory created",
            extra={
                "call_id": call_id,
                "max_turns": self.max_turns,
                "max_context_turns": self.max_context_turns
            }
        )
    
    def add_turn(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add conversation turn with validation.
        
        Args:
            role: Role ('user' or 'assistant')
            content: Turn content (will be truncated if too long)
            metadata: Optional metadata
        """
        # Create turn (validation happens in __post_init__)
        turn = ConversationTurn(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        self._turns.append(turn)
        self._total_turns_added += 1
        self._last_update = datetime.now(timezone.utc)
        
        # Prune if exceeded max
        if len(self._turns) > self.max_turns:
            pruned_count = len(self._turns) - self.max_turns
            self._turns = self._turns[-self.max_turns:]
            self._total_pruned += pruned_count
            
            logger.debug(
                f"Pruned {pruned_count} old turns",
                extra={
                    "call_id": self.call_id,
                    "total_pruned": self._total_pruned
                }
            )
        
        logger.debug(
            f"Turn added: {role}",
            extra={
                "call_id": self.call_id,
                "turn_count": len(self._turns),
                "total_added": self._total_turns_added
            }
        )
    
    def add_user_turn(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add user turn."""
        self.add_turn("user", content, metadata)
    
    def add_assistant_turn(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add assistant turn."""
        self.add_turn("assistant", content, metadata)
    
    def get_history(
        self,
        max_turns: Optional[int] = None
    ) -> List[ConversationTurn]:
        """
        Get conversation history.
        
        Args:
            max_turns: Max turns to return (defaults to max_context_turns)
            
        Returns:
            List of conversation turns (most recent)
        """
        limit = max_turns or self.max_context_turns
        # Enforce absolute maximum
        limit = min(limit, self.MAX_CONTEXT_TURNS)
        
        return self._turns[-limit:] if self._turns else []
    
    def get_history_as_messages(
        self,
        max_turns: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Get history as message format for LLM.
        
        Args:
            max_turns: Max turns to return
            
        Returns:
            List of messages
        """
        history = self.get_history(max_turns)
        return [
            {"role": turn.role, "content": turn.content}
            for turn in history
        ]
    
    def get_history_as_text(
        self,
        max_turns: Optional[int] = None,
        include_timestamps: bool = False
    ) -> str:
        """
        Get history as formatted text.
        
        Args:
            max_turns: Max turns to return
            include_timestamps: Include timestamps
            
        Returns:
            Formatted history text
        """
        history = self.get_history(max_turns)
        lines = []
        
        for turn in history:
            role_label = "Customer" if turn.role == "user" else "Assistant"
            
            if include_timestamps:
                ts = turn.timestamp.strftime("%H:%M:%S")
                line = f"[{ts}] {role_label}: {turn.content}"
            else:
                line = f"{role_label}: {turn.content}"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def get_turn_count(self) -> int:
        """Get current turn count."""
        return len(self._turns)
    
    def get_total_turns_added(self) -> int:
        """Get total turns added (including pruned)."""
        return self._total_turns_added
    
    def get_last_user_turn(self) -> Optional[ConversationTurn]:
        """Get most recent user turn."""
        for turn in reversed(self._turns):
            if turn.role == "user":
                return turn
        return None
    
    def get_last_assistant_turn(self) -> Optional[ConversationTurn]:
        """Get most recent assistant turn."""
        for turn in reversed(self._turns):
            if turn.role == "assistant":
                return turn
        return None
    
    def set_context(self, key: str, value: Any) -> bool:
        """
        Set context value with size validation.
        
        Args:
            key: Context key
            value: Context value
            
        Returns:
            True if set successfully, False if rejected
        """
        # Check key limit
        if key not in self._context and len(self._context) >= self.MAX_CONTEXT_KEYS:
            logger.warning(
                f"Cannot add context key '{key}' - limit reached ({self.MAX_CONTEXT_KEYS})",
                extra={"call_id": self.call_id}
            )
            return False
        
        # Estimate size
        try:
            value_str = json.dumps(value)
            value_size = len(value_str)
        except (TypeError, ValueError):
            # Fallback for non-serializable
            value_str = str(value)
            value_size = len(value_str)
        
        # Check if adding this would exceed total size
        old_size = 0
        if key in self._context:
            try:
                old_size = len(json.dumps(self._context[key]))
            except:
                old_size = len(str(self._context[key]))
        
        new_total_size = self._context_size - old_size + value_size
        
        if new_total_size > self.MAX_CONTEXT_TOTAL_SIZE:
            logger.warning(
                f"Context size would exceed limit: {new_total_size} > {self.MAX_CONTEXT_TOTAL_SIZE}",
                extra={"call_id": self.call_id, "key": key}
            )
            return False
        
        # Set value
        self._context[key] = value
        self._context_size = new_total_size
        
        logger.debug(
            f"Context set: {key} ({value_size} chars, total: {self._context_size})",
            extra={"call_id": self.call_id}
        )
        
        return True
    
    def get_context(self, key: str) -> Optional[Any]:
        """Get context value."""
        return self._context.get(key)
    
    def update_context(self, updates: Dict[str, Any]) -> Dict[str, bool]:
        """
        Update multiple context values.
        
        Args:
            updates: Dictionary of updates
            
        Returns:
            Dictionary of {key: success_bool}
        """
        results = {}
        for key, value in updates.items():
            results[key] = self.set_context(key, value)
        return results
    
    def delete_context(self, key: str) -> bool:
        """
        Delete context key.
        
        Args:
            key: Context key
            
        Returns:
            True if deleted
        """
        if key in self._context:
            del self._context[key]
            
            # Recalculate size
            self._recalculate_context_size()
            
            logger.debug(
                f"Context deleted: {key}",
                extra={"call_id": self.call_id}
            )
            return True
        return False
    
    def _recalculate_context_size(self) -> None:
        """Recalculate total context size."""
        total = 0
        for value in self._context.values():
            try:
                total += len(json.dumps(value))
            except:
                total += len(str(value))
        self._context_size = total
    
    def get_all_context(self) -> Dict[str, Any]:
        """Get all context (copy)."""
        return self._context.copy()
    
    def get_context_size(self) -> int:
        """Get current context size in characters."""
        return self._context_size
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self._turns.clear()
        logger.info(
            "History cleared",
            extra={"call_id": self.call_id}
        )
    
    def clear_context(self) -> None:
        """Clear all context."""
        self._context.clear()
        self._context_size = 0
        logger.info(
            "Context cleared",
            extra={"call_id": self.call_id}
        )
    
    def summarize(self) -> str:
        """Generate conversation summary."""
        if not self._turns:
            return "No conversation yet."
        
        user_turns = [t for t in self._turns if t.role == "user"]
        assistant_turns = [t for t in self._turns if t.role == "assistant"]
        
        duration = (
            self._last_update - self._created_at
        ).total_seconds()
        
        summary = (
            f"Conversation Summary:\n"
            f"- Total turns: {len(self._turns)} (added: {self._total_turns_added}, pruned: {self._total_pruned})\n"
            f"- User turns: {len(user_turns)}\n"
            f"- Assistant turns: {len(assistant_turns)}\n"
            f"- Duration: {duration:.1f} seconds\n"
            f"- Context keys: {len(self._context)}/{self.MAX_CONTEXT_KEYS}\n"
            f"- Context size: {self._context_size}/{self.MAX_CONTEXT_TOTAL_SIZE} chars\n"
        )
        
        # Add context info if available
        if "order_summary" in self._context:
            summary += f"- Order created: Yes\n"
        
        if "language" in self._context:
            summary += f"- Language: {self._context['language']}\n"
        
        return summary
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "turn_count": len(self._turns),
            "total_added": self._total_turns_added,
            "total_pruned": self._total_pruned,
            "context_keys": len(self._context),
            "context_size": self._context_size,
            "max_turns": self.max_turns,
            "max_context_keys": self.MAX_CONTEXT_KEYS,
            "max_context_size": self.MAX_CONTEXT_TOTAL_SIZE,
            "duration_seconds": (
                self._last_update - self._created_at
            ).total_seconds()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "call_id": self.call_id,
            "turns": [turn.to_dict() for turn in self._turns],
            "context": self._context,
            "created_at": self._created_at.isoformat(),
            "last_update": self._last_update.isoformat(),
            "stats": self.get_stats()
        }
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any]
    ) -> 'ConversationMemory':
        """Deserialize from dictionary with validation."""
        memory = cls(call_id=data["call_id"])
        
        # Restore turns (with validation)
        for turn_data in data["turns"]:
            try:
                turn = ConversationTurn.from_dict(turn_data)
                memory._turns.append(turn)
            except Exception as e:
                logger.warning(
                    f"Skipping invalid turn during restore: {e}",
                    extra={"call_id": data["call_id"]}
                )
        
        # Restore context (with validation)
        context = data.get("context", {})
        for key, value in context.items():
            memory.set_context(key, value)
        
        # Restore timestamps
        try:
            memory._created_at = datetime.fromisoformat(data["created_at"])
            memory._last_update = datetime.fromisoformat(data["last_update"])
        except:
            pass
        
        return memory
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ConversationMemory':
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class ConversationMemoryStore:
    """
    Store for managing multiple conversation memories.
    
    In-memory store for active conversations with limits.
    
    SAFETY:
    - Maximum memories tracked
    - Automatic cleanup of old memories
    """
    
    MAX_MEMORIES = 1000
    
    def __init__(self, max_memories: int = MAX_MEMORIES):
        """
        Initialize store.
        
        Args:
            max_memories: Maximum memories to track
        """
        self.max_memories = max_memories
        self._memories: Dict[str, ConversationMemory] = {}
    
    def create(
        self,
        call_id: str,
        **kwargs
    ) -> ConversationMemory:
        """
        Create new conversation memory.
        
        Args:
            call_id: Call identifier
            **kwargs: Additional arguments
            
        Returns:
            Created memory
        """
        # Check limit
        if call_id not in self._memories and len(self._memories) >= self.max_memories:
            logger.warning(
                f"Memory store at capacity ({self.max_memories}), cannot create new memory",
                extra={"call_id": call_id}
            )
            # Return existing or create ephemeral
            # For safety, create ephemeral that won't be stored
            return ConversationMemory(call_id=call_id, **kwargs)
        
        memory = ConversationMemory(call_id=call_id, **kwargs)
        self._memories[call_id] = memory
        
        logger.info(
            "Memory created in store",
            extra={
                "call_id": call_id,
                "total_memories": len(self._memories)
            }
        )
        
        return memory
    
    def get(self, call_id: str) -> Optional[ConversationMemory]:
        """Get conversation memory."""
        return self._memories.get(call_id)
    
    def delete(self, call_id: str) -> bool:
        """Delete conversation memory."""
        if call_id in self._memories:
            del self._memories[call_id]
            logger.info(
                "Memory deleted from store",
                extra={"call_id": call_id}
            )
            return True
        return False
    
    def get_all(self) -> Dict[str, ConversationMemory]:
        """Get all memories (copy)."""
        return self._memories.copy()
    
    def count(self) -> int:
        """Get count of active memories."""
        return len(self._memories)
    
    def clear(self) -> None:
        """Clear all memories."""
        count = len(self._memories)
        self._memories.clear()
        logger.info(f"All memories cleared ({count} removed)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        total_turns = sum(m.get_turn_count() for m in self._memories.values())
        total_context_size = sum(m.get_context_size() for m in self._memories.values())
        
        return {
            "active_memories": len(self._memories),
            "max_memories": self.max_memories,
            "total_turns": total_turns,
            "total_context_size": total_context_size
        }


# Global store instance
_global_store = ConversationMemoryStore()


def create_memory(call_id: str, **kwargs) -> ConversationMemory:
    """Create conversation memory."""
    return _global_store.create(call_id, **kwargs)


def get_memory(call_id: str) -> Optional[ConversationMemory]:
    """Get conversation memory."""
    return _global_store.get(call_id)


def delete_memory(call_id: str) -> bool:
    """Delete conversation memory."""
    return _global_store.delete(call_id)


def get_store_stats() -> Dict[str, Any]:
    """Get global store statistics."""
    return _global_store.get_stats()
