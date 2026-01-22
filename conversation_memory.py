"""
Conversation Memory
===================
Per-call conversation history and context management.

Responsibilities:
- Store conversation turns
- Track context across turns
- Manage conversation history
- Provide context for AI
- Support conversation summarization
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single conversation turn."""
    
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
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
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )


class ConversationMemory:
    """
    Manages conversation history and context.
    
    Features:
    - Turn-by-turn history
    - Context accumulation
    - History windowing
    - Summarization support
    """
    
    def __init__(
        self,
        call_id: str,
        max_turns: int = 50,
        max_context_turns: int = 10
    ):
        """
        Initialize conversation memory.
        
        Args:
            call_id: Call identifier
            max_turns: Maximum turns to store
            max_context_turns: Max turns to include in context
        """
        self.call_id = call_id
        self.max_turns = max_turns
        self.max_context_turns = max_context_turns
        
        # Conversation history
        self._turns: List[ConversationTurn] = []
        
        # Context snapshots (menu, order, etc.)
        self._context: Dict[str, Any] = {}
        
        # Metadata
        self._created_at = datetime.now(timezone.utc)
        self._last_update = self._created_at
        
        logger.info(
            "ConversationMemory created",
            extra={"call_id": call_id}
        )
    
    def add_turn(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add conversation turn.
        
        Args:
            role: Role ('user' or 'assistant')
            content: Turn content
            metadata: Optional metadata
        """
        turn = ConversationTurn(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        self._turns.append(turn)
        self._last_update = datetime.now(timezone.utc)
        
        # Prune if exceeded max
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns:]
        
        logger.debug(
            f"Turn added: {role}",
            extra={
                "call_id": self.call_id,
                "turn_count": len(self._turns)
            }
        )
    
    def add_user_turn(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add user turn.
        
        Args:
            content: User message
            metadata: Optional metadata
        """
        self.add_turn("user", content, metadata)
    
    def add_assistant_turn(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add assistant turn.
        
        Args:
            content: Assistant message
            metadata: Optional metadata
        """
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
            List of conversation turns
        """
        limit = max_turns or self.max_context_turns
        return self._turns[-limit:]
    
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
        """Get total turn count."""
        return len(self._turns)
    
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
    
    def set_context(self, key: str, value: Any) -> None:
        """
        Set context value.
        
        Args:
            key: Context key
            value: Context value
        """
        self._context[key] = value
        logger.debug(
            f"Context set: {key}",
            extra={"call_id": self.call_id}
        )
    
    def get_context(self, key: str) -> Optional[Any]:
        """
        Get context value.
        
        Args:
            key: Context key
            
        Returns:
            Context value or None
        """
        return self._context.get(key)
    
    def update_context(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple context values.
        
        Args:
            updates: Dictionary of updates
        """
        self._context.update(updates)
    
    def get_all_context(self) -> Dict[str, Any]:
        """Get all context."""
        return self._context.copy()
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self._turns.clear()
        logger.info(
            "History cleared",
            extra={"call_id": self.call_id}
        )
    
    def summarize(self) -> str:
        """
        Generate conversation summary.
        
        Returns:
            Summary text
        """
        if not self._turns:
            return "No conversation yet."
        
        user_turns = [t for t in self._turns if t.role == "user"]
        assistant_turns = [t for t in self._turns if t.role == "assistant"]
        
        duration = (
            self._last_update - self._created_at
        ).total_seconds()
        
        summary = (
            f"Conversation Summary:\n"
            f"- Total turns: {len(self._turns)}\n"
            f"- User turns: {len(user_turns)}\n"
            f"- Assistant turns: {len(assistant_turns)}\n"
            f"- Duration: {duration:.1f} seconds\n"
        )
        
        # Add context info if available
        if "order_summary" in self._context:
            summary += f"- Order created: Yes\n"
        
        if "language" in self._context:
            summary += f"- Language: {self._context['language']}\n"
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "call_id": self.call_id,
            "turns": [turn.to_dict() for turn in self._turns],
            "context": self._context,
            "created_at": self._created_at.isoformat(),
            "last_update": self._last_update.isoformat(),
            "turn_count": len(self._turns)
        }
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any]
    ) -> 'ConversationMemory':
        """
        Deserialize from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            ConversationMemory instance
        """
        memory = cls(call_id=data["call_id"])
        
        # Restore turns
        memory._turns = [
            ConversationTurn.from_dict(turn_data)
            for turn_data in data["turns"]
        ]
        
        # Restore context
        memory._context = data.get("context", {})
        
        # Restore timestamps
        memory._created_at = datetime.fromisoformat(data["created_at"])
        memory._last_update = datetime.fromisoformat(data["last_update"])
        
        return memory
    
    def to_json(self) -> str:
        """
        Serialize to JSON.
        
        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ConversationMemory':
        """
        Deserialize from JSON.
        
        Args:
            json_str: JSON string
            
        Returns:
            ConversationMemory instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


class ConversationMemoryStore:
    """
    Store for managing multiple conversation memories.
    
    In-memory store for active conversations.
    """
    
    def __init__(self):
        """Initialize store."""
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
        """
        Get conversation memory.
        
        Args:
            call_id: Call identifier
            
        Returns:
            Memory or None
        """
        return self._memories.get(call_id)
    
    def delete(self, call_id: str) -> bool:
        """
        Delete conversation memory.
        
        Args:
            call_id: Call identifier
            
        Returns:
            True if deleted
        """
        if call_id in self._memories:
            del self._memories[call_id]
            logger.info(
                "Memory deleted from store",
                extra={"call_id": call_id}
            )
            return True
        return False
    
    def get_all(self) -> Dict[str, ConversationMemory]:
        """Get all memories."""
        return self._memories.copy()
    
    def count(self) -> int:
        """Get count of active memories."""
        return len(self._memories)
    
    def clear(self) -> None:
        """Clear all memories."""
        self._memories.clear()
        logger.info("All memories cleared")


# Global store instance
_global_store = ConversationMemoryStore()


def create_memory(call_id: str, **kwargs) -> ConversationMemory:
    """
    Create conversation memory.
    
    Args:
        call_id: Call identifier
        **kwargs: Additional arguments
        
    Returns:
        Created memory
    """
    return _global_store.create(call_id, **kwargs)


def get_memory(call_id: str) -> Optional[ConversationMemory]:
    """
    Get conversation memory.
    
    Args:
        call_id: Call identifier
        
    Returns:
        Memory or None
    """
    return _global_store.get(call_id)


def delete_memory(call_id: str) -> bool:
    """
    Delete conversation memory.
    
    Args:
        call_id: Call identifier
        
    Returns:
        True if deleted
    """
    return _global_store.delete(call_id)
