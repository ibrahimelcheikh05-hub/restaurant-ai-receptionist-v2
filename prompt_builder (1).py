"""
Prompt Builder
==============
Dynamic prompt construction for restaurant AI assistant.

Responsibilities:
- Build system prompts
- Inject menu context
- Inject order context
- Format conversation history
- Apply prompt templates
- Handle multilingual prompts
"""

import logging
from typing import List, Dict, Any, Optional
import json
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class PromptTemplate:
    """Template for system prompts."""
    
    # Base system prompt
    BASE_SYSTEM = """You are a professional restaurant phone order assistant.

Your role:
- Help customers place orders
- Answer menu questions
- Suggest items when appropriate
- Be friendly and efficient
- Confirm order details

Guidelines:
- Keep responses concise (2-3 sentences max)
- Use natural, conversational language
- Never mention you're an AI
- If you don't know something, say so
- Always confirm before finalizing orders
"""
    
    # Menu context template
    MENU_CONTEXT = """
CURRENT MENU:
{menu}

Available categories: {categories}
Price range: ${min_price} - ${max_price}
"""
    
    # Order context template
    ORDER_CONTEXT = """
CURRENT ORDER:
{order}

Order total: ${total}
Item count: {item_count}
"""
    
    # Upsell context template
    UPSELL_CONTEXT = """
SUGGESTED UPSELLS:
{suggestions}

Mention these naturally if appropriate, but don't be pushy.
"""
    
    # Conversation context template
    CONVERSATION_CONTEXT = """
RECENT CONVERSATION:
{history}
"""
    
    # Special instructions template
    SPECIAL_INSTRUCTIONS = """
SPECIAL INSTRUCTIONS:
{instructions}
"""


class PromptBuilder:
    """
    Builds dynamic prompts with context injection.
    
    Constructs prompts that include:
    - System instructions
    - Menu information
    - Current order state
    - Conversation history
    - Upsell suggestions
    - Special instructions
    """
    
    def __init__(
        self,
        base_system_prompt: Optional[str] = None,
        max_history_turns: int = 5,
        include_timestamps: bool = False
    ):
        """
        Initialize prompt builder.
        
        Args:
            base_system_prompt: Override base system prompt
            max_history_turns: Max conversation turns to include
            include_timestamps: Include timestamps in history
        """
        self.base_system_prompt = (
            base_system_prompt or PromptTemplate.BASE_SYSTEM
        )
        self.max_history_turns = max_history_turns
        self.include_timestamps = include_timestamps
        
        logger.info("PromptBuilder initialized")
    
    def build_system_prompt(
        self,
        menu_data: Optional[Dict[str, Any]] = None,
        order_data: Optional[Dict[str, Any]] = None,
        upsell_suggestions: Optional[List[str]] = None,
        special_instructions: Optional[str] = None
    ) -> str:
        """
        Build complete system prompt with all context.
        
        Args:
            menu_data: Menu information
            order_data: Current order data
            upsell_suggestions: Upsell suggestion strings
            special_instructions: Special instructions
            
        Returns:
            Complete system prompt
        """
        sections = [self.base_system_prompt]
        
        # Add menu context
        if menu_data:
            menu_section = self._build_menu_section(menu_data)
            if menu_section:
                sections.append(menu_section)
        
        # Add order context
        if order_data:
            order_section = self._build_order_section(order_data)
            if order_section:
                sections.append(order_section)
        
        # Add upsell context
        if upsell_suggestions:
            upsell_section = self._build_upsell_section(upsell_suggestions)
            if upsell_section:
                sections.append(upsell_section)
        
        # Add special instructions
        if special_instructions:
            sections.append(
                PromptTemplate.SPECIAL_INSTRUCTIONS.format(
                    instructions=special_instructions
                )
            )
        
        return "\n\n".join(sections)
    
    def build_user_message(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Build user message with conversation history.
        
        Args:
            user_input: Current user input
            conversation_history: Previous conversation turns
            
        Returns:
            Formatted user message
        """
        sections = []
        
        # Add conversation history
        if conversation_history:
            history_section = self._build_history_section(
                conversation_history
            )
            if history_section:
                sections.append(history_section)
        
        # Add current input
        sections.append(f"Customer: {user_input}")
        sections.append("\nRespond naturally and helpfully:")
        
        return "\n\n".join(sections)
    
    def build_messages(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        menu_data: Optional[Dict[str, Any]] = None,
        order_data: Optional[Dict[str, Any]] = None,
        upsell_suggestions: Optional[List[str]] = None,
        special_instructions: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build complete message list for LLM.
        
        Args:
            user_input: Current user input
            conversation_history: Conversation history
            menu_data: Menu information
            order_data: Order data
            upsell_suggestions: Upsell suggestions
            special_instructions: Special instructions
            
        Returns:
            List of messages for LLM
        """
        messages = []
        
        # System message
        system_prompt = self.build_system_prompt(
            menu_data=menu_data,
            order_data=order_data,
            upsell_suggestions=upsell_suggestions,
            special_instructions=special_instructions
        )
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # Add recent conversation history as messages
        if conversation_history:
            history_messages = self._history_to_messages(
                conversation_history
            )
            messages.extend(history_messages)
        
        # Current user input
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        return messages
    
    def _build_menu_section(self, menu_data: Dict[str, Any]) -> str:
        """
        Build menu section of prompt.
        
        Args:
            menu_data: Menu information
            
        Returns:
            Formatted menu section
        """
        try:
            # Extract menu items
            items = menu_data.get("items", [])
            if not items:
                return ""
            
            # Get categories
            categories = set()
            prices = []
            
            menu_lines = []
            for item in items:
                name = item.get("name", "Unknown")
                price = item.get("price", 0.0)
                category = item.get("category", "Other")
                description = item.get("description", "")
                
                categories.add(category)
                prices.append(price)
                
                # Format item
                item_line = f"- {name} (${price:.2f})"
                if description:
                    item_line += f": {description}"
                
                menu_lines.append(item_line)
            
            menu_text = "\n".join(menu_lines)
            
            # Get price range
            min_price = min(prices) if prices else 0.0
            max_price = max(prices) if prices else 0.0
            
            return PromptTemplate.MENU_CONTEXT.format(
                menu=menu_text,
                categories=", ".join(sorted(categories)),
                min_price=f"{min_price:.2f}",
                max_price=f"{max_price:.2f}"
            )
        
        except Exception as e:
            logger.error(f"Error building menu section: {e}", exc_info=True)
            return ""
    
    def _build_order_section(self, order_data: Dict[str, Any]) -> str:
        """
        Build order section of prompt.
        
        Args:
            order_data: Order information
            
        Returns:
            Formatted order section
        """
        try:
            items = order_data.get("items", [])
            if not items:
                return PromptTemplate.ORDER_CONTEXT.format(
                    order="No items yet",
                    total="0.00",
                    item_count=0
                )
            
            # Format items
            order_lines = []
            total = 0.0
            
            for idx, item in enumerate(items, 1):
                name = item.get("name", "Unknown")
                quantity = item.get("quantity", 1)
                price = item.get("price", 0.0)
                modifications = item.get("modifications", [])
                
                item_total = price * quantity
                total += item_total
                
                item_line = f"{idx}. {name} x{quantity} (${item_total:.2f})"
                
                if modifications:
                    mods_text = ", ".join(modifications)
                    item_line += f" - {mods_text}"
                
                order_lines.append(item_line)
            
            order_text = "\n".join(order_lines)
            
            return PromptTemplate.ORDER_CONTEXT.format(
                order=order_text,
                total=f"{total:.2f}",
                item_count=len(items)
            )
        
        except Exception as e:
            logger.error(f"Error building order section: {e}", exc_info=True)
            return ""
    
    def _build_upsell_section(
        self,
        suggestions: List[str]
    ) -> str:
        """
        Build upsell section of prompt.
        
        Args:
            suggestions: List of suggestion strings
            
        Returns:
            Formatted upsell section
        """
        if not suggestions:
            return ""
        
        suggestions_text = "\n".join(f"- {s}" for s in suggestions)
        
        return PromptTemplate.UPSELL_CONTEXT.format(
            suggestions=suggestions_text
        )
    
    def _build_history_section(
        self,
        history: List[Dict[str, Any]]
    ) -> str:
        """
        Build conversation history section.
        
        Args:
            history: Conversation history
            
        Returns:
            Formatted history section
        """
        if not history:
            return ""
        
        # Take last N turns
        recent_history = history[-self.max_history_turns:]
        
        # Format turns
        history_lines = []
        for turn in recent_history:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            timestamp = turn.get("timestamp")
            
            if role == "user":
                prefix = "Customer"
            elif role == "assistant":
                prefix = "Assistant"
            else:
                prefix = role.capitalize()
            
            line = f"{prefix}: {content}"
            
            if self.include_timestamps and timestamp:
                line = f"[{timestamp}] {line}"
            
            history_lines.append(line)
        
        history_text = "\n".join(history_lines)
        
        return PromptTemplate.CONVERSATION_CONTEXT.format(
            history=history_text
        )
    
    def _history_to_messages(
        self,
        history: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Convert history to message format.
        
        Args:
            history: Conversation history
            
        Returns:
            List of messages
        """
        messages = []
        
        # Take last N turns
        recent_history = history[-self.max_history_turns:]
        
        for turn in recent_history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            
            # Map roles
            if role == "assistant":
                msg_role = "assistant"
            else:
                msg_role = "user"
            
            messages.append({
                "role": msg_role,
                "content": content
            })
        
        return messages
    
    def format_menu_for_display(
        self,
        menu_data: Dict[str, Any]
    ) -> str:
        """
        Format menu for human-readable display.
        
        Args:
            menu_data: Menu information
            
        Returns:
            Formatted menu text
        """
        items = menu_data.get("items", [])
        if not items:
            return "No menu items available."
        
        # Group by category
        by_category: Dict[str, List[Dict]] = {}
        for item in items:
            category = item.get("category", "Other")
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(item)
        
        # Format each category
        sections = []
        for category, category_items in sorted(by_category.items()):
            sections.append(f"\n{category.upper()}")
            sections.append("-" * len(category))
            
            for item in category_items:
                name = item.get("name", "Unknown")
                price = item.get("price", 0.0)
                description = item.get("description", "")
                
                line = f"{name} - ${price:.2f}"
                if description:
                    line += f"\n  {description}"
                
                sections.append(line)
        
        return "\n".join(sections)


# Default instance
_default_builder: Optional[PromptBuilder] = None


def get_default_builder() -> PromptBuilder:
    """
    Get or create default prompt builder.
    
    Returns:
        Default builder instance
    """
    global _default_builder
    
    if _default_builder is None:
        _default_builder = PromptBuilder()
    
    return _default_builder
