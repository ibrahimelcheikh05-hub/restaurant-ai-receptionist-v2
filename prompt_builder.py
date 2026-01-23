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
- DEFEND against prompt injection
- SANITIZE all user inputs
- VALIDATE all data before injection

SECURITY:
- User input is NEVER trusted
- All injected data is sanitized
- Prompt injection patterns are detected
- Length limits are enforced
- Special characters are escaped
"""

import logging
from typing import List, Dict, Any, Optional
import json
import re
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

CRITICAL RULES:
- You can ONLY suggest items that are on the menu
- You can ONLY modify items with options that exist
- You MUST validate all prices against the menu
- You CANNOT invent items, prices, or modifications
- If a customer asks for something not on the menu, politely decline
"""
    
    # Menu context template
    MENU_CONTEXT = """
CURRENT MENU:
{menu}

Available categories: {categories}
Price range: ${min_price} - ${max_price}

IMPORTANT: You can ONLY suggest items from this menu. Do not make up items.
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


class PromptSanitizer:
    """
    Sanitizes user inputs to prevent prompt injection.
    
    SECURITY FEATURES:
    - Detects and blocks injection patterns
    - Removes control characters
    - Enforces length limits
    - Escapes special sequences
    """
    
    # Prompt injection patterns to detect
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|all|above|prior)\s+(instructions|prompts|rules)",
        r"disregard\s+(previous|all|above|prior)",
        r"forget\s+(everything|all|previous)",
        r"you\s+are\s+now",
        r"new\s+(instructions|rules|prompt)",
        r"system\s*:\s*",
        r"assistant\s*:\s*",
        r"\[system\]",
        r"\[assistant\]",
        r"<\s*system\s*>",
        r"<\s*assistant\s*>",
        r"jailbreak",
        r"DAN\s+mode",
    ]
    
    # Compile patterns
    COMPILED_PATTERNS = [
        re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_PATTERNS
    ]
    
    # Control characters to remove
    CONTROL_CHARS = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
    
    # Maximum lengths
    MAX_USER_INPUT_LENGTH = 500
    MAX_MENU_ITEM_NAME_LENGTH = 100
    MAX_MENU_DESCRIPTION_LENGTH = 200
    MAX_INSTRUCTION_LENGTH = 500
    
    @classmethod
    def sanitize_user_input(cls, text: str) -> str:
        """
        Sanitize user input to prevent prompt injection.
        
        Args:
            text: Raw user input
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Enforce length limit
        if len(text) > cls.MAX_USER_INPUT_LENGTH:
            logger.warning(
                f"User input truncated from {len(text)} to {cls.MAX_USER_INPUT_LENGTH} chars"
            )
            text = text[:cls.MAX_USER_INPUT_LENGTH]
        
        # Remove control characters
        text = cls.CONTROL_CHARS.sub('', text)
        
        # Check for injection patterns
        for pattern in cls.COMPILED_PATTERNS:
            if pattern.search(text):
                logger.warning(
                    f"Potential prompt injection detected: {pattern.pattern}",
                    extra={"text_preview": text[:50]}
                )
                # Replace injection attempt with safe placeholder
                text = pattern.sub('[FILTERED]', text)
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    @classmethod
    def sanitize_menu_text(cls, text: str, max_length: int = 200) -> str:
        """
        Sanitize menu text (names, descriptions).
        
        Args:
            text: Menu text
            max_length: Maximum length
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Enforce length
        if len(text) > max_length:
            text = text[:max_length]
        
        # Remove control characters
        text = cls.CONTROL_CHARS.sub('', text)
        
        # Remove potential injection sequences
        text = text.replace('[system]', '').replace('[assistant]', '')
        text = text.replace('<system>', '').replace('<assistant>', '')
        
        return text.strip()
    
    @classmethod
    def validate_numeric(cls, value: Any, field_name: str) -> float:
        """
        Validate numeric field.
        
        Args:
            value: Value to validate
            field_name: Field name for logging
            
        Returns:
            Validated float
        """
        try:
            numeric = float(value)
            if numeric < 0:
                logger.warning(f"Negative {field_name}: {numeric}, using 0")
                return 0.0
            if numeric > 10000:  # Sanity check
                logger.warning(f"Excessive {field_name}: {numeric}, capping at 10000")
                return 10000.0
            return numeric
        except (ValueError, TypeError):
            logger.error(f"Invalid {field_name}: {value}, using 0")
            return 0.0


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
    
    SECURITY:
    - All inputs are sanitized
    - Injection patterns are blocked
    - Length limits are enforced
    - Data is validated before injection
    """
    
    def __init__(
        self,
        base_system_prompt: Optional[str] = None,
        max_history_turns: int = 5,
        include_timestamps: bool = False,
        enable_injection_defense: bool = True
    ):
        """
        Initialize prompt builder.
        
        Args:
            base_system_prompt: Override base system prompt
            max_history_turns: Max conversation turns to include
            include_timestamps: Include timestamps in history
            enable_injection_defense: Enable prompt injection defense
        """
        self.base_system_prompt = (
            base_system_prompt or PromptTemplate.BASE_SYSTEM
        )
        self.max_history_turns = max(1, min(max_history_turns, 10))  # Limit range
        self.include_timestamps = include_timestamps
        self.enable_injection_defense = enable_injection_defense
        
        logger.info(
            "PromptBuilder initialized",
            extra={
                "max_history_turns": self.max_history_turns,
                "injection_defense": self.enable_injection_defense
            }
        )
    
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
        
        # Add menu context (validated)
        if menu_data:
            menu_section = self._build_menu_section(menu_data)
            if menu_section:
                sections.append(menu_section)
        
        # Add order context (validated)
        if order_data:
            order_section = self._build_order_section(order_data)
            if order_section:
                sections.append(order_section)
        
        # Add upsell context (sanitized)
        if upsell_suggestions:
            upsell_section = self._build_upsell_section(upsell_suggestions)
            if upsell_section:
                sections.append(upsell_section)
        
        # Add special instructions (sanitized)
        if special_instructions:
            sanitized_instructions = PromptSanitizer.sanitize_menu_text(
                special_instructions,
                PromptSanitizer.MAX_INSTRUCTION_LENGTH
            )
            if sanitized_instructions:
                sections.append(
                    PromptTemplate.SPECIAL_INSTRUCTIONS.format(
                        instructions=sanitized_instructions
                    )
                )
        
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
        
        # Current user input (SANITIZED)
        sanitized_input = self._sanitize_input(user_input)
        messages.append({
            "role": "user",
            "content": sanitized_input
        })
        
        return messages
    
    def _sanitize_input(self, text: str) -> str:
        """
        Sanitize user input.
        
        Args:
            text: Raw input
            
        Returns:
            Sanitized input
        """
        if not self.enable_injection_defense:
            return text
        
        return PromptSanitizer.sanitize_user_input(text)
    
    def _build_menu_section(self, menu_data: Dict[str, Any]) -> str:
        """
        Build menu section of prompt with validation.
        
        Args:
            menu_data: Menu information
            
        Returns:
            Formatted menu section
        """
        try:
            # Extract and validate menu items
            items = menu_data.get("items", [])
            if not items:
                return ""
            
            # Validate items is a list
            if not isinstance(items, list):
                logger.error("Menu items is not a list")
                return ""
            
            # Limit number of items to prevent prompt bloat
            MAX_MENU_ITEMS = 50
            if len(items) > MAX_MENU_ITEMS:
                logger.warning(
                    f"Menu has {len(items)} items, limiting to {MAX_MENU_ITEMS}"
                )
                items = items[:MAX_MENU_ITEMS]
            
            # Get categories
            categories = set()
            prices = []
            
            menu_lines = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                
                # Sanitize and validate fields
                name = PromptSanitizer.sanitize_menu_text(
                    item.get("name", "Unknown"),
                    PromptSanitizer.MAX_MENU_ITEM_NAME_LENGTH
                )
                price = PromptSanitizer.validate_numeric(
                    item.get("price", 0.0),
                    "price"
                )
                category = PromptSanitizer.sanitize_menu_text(
                    item.get("category", "Other"),
                    50
                )
                description = PromptSanitizer.sanitize_menu_text(
                    item.get("description", ""),
                    PromptSanitizer.MAX_MENU_DESCRIPTION_LENGTH
                )
                
                categories.add(category)
                prices.append(price)
                
                # Format item
                item_line = f"- {name} (${price:.2f})"
                if description:
                    item_line += f": {description}"
                
                menu_lines.append(item_line)
            
            if not menu_lines:
                return ""
            
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
            logger.error(
                f"Error building menu section: {e}",
                exc_info=True
            )
            return ""
    
    def _build_order_section(self, order_data: Dict[str, Any]) -> str:
        """
        Build order section of prompt with validation.
        
        Args:
            order_data: Order information
            
        Returns:
            Formatted order section
        """
        try:
            items = order_data.get("items", [])
            
            if not isinstance(items, list):
                logger.error("Order items is not a list")
                items = []
            
            if not items:
                return PromptTemplate.ORDER_CONTEXT.format(
                    order="No items yet",
                    total="0.00",
                    item_count=0
                )
            
            # Limit number of items
            MAX_ORDER_ITEMS = 20
            if len(items) > MAX_ORDER_ITEMS:
                logger.warning(
                    f"Order has {len(items)} items, limiting to {MAX_ORDER_ITEMS}"
                )
                items = items[:MAX_ORDER_ITEMS]
            
            # Format items
            order_lines = []
            total = 0.0
            
            for idx, item in enumerate(items, 1):
                if not isinstance(item, dict):
                    continue
                
                # Sanitize and validate
                name = PromptSanitizer.sanitize_menu_text(
                    item.get("name", "Unknown"),
                    PromptSanitizer.MAX_MENU_ITEM_NAME_LENGTH
                )
                quantity = max(1, min(int(item.get("quantity", 1)), 100))  # Limit quantity
                price = PromptSanitizer.validate_numeric(
                    item.get("price", 0.0),
                    "price"
                )
                modifications = item.get("modifications", [])
                
                # Validate modifications
                if not isinstance(modifications, list):
                    modifications = []
                
                item_total = price * quantity
                total += item_total
                
                item_line = f"{idx}. {name} x{quantity} (${item_total:.2f})"
                
                if modifications:
                    # Sanitize modification text
                    safe_mods = [
                        PromptSanitizer.sanitize_menu_text(str(mod), 50)
                        for mod in modifications[:5]  # Limit modifications
                    ]
                    mods_text = ", ".join(safe_mods)
                    item_line += f" - {mods_text}"
                
                order_lines.append(item_line)
            
            order_text = "\n".join(order_lines)
            
            return PromptTemplate.ORDER_CONTEXT.format(
                order=order_text,
                total=f"{total:.2f}",
                item_count=len(order_lines)
            )
        
        except Exception as e:
            logger.error(
                f"Error building order section: {e}",
                exc_info=True
            )
            return ""
    
    def _build_upsell_section(
        self,
        suggestions: List[str]
    ) -> str:
        """
        Build upsell section of prompt with sanitization.
        
        Args:
            suggestions: List of suggestion strings
            
        Returns:
            Formatted upsell section
        """
        if not suggestions or not isinstance(suggestions, list):
            return ""
        
        # Sanitize and limit suggestions
        MAX_UPSELLS = 5
        safe_suggestions = [
            PromptSanitizer.sanitize_menu_text(str(s), 100)
            for s in suggestions[:MAX_UPSELLS]
        ]
        
        safe_suggestions = [s for s in safe_suggestions if s]  # Remove empty
        
        if not safe_suggestions:
            return ""
        
        suggestions_text = "\n".join(f"- {s}" for s in safe_suggestions)
        
        return PromptTemplate.UPSELL_CONTEXT.format(
            suggestions=suggestions_text
        )
    
    def _history_to_messages(
        self,
        history: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Convert history to message format with sanitization.
        
        Args:
            history: Conversation history
            
        Returns:
            List of messages
        """
        if not history or not isinstance(history, list):
            return []
        
        messages = []
        
        # Take last N turns
        recent_history = history[-self.max_history_turns:]
        
        for turn in recent_history:
            if not isinstance(turn, dict):
                continue
            
            role = turn.get("role", "user")
            content = turn.get("content", "")
            
            # Sanitize content
            if self.enable_injection_defense:
                content = PromptSanitizer.sanitize_user_input(content)
            
            if not content:
                continue
            
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
        if not items or not isinstance(items, list):
            return "No menu items available."
        
        # Group by category
        by_category: Dict[str, List[Dict]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            
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
                name = PromptSanitizer.sanitize_menu_text(
                    item.get("name", "Unknown"),
                    100
                )
                price = PromptSanitizer.validate_numeric(
                    item.get("price", 0.0),
                    "price"
                )
                description = PromptSanitizer.sanitize_menu_text(
                    item.get("description", ""),
                    200
                )
                
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


def set_default_builder(builder: PromptBuilder) -> None:
    """
    Set default prompt builder.
    
    Args:
        builder: Prompt builder to use as default
    """
    global _default_builder
    _default_builder = builder
    logger.info("Default prompt builder set")
