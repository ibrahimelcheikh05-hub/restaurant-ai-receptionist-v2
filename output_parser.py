"""
Output Parser
=============
Parse and validate LLM outputs.

Responsibilities:
- Parse LLM text responses
- Extract structured actions
- Validate output format
- Detect intent/commands
- Extract entities (items, quantities, etc.)
- Sanitize outputs
- VALIDATE all AI suggestions
- REJECT hallucinated content
- ENFORCE safety boundaries

CRITICAL SAFETY:
- AI outputs are NEVER trusted blindly
- All extracted entities must be validated
- Menu items must exist in actual menu
- Prices must match menu prices
- Quantities must be reasonable
- No AI output is executed without validation
"""

import logging
import re
from typing import Dict, Any, Optional, List, Tuple, Set
import json

logger = logging.getLogger(__name__)


class ParsedOutput:
    """Structured parsed output from LLM."""
    
    def __init__(
        self,
        text: str,
        intent: Optional[str] = None,
        action: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        validation_errors: Optional[List[str]] = None
    ):
        """
        Initialize parsed output.
        
        Args:
            text: Response text
            intent: Detected intent
            action: Suggested action (e.g., 'transfer', 'end_call')
            entities: Extracted entities
            confidence: Parsing confidence (0-1)
            validation_errors: List of validation errors
        """
        self.text = text
        self.intent = intent
        self.action = action
        self.entities = entities or {}
        self.confidence = confidence
        self.validation_errors = validation_errors or []
    
    def is_valid(self) -> bool:
        """Check if output passed validation."""
        return len(self.validation_errors) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "intent": self.intent,
            "action": self.action,
            "entities": self.entities,
            "confidence": self.confidence,
            "validation_errors": self.validation_errors,
            "is_valid": self.is_valid()
        }


class OutputParser:
    """
    Parses LLM outputs into structured format.
    
    Detects:
    - Intents (ordering, question, complaint, etc.)
    - Actions (transfer, end call, etc.)
    - Entities (menu items, quantities, prices, etc.)
    - Special commands
    
    VALIDATES:
    - All extracted entities against known data
    - Menu items against actual menu
    - Prices against menu prices
    - Quantities are reasonable
    - AI is not hallucinating
    """
    
    # Action detection patterns
    ACTION_PATTERNS = {
        "transfer": [
            r"(?:speak|talk)\s+(?:to|with)\s+(?:a\s+)?(?:human|person|manager|someone)",
            r"transfer\s+(?:me|call)",
            r"(?:can|could|may)\s+i\s+speak\s+(?:to|with)",
            r"connect\s+me\s+(?:to|with)"
        ],
        "end_call": [
            r"\b(?:goodbye|bye|thanks|thank you)\b.*\b(?:bye|goodbye)\b",
            r"that'?s?\s+(?:all|everything|it)",
            r"have\s+a\s+(?:good|great|nice)\s+(?:day|night|evening)",
            r"talk\s+to\s+you\s+(?:later|soon)",
            r"see\s+you"
        ],
        "repeat": [
            r"(?:say|repeat)\s+that\s+again",
            r"what\s+(?:did|was)\s+that",
            r"(?:didn'?t|did\s+not)\s+(?:hear|catch)\s+that",
            r"come\s+again",
            r"pardon"
        ]
    }
    
    # Intent detection patterns
    INTENT_PATTERNS = {
        "add_item": [
            r"(?:i'?d?\s+like|i\s+want|(?:can|could)\s+i\s+(?:get|have))\s+(?:a|an|the)?",
            r"add\s+(?:a|an|the)?",
            r"give\s+me\s+(?:a|an|the)?",
            r"order\s+(?:a|an|the)?"
        ],
        "remove_item": [
            r"(?:remove|cancel|delete)\s+(?:the)?",
            r"(?:don'?t|do\s+not)\s+want",
            r"take\s+(?:off|out)\s+(?:the)?",
            r"scratch\s+(?:the)?"
        ],
        "modify_item": [
            r"(?:change|modify|update)\s+(?:the)?",
            r"(?:no|without|hold)\s+the",
            r"(?:add|extra|with)\s+(?:extra)?",
            r"instead\s+of"
        ],
        "question_menu": [
            r"what\s+(?:is|are|do\s+you\s+have)",
            r"(?:tell|show)\s+me\s+(?:what|your)",
            r"what'?s?\s+(?:on|in)\s+(?:the|your)\s+menu",
            r"menu\s+options"
        ],
        "question_price": [
            r"how\s+much\s+(?:is|are|does|do|cost)",
            r"what'?s?\s+the\s+(?:price|cost)",
            r"price\s+of",
            r"cost\s+of"
        ],
        "confirm_order": [
            r"that'?s?\s+(?:correct|right|good)",
            r"(?:yes|yeah|yep),?\s+(?:that'?s?\s+)?(?:correct|right|good|it)",
            r"sounds\s+good",
            r"looks\s+good"
        ],
        "ready_to_order": [
            r"(?:i'?m?\s+)?ready\s+to\s+order",
            r"(?:can|may)\s+i\s+(?:place|make)\s+(?:an\s+)?order",
            r"want\s+to\s+order"
        ]
    }
    
    # Safety limits
    MAX_OUTPUT_LENGTH = 1000
    MAX_QUANTITY = 99
    MIN_QUANTITY = 1
    MAX_PRICE = 999.99
    MIN_PRICE = 0.01
    
    def __init__(self, strict_validation: bool = True):
        """
        Initialize output parser.
        
        Args:
            strict_validation: Enable strict validation of AI outputs
        """
        self.strict_validation = strict_validation
        
        # Compile patterns for efficiency
        self._action_patterns_compiled = {
            action: [re.compile(p, re.IGNORECASE) for p in patterns]
            for action, patterns in self.ACTION_PATTERNS.items()
        }
        
        self._intent_patterns_compiled = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }
        
        logger.info(
            "OutputParser initialized",
            extra={"strict_validation": strict_validation}
        )
    
    def parse(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ParsedOutput:
        """
        Parse LLM output text with validation.
        
        Args:
            text: LLM response text
            context: Optional parsing context (menu, order, etc.)
            
        Returns:
            Parsed output with validation results
        """
        validation_errors = []
        
        # Validate and sanitize text
        clean_text, text_errors = self._sanitize_and_validate_text(text)
        validation_errors.extend(text_errors)
        
        # Detect action
        action = self._detect_action(clean_text)
        
        # Detect intent
        intent = self._detect_intent(clean_text)
        
        # Extract entities with validation
        entities = {}
        if context:
            entities, entity_errors = self._extract_and_validate_entities(
                clean_text,
                context
            )
            validation_errors.extend(entity_errors)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            text=clean_text,
            action=action,
            intent=intent,
            entities=entities,
            has_errors=len(validation_errors) > 0
        )
        
        # Log validation errors
        if validation_errors:
            logger.warning(
                "Output validation errors",
                extra={
                    "errors": validation_errors,
                    "text_preview": clean_text[:100]
                }
            )
        
        return ParsedOutput(
            text=clean_text,
            intent=intent,
            action=action,
            entities=entities,
            confidence=confidence,
            validation_errors=validation_errors
        )
    
    def _sanitize_and_validate_text(self, text: str) -> Tuple[str, List[str]]:
        """
        Sanitize and validate output text.
        
        Args:
            text: Raw text
            
        Returns:
            Tuple of (sanitized_text, validation_errors)
        """
        errors = []
        
        if not text:
            errors.append("Empty output text")
            return "", errors
        
        # Check length
        if len(text) > self.MAX_OUTPUT_LENGTH:
            errors.append(
                f"Output exceeds max length ({len(text)} > {self.MAX_OUTPUT_LENGTH})"
            )
            text = text[:self.MAX_OUTPUT_LENGTH]
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove markdown artifacts
        text = re.sub(r'\*\*', '', text)
        text = re.sub(r'__', '', text)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        
        # Check for suspiciously long words (possible garbage)
        words = text.split()
        for word in words:
            if len(word) > 50:
                errors.append(f"Suspiciously long word: {word[:20]}...")
        
        return text, errors
    
    def _detect_action(self, text: str) -> Optional[str]:
        """
        Detect suggested action from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Action name or None
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        for action, patterns in self._action_patterns_compiled.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    logger.debug(f"Detected action: {action}")
                    return action
        
        return None
    
    def _detect_intent(self, text: str) -> Optional[str]:
        """
        Detect user intent from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Intent name or None
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        # Score each intent
        intent_scores: Dict[str, int] = {}
        
        for intent, patterns in self._intent_patterns_compiled.items():
            score = 0
            for pattern in patterns:
                if pattern.search(text_lower):
                    score += 1
            
            if score > 0:
                intent_scores[intent] = score
        
        # Return highest scoring intent
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            logger.debug(f"Detected intent: {best_intent[0]}")
            return best_intent[0]
        
        return None
    
    def _extract_and_validate_entities(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Extract and validate entities from text.
        
        Args:
            text: Text to analyze
            context: Context (menu items, etc.)
            
        Returns:
            Tuple of (entities, validation_errors)
        """
        entities = {}
        errors = []
        
        # Extract menu items (with validation)
        menu_items = context.get("menu_items", [])
        if menu_items:
            found_items, item_errors = self._extract_and_validate_menu_items(
                text,
                menu_items
            )
            if found_items:
                entities["menu_items"] = found_items
            errors.extend(item_errors)
        
        # Extract quantities (with validation)
        quantities, qty_errors = self._extract_and_validate_quantities(text)
        if quantities:
            entities["quantities"] = quantities
        errors.extend(qty_errors)
        
        # Extract prices (with validation)
        prices, price_errors = self._extract_and_validate_prices(text, context)
        if prices:
            entities["prices"] = prices
        errors.extend(price_errors)
        
        return entities, errors
    
    def _extract_and_validate_menu_items(
        self,
        text: str,
        menu_items: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Extract and validate menu item mentions.
        
        Args:
            text: Text to search
            menu_items: List of valid menu item names
            
        Returns:
            Tuple of (found_items, validation_errors)
        """
        found = []
        errors = []
        text_lower = text.lower()
        
        # Build set of valid items (lowercase) for fast lookup
        valid_items_lower = {item.lower(): item for item in menu_items}
        
        for item_lower, item_original in valid_items_lower.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(item_lower) + r'\b'
            if re.search(pattern, text_lower):
                found.append(item_original)
        
        # Detect potential hallucination
        # Look for food-related words that aren't in the menu
        food_keywords = [
            'pizza', 'burger', 'sandwich', 'salad', 'pasta', 'chicken',
            'beef', 'pork', 'fish', 'soup', 'rice', 'noodles'
        ]
        
        if self.strict_validation:
            for keyword in food_keywords:
                if re.search(r'\b' + keyword + r'\b', text_lower):
                    # Check if this keyword is part of any valid menu item
                    if not any(keyword in item.lower() for item in menu_items):
                        errors.append(
                            f"Potential hallucination: mentioned '{keyword}' not in menu"
                        )
        
        return found, errors
    
    def _extract_and_validate_quantities(
        self,
        text: str
    ) -> Tuple[List[int], List[str]]:
        """
        Extract and validate quantity numbers.
        
        Args:
            text: Text to search
            
        Returns:
            Tuple of (quantities, validation_errors)
        """
        quantities = []
        errors = []
        
        # Pattern: number followed by item-like words
        pattern = r'\b(\d+)\s+(?:of|x|pieces?|orders?)?'
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            try:
                qty = int(match.group(1))
                
                # Validate quantity
                if qty < self.MIN_QUANTITY:
                    errors.append(
                        f"Invalid quantity {qty} (minimum is {self.MIN_QUANTITY})"
                    )
                elif qty > self.MAX_QUANTITY:
                    errors.append(
                        f"Invalid quantity {qty} (maximum is {self.MAX_QUANTITY})"
                    )
                    qty = self.MAX_QUANTITY
                
                quantities.append(qty)
            except ValueError:
                errors.append(f"Invalid quantity format: {match.group(1)}")
        
        # Also look for word numbers
        word_numbers = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        
        for word, num in word_numbers.items():
            if re.search(r'\b' + word + r'\b', text, re.IGNORECASE):
                quantities.append(num)
        
        return quantities, errors
    
    def _extract_and_validate_prices(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> Tuple[List[float], List[str]]:
        """
        Extract and validate prices.
        
        Args:
            text: Text to search
            context: Context with menu pricing
            
        Returns:
            Tuple of (prices, validation_errors)
        """
        prices = []
        errors = []
        
        # Pattern: $X.XX or X.XX
        pattern = r'\$?(\d+(?:\.\d{2})?)'
        matches = re.finditer(pattern, text)
        
        # Get valid prices from menu
        valid_prices = set()
        menu_items = context.get("menu_items_with_prices", [])
        for item in menu_items:
            if isinstance(item, dict):
                price = item.get("price")
                if price is not None:
                    valid_prices.add(float(price))
        
        for match in matches:
            try:
                price = float(match.group(1))
                
                # Validate price
                if price < self.MIN_PRICE:
                    errors.append(
                        f"Invalid price ${price:.2f} (too low)"
                    )
                elif price > self.MAX_PRICE:
                    errors.append(
                        f"Invalid price ${price:.2f} (exceeds maximum)"
                    )
                    continue
                
                # If we have menu prices, check against them
                if self.strict_validation and valid_prices:
                    # Allow small floating point differences
                    if not any(abs(price - vp) < 0.01 for vp in valid_prices):
                        errors.append(
                            f"Price ${price:.2f} does not match any menu item"
                        )
                
                prices.append(price)
            except ValueError:
                errors.append(f"Invalid price format: {match.group(1)}")
        
        return prices, errors
    
    def _calculate_confidence(
        self,
        text: str,
        action: Optional[str],
        intent: Optional[str],
        entities: Dict[str, Any],
        has_errors: bool
    ) -> float:
        """
        Calculate parsing confidence score.
        
        Args:
            text: Parsed text
            action: Detected action
            intent: Detected intent
            entities: Extracted entities
            has_errors: Whether validation errors occurred
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 1.0
        
        # Lower confidence if text is very short
        if len(text) < 5:
            confidence *= 0.5
        
        # Lower confidence if no clear intent/action
        if not action and not intent:
            confidence *= 0.7
        
        # Boost confidence if entities found and validated
        if entities and not has_errors:
            confidence = min(1.0, confidence * 1.1)
        
        # Reduce confidence if validation errors
        if has_errors:
            confidence *= 0.6
        
        # Floor at 0
        return max(0.0, confidence)
    
    def detect_goodbye(self, text: str) -> bool:
        """
        Detect if text is a goodbye.
        
        Args:
            text: Text to check
            
        Returns:
            True if goodbye detected
        """
        if not text:
            return False
        
        action = self._detect_action(text)
        return action == "end_call"
    
    def detect_transfer_request(self, text: str) -> bool:
        """
        Detect if text requests transfer.
        
        Args:
            text: Text to check
            
        Returns:
            True if transfer requested
        """
        if not text:
            return False
        
        action = self._detect_action(text)
        return action == "transfer"
    
    def extract_order_confirmation(
        self,
        text: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Extract order confirmation from text.
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (is_confirmation, confirmation_type)
            confirmation_type: 'yes', 'no', or None
        """
        if not text:
            return (False, None)
        
        text_lower = text.lower()
        
        # Positive confirmations
        yes_patterns = [
            r'\b(?:yes|yeah|yep|yup|correct|right|good|fine|ok|okay)\b',
            r'\bthat\'?s?\s+(?:correct|right|good|fine)\b',
            r'\bsounds\s+good\b',
            r'\blooks\s+good\b'
        ]
        
        for pattern in yes_patterns:
            if re.search(pattern, text_lower):
                return (True, 'yes')
        
        # Negative confirmations
        no_patterns = [
            r'\b(?:no|nope|nah|wrong|incorrect)\b',
            r'\bthat\'?s?\s+(?:wrong|incorrect|not\s+right)\b',
            r'\bnot\s+(?:correct|right)\b'
        ]
        
        for pattern in no_patterns:
            if re.search(pattern, text_lower):
                return (True, 'no')
        
        return (False, None)


# Default instance
_default_parser: Optional[OutputParser] = None


def get_default_parser() -> OutputParser:
    """
    Get or create default output parser.
    
    Returns:
        Default parser instance
    """
    global _default_parser
    
    if _default_parser is None:
        _default_parser = OutputParser(strict_validation=True)
    
    return _default_parser


def set_default_parser(parser: OutputParser) -> None:
    """
    Set default output parser.
    
    Args:
        parser: Output parser to use as default
    """
    global _default_parser
    _default_parser = parser
    logger.info("Default output parser set")


def parse_output(text: str, **kwargs) -> ParsedOutput:
    """
    Quick parse using default parser.
    
    Args:
        text: Text to parse
        **kwargs: Additional arguments
        
    Returns:
        Parsed output
    """
    parser = get_default_parser()
    return parser.parse(text, **kwargs)
