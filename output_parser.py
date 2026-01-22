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
"""

import logging
import re
from typing import Dict, Any, Optional, List, Tuple
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
        confidence: float = 1.0
    ):
        """
        Initialize parsed output.
        
        Args:
            text: Response text
            intent: Detected intent
            action: Suggested action (e.g., 'transfer', 'end_call')
            entities: Extracted entities
            confidence: Parsing confidence (0-1)
        """
        self.text = text
        self.intent = intent
        self.action = action
        self.entities = entities or {}
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "intent": self.intent,
            "action": self.action,
            "entities": self.entities,
            "confidence": self.confidence
        }


class OutputParser:
    """
    Parses LLM outputs into structured format.
    
    Detects:
    - Intents (ordering, question, complaint, etc.)
    - Actions (transfer, end call, etc.)
    - Entities (menu items, quantities, prices, etc.)
    - Special commands
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
            r"have\s+a\s+(?:good|great|nice)\s+(?:day|night)",
            r"talk\s+to\s+you\s+(?:later|soon)",
            r"see\s+you"
        ],
        "repeat": [
            r"(?:say|repeat)\s+that\s+again",
            r"what\s+(?:did|was)\s+that",
            r"(?:didn'?t|did\s+not)\s+(?:hear|catch)\s+that",
            r"come\s+again"
        ]
    }
    
    # Intent detection patterns
    INTENT_PATTERNS = {
        "add_item": [
            r"(?:i'?d?\s+like|i\s+want|(?:can|could)\s+i\s+(?:get|have))\s+(?:a|an|the)?",
            r"add\s+(?:a|an|the)?",
            r"give\s+me\s+(?:a|an|the)?"
        ],
        "remove_item": [
            r"(?:remove|cancel|delete)\s+(?:the)?",
            r"(?:don'?t|do\s+not)\s+want",
            r"take\s+(?:off|out)\s+(?:the)?"
        ],
        "modify_item": [
            r"(?:change|modify|update)\s+(?:the)?",
            r"(?:no|without|hold)\s+the",
            r"(?:add|extra|with)\s+(?:extra)?"
        ],
        "question_menu": [
            r"what\s+(?:is|are|do\s+you\s+have)",
            r"(?:tell|show)\s+me\s+(?:what|your)",
            r"what'?s?\s+(?:on|in)\s+(?:the|your)\s+menu"
        ],
        "question_price": [
            r"how\s+much\s+(?:is|are|does|do)",
            r"what'?s?\s+the\s+(?:price|cost)",
            r"price\s+of"
        ],
        "confirm_order": [
            r"that'?s?\s+(?:correct|right|good)",
            r"(?:yes|yeah|yep),?\s+(?:that'?s?\s+)?(?:correct|right|good|it)",
            r"sounds\s+good"
        ],
        "ready_to_order": [
            r"(?:i'?m?\s+)?ready\s+to\s+order",
            r"(?:can|may)\s+i\s+(?:place|make)\s+(?:an\s+)?order"
        ]
    }
    
    def __init__(self):
        """Initialize output parser."""
        # Compile patterns for efficiency
        self._action_patterns_compiled = {
            action: [re.compile(p, re.IGNORECASE) for p in patterns]
            for action, patterns in self.ACTION_PATTERNS.items()
        }
        
        self._intent_patterns_compiled = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }
        
        logger.info("OutputParser initialized")
    
    def parse(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ParsedOutput:
        """
        Parse LLM output text.
        
        Args:
            text: LLM response text
            context: Optional parsing context
            
        Returns:
            Parsed output
        """
        # Sanitize text
        clean_text = self._sanitize_text(text)
        
        # Detect action
        action = self._detect_action(clean_text)
        
        # Detect intent
        intent = self._detect_intent(clean_text)
        
        # Extract entities (if context provided)
        entities = {}
        if context:
            entities = self._extract_entities(clean_text, context)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            text=clean_text,
            action=action,
            intent=intent,
            entities=entities
        )
        
        return ParsedOutput(
            text=clean_text,
            intent=intent,
            action=action,
            entities=entities,
            confidence=confidence
        )
    
    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize text output.
        
        Args:
            text: Raw text
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove markdown artifacts
        text = re.sub(r'\*\*', '', text)
        text = re.sub(r'__', '', text)
        
        return text
    
    def _detect_action(self, text: str) -> Optional[str]:
        """
        Detect suggested action from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Action name or None
        """
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
    
    def _extract_entities(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract entities from text.
        
        Args:
            text: Text to analyze
            context: Context (menu items, etc.)
            
        Returns:
            Extracted entities
        """
        entities = {}
        
        # Extract menu items (if menu in context)
        menu_items = context.get("menu_items", [])
        if menu_items:
            found_items = self._extract_menu_items(text, menu_items)
            if found_items:
                entities["menu_items"] = found_items
        
        # Extract quantities
        quantities = self._extract_quantities(text)
        if quantities:
            entities["quantities"] = quantities
        
        # Extract prices
        prices = self._extract_prices(text)
        if prices:
            entities["prices"] = prices
        
        return entities
    
    def _extract_menu_items(
        self,
        text: str,
        menu_items: List[str]
    ) -> List[str]:
        """
        Extract menu item mentions from text.
        
        Args:
            text: Text to search
            menu_items: List of menu item names
            
        Returns:
            List of found menu items
        """
        found = []
        text_lower = text.lower()
        
        for item in menu_items:
            item_lower = item.lower()
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(item_lower) + r'\b'
            if re.search(pattern, text_lower):
                found.append(item)
        
        return found
    
    def _extract_quantities(self, text: str) -> List[int]:
        """
        Extract quantity numbers from text.
        
        Args:
            text: Text to search
            
        Returns:
            List of quantities
        """
        quantities = []
        
        # Pattern: number followed by item-like words
        pattern = r'\b(\d+)\s+(?:of|x|pieces?|orders?)?'
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            qty = int(match.group(1))
            quantities.append(qty)
        
        # Also look for word numbers
        word_numbers = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        
        for word, num in word_numbers.items():
            if re.search(r'\b' + word + r'\b', text, re.IGNORECASE):
                quantities.append(num)
        
        return quantities
    
    def _extract_prices(self, text: str) -> List[float]:
        """
        Extract prices from text.
        
        Args:
            text: Text to search
            
        Returns:
            List of prices
        """
        prices = []
        
        # Pattern: $X.XX or X.XX
        pattern = r'\$?(\d+\.\d{2})'
        matches = re.finditer(pattern, text)
        
        for match in matches:
            price = float(match.group(1))
            prices.append(price)
        
        return prices
    
    def _calculate_confidence(
        self,
        text: str,
        action: Optional[str],
        intent: Optional[str],
        entities: Dict[str, Any]
    ) -> float:
        """
        Calculate parsing confidence score.
        
        Args:
            text: Parsed text
            action: Detected action
            intent: Detected intent
            entities: Extracted entities
            
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
        
        # Boost confidence if entities found
        if entities:
            confidence = min(1.0, confidence * 1.1)
        
        return confidence
    
    def detect_goodbye(self, text: str) -> bool:
        """
        Detect if text is a goodbye.
        
        Args:
            text: Text to check
            
        Returns:
            True if goodbye detected
        """
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
        text_lower = text.lower()
        
        # Positive confirmations
        yes_patterns = [
            r'\b(?:yes|yeah|yep|yup|correct|right|good|fine)\b',
            r'\bthat\'?s?\s+(?:correct|right|good|fine)\b',
            r'\bsounds\s+good\b'
        ]
        
        for pattern in yes_patterns:
            if re.search(pattern, text_lower):
                return (True, 'yes')
        
        # Negative confirmations
        no_patterns = [
            r'\b(?:no|nope|nah|wrong|incorrect)\b',
            r'\bthat\'?s?\s+(?:wrong|incorrect|not\s+right)\b'
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
        _default_parser = OutputParser()
    
    return _default_parser


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
