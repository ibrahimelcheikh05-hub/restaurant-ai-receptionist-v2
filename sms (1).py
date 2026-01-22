"""
SMS
===
SMS delivery via Twilio.

Responsibilities:
- Send SMS messages
- Retry on failure
- Track delivery status
- Cost tracking
- Phone number validation
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import asyncio
import os
import re

logger = logging.getLogger(__name__)

# Try to import Twilio
try:
    from twilio.rest import Client
    from twilio.base.exceptions import TwilioRestException
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logger.warning("Twilio not available - SMS disabled")


class SMSResult:
    """SMS send result."""
    
    def __init__(
        self,
        success: bool,
        message_sid: Optional[str] = None,
        error: Optional[str] = None
    ):
        """
        Initialize SMS result.
        
        Args:
            success: Whether SMS was sent successfully
            message_sid: Twilio message SID
            error: Error message if failed
        """
        self.success = success
        self.message_sid = message_sid
        self.error = error
        self.sent_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "message_sid": self.message_sid,
            "error": self.error,
            "sent_at": self.sent_at.isoformat()
        }


class SMSClient:
    """
    SMS client using Twilio.
    
    Features:
    - Send SMS messages
    - Retry on transient failures
    - Phone number validation
    - Delivery tracking
    """
    
    def __init__(
        self,
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        from_number: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize SMS client.
        
        Args:
            account_sid: Twilio account SID
            auth_token: Twilio auth token
            from_number: From phone number
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (seconds)
        """
        self.account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = from_number or os.getenv("TWILIO_PHONE_NUMBER")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize Twilio client
        self.client = None
        if TWILIO_AVAILABLE and self.account_sid and self.auth_token:
            try:
                self.client = Client(self.account_sid, self.auth_token)
                logger.info("Twilio SMS client initialized")
            except Exception as e:
                logger.error(
                    f"Failed to initialize Twilio: {e}",
                    exc_info=True
                )
        
        # Metrics
        self._total_sent = 0
        self._total_failed = 0
        self._total_cost = 0.0
        
        logger.info(
            "SMSClient initialized",
            extra={
                "twilio_available": TWILIO_AVAILABLE,
                "from_number": self.from_number
            }
        )
    
    def _validate_phone(self, phone: str) -> bool:
        """
        Validate phone number format.
        
        Args:
            phone: Phone number
            
        Returns:
            True if valid
        """
        if not phone:
            return False
        
        # Remove non-digits
        digits = re.sub(r'\D', '', phone)
        
        # Should be 10-15 digits
        if len(digits) < 10 or len(digits) > 15:
            return False
        
        return True
    
    async def send_sms(
        self,
        to_number: str,
        message: str,
        order_id: Optional[str] = None
    ) -> SMSResult:
        """
        Send SMS message.
        
        Args:
            to_number: Recipient phone number
            message: Message text
            order_id: Optional order ID for tracking
            
        Returns:
            SMS result
        """
        # Validate inputs
        if not self._validate_phone(to_number):
            logger.error(f"Invalid phone number: {to_number}")
            return SMSResult(
                success=False,
                error="Invalid phone number"
            )
        
        if not message or not message.strip():
            logger.error("Empty message")
            return SMSResult(
                success=False,
                error="Empty message"
            )
        
        # Check if Twilio is available
        if not self.client:
            logger.warning("Twilio client not available")
            return SMSResult(
                success=False,
                error="SMS service not available"
            )
        
        # Normalize phone number
        if not to_number.startswith('+'):
            # Assume US number
            digits = re.sub(r'\D', '', to_number)
            if len(digits) == 10:
                to_number = f"+1{digits}"
            else:
                to_number = f"+{digits}"
        
        # Try sending with retries
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Sending SMS (attempt {attempt + 1})",
                    extra={
                        "to": to_number,
                        "order_id": order_id
                    }
                )
                
                # Send via Twilio (sync call in executor)
                loop = asyncio.get_event_loop()
                
                def _send_sync():
                    return self.client.messages.create(
                        to=to_number,
                        from_=self.from_number,
                        body=message
                    )
                
                twilio_message = await loop.run_in_executor(
                    None,
                    _send_sync
                )
                
                # Success
                self._total_sent += 1
                self._total_cost += 0.0079  # Approximate cost per SMS
                
                logger.info(
                    "SMS sent successfully",
                    extra={
                        "to": to_number,
                        "sid": twilio_message.sid,
                        "order_id": order_id
                    }
                )
                
                return SMSResult(
                    success=True,
                    message_sid=twilio_message.sid
                )
            
            except TwilioRestException as e:
                last_error = str(e)
                error_code = e.code
                
                logger.warning(
                    f"Twilio error (attempt {attempt + 1}): {e}",
                    extra={
                        "to": to_number,
                        "error_code": error_code
                    }
                )
                
                # Check if permanent error (don't retry)
                permanent_codes = {21211, 21614, 21408}
                if error_code in permanent_codes:
                    logger.error(
                        f"Permanent error code {error_code} - not retrying"
                    )
                    break
                
                # Wait before retry
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
            
            except Exception as e:
                last_error = str(e)
                logger.error(
                    f"SMS send error (attempt {attempt + 1}): {e}",
                    exc_info=True
                )
                
                # Wait before retry
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
        
        # All retries failed
        self._total_failed += 1
        
        logger.error(
            "SMS send failed after all retries",
            extra={
                "to": to_number,
                "attempts": self.max_retries,
                "last_error": last_error
            }
        )
        
        return SMSResult(
            success=False,
            error=last_error
        )
    
    async def send_order_confirmation(
        self,
        to_number: str,
        order_summary: Dict[str, Any]
    ) -> SMSResult:
        """
        Send order confirmation SMS.
        
        Args:
            to_number: Customer phone number
            order_summary: Order details
            
        Returns:
            SMS result
        """
        # Format message
        items = order_summary.get("items", [])
        total = order_summary.get("total", 0.0)
        
        message_lines = [
            "Thank you for your order!",
            "",
            "Your order:"
        ]
        
        for item in items:
            message_lines.append(f"- {item}")
        
        message_lines.append("")
        message_lines.append(f"Total: ${total:.2f}")
        message_lines.append("")
        message_lines.append("We'll have it ready soon!")
        
        message = "\n".join(message_lines)
        
        return await self.send_sms(
            to_number=to_number,
            message=message,
            order_id=order_summary.get("order_id")
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get SMS statistics.
        
        Returns:
            Statistics dictionary
        """
        total = self._total_sent + self._total_failed
        
        stats = {
            "total_sent": self._total_sent,
            "total_failed": self._total_failed,
            "total_cost": self._total_cost
        }
        
        if total > 0:
            stats["success_rate"] = self._total_sent / total
        
        return stats


# Default instance
_default_client: Optional[SMSClient] = None


def get_default_client() -> SMSClient:
    """
    Get or create default SMS client.
    
    Returns:
        Default client instance
    """
    global _default_client
    
    if _default_client is None:
        _default_client = SMSClient()
    
    return _default_client


async def send_sms(
    to_number: str,
    message: str,
    **kwargs
) -> SMSResult:
    """
    Quick SMS send.
    
    Args:
        to_number: Recipient phone
        message: Message text
        **kwargs: Additional arguments
        
    Returns:
        SMS result
    """
    client = get_default_client()
    return await client.send_sms(to_number, message, **kwargs)


async def send_order_confirmation(
    to_number: str,
    order_summary: Dict[str, Any]
) -> SMSResult:
    """
    Quick order confirmation SMS.
    
    Args:
        to_number: Customer phone
        order_summary: Order details
        
    Returns:
        SMS result
    """
    client = get_default_client()
    return await client.send_order_confirmation(to_number, order_summary)
