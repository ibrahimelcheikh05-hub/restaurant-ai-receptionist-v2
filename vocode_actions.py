"""
Vocode Actions
==============
Vocode action wrappers that route through CallController.

Responsibilities:
- Wrap Vocode's TransferCall action
- Wrap Vocode's EndConversation action
- Ensure all actions go through CallController approval
- Prevent AI from directly controlling system flow

These actions are registered with Vocode but defer decisions to CallController.
"""

import asyncio
import logging
from typing import Type, Optional

from pydantic.v1 import BaseModel

from vocode.streaming.action.phone_call_action import TwilioPhoneConversationAction
from vocode.streaming.action.base_action import BaseAction
from vocode.streaming.models.actions import ActionConfig as VocodeActionConfig
from vocode.streaming.models.actions import ActionInput, ActionOutput
from vocode.streaming.utils.state_manager import TwilioPhoneConversationStateManager

logger = logging.getLogger(__name__)


# =============================================================================
# Transfer Call Action
# =============================================================================

class ControlledTransferCallParameters(BaseModel):
    """Parameters for controlled transfer."""
    reason: Optional[str] = "user_requested"


class ControlledTransferCallResponse(BaseModel):
    """Response from controlled transfer."""
    success: bool
    approved: bool
    transfer_number: Optional[str] = None
    message: Optional[str] = None


class ControlledTransferCallActionConfig(
    VocodeActionConfig,
    type="action_controlled_transfer_call"  # type: ignore
):
    """
    Configuration for controlled call transfer.
    
    Unlike Vocode's default TransferCall, this routes through
    CallController for approval.
    """
    
    def action_attempt_to_string(self, input: ActionInput) -> str:
        return "Requesting call transfer approval"
    
    def action_result_to_string(
        self,
        input: ActionInput,
        output: ActionOutput
    ) -> str:
        assert isinstance(output.response, ControlledTransferCallResponse)
        
        if output.response.approved and output.response.success:
            return f"Call transfer approved and initiated to {output.response.transfer_number}"
        elif output.response.approved and not output.response.success:
            return "Call transfer approved but failed to execute"
        else:
            return "Call transfer request denied"


class ControlledTransferCall(
    TwilioPhoneConversationAction[
        ControlledTransferCallActionConfig,
        ControlledTransferCallParameters,
        ControlledTransferCallResponse
    ]
):
    """
    Controlled call transfer action.
    
    Flow:
    1. AI suggests transfer
    2. This action is triggered
    3. We route to CallController for approval
    4. If approved, we execute Twilio transfer
    5. If denied, we return denial message
    
    CallController maintains control - AI cannot force transfers.
    """
    
    description: str = (
        "Request to transfer the call to a human agent. "
        "Use this when the caller requests to speak with a person, "
        "or when you cannot help with their request."
    )
    parameters_type: Type[ControlledTransferCallParameters] = (
        ControlledTransferCallParameters
    )
    response_type: Type[ControlledTransferCallResponse] = (
        ControlledTransferCallResponse
    )
    conversation_state_manager: TwilioPhoneConversationStateManager
    
    def __init__(
        self,
        action_config: ControlledTransferCallActionConfig,
        call_controller=None
    ):
        """
        Initialize controlled transfer action.
        
        Args:
            action_config: Action configuration
            call_controller: CallController instance (injected)
        """
        super().__init__(
            action_config,
            quiet=False,  # We want to announce transfer
            is_interruptible=False,  # Don't allow interruption during transfer
            should_respond="always"
        )
        
        self.call_controller = call_controller
    
    async def run(
        self,
        action_input: ActionInput[ControlledTransferCallParameters]
    ) -> ActionOutput[ControlledTransferCallResponse]:
        """
        Execute controlled transfer.
        
        Args:
            action_input: Action input with parameters
            
        Returns:
            Transfer result
        """
        logger.info(
            "Transfer action triggered",
            extra={
                "reason": action_input.params.reason
            }
        )
        
        # Wait for user message to complete if needed
        if action_input.user_message_tracker is not None:
            await action_input.user_message_tracker.wait()
        
        # Check if last message was interrupted
        if self.conversation_state_manager.transcript.was_last_message_interrupted():
            logger.info("Last message interrupted, aborting transfer")
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=ControlledTransferCallResponse(
                    success=False,
                    approved=False,
                    message="Transfer cancelled due to interruption"
                )
            )
        
        # Route through CallController for approval
        if not self.call_controller:
            logger.error("No CallController - cannot approve transfer")
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=ControlledTransferCallResponse(
                    success=False,
                    approved=False,
                    message="System error - transfer unavailable"
                )
            )
        
        # Request transfer from controller
        transfer_result = await self.call_controller._handle_transfer_request(
            reason=action_input.params.reason or "ai_suggested"
        )
        
        approved = transfer_result.get("status") == "transfer_approved"
        transfer_number = transfer_result.get("transfer_number")
        
        if not approved:
            logger.info("Transfer denied by controller")
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=ControlledTransferCallResponse(
                    success=False,
                    approved=False,
                    message=transfer_result.get("response_text")
                )
            )
        
        # Transfer approved - execute Twilio transfer
        logger.info(f"Transfer approved - executing to {transfer_number}")
        
        try:
            # Get Twilio call SID
            twilio_call_sid = self.get_twilio_sid(action_input)
            
            # Execute transfer via Twilio
            await self._execute_twilio_transfer(
                twilio_call_sid=twilio_call_sid,
                to_phone=transfer_number
            )
            
            logger.info("Transfer executed successfully")
            
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=ControlledTransferCallResponse(
                    success=True,
                    approved=True,
                    transfer_number=transfer_number,
                    message=transfer_result.get("response_text")
                )
            )
        
        except Exception as e:
            logger.error(
                f"Transfer execution failed: {e}",
                exc_info=True
            )
            
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=ControlledTransferCallResponse(
                    success=False,
                    approved=True,
                    transfer_number=transfer_number,
                    message="Transfer approved but failed to execute"
                )
            )
    
    async def _execute_twilio_transfer(
        self,
        twilio_call_sid: str,
        to_phone: str
    ) -> dict:
        """
        Execute Twilio call transfer.
        
        Args:
            twilio_call_sid: Twilio call SID
            to_phone: Phone number to transfer to
            
        Returns:
            Twilio API response
        """
        from vocode.streaming.utils.async_requester import AsyncRequestor
        from vocode.streaming.utils.phone_numbers import sanitize_phone_number
        
        twilio_client = self.conversation_state_manager.create_twilio_client()
        
        sanitized_phone = sanitize_phone_number(to_phone)
        
        url = (
            f"https://api.twilio.com/2010-04-01/Accounts/"
            f"{twilio_client.get_telephony_config().account_sid}/"
            f"Calls/{twilio_call_sid}.json"
        )
        
        twiml_data = f"<Response><Dial>{sanitized_phone}</Dial></Response>"
        payload = {"Twiml": twiml_data}
        
        async with AsyncRequestor().get_session().post(
            url,
            data=payload,
            auth=twilio_client.auth
        ) as response:
            if response.status != 200:
                error_msg = f"Twilio transfer failed: {response.status}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            return await response.json()


# =============================================================================
# End Conversation Action
# =============================================================================

class ControlledEndConversationParameters(BaseModel):
    """Parameters for controlled end conversation."""
    reason: Optional[str] = "conversation_complete"


class ControlledEndConversationResponse(BaseModel):
    """Response from controlled end conversation."""
    success: bool
    message: Optional[str] = None


class ControlledEndConversationActionConfig(
    VocodeActionConfig,
    type="action_controlled_end_conversation"  # type: ignore
):
    """Configuration for controlled conversation ending."""
    
    def action_attempt_to_string(self, input: ActionInput) -> str:
        return "Attempting to end conversation"
    
    def action_result_to_string(
        self,
        input: ActionInput,
        output: ActionOutput
    ) -> str:
        assert isinstance(output.response, ControlledEndConversationResponse)
        
        if output.response.success:
            return "Conversation ended successfully"
        else:
            return "Did not end conversation due to interruption"


class ControlledEndConversation(
    BaseAction[
        ControlledEndConversationActionConfig,
        ControlledEndConversationParameters,
        ControlledEndConversationResponse
    ]
):
    """
    Controlled end conversation action.
    
    Routes through CallController before ending call.
    Ensures proper cleanup and state management.
    """
    
    description: str = (
        "End the conversation. Use this when the caller says goodbye "
        "or when the conversation has naturally concluded."
    )
    parameters_type: Type[ControlledEndConversationParameters] = (
        ControlledEndConversationParameters
    )
    response_type: Type[ControlledEndConversationResponse] = (
        ControlledEndConversationResponse
    )
    
    def __init__(
        self,
        action_config: ControlledEndConversationActionConfig,
        call_controller=None
    ):
        """
        Initialize controlled end conversation action.
        
        Args:
            action_config: Action configuration
            call_controller: CallController instance (injected)
        """
        super().__init__(
            action_config,
            quiet=True,  # Don't announce ending
            is_interruptible=False,  # Can't interrupt goodbye
            should_respond="sometimes"
        )
        
        self.call_controller = call_controller
    
    async def run(
        self,
        action_input: ActionInput[ControlledEndConversationParameters]
    ) -> ActionOutput[ControlledEndConversationResponse]:
        """
        Execute controlled conversation end.
        
        Args:
            action_input: Action input with parameters
            
        Returns:
            End conversation result
        """
        logger.info(
            "End conversation action triggered",
            extra={
                "reason": action_input.params.reason
            }
        )
        
        # Wait for user message to complete if needed
        if action_input.user_message_tracker is not None:
            await action_input.user_message_tracker.wait()
        
        # Check if last message was interrupted
        if self.conversation_state_manager.transcript.was_last_message_interrupted():
            logger.info("Last message interrupted, not ending conversation")
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=ControlledEndConversationResponse(
                    success=False,
                    message="User interrupted - continuing conversation"
                )
            )
        
        # Route through CallController
        if self.call_controller:
            await self.call_controller.close(
                reason=action_input.params.reason or "ai_suggested_end"
            )
        else:
            # Fallback to Vocode's termination
            logger.warning("No CallController - using direct termination")
            await self.conversation_state_manager.terminate_conversation()
        
        logger.info("Conversation ended")
        
        return ActionOutput(
            action_type=action_input.action_config.type,
            response=ControlledEndConversationResponse(
                success=True,
                message="Conversation ended"
            )
        )


# =============================================================================
# Action Factory
# =============================================================================

class ControlledActionFactory:
    """
    Factory for creating controlled actions.
    
    Creates actions with CallController injected.
    """
    
    def __init__(self, call_controller):
        """
        Initialize factory.
        
        Args:
            call_controller: CallController instance to inject
        """
        self.call_controller = call_controller
    
    def create_transfer_action(
        self,
        config: Optional[ControlledTransferCallActionConfig] = None
    ) -> ControlledTransferCall:
        """
        Create transfer call action.
        
        Args:
            config: Optional action config
            
        Returns:
            Transfer action
        """
        if not config:
            config = ControlledTransferCallActionConfig()
        
        return ControlledTransferCall(
            action_config=config,
            call_controller=self.call_controller
        )
    
    def create_end_conversation_action(
        self,
        config: Optional[ControlledEndConversationActionConfig] = None
    ) -> ControlledEndConversation:
        """
        Create end conversation action.
        
        Args:
            config: Optional action config
            
        Returns:
            End conversation action
        """
        if not config:
            config = ControlledEndConversationActionConfig()
        
        return ControlledEndConversation(
            action_config=config,
            call_controller=self.call_controller
        )
    
    def get_action_configs(self) -> list:
        """
        Get list of action configs for agent.
        
        Returns:
            List of action configs
        """
        return [
            ControlledTransferCallActionConfig(),
            ControlledEndConversationActionConfig()
        ]
