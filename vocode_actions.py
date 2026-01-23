"""
Vocode Actions
==============
Vocode action wrappers that route through CallController.

Responsibilities:
- Wrap Vocode's TransferCall action
- Wrap Vocode's EndConversation action
- Ensure all actions go through CallController approval
- Prevent AI from directly controlling system flow
- Validate state before action execution
- NEVER allow AI to bypass controller authority

These actions are registered with Vocode but defer decisions to CallController.

CRITICAL SAFETY RULES:
- Actions MUST check controller state
- Actions MUST be approved by controller
- Actions CANNOT execute in frozen states
- Actions CANNOT bypass termination checks
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
    transfer_number: Optional[str] = None


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
    3. Validate controller state
    4. Route to CallController for approval
    5. If approved, CallController executes transfer sequence
    6. If denied, return denial message
    
    CallController maintains control - AI cannot force transfers.
    
    SAFETY:
    - Cannot execute if controller is terminating
    - Cannot execute in frozen state
    - Must be approved by controller
    - Transfer is handled by controller, not this action
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
        
        logger.debug(
            "ControlledTransferCall initialized",
            extra={"has_controller": call_controller is not None}
        )
    
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
        reason = action_input.params.reason or "ai_suggested"
        transfer_number = action_input.params.transfer_number
        
        logger.info(
            "Transfer action triggered",
            extra={
                "reason": reason,
                "number": transfer_number
            }
        )
        
        # SAFETY CHECK 1: Verify controller exists
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
        
        # SAFETY CHECK 2: Verify controller is not terminating
        if self.call_controller._termination_in_progress:
            logger.warning(
                "Transfer blocked - controller terminating",
                extra={"call_id": self.call_controller.call_id}
            )
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=ControlledTransferCallResponse(
                    success=False,
                    approved=False,
                    message="Call is ending"
                )
            )
        
        # SAFETY CHECK 3: Verify not in frozen state
        if self.call_controller.state_machine.is_frozen():
            logger.warning(
                "Transfer blocked - frozen state",
                extra={
                    "call_id": self.call_controller.call_id,
                    "state": self.call_controller.state_machine.current_state.value
                }
            )
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=ControlledTransferCallResponse(
                    success=False,
                    approved=False,
                    message="Transfer not available in current state"
                )
            )
        
        # Wait for user message to complete if needed
        if action_input.user_message_tracker is not None:
            try:
                await asyncio.wait_for(
                    action_input.user_message_tracker.wait(),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("User message tracker wait timed out")
        
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
        
        # Route through CallController for approval and execution
        # Controller will handle the full transfer sequence
        try:
            transfer_result = await self.call_controller._handle_transfer_request(
                reason=reason,
                transfer_number=transfer_number
            )
            
            status = transfer_result.get("status")
            
            # Check result
            if status == "denied":
                logger.info("Transfer denied by controller")
                return ActionOutput(
                    action_type=action_input.action_config.type,
                    response=ControlledTransferCallResponse(
                        success=False,
                        approved=False,
                        message="Transfer not approved"
                    )
                )
            
            elif status == "approved":
                # Transfer was approved and executed by controller
                logger.info("Transfer approved and executed by controller")
                return ActionOutput(
                    action_type=action_input.action_config.type,
                    response=ControlledTransferCallResponse(
                        success=True,
                        approved=True,
                        transfer_number=transfer_result.get("number"),
                        message="Transfer initiated"
                    )
                )
            
            elif status in ["already_in_progress", "already_transferred"]:
                logger.info(f"Transfer status: {status}")
                return ActionOutput(
                    action_type=action_input.action_config.type,
                    response=ControlledTransferCallResponse(
                        success=False,
                        approved=False,
                        message="Transfer already in progress"
                    )
                )
            
            elif status == "terminating":
                logger.info("Transfer blocked - call terminating")
                return ActionOutput(
                    action_type=action_input.action_config.type,
                    response=ControlledTransferCallResponse(
                        success=False,
                        approved=False,
                        message="Call is ending"
                    )
                )
            
            else:
                logger.warning(f"Unknown transfer status: {status}")
                return ActionOutput(
                    action_type=action_input.action_config.type,
                    response=ControlledTransferCallResponse(
                        success=False,
                        approved=False,
                        message="Transfer failed"
                    )
                )
        
        except Exception as e:
            logger.error(
                f"Transfer request failed: {e}",
                extra={"call_id": self.call_controller.call_id},
                exc_info=True
            )
            
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=ControlledTransferCallResponse(
                    success=False,
                    approved=False,
                    message="Transfer request failed"
                )
            )


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
    
    SAFETY:
    - Cannot execute if already terminating
    - Routes through controller terminate_call()
    - Never bypasses controller authority
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
        
        logger.debug(
            "ControlledEndConversation initialized",
            extra={"has_controller": call_controller is not None}
        )
    
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
        reason = action_input.params.reason or "ai_suggested_end"
        
        logger.info(
            "End conversation action triggered",
            extra={"reason": reason}
        )
        
        # SAFETY CHECK 1: Verify controller exists
        if not self.call_controller:
            logger.error("No CallController - using fallback termination")
            # Fallback to Vocode's termination
            try:
                await self.conversation_state_manager.terminate_conversation()
                return ActionOutput(
                    action_type=action_input.action_config.type,
                    response=ControlledEndConversationResponse(
                        success=True,
                        message="Conversation ended (fallback)"
                    )
                )
            except Exception as e:
                logger.error(f"Fallback termination failed: {e}", exc_info=True)
                return ActionOutput(
                    action_type=action_input.action_config.type,
                    response=ControlledEndConversationResponse(
                        success=False,
                        message="Failed to end conversation"
                    )
                )
        
        # SAFETY CHECK 2: Check if already terminating
        if self.call_controller._termination_in_progress:
            logger.info("End conversation blocked - already terminating")
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=ControlledEndConversationResponse(
                    success=True,
                    message="Conversation already ending"
                )
            )
        
        # Wait for user message to complete if needed
        if action_input.user_message_tracker is not None:
            try:
                await asyncio.wait_for(
                    action_input.user_message_tracker.wait(),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("User message tracker wait timed out")
        
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
        
        # Route through CallController's close method
        # This ensures proper shutdown sequence
        try:
            # Use close instead of terminate_call for graceful shutdown
            await self.call_controller.close(reason=reason)
            
            logger.info("Conversation end initiated")
            
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=ControlledEndConversationResponse(
                    success=True,
                    message="Conversation ended"
                )
            )
        
        except Exception as e:
            logger.error(
                f"End conversation failed: {e}",
                extra={"call_id": self.call_controller.call_id},
                exc_info=True
            )
            
            return ActionOutput(
                action_type=action_input.action_config.type,
                response=ControlledEndConversationResponse(
                    success=False,
                    message="Failed to end conversation"
                )
            )


# =============================================================================
# Action Factory
# =============================================================================

class ControlledActionFactory:
    """
    Factory for creating controlled actions.
    
    Creates actions with CallController injected.
    
    SAFETY:
    - All actions must have controller reference
    - Actions validate controller state before execution
    - Actions cannot bypass controller authority
    """
    
    def __init__(self, call_controller):
        """
        Initialize factory.
        
        Args:
            call_controller: CallController instance to inject
        """
        if not call_controller:
            raise ValueError("CallController is required")
        
        self.call_controller = call_controller
        
        logger.info(
            "ControlledActionFactory initialized",
            extra={"call_id": call_controller.call_id}
        )
    
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
        
        action = ControlledTransferCall(
            action_config=config,
            call_controller=self.call_controller
        )
        
        logger.debug(
            "Transfer action created",
            extra={"call_id": self.call_controller.call_id}
        )
        
        return action
    
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
        
        action = ControlledEndConversation(
            action_config=config,
            call_controller=self.call_controller
        )
        
        logger.debug(
            "End conversation action created",
            extra={"call_id": self.call_controller.call_id}
        )
        
        return action
    
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
