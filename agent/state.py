"""
Conversation state schema for the AutoStream AI Agent.
"""
from typing import TypedDict, Optional, List, Literal
from langchain_core.messages import BaseMessage


class LeadInfo(TypedDict, total=False):
    """Information collected from a potential lead."""
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]


class ConversationState(TypedDict):
    """
    State schema for the AutoStream agent conversation.
    Maintains context across multiple turns.
    """
    # Conversation history
    messages: List[BaseMessage]
    
    # Current classified intent
    intent: Literal["greeting", "inquiry", "high_intent", "unknown"]
    
    # Lead information being collected
    lead_info: LeadInfo
    
    # Whether lead has been successfully captured
    lead_captured: bool
    
    # Current response to send to user
    response: str
