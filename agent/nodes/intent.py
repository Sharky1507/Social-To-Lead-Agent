"""
Intent classification node for the AutoStream AI Agent.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import ConversationState


def classify_intent(state: ConversationState, llm: ChatGoogleGenerativeAI) -> str:
    """
    Classify the user's intent based on their latest message.
    
    Returns one of:
    - 'greeting': Casual hello/hi
    - 'inquiry': Product or pricing questions
    - 'high_intent': Ready to sign up/try the product
    - 'unknown': Cannot determine intent
    """
    messages = state.get("messages", [])
    if not messages:
        return "unknown"
    
    # Get the last user message
    last_message = messages[-1].content if messages else ""
    
    # Check if we're in the middle of collecting lead info
    lead_info = state.get("lead_info", {})
    if state.get("intent") == "high_intent" and not state.get("lead_captured", False):
        # If we already detected high intent and haven't captured lead yet,
        # this is likely follow-up info
        has_name = lead_info.get("name")
        has_email = lead_info.get("email")
        has_platform = lead_info.get("platform")
        
        if not (has_name and has_email and has_platform):
            return "high_intent"  # Continue lead collection
    
    system_prompt = """You are an intent classifier for AutoStream, a video editing SaaS.
Classify the user's message into exactly one of these categories:
- greeting: Casual greetings like hi, hello, hey, what's up
- inquiry: Questions about pricing, features, plans, refunds, support, or the product
- high_intent: User shows interest in signing up, trying, buying, subscribing, or mentions their platform (YouTube, Instagram, etc.)

Respond with ONLY the category name, nothing else."""

    classification_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Classify this message: {last_message}")
    ]
    
    response = llm.invoke(classification_messages)
    intent = response.content.strip().lower()
    
    # Validate the intent
    valid_intents = ["greeting", "inquiry", "high_intent"]
    if intent not in valid_intents:
        return "unknown"
    
    return intent


def intent_node(state: ConversationState, llm: ChatGoogleGenerativeAI) -> dict:
    """
    LangGraph node that classifies intent and updates state.
    """
    intent = classify_intent(state, llm)
    return {"intent": intent}
