"""
LangGraph workflow definition for the AutoStream AI Agent.
"""
from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

from agent.state import ConversationState
from agent.nodes.intent import intent_node
from agent.nodes.rag import rag_node
from agent.nodes.lead import lead_node


def create_agent(api_key: str):
    """
    Create and return the AutoStream agent graph.
    
    Args:
        api_key: Google API key for Gemini
    
    Returns:
        Compiled LangGraph workflow
    """
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7
    )
    
    # Create node functions with LLM binding
    def intent_classifier(state: ConversationState) -> dict:
        return intent_node(state, llm)
    
    def rag_retriever(state: ConversationState) -> dict:
        return rag_node(state, llm)
    
    def lead_qualifier(state: ConversationState) -> dict:
        return lead_node(state, llm)
    
    def greeting_responder(state: ConversationState) -> dict:
        """Generate a friendly greeting response."""
        messages = state.get("messages", [])
        context_messages = [
            {"role": "system", "content": "You are a friendly sales assistant for AutoStream, a video editing SaaS for content creators. Respond warmly to greetings and offer to help with any questions about our product or pricing."},
        ]
        for msg in messages:
            if isinstance(msg, HumanMessage):
                context_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                context_messages.append({"role": "assistant", "content": msg.content})
        
        response = llm.invoke(context_messages)
        return {"response": response.content}
    
    # Define the routing function
    def route_by_intent(state: ConversationState) -> Literal["greeting", "inquiry", "high_intent"]:
        intent = state.get("intent", "unknown")
        
        # If lead already captured, treat as inquiry for follow-up
        if state.get("lead_captured", False) and intent == "high_intent":
            return "inquiry"
        
        if intent == "greeting":
            return "greeting"
        elif intent == "inquiry":
            return "inquiry"
        elif intent == "high_intent":
            return "high_intent"
        else:
            return "inquiry"  # Default to inquiry for unknown
    
    # Build the graph
    workflow = StateGraph(ConversationState)
    
    # Add nodes
    workflow.add_node("classify_intent", intent_classifier)
    workflow.add_node("greeting_response", greeting_responder)
    workflow.add_node("rag_response", rag_retriever)
    workflow.add_node("lead_qualification", lead_qualifier)
    
    # Set entry point
    workflow.set_entry_point("classify_intent")
    
    # Add conditional edges from intent classification
    workflow.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "greeting": "greeting_response",
            "inquiry": "rag_response",
            "high_intent": "lead_qualification"
        }
    )
    
    # All response nodes go to END
    workflow.add_edge("greeting_response", END)
    workflow.add_edge("rag_response", END)
    workflow.add_edge("lead_qualification", END)
    
    # Compile the graph
    return workflow.compile()


def run_conversation(agent, state: ConversationState, user_message: str) -> tuple[ConversationState, str]:
    """
    Process a user message through the agent.
    
    Args:
        agent: Compiled LangGraph agent
        state: Current conversation state
        user_message: User's input message
    
    Returns:
        Tuple of (updated_state, agent_response)
    """
    # Add user message to state
    new_messages = state.get("messages", []).copy()
    new_messages.append(HumanMessage(content=user_message))
    
    # Update state with new message
    current_state = {
        **state,
        "messages": new_messages
    }
    
    # Run the agent
    result = agent.invoke(current_state)
    
    # Get the response and add it to messages
    response = result.get("response", "I'm here to help! What would you like to know about AutoStream?")
    result["messages"] = result.get("messages", []).copy()
    result["messages"].append(AIMessage(content=response))
    
    return result, response
