"""
RAG-powered knowledge retrieval node for the AutoStream AI Agent.
"""
import json
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import ConversationState


def load_knowledge_base() -> dict:
    """Load the AutoStream knowledge base from JSON."""
    kb_path = Path(__file__).parent.parent.parent / "knowledge" / "autostream_kb.json"
    with open(kb_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_knowledge_base(kb: dict) -> str:
    """Format the knowledge base into a readable string for the LLM."""
    lines = []
    
    # Company info
    lines.append(f"Company: {kb['company']}")
    lines.append(f"Description: {kb['tagline']}")
    lines.append("")
    
    # Pricing
    lines.append("PRICING:")
    for plan_key, plan in kb["pricing"].items():
        lines.append(f"  {plan['name']}: {plan['price']}")
        for feature in plan["features"]:
            lines.append(f"    - {feature}")
    lines.append("")
    
    # Policies
    lines.append("POLICIES:")
    lines.append(f"  Refund Policy: {kb['policies']['refund']}")
    lines.append(f"  Basic Plan Support: {kb['policies']['support']['basic']}")
    lines.append(f"  Pro Plan Support: {kb['policies']['support']['pro']}")
    lines.append("")
    
    # FAQ
    lines.append("FAQ:")
    for faq in kb["faq"]:
        lines.append(f"  Q: {faq['question']}")
        lines.append(f"  A: {faq['answer']}")
        lines.append("")
    
    return "\n".join(lines)


def retrieve_and_respond(state: ConversationState, llm: ChatGoogleGenerativeAI) -> str:
    """
    Retrieve relevant knowledge and generate a response.
    """
    messages = state.get("messages", [])
    if not messages:
        return "How can I help you today?"
    
    last_message = messages[-1].content
    
    # Load and format knowledge base
    kb = load_knowledge_base()
    kb_text = format_knowledge_base(kb)
    
    system_prompt = f"""You are a helpful sales assistant for AutoStream, a SaaS product that provides automated video editing tools for content creators.

Use the following knowledge base to answer questions accurately:

{kb_text}

Guidelines:
- Be friendly and helpful
- Answer questions accurately based on the knowledge base
- If asked about pricing, clearly explain both plans
- If the user shows interest in signing up, encourage them and let them know we'd love to help them get started
- Keep responses concise but informative"""

    # Build conversation context
    conversation_messages = [SystemMessage(content=system_prompt)]
    for msg in messages:
        conversation_messages.append(msg)
    
    response = llm.invoke(conversation_messages)
    return response.content


def rag_node(state: ConversationState, llm: ChatGoogleGenerativeAI) -> dict:
    """
    LangGraph node that performs RAG retrieval and generates response.
    """
    response = retrieve_and_respond(state, llm)
    return {"response": response}
