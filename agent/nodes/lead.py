"""
Lead qualification and capture node for the AutoStream AI Agent.
"""
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import ConversationState
from agent.tools.lead_capture import mock_lead_capture


def extract_lead_info(state: ConversationState, llm: ChatGoogleGenerativeAI) -> dict:
    """
    Extract lead information from the conversation.
    Uses LLM to identify name, email, and platform from messages.
    """
    messages = state.get("messages", [])
    lead_info = state.get("lead_info", {}).copy()
    
    # Get all user messages as context
    conversation_text = "\n".join([
        msg.content for msg in messages 
        if isinstance(msg, HumanMessage)
    ])
    
    # Check for email in the latest messages using regex first
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    emails = re.findall(email_pattern, conversation_text)
    if emails and not lead_info.get("email"):
        lead_info["email"] = emails[-1]  # Use most recent email
    
    # Use LLM to extract other info
    system_prompt = """You are extracting lead information from a conversation.
Extract the following if mentioned:
- name: The person's name
- platform: Their content platform (YouTube, Instagram, TikTok, etc.)

Respond in this exact format (use null if not found):
name: [extracted name or null]
platform: [extracted platform or null]

Only extract information that is clearly stated. Do not guess."""

    extraction_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Extract info from this conversation:\n{conversation_text}")
    ]
    
    response = llm.invoke(extraction_messages)
    response_text = response.content.strip()
    
    # Parse the response
    for line in response_text.split("\n"):
        if line.startswith("name:"):
            value = line.split(":", 1)[1].strip()
            if value.lower() != "null" and not lead_info.get("name"):
                lead_info["name"] = value
        elif line.startswith("platform:"):
            value = line.split(":", 1)[1].strip()
            if value.lower() != "null" and not lead_info.get("platform"):
                lead_info["platform"] = value
    
    return lead_info


def generate_lead_response(state: ConversationState, lead_info: dict, llm: ChatGoogleGenerativeAI) -> str:
    """
    Generate appropriate response for lead qualification.
    """
    has_name = lead_info.get("name")
    has_email = lead_info.get("email")
    has_platform = lead_info.get("platform")
    
    # If all info collected, confirm and capture
    if has_name and has_email and has_platform:
        # Call the mock lead capture
        capture_result = mock_lead_capture(
            name=lead_info["name"],
            email=lead_info["email"],
            platform=lead_info["platform"]
        )
        return f"""Thank you {lead_info['name']}! I've got all your information:
- Name: {lead_info['name']}
- Email: {lead_info['email']}
- Platform: {lead_info['platform']}

Our team will reach out to you shortly to help you get started with AutoStream Pro. You're going to love creating amazing content with our 4K editing and AI captions!"""

    # Ask for missing info
    missing = []
    if not has_name:
        missing.append("your name")
    if not has_email:
        missing.append("your email address")
    if not has_platform:
        missing.append("which platform you create content for (YouTube, Instagram, TikTok, etc.)")
    
    if len(missing) == 3:
        return """That's great to hear! I'd love to help you get started with AutoStream.
To set you up, could you please share:
1. Your name
2. Your email address
3. Which platform you create content for (YouTube, Instagram, TikTok, etc.)?"""
    elif len(missing) == 2:
        return f"Thanks! I just need a couple more things: {' and '.join(missing)}."
    else:
        return f"Almost there! Just need {missing[0]}."


def lead_node(state: ConversationState, llm: ChatGoogleGenerativeAI) -> dict:
    """
    LangGraph node for lead qualification and capture.
    """
    # Extract any new lead info from conversation
    lead_info = extract_lead_info(state, llm)
    
    # Check if all info is now available
    has_all = (
        lead_info.get("name") and 
        lead_info.get("email") and 
        lead_info.get("platform")
    )
    
    # Generate response
    response = generate_lead_response(state, lead_info, llm)
    
    return {
        "lead_info": lead_info,
        "lead_captured": has_all,
        "response": response
    }
