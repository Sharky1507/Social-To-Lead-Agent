"""
Mock lead capture tool for the AutoStream AI Agent.
"""


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Capture a qualified lead's information.
    
    Args:
        name: The lead's full name
        email: The lead's email address
        platform: The creator platform (YouTube, Instagram, etc.)
    
    Returns:
        Confirmation message
    """
    print(f"Lead captured successfully: {name}, {email}, {platform}")
    return f"Lead captured successfully: {name}, {email}, {platform}"
