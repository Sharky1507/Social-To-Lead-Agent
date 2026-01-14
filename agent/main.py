"""
CLI entry point for the AutoStream AI Agent.
"""
import os
from dotenv import load_dotenv
from agent.graph import create_agent, run_conversation
from agent.state import ConversationState


def main():
    """Run the AutoStream agent in CLI mode."""
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please create a .env file with your API key:")
        print("  GOOGLE_API_KEY=your_key_here")
        return
    
    # Create the agent
    print("Initializing AutoStream AI Agent...")
    agent = create_agent(api_key)
    
    # Initialize state
    state: ConversationState = {
        "messages": [],
        "intent": "unknown",
        "lead_info": {},
        "lead_captured": False,
        "response": ""
    }
    
    print("\n" + "="*50)
    print("Welcome to AutoStream!")
    print("I'm your AI assistant. Ask me about our video")
    print("editing tools, pricing, or get started today!")
    print("Type 'quit' to exit.")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThanks for chatting with AutoStream! Goodbye!")
                break
            
            # Run the conversation
            state, response = run_conversation(agent, state, user_input)
            print(f"\nAutoStream: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
