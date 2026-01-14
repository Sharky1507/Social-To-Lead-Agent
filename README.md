# AutoStream Social-to-Lead AI Agent

A conversational AI agent for AutoStream, a SaaS product providing automated video editing tools for content creators.

## Features

- **Intent Classification**: Identifies user intent (greeting, inquiry, high-intent lead)
- **RAG-Powered Knowledge Retrieval**: Answers questions from a local knowledge base
- **Lead Capture**: Collects name, email, and platform for qualified leads
- **State Management**: Maintains context across 5-6 conversation turns

## Tech Stack

- **Python 3.9+**
- **LangGraph** - Workflow orchestration
- **LangChain** - LLM integration
- **Gemini 2.5 Flash** - Language model
- **Streamlit** - Web interface

## Setup

1. **Create virtual environment:**
   ```bash
   uv venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

2. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Configure API key:**
   ```bash
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

## Usage

### Streamlit Web Interface
```bash
.venv\Scripts\activate
streamlit run app.py
```

### CLI Mode
```bash
.venv\Scripts\activate
python -m agent.main
```

## Project Structure

```
├── agent/
│   ├── __init__.py
│   ├── main.py          # CLI entry point
│   ├── graph.py         # LangGraph workflow
│   ├── state.py         # Conversation state schema
│   ├── nodes/
│   │   ├── intent.py    # Intent classification
│   │   ├── rag.py       # Knowledge retrieval
│   │   └── lead.py      # Lead qualification
│   └── tools/
│       └── lead_capture.py
├── knowledge/
│   └── autostream_kb.json
├── app.py               # Streamlit interface
├── requirements.txt
└── .env.example
```

## Conversation Flow

1. **Greeting** → Agent responds warmly
2. **Product/Pricing Inquiry** → RAG retrieves from knowledge base
3. **High-Intent Detection** → Agent asks for name, email, platform
4. **Lead Capture** → Calls `mock_lead_capture()` with collected info

## License

MIT
