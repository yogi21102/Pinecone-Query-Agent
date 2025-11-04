## Groq Pinecone Query Agent (LangGraph)

This repository demonstrates a small agent that converts natural-language queries into a structured Pinecone search query using Groq (via `langchain_groq`) and LangGraph to manage the agent state flow.

The core script is `app.py` which:
- Defines a Pydantic schema (`PineconeQuerySchema`) representing the structured query the LLM should return.
- Wraps a Groq chat model (`ChatGroq`) and converts the Pydantic schema into an OpenAI-style tool for the model to "call".
- Uses `langgraph`'s `StateGraph` to construct a two-node workflow:
  - `llm_call`: asks the model to fill the `PineconeQuerySchema` using the system prompt.
  - `tool_node`: formats the model's tool-call output into a Pinecone filter dict and final query text.
- Saves a generated agent graph image as `agent_graph.png` when run.

Why this project
- Shows how to convert free-form user queries into structured search inputs suitable for vector search (e.g., Pinecone).
- Demonstrates using function-calling / tool calling with Pydantic schemas.
- Uses LangGraph to compose and run a simple stateful agent loop.

Prerequisites
- Python 3.10+ (this project was created with Python 3.12 in a venv layout).
- A Groq API key set in the environment variable `GROQ_API_KEY`.

Quickstart

1. Create and activate a virtual environment (optional but recommended):

```bash
python3 -m venv myenv
source myenv/bin/activate
```

2. Install dependencies (project provides `req.txt`):

```bash
python -m pip install -r req.txt
```

3. Set your Groq API key (macOS / Linux / zsh):

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

You can also place environment variables in a `.env` file at the project root and the script will load it via `python-dotenv`.

4. Run the agent script:

```bash
python app.py
```

What to expect
- The script will attempt to initialize the Groq model. If `GROQ_API_KEY` is missing, you'll see a warning and the script may fail.
- The script will generate and save `agent_graph.png` (if graph rendering is available in your environment).
- It will run several sample queries and print the resulting `query_text_for_embedding` and `pinecone_filter` values.

Files of interest
- `app.py` — main agent script.
- `req.txt` — dependency list used for pip install.
- `agent_graph.png` — generated when running `app.py` (if available).

Notes & Troubleshooting
- If the Groq model fails to initialize, confirm `GROQ_API_KEY` is set and valid.
- The script depends on `langchain_groq`, `langgraph`, and `langchain` packages. Versions may matter; if you run into compatibility issues, try using the same Python version referenced in the venv layout (Python 3.12) or pin package versions.
- Rendering `agent_graph.png` requires langgraph's graph rendering helpers and a working image backend; the script catches failures when saving the graph and will continue to run the sample queries.

