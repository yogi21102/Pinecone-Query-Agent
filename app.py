import os
import operator
import json
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from typing_extensions import TypedDict, Annotated

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.messages import AnyMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from dotenv import load_dotenv

load_dotenv()

# --- PRE-REQUISITE: Set Groq API Key ---
if not os.environ.get("GROQ_API_KEY"):
    print("Warning: GROQ_API_KEY environment variable not set.")

# --- Step 1: Define tools and model ---

# We define the structured data we want the LLM to extract.
class PineconeQuerySchema(BaseModel):
    """The structured representation of a Pinecone query."""
    query_text: str = Field(
        description="The core semantic meaning of the query, to be used for generating a vector embedding. E.g., for 'articles by Alice about machine learning', this would be 'machine learning'."
    )
    author: Optional[str] = Field(
        description="The author to filter by.",
        default=None
    )
    tags: Optional[List[str]] = Field(
        description="A list of tags to filter by. E.g., 'LLMs' or ['LLMs', 'Python']",
        default=None
    )
    published_year: Optional[int] = Field(
        description="The specific year to filter by.",
        default=None
    )
    published_month: Optional[int] = Field(
        description="The specific month (as an integer 1-12) to filter by.",
        default=None
    )
    published_day: Optional[int] = Field(
        description="The specific day (1-31) to filter by.",
        default=None
    )

# --- Initialize Model ---
try:
    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )
except Exception as e:
    print(f"Error initializing Groq model: {e}")
    print("Please ensure your GROQ_API_KEY is set correctly.")
    exit()

# --- Augment the LLM with our "tool" ---
# We convert the Pydantic schema to a "tool" format and bind it to the model.
# The LLM will now try to call this "tool" by filling out the schema.
schema_tool = convert_to_openai_tool(PineconeQuerySchema)
model_with_tools = model.bind_tools([schema_tool])


# --- Step 2: Define state ---

class AgentState(TypedDict):
    # The history of messages
    messages: Annotated[list[AnyMessage], operator.add]
    
    # Final, formatted output will be stored here
    query_text_for_embedding: Optional[str]
    pinecone_filter: Optional[Dict[str, Any]]


# --- Step 3: Define model node ---

# This is the prompt from the original script
system_template = """You are an expert at converting natural language (NL) queries into a structured Pinecone search query.
You must extract the vector search component and metadata filters by calling the `PineconeQuerySchema` tool.

The available metadata fields for filtering are:
- 'author' (string)
- 'tags' (list of strings)
- 'published_year' (int)
- 'published_month' (int, 1-12)
- 'published_day' (int, 1-31)

Today's date is: {current_date}.
Use this date to resolve relative date queries:
- 'last year' means the full previous calendar year (e.g., if today is 2025, 'last year' is 2024).
- 'this year' means the current calendar year.
- 'last month' means the full previous calendar month.
- 'yesterday' means the previous day.

Disambiguation Rules:
- 'query_text': This should be the *semantic topic* of the search. 
  - For "articles by Alice about machine learning", query_text is "machine learning".
  - For "anything by John Doe on vector search", query_text is "vector search".
- 'tags': Extract tags mentioned with 'tagged with' or similar phrasing.
- 'author': Extract author names.
- 'dates': Parse dates like "June, 2023" into month=6, year=2023.
"""

def llm_call(state: AgentState):
    """LLM node: extracts structured data by "calling" the Pydantic schema tool"""
    
    # 1. Inject the current date into the system prompt
    current_date = datetime.now().strftime("%B %d, %Y")
    system_message = SystemMessage(content=system_template.format(current_date=current_date))
    
    # 2. Create the message list
    messages_with_system_prompt = [system_message] + state["messages"]
    
    # 3. Invoke the model
    ai_message = model_with_tools.invoke(messages_with_system_prompt)
    
    return {"messages": [ai_message]}


# --- Step 4: Define "tool" node (formatter node) ---

def format_pinecone_filter_node(state: AgentState):
    """
    This node acts like the "tool" execution.
    It takes the structured data from the LLM's tool_call,
    formats it into a Pinecone filter, and updates the final state.
    """
    
    # 1. Get the last AI message
    last_message = state["messages"][-1]
    
    if not last_message.tool_calls:
        print("Error: Formatter node reached but no tool calls found.")
        return {}

    # 2. Extract the arguments from the tool call
    # We assume the LLM correctly calls our one-and-only tool
    tool_call = last_message.tool_calls[0]
    if tool_call['name'] != "PineconeQuerySchema":
         print(f"Error: LLM called unexpected tool: {tool_call['name']}")
         return {}
         
    schema_data = tool_call['args']
    
    # 3. Use the formatting logic from the original script
    filter_dict = {}
    
    if schema_data.get('author'):
        filter_dict["author"] = {"$eq": schema_data['author']}
    
    if schema_data.get('published_year'):
        filter_dict["published_year"] = {"$eq": schema_data['published_year']}
        
    if schema_data.get('published_month'):
        filter_dict["published_month"] = {"$eq": schema_data['published_month']}
        
    if schema_data.get('published_day'):
        filter_dict["published_day"] = {"$eq": schema_data['published_day']}
        
    if schema_data.get('tags'):
        filter_dict["tags"] = {"$in": schema_data['tags']}

    # 4. Return the final, formatted data to update the state
    return {
        "query_text_for_embedding": schema_data.get('query_text'),
        "pinecone_filter": filter_dict
    }


# --- Step 5: Define logic to determine whether to end ---

def should_continue(state: AgentState) -> Literal["tool_node", "_ _end_ _"]:
    """
    Decide if we should route to the formatter node or stop.
    """
    last_message = state["messages"][-1]

    # If the LLM makes a tool call, route to the formatter node
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise (e.g., LLM responded with text), we stop
    return END

# --- Step 6: Build agent ---

# Build workflow
agent_builder = StateGraph(AgentState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", format_pinecone_filter_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",
        END: END
    }
)
# The formatter node is the last step
agent_builder.add_edge("tool_node", END)

# Compile the agent
agent = agent_builder.compile()


# --- Main execution block to run the agent ---
if __name__ == "__main__":
    print("--- Groq Pinecone Query Agent (LangGraph) ---")
    
    try:
        
        print("Generating agent graph and saving to 'agent_graph.png'...")
        
        # 1. Get the raw PNG bytes from the graph
        png_bytes = agent.get_graph(xray=True).draw_mermaid_png()
        
        # 2. Write those bytes to a new file
        with open("agent_graph.png", "wb") as f:
            f.write(png_bytes)
            
        print("Successfully saved 'agent_graph.png'. Open the file to see the graph.")
        

    except Exception as e:
        print(f"Could not display or save graph: {e}. (This is non-essential).")

    # --- Test Queries ---
    sample_queries = [
        "Show me articles by Alice Zhang from last year about machine learning.",
        "Find posts tagged with 'LLMs' published in June, 2023.",
        "Anything by John Doe on vector search?",
        "I need finance articles from this year.",
        "What did 'Dr. Eva' write about 'quantum computing' in 2022?",
        "Search for posts by 'admin' tagged 'news' or 'updates'."
    ]
    
    print(f"\nToday's Date: {datetime.now().strftime('%B %d, %Y')}\n")
    
    for i, nl_query in enumerate(sample_queries):
        print(f"--- Query {i+1} ---")
        print(f"Natural Language: \"{nl_query}\"")
        
        # 1. Prepare the input
        messages_in = [HumanMessage(content=nl_query)]
        
        # 2. Invoke the agent
        # The .invoke() call will return the *final* state of the graph
        final_state = agent.invoke({"messages": messages_in})
        
        # 3. Print the formatted results from the final state
        print(f"Vector Search Text: \"{final_state['query_text_for_embedding']}\"")
        print(f"Pinecone Filter: {final_state['pinecone_filter']}")
        print(f"LLM Calls: {len(final_state['messages']) // 2}") # Each cycle is 1 Human + 1 AI
        print("-" * (12 + len(str(i+1))))
