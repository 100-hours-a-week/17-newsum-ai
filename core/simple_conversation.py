"""
Simple conversation example using LangGraph.
"""
from typing import Dict, Any, List, Tuple
from langgraph.graph import StateGraph, END

# Define the state type for our conversation
# In a real application, you would use Pydantic models
State = Dict[str, Any]

# Define some node functions
def user_input(state: State) -> State:
    """Placeholder for user input node."""
    # In a real application, this would get input from the user
    user_message = input("User: ")
    state["messages"] = state.get("messages", []) + [{"role": "user", "content": user_message}]
    return state

def process_input(state: State) -> State:
    """Process the user input."""
    # In a real application, this might do NLP preprocessing
    messages = state["messages"]
    last_message = messages[-1]["content"]
    
    # Simple analysis
    state["input_analysis"] = {
        "length": len(last_message),
        "question": last_message.endswith("?"),
        "greeting": any(x in last_message.lower() for x in ["hello", "hi", "hey"])
    }
    
    return state

def generate_response(state: State) -> State:
    """Generate a response based on the conversation so far."""
    # In a real application, this would call an LLM API
    messages = state["messages"]
    analysis = state["input_analysis"]
    
    # Simple rule-based response
    if analysis["greeting"]:
        response = "Hello! How can I help you today?"
    elif analysis["question"]:
        response = "That's an interesting question. I'll do my best to answer."
    else:
        response = "I understand. Please tell me more."
    
    state["messages"] = messages + [{"role": "assistant", "content": response}]
    return state

def should_continue(state: State) -> str:
    """Check if the conversation should continue."""
    # In a real application, this might check for exit keywords
    last_message = state["messages"][-2]["content"]  # User's last message
    
    if last_message.lower() in ["exit", "quit", "bye", "goodbye"]:
        return "end"
    return "continue"

# Create the graph
def create_conversation_graph():
    """Create a simple conversation graph."""
    # Initialize the graph
    graph = StateGraph(State)
    
    # Add nodes
    graph.add_node("user_input", user_input)
    graph.add_node("process_input", process_input)
    graph.add_node("generate_response", generate_response)
    
    # Add edges
    graph.add_edge("user_input", "process_input")
    graph.add_edge("process_input", "generate_response")
    
    # Add conditional edge from generate_response
    graph.add_conditional_edges(
        "generate_response",
        should_continue,
        {
            "continue": "user_input",
            "end": END
        }
    )
    
    # Set the entry point
    graph.set_entry_point("user_input")
    
    # Compile the graph
    return graph.compile()

def main():
    """Run the conversation graph."""
    # Create the graph
    graph = create_conversation_graph()
    
    # Initialize the state
    state = {"messages": []}
    
    # Run the graph
    for output in graph.stream(state):
        if "messages" in output:
            messages = output["messages"]
            if messages and messages[-1]["role"] == "assistant":
                print(f"Assistant: {messages[-1]['content']}")

if __name__ == "__main__":
    print("Simple conversation example. Type 'exit' to quit.")
    main()
