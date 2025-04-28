# LangGraph Project for 17-team-4cut

This project implements a LangGraph-based application for the 17-team-4cut project.

## Project Structure

```
langgraph/
├── __init__.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── graphs/
│   ├── __init__.py
│   └── base_graph.py
├── nodes/
│   ├── __init__.py
│   └── base_node.py
├── utils/
│   ├── __init__.py
│   └── helpers.py
└── examples/
    ├── __init__.py
    └── simple_conversation.py
```

## Getting Started

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the example:
   ```
   python -m langgraph.examples.simple_conversation
   ```

## Development

### Creating a New Graph

To create a new LangGraph application, extend the `BaseGraph` class in the `graphs` directory:

```python
from langgraph.graphs.base_graph import BaseGraph

class MyCustomGraph(BaseGraph):
    def __init__(self):
        super().__init__()
        # Add custom initialization here
```

### Creating New Nodes

To create new node types, extend the `BaseNode` or `LLMNode` classes in the `nodes` directory:

```python
from langgraph.nodes.base_node import BaseNode

class MyCustomNode(BaseNode):
    def __call__(self, state):
        # Implement custom node logic
        return updated_state
```
