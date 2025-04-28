"""
Base graph implementation for LangGraph project.
"""
from typing import Dict, Any, Tuple, List
from langgraph.graph import StateGraph, END

class BaseGraph:
    """Base graph class for LangGraph applications."""
    
    def __init__(self):
        """Initialize the base graph."""
        self.graph = StateGraph(nodes={"state": Dict})
        
    def add_node(self, name: str, function: callable):
        """Add a node to the graph.
        
        Args:
            name: Name of the node
            function: Function to execute at this node
        """
        self.graph.add_node(name, function)
        
    def add_edge(self, start: str, end: str):
        """Add an edge between nodes.
        
        Args:
            start: Starting node
            end: Ending node
        """
        self.graph.add_edge(start, end)
        
    def add_conditional_edge(self, start: str, condition: callable, 
                          destinations: Dict[str, str]):
        """Add a conditional edge based on a condition function.
        
        Args:
            start: Starting node
            condition: Function that determines the next node
            destinations: Mapping of condition results to destination nodes
        """
        self.graph.add_conditional_edges(start, condition, destinations)
        
    def set_entry_point(self, node: str):
        """Set the entry point for the graph.
        
        Args:
            node: Name of the entry node
        """
        self.graph.set_entry_point(node)
        
    def set_finish_point(self, node: str):
        """Set a node as a finish point that ends execution.
        
        Args:
            node: Name of the node to mark as a finish point
        """
        self.graph.add_edge(node, END)
        
    def compile(self):
        """Compile the graph into an executable form."""
        return self.graph.compile()
