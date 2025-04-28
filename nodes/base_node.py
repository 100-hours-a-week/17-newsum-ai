"""
Base node implementation for LangGraph project.
"""
from typing import Dict, Any, Optional


class BaseNode:
    """Base class for all nodes in the graph."""
    
    def __init__(self, name: str, description: str = ""):
        """Initialize a base node.
        
        Args:
            name: Name of the node
            description: Description of the node's purpose
        """
        self.name = name
        self.description = description
        
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the node's operation.
        
        Args:
            state: Current state of the graph
            
        Returns:
            Updated state after node execution
        """
        # This should be implemented by child classes
        raise NotImplementedError("Subclasses must implement __call__")
    
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """Validate that the input state has required keys.
        
        Args:
            state: Current state to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Default implementation accepts any state
        return True


class LLMNode(BaseNode):
    """Node that interacts with a language model."""
    
    def __init__(self, name: str, prompt_template: str, 
                 output_key: str, description: str = ""):
        """Initialize an LLM node.
        
        Args:
            name: Name of the node
            prompt_template: Template for the prompt to send to the LLM
            output_key: Key to store the LLM response in the state
            description: Description of the node's purpose
        """
        super().__init__(name, description)
        self.prompt_template = prompt_template
        self.output_key = output_key
        
    def format_prompt(self, state: Dict[str, Any]) -> str:
        """Format the prompt template with values from the state.
        
        Args:
            state: Current state with values to format the prompt
            
        Returns:
            Formatted prompt string
        """
        try:
            return self.prompt_template.format(**state)
        except KeyError as e:
            missing_key = str(e).strip("'")
            raise ValueError(f"Missing required key '{missing_key}' in state for prompt formatting")

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the LLM operation.
        
        Args:
            state: Current state of the graph
            
        Returns:
            Updated state with LLM response
        """
        # This is a placeholder - in a real implementation, this would call an LLM API
        prompt = self.format_prompt(state)
        
        # Here you would make an API call to an LLM
        response = f"Placeholder response for prompt: {prompt[:50]}..."
        
        # Update the state with the LLM response
        state[self.output_key] = response
        return state
