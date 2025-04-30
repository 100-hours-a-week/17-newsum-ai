import os
import sys
import asyncio
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.agents.scenariowriter_agent import ScenarioWriterAgent
from app.workflows.state import ComicState

async def test_scenariowriter():
    """Manually test the ScenarioWriterAgent with sample data"""
    print("Testing ScenarioWriterAgent with sample data...")
    
    # Create a test state with realistic data
    state = ComicState(
        final_summary=(
            "Global temperatures continue to rise as world leaders debate climate policies. "
            "The latest UN report warns of severe consequences if immediate action is not taken. "
            "Meanwhile, fossil fuel companies report record profits amid energy crisis."
        ),
        humor_texts=[
            "World leaders debating climate policies while the planet burns is like firefighters arguing about water usage protocols during a five-alarm fire.",
            "Fossil fuel companies making record profits during a climate crisis is like selling portable fans on a sinking Titanic.",
            "The UN issuing 'severe warnings' about climate change is becoming the planetary equivalent of 'This is your last warning!' that parents give their kids... 37 times in a row."
        ],
        public_sentiment={
            "sentiment": {"positive": 0.1, "negative": 0.7, "neutral": 0.2},
            "emotions": {"anger": 0.4, "sadness": 0.3, "joy": 0.1, "fear": 0.1, "surprise": 0.1}
        }
    )
    
    # Initialize the agent
    agent = ScenarioWriterAgent()
    
    # Run the agent
    try:
        print("Generating comic scenario...")
        result = await agent.run(state)
        
        # Print results
        print("\nResults:")
        if 'scenarios' in result and result['scenarios']:
            print(f"Generated {len(result['scenarios'])} panels:")
            for i, panel in enumerate(result['scenarios'], 1):
                print(f"\nPanel {i}:")
                print(f"Scene: {panel['description']}")
                print(f"Dialogue: {panel['dialogue']}")
        else:
            print("No scenario generated.")
            if 'error_message' in result and result['error_message']:
                print(f"Error: {result['error_message']}")
    
    except Exception as e:
        print(f"Error testing scenariowriter: {e}")

async def test_scenariowriter_minimal_data():
    """Test the ScenarioWriterAgent with minimal data"""
    print("\nTesting ScenarioWriterAgent with minimal data...")
    
    # Create a test state with only the required fields
    state = ComicState(
        final_summary="Climate change impacts are accelerating worldwide."
        # No humor_texts or public_sentiment
    )
    
    # Initialize the agent
    agent = ScenarioWriterAgent()
    
    # Run the agent
    try:
        result = await agent.run(state)
        
        # Print results
        print("\nResults with minimal data:")
        if 'scenarios' in result and result['scenarios']:
            print(f"Generated {len(result['scenarios'])} panels:")
            for i, panel in enumerate(result['scenarios'], 1):
                print(f"\nPanel {i}:")
                print(f"Scene: {panel['description']}")
                print(f"Dialogue: {panel['dialogue']}")
        else:
            print("No scenario generated.")
            if 'error_message' in result and result['error_message']:
                print(f"Error: {result['error_message']}")
    
    except Exception as e:
        print(f"Error with minimal data test: {e}")

if __name__ == "__main__":
    # Run the tests
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_scenariowriter())
    loop.run_until_complete(test_scenariowriter_minimal_data())
