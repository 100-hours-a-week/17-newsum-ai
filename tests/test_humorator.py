import os
import sys
import asyncio
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.agents.humorator_agent import HumoratorAgent
from app.workflows.state import ComicState

async def test_humorator():
    """Manually test the HumoratorAgent with sample data"""
    print("Testing HumoratorAgent with sample data...")
    
    # Create a test state with realistic data
    state = ComicState(
        final_summary=(
            "Global temperatures continue to rise as world leaders debate climate policies. "
            "The latest UN report warns of severe consequences if immediate action is not taken. "
            "Meanwhile, fossil fuel companies report record profits amid energy crisis."
        ),
        public_sentiment={
            "sentiment": {"positive": 0.1, "negative": 0.7, "neutral": 0.2},
            "emotions": {"anger": 0.4, "sadness": 0.3, "joy": 0.1, "fear": 0.1, "surprise": 0.1}
        }
    )
    
    # Initialize the agent
    agent = HumoratorAgent()
    
    # Run the agent
    try:
        print("Generating humor points...")
        result = await agent.run(state)
        
        # Print results
        print("\nResults:")
        if 'humor_texts' in result and result['humor_texts']:
            print(f"Generated {len(result['humor_texts'])} humor points:")
            for i, humor_point in enumerate(result['humor_texts'], 1):
                print(f"{i}. {humor_point}")
        else:
            print("No humor points generated.")
            if 'error_message' in result and result['error_message']:
                print(f"Error: {result['error_message']}")
    
    except Exception as e:
        print(f"Error testing humorator: {e}")

async def test_humorator_with_different_emotions():
    """Test the HumoratorAgent with different dominant emotions"""
    print("\nTesting HumoratorAgent with different emotional contexts...")
    
    # Test scenarios with different emotions
    emotion_scenarios = [
        {"main": "anger", "data": {"anger": 0.6, "sadness": 0.2, "joy": 0.1, "fear": 0.05, "surprise": 0.05}},
        {"main": "sadness", "data": {"anger": 0.1, "sadness": 0.7, "joy": 0.05, "fear": 0.1, "surprise": 0.05}},
        {"main": "fear", "data": {"anger": 0.1, "sadness": 0.2, "joy": 0.05, "fear": 0.6, "surprise": 0.05}}
    ]
    
    agent = HumoratorAgent()
    base_summary = "Climate change is accelerating with measurable impacts worldwide. Scientists warn of increasing natural disasters."
    
    for scenario in emotion_scenarios:
        print(f"\nTesting with dominant emotion: {scenario['main']}")
        
        state = ComicState(
            final_summary=base_summary,
            public_sentiment={
                "sentiment": {"positive": 0.1, "negative": 0.7, "neutral": 0.2},
                "emotions": scenario["data"]
            }
        )
        
        try:
            result = await agent.run(state)
            
            print(f"Generated {len(result.get('humor_texts', []))} humor points:")
            for i, humor_point in enumerate(result.get('humor_texts', []), 1):
                print(f"{i}. {humor_point}")
                
        except Exception as e:
            print(f"Error with {scenario['main']} scenario: {e}")

if __name__ == "__main__":
    # Run the tests
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_humorator())
    loop.run_until_complete(test_humorator_with_different_emotions())
