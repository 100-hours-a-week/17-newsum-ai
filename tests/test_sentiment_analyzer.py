import os
import sys
import asyncio
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.agents.sentiment_analyzer_agent import SentimentAnalyzerAgent
from app.workflows.state import ComicState

async def test_sentiment_analyzer():
    """Manually test the SentimentAnalyzerAgent with a real query"""
    print("Testing SentimentAnalyzerAgent with a real query...")
    
    # Create a test state
    state = ComicState(
        initial_query="climate change",
        final_summary="Global temperatures continue to rise as world leaders debate climate policies. The latest UN report warns of severe consequences if immediate action is not taken."
    )
    
    # Initialize the agent
    agent = SentimentAnalyzerAgent()
    
    # Run the agent
    try:
        print("Running sentiment analysis...")
        result = await agent.run(state)
        
        # Print results
        print("\nResults:")
        print(f"Success: {'public_sentiment' in result and result['public_sentiment'] is not None}")
        
        if 'public_sentiment' in result and result['public_sentiment']:
            print("\nSentiment distribution:")
            for sentiment, value in result['public_sentiment']['sentiment'].items():
                print(f"  {sentiment}: {value:.2f}")
                
            print("\nEmotion distribution:")
            for emotion, value in result['public_sentiment']['emotions'].items():
                print(f"  {emotion}: {value:.2f}")
                
            # Identify dominant sentiment and emotion
            dominant_sentiment = max(result['public_sentiment']['sentiment'].items(), key=lambda x: x[1])[0]
            dominant_emotion = max(result['public_sentiment']['emotions'].items(), key=lambda x: x[1])[0]
            
            print(f"\nDominant sentiment: {dominant_sentiment}")
            print(f"Dominant emotion: {dominant_emotion}")
        else:
            print("\nNo sentiment results returned.")
            if 'error_message' in result:
                print(f"Error: {result['error_message']}")
    
    except Exception as e:
        print(f"Error running sentiment analyzer: {e}")

async def test_sentiment_analyzer_with_keywords():
    """Test the keyword-based sentiment analysis directly"""
    print("\nTesting keyword-based sentiment analysis...")
    
    # Sample comments
    comments = [
        "This is really concerning news about climate change.",
        "I'm angry that politicians are not taking this seriously!",
        "We need to act now before it's too late.",
        "I'm worried about the future of our planet.",
        "This makes me so sad to see what's happening.",
    ]
    
    # Initialize the agent
    agent = SentimentAnalyzerAgent()
    
    # Run the keyword sentiment analysis
    try:
        result = await agent._simple_keyword_sentiment(comments)
        
        print("\nKeyword Analysis Results:")
        print(json.dumps(result, indent=2))
    
    except Exception as e:
        print(f"Error in keyword sentiment analysis: {e}")

if __name__ == "__main__":
    # Run the tests
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_sentiment_analyzer())
    loop.run_until_complete(test_sentiment_analyzer_with_keywords())
