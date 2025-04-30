import os
import sys
import asyncio
import json

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.workflows.main_workflow import build_main_workflow
from app.workflows.state import ComicState

async def test_end_to_end_workflow():
    """Run an end-to-end test of the workflow with a specific query"""
    print("Running end-to-end workflow test...")
    
    # Create test state with an initial query
    initial_state = ComicState(initial_query="Climate change impact")
    
    # Build the workflow
    workflow = build_main_workflow()
    
    # Execute the workflow
    try:
        print(f"Starting workflow with query: '{initial_state.initial_query}'")
        print("This may take a while, as it will run through all the steps including:")
        print("- Collecting news articles")
        print("- Scraping article content")
        print("- Summarizing articles")
        print("- Analyzing public sentiment")
        print("- Generating humor points")
        print("- Creating comic scenarios")
        print("\nRunning workflow...")
        
        final_state_dict = await workflow.ainvoke(initial_state.model_dump())
        
        # Print workflow results
        print("\n===== WORKFLOW RESULTS =====")
        
        # News URLs
        if 'news_urls' in final_state_dict and final_state_dict['news_urls']:
            print(f"\nCollected {len(final_state_dict['news_urls'])} news URLs:")
            for url in final_state_dict['news_urls'][:3]:  # Show first 3 only
                print(f"- {url}")
            if len(final_state_dict['news_urls']) > 3:
                print(f"- ...and {len(final_state_dict['news_urls']) - 3} more")
        else:
            print("\nNo news URLs collected.")
        
        # Final summary
        if 'final_summary' in final_state_dict and final_state_dict['final_summary']:
            print(f"\nFinal Summary:\n{final_state_dict['final_summary']}")
        else:
            print("\nNo final summary generated.")
        
        # Public sentiment
        if 'public_sentiment' in final_state_dict and final_state_dict['public_sentiment']:
            print("\nPublic Sentiment Analysis:")
            sentiment = final_state_dict['public_sentiment']['sentiment']
            emotions = final_state_dict['public_sentiment']['emotions']
            
            # Find dominant sentiment and emotion
            dominant_sentiment = max(sentiment.items(), key=lambda x: x[1])[0]
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
            print(f"Dominant sentiment: {dominant_sentiment} ({sentiment[dominant_sentiment]:.2f})")
            print(f"Dominant emotion: {dominant_emotion} ({emotions[dominant_emotion]:.2f})")
        else:
            print("\nNo public sentiment analysis.")
        
        # Humor texts
        if 'humor_texts' in final_state_dict and final_state_dict['humor_texts']:
            print(f"\nGenerated {len(final_state_dict['humor_texts'])} humor points:")
            for i, humor in enumerate(final_state_dict['humor_texts'], 1):
                print(f"{i}. {humor}")
        else:
            print("\nNo humor points generated.")
        
        # Scenarios (if implemented)
        if 'scenarios' in final_state_dict and final_state_dict['scenarios']:
            print(f"\nGenerated {len(final_state_dict['scenarios'])} panels:")
            for i, panel in enumerate(final_state_dict['scenarios'], 1):
                print(f"\nPanel {i}:")
                print(f"Scene: {panel['description']}")
                print(f"Dialogue: {panel['dialogue']}")
        
        # Check for errors
        if 'error_message' in final_state_dict and final_state_dict['error_message']:
            print(f"\nError in workflow: {final_state_dict['error_message']}")
        
        print("\n===== END OF RESULTS =====")
        
    except Exception as e:
        print(f"\nError running end-to-end test: {e}")

if __name__ == "__main__":
    # Run the test
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_end_to_end_workflow())
