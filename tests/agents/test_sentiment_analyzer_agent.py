import sys
import os
import asyncio
import unittest
from unittest.mock import patch, MagicMock

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.agents.sentiment_analyzer_agent import SentimentAnalyzerAgent
from app.workflows.state import ComicState

class TestSentimentAnalyzerAgent(unittest.TestCase):
    """Tests for the SentimentAnalyzerAgent class"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent = SentimentAnalyzerAgent()
        self.test_state = ComicState(
            initial_query="climate change",
            final_summary="Global temperatures continue to rise as world leaders fail to reach consensus on climate action. Scientists warn of severe consequences if immediate steps are not taken."
        )

    def test_initialization(self):
        """Test that the agent initializes correctly"""
        self.assertIsInstance(self.agent, SentimentAnalyzerAgent)

    @patch('app.agents.sentiment_analyzer_agent.SentimentAnalyzerAgent._fetch_youtube_videos')
    @patch('app.agents.sentiment_analyzer_agent.SentimentAnalyzerAgent._fetch_video_comments')
    @patch('app.agents.sentiment_analyzer_agent.SentimentAnalyzerAgent._simple_keyword_sentiment')
    def test_run_with_no_api_key(self, mock_sentiment, mock_comments, mock_videos):
        """Test the agent's run method when no API key is available"""
        # Setup mocks
        mock_videos.return_value = ["video1", "video2"]
        mock_comments.return_value = ["Comment 1", "Comment 2", "Comment 3"]
        mock_sentiment.return_value = {
            "sentiment": {"positive": 0.2, "negative": 0.5, "neutral": 0.3},
            "emotions": {"anger": 0.4, "sadness": 0.3, "joy": 0.1, "fear": 0.1, "surprise": 0.1}
        }

        # Run the agent
        result = asyncio.run(self.agent.run(self.test_state))

        # Assertions
        self.assertIsNotNone(result)
        self.assertIn("public_sentiment", result)
        self.assertIsNotNone(result["public_sentiment"])
        self.assertIn("sentiment", result["public_sentiment"])
        self.assertIn("emotions", result["public_sentiment"])
        
        # Verify correct method calls
        mock_videos.assert_called_once()
        mock_comments.assert_called()
        mock_sentiment.assert_called_once()

    @patch('app.agents.sentiment_analyzer_agent.SentimentAnalyzerAgent._fetch_youtube_videos')
    def test_run_with_no_videos(self, mock_videos):
        """Test the agent's run method when no videos are found"""
        # Setup mock to return empty list
        mock_videos.return_value = []

        # Run the agent
        result = asyncio.run(self.agent.run(self.test_state))

        # Assertions
        self.assertIsNotNone(result)
        self.assertIn("public_sentiment", result)
        self.assertIsNone(result["public_sentiment"])
        
        # Verify method call
        mock_videos.assert_called_once()

    @patch('app.agents.sentiment_analyzer_agent.SentimentAnalyzerAgent._simple_keyword_sentiment')
    def test_keyword_sentiment_analysis(self, mock_simple_sentiment):
        """Test the keyword-based sentiment analysis"""
        # Setup mock
        mock_sentiment_result = {
            "sentiment": {"positive": 0.3, "negative": 0.4, "neutral": 0.3},
            "emotions": {"anger": 0.2, "sadness": 0.3, "joy": 0.2, "fear": 0.2, "surprise": 0.1}
        }
        mock_simple_sentiment.return_value = mock_sentiment_result
        
        # Test comments
        test_comments = [
            "This is terrible news about climate change!",
            "I'm very concerned about our future.",
            "The government needs to act now!"
        ]
        
        # Call the method directly
        result = asyncio.run(self.agent._simple_keyword_sentiment(test_comments))
        
        # Assertions
        self.assertEqual(result, mock_sentiment_result)
        mock_simple_sentiment.assert_called_once_with(test_comments)

if __name__ == "__main__":
    unittest.main()
