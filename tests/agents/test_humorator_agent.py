import sys
import os
import asyncio
import unittest
from unittest.mock import patch, MagicMock

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.agents.humorator_agent import HumoratorAgent
from app.workflows.state import ComicState

class TestHumoratorAgent(unittest.TestCase):
    """Tests for the HumoratorAgent class"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent = HumoratorAgent()
        self.test_state = ComicState(
            final_summary="Global temperatures continue to rise as world leaders fail to reach consensus on climate action. Scientists warn of severe consequences if immediate steps are not taken.",
            public_sentiment={
                "sentiment": {"positive": 0.1, "negative": 0.7, "neutral": 0.2},
                "emotions": {"anger": 0.4, "sadness": 0.3, "joy": 0.1, "fear": 0.1, "surprise": 0.1}
            }
        )

    def test_initialization(self):
        """Test that the agent initializes correctly"""
        self.assertIsInstance(self.agent, HumoratorAgent)

    @patch('app.agents.humorator_agent.call_llm_api')
    def test_run_with_valid_data(self, mock_llm_api):
        """Test the agent's run method with valid data"""
        # Setup mock LLM response
        mock_llm_api.return_value = """- World leaders failing to reach consensus while the planet burns is like watching firefighters argue about water usage while a house is engulfed in flames.
- Scientists giving "severe consequences" warnings is the ultimate version of "I told you so" that nobody wants to hear.
- Politicians who promised climate action are proving that hot air isn't just a greenhouse gas problem."""

        # Run the agent
        result = asyncio.run(self.agent.run(self.test_state))

        # Assertions
        self.assertIsNotNone(result)
        self.assertIn("humor_texts", result)
        self.assertEqual(len(result["humor_texts"]), 3)
        self.assertIn("firefighters", result["humor_texts"][0])
        self.assertIsNone(result["error_message"])
        
        # Verify LLM was called with appropriate data
        mock_llm_api.assert_called_once()
        args, kwargs = mock_llm_api.call_args
        self.assertIn("News Summary", args[0])
        self.assertIn("Public Sentiment Analysis Results", args[0])
        self.assertIn("Global temperatures", args[0])
        self.assertEqual(kwargs["max_tokens"], 300)
        self.assertEqual(kwargs["temperature"], 0.6)

    @patch('app.agents.humorator_agent.call_llm_api')
    def test_run_with_none_response(self, mock_llm_api):
        """Test the agent's run method when LLM returns 'None'"""
        # Setup mock to return 'None'
        mock_llm_api.return_value = "None"

        # Run the agent
        result = asyncio.run(self.agent.run(self.test_state))

        # Assertions
        self.assertIsNotNone(result)
        self.assertIn("humor_texts", result)
        self.assertEqual(result["humor_texts"], [])
        self.assertIsNone(result["error_message"])
        
        # Verify LLM was called
        mock_llm_api.assert_called_once()

    def test_run_with_no_summary(self):
        """Test the agent's run method with no summary available"""
        # Create state with no summary
        empty_state = ComicState()
        
        # Run the agent
        result = asyncio.run(self.agent.run(empty_state))
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertIn("humor_texts", result)
        self.assertEqual(result["humor_texts"], [])
        self.assertIn("error_message", result)
        self.assertIsNotNone(result["error_message"])

    @patch('app.agents.humorator_agent.call_llm_api')
    def test_run_with_llm_exception(self, mock_llm_api):
        """Test the agent's run method when LLM API throws an exception"""
        # Setup mock to raise exception
        mock_llm_api.side_effect = Exception("API Error")

        # Run the agent
        result = asyncio.run(self.agent.run(self.test_state))

        # Assertions
        self.assertIsNotNone(result)
        self.assertIn("humor_texts", result)
        self.assertEqual(result["humor_texts"], [])
        self.assertIn("error_message", result)
        self.assertIn("Failed to generate humor points", result["error_message"])
        
        # Verify LLM was called
        mock_llm_api.assert_called_once()

if __name__ == "__main__":
    unittest.main()
