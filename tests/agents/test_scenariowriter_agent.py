import sys
import os
import asyncio
import unittest
from unittest.mock import patch, MagicMock

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from app.agents.scenariowriter_agent import ScenarioWriterAgent
from app.workflows.state import ComicState

class TestScenarioWriterAgent(unittest.TestCase):
    """Tests for the ScenarioWriterAgent class"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent = ScenarioWriterAgent()
        self.test_state = ComicState(
            final_summary="Global temperatures continue to rise as world leaders fail to reach consensus on climate action. Scientists warn of severe consequences if immediate steps are not taken.",
            humor_texts=[
                "World leaders failing to reach consensus while the planet burns is like watching firefighters argue about water usage while a house is engulfed in flames.",
                "Scientists giving 'severe consequences' warnings is the ultimate version of 'I told you so' that nobody wants to hear.",
                "Politicians who promised climate action are proving that hot air isn't just a greenhouse gas problem."
            ],
            public_sentiment={
                "sentiment": {"positive": 0.1, "negative": 0.7, "neutral": 0.2},
                "emotions": {"anger": 0.4, "sadness": 0.3, "joy": 0.1, "fear": 0.1, "surprise": 0.1}
            }
        )

    def test_initialization(self):
        """Test that the agent initializes correctly"""
        self.assertIsInstance(self.agent, ScenarioWriterAgent)

    @patch('app.agents.scenariowriter_agent.call_llm_api')
    def test_run_with_valid_data(self, mock_llm_api):
        """Test the agent's run method with valid data"""
        # Setup mock LLM response
        mock_llm_api.return_value = """[Panel 1]
Scene: A conference room filled with world leaders arguing heatedly, papers flying. A small globe in the center of the table is on fire, but no one is paying attention to it.
Dialogue: "We can't agree to reduce emissions by 50%! The economic impact would be..."

[Panel 2]
Scene: Outside the conference building, a scientist in a lab coat is monitoring a thermometer that's about to burst. Sweat is dripping from the scientist's forehead.
Dialogue: "Um, excuse me? The data shows we're reaching critical temperatures..."

[Panel 3]
Scene: Back in the conference room, the leaders continue arguing while the small globe is now engulfed in flames. A firefighter has entered but looks confused as the leaders block the way.
Dialogue: Firefighter: "Excuse me, I need to put out that fire!" Leader: "Not until we finalize the water usage protocol!"

[Panel 4]
Scene: The scientist outside the building is now joined by a crowd of concerned citizens. They all look up at the conference building which has smoke coming out of the windows.
Dialogue: Scientist to crowd: "On the bright side, at least all that hot air from politicians is finally being recognized as a greenhouse gas."
"""

        # Run the agent
        result = asyncio.run(self.agent.run(self.test_state))

        # Assertions
        self.assertIsNotNone(result)
        self.assertIn("scenarios", result)
        self.assertEqual(len(result["scenarios"]), 4)
        
        # Check structure of scenarios
        for i, scenario in enumerate(result["scenarios"]):
            self.assertIn("description", scenario)
            self.assertIn("dialogue", scenario)
            if i == 0:
                self.assertIn("conference room", scenario["description"].lower())
            
        self.assertIsNone(result["error_message"])
        
        # Verify LLM was called with appropriate data
        mock_llm_api.assert_called_once()
        args, kwargs = mock_llm_api.call_args
        self.assertIn("News Summary", args[0])
        self.assertIn("Humor Points", args[0])
        self.assertIn("Public Sentiment Analysis", args[0])
        self.assertEqual(kwargs["max_tokens"], 800)
        self.assertEqual(kwargs["temperature"], 0.7)

    def test_run_with_no_summary(self):
        """Test the agent's run method with no summary available"""
        # Create state with no summary
        empty_state = ComicState()
        
        # Run the agent
        result = asyncio.run(self.agent.run(empty_state))
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertIn("scenarios", result)
        self.assertEqual(len(result["scenarios"]), 0)

    @patch('app.agents.scenariowriter_agent.call_llm_api')
    def test_run_with_malformed_response(self, mock_llm_api):
        """Test the agent's run method with malformed LLM response"""
        # Setup mock to return malformed response
        mock_llm_api.return_value = """This is not properly formatted.
It doesn't have the panel structure.
Just some random text."""

        # Run the agent
        result = asyncio.run(self.agent.run(self.test_state))

        # Assertions
        self.assertIsNotNone(result)
        self.assertIn("scenarios", result)
        # Even with malformed input, it should ensure 4 panels
        self.assertEqual(len(result["scenarios"]), 4)
        
        # Verify each panel has necessary fields
        for scenario in result["scenarios"]:
            self.assertIn("description", scenario)
            self.assertIn("dialogue", scenario)
        
        # Verify LLM was called
        mock_llm_api.assert_called_once()

    @patch('app.agents.scenariowriter_agent.call_llm_api')
    def test_run_with_llm_exception(self, mock_llm_api):
        """Test the agent's run method when LLM API throws an exception"""
        # Setup mock to raise exception
        mock_llm_api.side_effect = Exception("API Error")

        # Run the agent
        result = asyncio.run(self.agent.run(self.test_state))

        # Assertions
        self.assertIsNotNone(result)
        self.assertIn("scenarios", result)
        self.assertEqual(len(result["scenarios"]), 4)  # Should return basic scenarios
        self.assertIn("error_message", result)
        self.assertIn("Failed to generate scenarios", result["error_message"])
        
        # Verify LLM was called
        mock_llm_api.assert_called_once()

if __name__ == "__main__":
    unittest.main()
