import sys
import os
import asyncio
import unittest
from unittest.mock import patch, MagicMock

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.workflows.main_workflow import build_main_workflow
from app.workflows.state import ComicState

class TestWorkflow(unittest.TestCase):
    """Tests for the main workflow"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_query = "climate change news"
        self.workflow = build_main_workflow()

    @patch('app.agents.collector_agent.collect_news')
    @patch('app.agents.scraper_agent.ScraperAgent.run')
    @patch('app.agents.individual_summarizer_agent.IndividualSummarizerAgent.run')
    @patch('app.agents.synthesis_summarizer_agent.SynthesisSummarizerAgent.run')
    @patch('app.agents.content_summarizer_agent.ContentSummarizerAgent.run')  # SentimentAnalyzer → ContentSummarizer
    @patch('app.agents.humorator_agent.HumoratorAgent.run')
    def test_workflow_execution(self, mock_humorator, mock_content, mock_synthesis,
                                mock_individual, mock_scraper, mock_collector):
        """Test the execution of the complete workflow"""
        # Setup mock returns for each agent
        mock_collector.return_value = {
            "news_urls": [
                "https://www.bbc.com/news/science-environment-12345678",  # 예시 BBC URL
                "https://www.reuters.com/business/environment/global-climate-report-released-98765432",
                # 예시 Reuters URL
                "https://www.nytimes.com/2025/04/29/climate/summit-update.html"  # 예시 NYT URL (미래 날짜지만 형식은 현실적)
            ],
            "selected_url": "https://www.bbc.com/news/science-environment-12345678"  # 위 목록 중 하나
        }

        mock_scraper.return_value = {
            "articles": [
                # 각 URL에서 가져왔을 법한 짧은 기사 내용 스니펫
                "London - A new UN report highlights accelerating sea level rise due to significant polar ice melt. Governments worldwide are urged to take immediate and decisive action.",
                "Geneva - The World Meteorological Organization (WMO) confirmed last year was the warmest on record globally, citing increased greenhouse gas emissions as the primary driver.",
                "New York - International delegates gathered today at the climate summit to discuss revised carbon emission targets, facing mounting pressure from environmental activists demanding stricter regulations and faster transitions."
            ]
        }

        mock_individual.return_value = {
            "summaries": [
                # 각 기사 내용에 대한 간결한 요약
                "UN report shows faster sea level rise from polar ice melt, demanding urgent government action.",
                "WMO confirms last year as warmest ever recorded due to increased greenhouse gas emissions.",
                "Delegates discuss revised carbon targets at climate summit amid activist pressure for stricter rules."
            ]
        }

        mock_synthesis.return_value = {
            "final_summary": "Recent reports confirm record global warming and accelerating sea level rise, driven primarily by greenhouse gas emissions from human activity. During international climate talks focused on revising carbon targets, governments face increasing pressure from activists for more urgent and stricter regulations to mitigate the crisis."
        }

        # 기존 SentimentAnalyzerAgent → ContentSummarizerAgent 반환값 수정
        mock_content.return_value = {
            "additional_context": {
                "summary": "Analysis of recent YouTube trends indicates growing public concern regarding extreme weather events linked to climate change. Popular videos discuss renewable energy solutions like solar and wind power, alongside debates on the effectiveness of current government policies.",
                "perspectives": [
                    # 시뮬레이션된 유튜브 댓글 또는 관련 관점
                    "We need massive investment in green technology now!",
                    "Individual actions matter, but systemic change is key.",
                    "Are carbon offset programs truly effective?",
                    "Politicians seem to be lagging behind the science."
                ]
            }
        }

        mock_humorator.return_value = {
            "humor_texts": [
                # 기후 변화 주제와 관련된 유머 시도 (테스트용)
                "My therapist told me to face my fears... so I bought a beachfront property. #ClimateChangeHumor",
                "Maybe we can negotiate with the weather? Offer it a tax break?",
                "Sea levels rising? My plan is simple: just learn to breathe underwater. Problem solved."
            ]
        }

        # Create initial state
        initial_state = ComicState(initial_query=self.test_query)
        initial_state_dict = initial_state.model_dump()

        # Run the workflow
        # 비동기 호출을 동기적으로 실행
        final_state = asyncio.run(self.workflow.ainvoke(initial_state_dict))

        # Assertions
        self.assertIsNotNone(final_state)

        # Check that each agent was called
        # mock_collector.assert_called_once()
        # mock_scraper.assert_called_once()
        # mock_individual.assert_called_once()
        # mock_synthesis.assert_called_once()
        # mock_content.assert_called_once()  # sentiment → content 이름 변경
        # mock_humorator.assert_called_once()

        # Check final state has all expected data
        self.assertEqual(final_state["initial_query"], self.test_query)
        self.assertEqual(final_state["news_urls"], ["https://example.com/news1", "https://example.com/news2"])
        self.assertEqual(final_state["articles"], ["Article 1 content", "Article 2 content"])
        self.assertEqual(final_state["summaries"], ["Summary of article 1", "Summary of article 2"])
        self.assertEqual(final_state["final_summary"], "Combined summary of all articles about climate change")
        self.assertIn("additional_context", final_state)  # public_sentiment → additional_context
        self.assertEqual(final_state["humor_texts"], [
            "Humor point 1 about climate change",
            "Humor point 2 about climate change"
        ])

    # @patch('app.agents.collector_agent.collect_news')
    # def test_workflow_with_error(self, mock_collector):
    #     """Test workflow when an error occurs in one of the agents"""
    #     # Setup collector agent to return an error
    #     mock_collector.return_value = {
    #         "error_message": "Failed to find news articles",
    #         "news_urls": []  # 명시적으로 빈 URL 목록 반환
    #     }
    #
    #     # Create initial state
    #     initial_state = ComicState(initial_query=self.test_query)
    #     initial_state_dict = initial_state.model_dump()
    #
    #     # Run the workflow
    #     # 비동기 호출을 동기적으로 실행
    #     final_state = asyncio.run(self.workflow.ainvoke(initial_state_dict))
    #
    #     # Assertions
    #     self.assertIsNotNone(final_state)
    #     self.assertEqual(final_state["initial_query"], self.test_query)
    #
    #     # 실제 동작을 기반으로 테스트 케이스 조정 (두 접근 방식 중 하나 선택)
    #     # 방식 1: 빈 URL 목록이 기대되는 경우 (권장 방식)
    #     self.assertEqual(final_state["news_urls"], [])
    #
    #     # 방식 2: 오류가 있어도 수집된 URL이 유지되는 경우
    #     # if "news_urls" in final_state and final_state["news_urls"]:
    #     #     print(f"Warning: news_urls not empty despite error: {final_state['news_urls']}")
    #
    #     self.assertIn("error_message", final_state)
    #     self.assertEqual(final_state["error_message"], "Failed to find news articles")
    #
    #     # Verify only collector was called
    #     mock_collector.assert_called_once()

if __name__ == "__main__":
    unittest.main()
