# app/workflows/state.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ComicState(BaseModel):
    """
    Defines the state passed between nodes in the LangGraph workflow.
    Based on the specification document and README.md.
    """
    # Core IDs and Metadata
    comic_id: Optional[str] = Field(default=None, description="Unique ID for the comic generation task.")
    trace_id: Optional[str] = Field(default=None, description="Trace ID for observability (e.g., LangSmith).")
    timestamp: Optional[str] = Field(default=None, description="ISO 8601 timestamp of workflow initiation.")
    initial_query: Optional[str] = Field(default=None, description="The user's initial query.")

    # Configuration
    config: Dict[str, Any] = Field(default_factory=dict, description="Common configuration for the workflow (e.g., model names, feature flags).")

    # Topic Analysis (Node 02 output)
    topic_analysis: Dict[str, Any] = Field(default_factory=dict, description="Structured analysis of the initial query (main topic, entities, keywords).")
    search_keywords: List[str] = Field(default_factory=list, description="List of keywords for information gathering.")

    # Collection URLs (Node 03 & 04 output)
    fact_urls: List[Dict[str, str]] = Field(default_factory=list, description="List of collected news article URLs [{'url': str, 'source': str, 'search_keyword': str}].")
    opinion_urls: List[Dict[str, str]] = Field(default_factory=list, description="List of collected opinion URLs [{'url': str, 'source': str, 'search_keyword': str}].")

    # Scraped Content (Node 05 & 06 output)
    articles: List[Dict[str, Any]] = Field(default_factory=list, description="List of scraped news article content [{'url': str, 'title': str, 'text': str, ...}].")
    opinions_raw: List[Dict[str, Any]] = Field(default_factory=list, description="List of raw scraped opinion content [{'url': str, 'text': str, 'author': str, ...}].")

    # Filtered & Clustered Opinions (Node 07 output)
    opinions_clean: List[Dict[str, Any]] = Field(default_factory=list, description="Filtered, deduplicated, and clustered opinions with added 'cluster_id', 'is_representative'.")

    # Summaries (Node 08, 09, 10 output)
    news_summaries: List[Dict[str, Any]] = Field(default_factory=list, description="List of news summaries with FEQA scores [{'original_url': str, 'summary_text': str, 'feqa_score': float}].")
    opinion_summaries: Dict[str, Any] = Field(default_factory=dict, description="Opinion summary including stance clusters and sentiment distribution {'summary_text': str, ...}.")
    final_summary: Optional[str] = Field(default=None, description="Synthesized final summary combining news and opinions.")

    # Evaluation & Trend (Node 11 & 12 output)
    evaluation_metrics: Dict[str, float] = Field(default_factory=dict, description="Evaluation metrics for the final summary {'rouge_l': float, 'bert_score': float, 'topic_coverage': float}.")
    decision: Optional[str] = Field(default=None, description="Decision based on evaluation ('proceed', 'research_again', 'refine_topic').")
    trend_scores: List[Dict[str, Any]] = Field(default_factory=list, description="List of trend scores for keywords [{'keyword': str, 'score': float, ...}].")

    # Creative Generation (Node 14, 15, 17 output)
    comic_ideas: List[Dict[str, Any]] = Field(default_factory=list, description="List of generated comic ideas [{'idea_title': str, 'concept': str, 'creative_score': float}].")
    # NOTE: 'chosen_idea' needs to be populated *between* Node 14 and Node 15
    # This might require a human-in-the-loop step or an automated selection node.
    chosen_idea: Optional[Dict[str, Any]] = Field(default=None, description="The selected comic idea to proceed with.")
    scenarios: List[Dict[str, Any]] = Field(default_factory=list, description="List of 4 panel scenarios [{'scene': int, 'panel_description': str, 'dialogue': str, 'seed_tags': List[str]}].")
    scenario_prompt: Optional[str] = Field(default=None, description="The prompt used to generate the scenarios (for Node 16 report).")
    image_urls: List[str] = Field(default_factory=list, description="List of generated image URLs for the 4 panels.")

    # Optional Translation (Node 18 output)
    translated_text: Optional[List[Dict[str, str]]] = Field(default=None, description="List of translated dialogues [{'scene': int, 'original_dialogue': str, 'translated_dialogue': str}].")

    # Final Output (Node 19 output)
    final_comic: Dict[str, Optional[str]] = Field(default_factory=dict, description="Final comic output details, e.g., {'png_url': ..., 'webp_url': ..., 'alt_text': ...}")

    # Tracking & Stats
    used_links: List[Dict[str, str]] = Field(default_factory=list, description="List of URLs used during the process [{'url': str, 'purpose': str, 'status': Optional[str]}].")
    processing_stats: Dict[str, float] = Field(default_factory=dict, description="Dictionary to store processing time per node {'node_name_time': float, ...}.")

    # Error Handling
    error_message: Optional[str] = Field(default=None, description="Stores error messages if a node fails.")

    class Config:
        # Allows for arbitrary types, useful if custom objects are stored
        arbitrary_types_allowed = True