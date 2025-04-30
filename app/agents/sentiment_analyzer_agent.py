# app/agents/sentiment_analyzer_agent.py
import logging
import json
import httpx
from typing import Dict, Optional, Any, List
from app.workflows.state import ComicState
from app.services.llm_server_client import call_llm_api
from app.config.settings import settings

logger = logging.getLogger(__name__)

class SentimentAnalyzerAgent:
    """
    Agent that collects comments from news-related YouTube videos and performs sentiment analysis
    """
    
    async def _fetch_youtube_videos(self, query: str, max_results: int = 3) -> List[str]:
        """Searches for YouTube video IDs related to the news query."""
        if not settings.YOUTUBE_API_KEY:
            logger.warning("[SentimentAnalyzer] YouTube API key not configured.")
            return []
            
        search_url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "key": settings.YOUTUBE_API_KEY,
            "q": query + " news",  # Add news keyword
            "part": "snippet",
            "type": "video",
            "maxResults": max_results,
            "order": "relevance"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(search_url, params=params, timeout=20.0)
                response.raise_for_status()
                data = response.json()
                
                # Extract video IDs
                video_ids = [item["id"]["videoId"] for item in data.get("items", [])]
                logger.info(f"[SentimentAnalyzer] YouTube search results: {len(video_ids)} videos found")
                return video_ids
                
        except Exception as e:
            logger.error(f"[SentimentAnalyzer] Failed to search YouTube videos: {e}")
            return []
    
    async def _fetch_video_comments(self, video_id: str, max_comments: int = 50) -> List[str]:
        """Collects comments from a specific video."""
        if not settings.YOUTUBE_API_KEY:
            return []
            
        comments_url = "https://www.googleapis.com/youtube/v3/commentThreads"
        params = {
            "key": settings.YOUTUBE_API_KEY,
            "videoId": video_id,
            "part": "snippet",
            "maxResults": max_comments,
            "order": "relevance"  # Prioritize relevant comments
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(comments_url, params=params, timeout=20.0)
                response.raise_for_status()
                data = response.json()
                
                # Extract comment text
                comments = []
                for item in data.get("items", []):
                    comment_text = item["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
                    comments.append(comment_text)
                
                logger.info(f"[SentimentAnalyzer] Collected {len(comments)} comments from video {video_id}")
                return comments
                
        except Exception as e:
            logger.error(f"[SentimentAnalyzer] Failed to collect comments from video {video_id}: {e}")
            return []
    
    async def _simple_keyword_sentiment(self, comments: List[str]) -> Dict[str, Any]:
        """
        Performs simple keyword-based sentiment analysis (backup when LLM API call is difficult)
        """
        if not comments:
            return {
                "sentiment": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                "emotions": {"anger": 0.2, "sadness": 0.2, "joy": 0.2, "fear": 0.2, "surprise": 0.2}
            }
            
        # Simple sentiment/emotion word dictionaries
        positive_words = ["good", "nice", "love", "happy", "hope", "support", "great", "awesome", "thanks"]
        negative_words = ["bad", "disappointed", "angry", "hate", "sad", "anxious", "worry", "terrible"]
        
        # Emotion keywords
        emotion_words = {
            "anger": ["angry", "rage", "annoyed", "furious", "mad", "upset", "irritated"],
            "sadness": ["sad", "depressed", "unhappy", "miserable", "tears", "cry", "heartbroken"],
            "joy": ["happy", "joy", "delighted", "fun", "excited", "pleased", "wonderful"],
            "fear": ["afraid", "scared", "fear", "worry", "anxious", "terrified", "dread"],
            "surprise": ["surprised", "shocked", "unexpected", "wow", "amazing", "astonished"]
        }
        
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        emotion_counts = {k: 0 for k in emotion_words.keys()}
        
        for comment in comments:
            comment_lower = comment.lower()
            # Sentiment analysis
            pos_count = sum(1 for word in positive_words if word in comment_lower)
            neg_count = sum(1 for word in negative_words if word in comment_lower)
            
            if pos_count > neg_count:
                sentiment_counts["positive"] += 1
            elif neg_count > pos_count:
                sentiment_counts["negative"] += 1
            else:
                sentiment_counts["neutral"] += 1
                
            # Emotion analysis
            for emotion, keywords in emotion_words.items():
                if any(keyword in comment_lower for keyword in keywords):
                    emotion_counts[emotion] += 1
        
        # Calculate ratios
        total_sentiment = sum(sentiment_counts.values()) or 1  # Prevent division by zero
        sentiment_ratios = {k: v/total_sentiment for k, v in sentiment_counts.items()}
        
        total_emotion = sum(emotion_counts.values()) or 1
        emotion_ratios = {k: v/total_emotion for k, v in emotion_counts.items()}
        
        return {
            "sentiment": sentiment_ratios,
            "emotions": emotion_ratios
        }
    
    async def _analyze_sentiment_with_llm(self, comments: List[str]) -> Dict[str, Any]:
        """Sentiment analysis using LLM API"""
        if not comments:
            return {
                "sentiment": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                "emotions": {"anger": 0.2, "sadness": 0.2, "joy": 0.2, "fear": 0.2, "surprise": 0.2}
            }
        
        # Sample comments (too many comments can't be processed due to context length limits)
        sample_size = min(30, len(comments))
        sampled_comments = comments[:sample_size]
        
        # LLM prompt
        prompt = f"""The following are comments from news-related YouTube videos. Analyze these comments to estimate the overall sentiment (positive/negative/neutral) and emotion (anger/sadness/joy/fear/surprise) distribution.

Comments:
---
{chr(10).join(sampled_comments)}
---

Return the results in the following JSON format:
{{
  "sentiment": {{"positive": 0.0, "negative": 0.0, "neutral": 0.0}},
  "emotions": {{"anger": 0.0, "sadness": 0.0, "joy": 0.0, "fear": 0.0, "surprise": 0.0}}
}}

Each value should be a ratio between 0.0 and 1.0, and the sum within each category should be 1.0.
"""
        
        try:
            response = await call_llm_api(prompt, max_tokens=200, temperature=0.1)
            
            # Parse JSON response
            # Extract JSON block from response
            json_str = ""
            in_json = False
            for line in response.split('\n'):
                line = line.strip()
                if line == '{' or line.startswith('{"'):
                    in_json = True
                    json_str += line
                elif in_json and ('}' in line):
                    json_str += line
                    in_json = False
                elif in_json:
                    json_str += line
            
            # Try to parse JSON
            try:
                result = json.loads(json_str)
                # Validate result
                if "sentiment" in result and "emotions" in result:
                    return result
            except json.JSONDecodeError:
                logger.error(f"[SentimentAnalyzer] JSON parsing failed: {json_str}")
            
            # Use default values if parsing fails
            return await self._simple_keyword_sentiment(comments)
            
        except Exception as e:
            logger.error(f"[SentimentAnalyzer] LLM sentiment analysis failed: {e}")
            # Use simple keyword-based analysis if LLM call fails
            return await self._simple_keyword_sentiment(comments)
    
    async def run(self, state: ComicState) -> Dict[str, Optional[Any]]:
        """
        1. Search for YouTube videos related to the news summary
        2. Collect comments from videos
        3. Perform sentiment analysis on comments
        4. Store results in state
        """
        logger.info("--- [Sentiment Analyzer Agent] Starting execution ---")
        updates: Dict[str, Optional[Any]] = {}
        
        # Verify news summary
        if not state.final_summary and not state.initial_query:
            logger.warning("[SentimentAnalyzer] No news summary or query available for analysis.")
            updates["public_sentiment"] = None
            return updates
            
        try:
            # 1. Extract search query from summary or initial query
            search_query = state.initial_query
            if not search_query and state.final_summary:
                # Simple approach: use the first part of the summary as query
                search_query = state.final_summary.split('.')[0]
            
            if not search_query:
                logger.warning("[SentimentAnalyzer] Cannot extract search query.")
                updates["public_sentiment"] = None
                return updates
            
            # 2. Search YouTube videos
            video_ids = await self._fetch_youtube_videos(search_query, max_results=3)
            
            # 3. Collect comments
            all_comments = []
            for video_id in video_ids:
                comments = await self._fetch_video_comments(video_id, max_comments=30)
                all_comments.extend(comments)
                
            # 4. Sentiment analysis
            if not all_comments:
                logger.warning("[SentimentAnalyzer] No comments collected.")
                updates["public_sentiment"] = None
                return updates
            
            # Choose analysis method based on API key availability
            if settings.LLM_API_ENDPOINT and settings.LLM_API_KEY:
                sentiment_results = await self._analyze_sentiment_with_llm(all_comments)
            else:
                sentiment_results = await self._simple_keyword_sentiment(all_comments)
            
            # 5. Store results
            updates["public_sentiment"] = sentiment_results
            logger.info(f"[SentimentAnalyzer] Sentiment analysis complete: {sentiment_results}")
            
        except Exception as e:
            logger.exception(f"[SentimentAnalyzer] Error during execution: {e}")
            updates["public_sentiment"] = None
            # Sentiment analysis failure is not treated as a workflow error (optional enhancement feature)
            # updates["error_message"] = f"Public sentiment analysis failed: {str(e)}"
            
        logger.info("--- [Sentiment Analyzer Agent] Execution complete ---")
        return updates