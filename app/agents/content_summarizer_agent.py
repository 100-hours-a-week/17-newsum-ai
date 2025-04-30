# app/agents/content_summarizer_agent.py
import logging
import json
import httpx
from typing import Dict, Optional, Any, List, Tuple
from app.workflows.state import ComicState
from app.services.llm_server_client import call_llm_api
from app.config.settings import settings

logger = logging.getLogger(__name__)

class ContentSummarizerAgent:
    """
    Agent that retrieves YouTube video captions and comments related to news topics,
    and provides a comprehensive summary of additional context and perspectives.
    """
    
    async def _fetch_youtube_videos(self, query: str, max_results: int = 3) -> List[str]:
        """Searches for YouTube video IDs related to the news query."""
        if not settings.YOUTUBE_API_KEY:
            logger.warning("[ContentSummarizer] YouTube API key not configured.")
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
                
                # Extract video IDs and titles
                videos = []
                for item in data.get("items", []):
                    videos.append({
                        "id": item["id"]["videoId"],
                        "title": item["snippet"]["title"],
                        "channel": item["snippet"]["channelTitle"]
                    })
                
                logger.info(f"[ContentSummarizer] YouTube search results: {len(videos)} videos found")
                return videos
                
        except Exception as e:
            logger.error(f"[ContentSummarizer] Failed to search YouTube videos: {e}")
            return []
    
    async def _fetch_video_captions(self, video_id: str) -> str:
        """Retrieves captions/transcripts for a video if available."""
        if not settings.YOUTUBE_API_KEY:
            return ""
            
        # First, get caption tracks available for the video
        captions_url = "https://www.googleapis.com/youtube/v3/captions"
        params = {
            "key": settings.YOUTUBE_API_KEY,
            "videoId": video_id,
            "part": "snippet"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(captions_url, params=params, timeout=20.0)
                
                if response.status_code == 403:
                    logger.warning(f"[ContentSummarizer] No permission to access captions for video {video_id}")
                    return ""
                
                response.raise_for_status()
                data = response.json()
                
                # Look for English captions (or any available caption)
                caption_id = None
                items = data.get("items", [])
                
                for item in items:
                    language = item["snippet"].get("language", "")
                    track_kind = item["snippet"].get("trackKind", "")
                    
                    # Prefer English auto-generated captions if available
                    if language == "en" and track_kind == "ASR":
                        caption_id = item["id"]
                        break
                    # Or any English captions
                    elif language == "en":
                        caption_id = item["id"]
                        break
                
                # If no English caption, take any available
                if not caption_id and items:
                    caption_id = items[0]["id"]
                
                if not caption_id:
                    logger.warning(f"[ContentSummarizer] No captions found for video {video_id}")
                    return ""
                
                # Now get the actual caption content
                # Note: This step requires OAuth 2.0 authentication and may not work with just API key
                # For simplicity, we'll return a placeholder message
                logger.info(f"[ContentSummarizer] Found caption ID for video {video_id}: {caption_id}")
                return "[Captions available - OAuth required for full access]"
                
        except Exception as e:
            logger.error(f"[ContentSummarizer] Failed to fetch captions for video {video_id}: {e}")
            return ""
    
    async def _fetch_video_comments(self, video_id: str, max_comments: int = 15) -> List[str]:
        """Collects most relevant comments from a video."""
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
                    like_count = item["snippet"]["topLevelComment"]["snippet"]["likeCount"]
                    
                    # Include only substantive comments (at least 15 chars) with high engagement
                    if len(comment_text) >= 15:
                        comments.append({
                            "text": comment_text,
                            "likes": like_count
                        })
                
                # Sort by likes and take top comments
                comments.sort(key=lambda x: x["likes"], reverse=True)
                top_comments = [c["text"] for c in comments[:10]]  # Take top 10 most liked
                
                logger.info(f"[ContentSummarizer] Collected {len(top_comments)} relevant comments from video {video_id}")
                return top_comments
                
        except Exception as e:
            logger.error(f"[ContentSummarizer] Failed to collect comments from video {video_id}: {e}")
            return []
    
    async def _get_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """Retrieves additional metadata for the video."""
        if not settings.YOUTUBE_API_KEY:
            return {}
            
        videos_url = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            "key": settings.YOUTUBE_API_KEY,
            "id": video_id,
            "part": "snippet,statistics"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(videos_url, params=params, timeout=20.0)
                response.raise_for_status()
                data = response.json()
                
                if not data.get("items"):
                    return {}
                
                item = data["items"][0]
                
                return {
                    "title": item["snippet"]["title"],
                    "description": item["snippet"]["description"],
                    "viewCount": item["statistics"].get("viewCount", "0"),
                    "likeCount": item["statistics"].get("likeCount", "0"),
                    "commentCount": item["statistics"].get("commentCount", "0"),
                    "publishedAt": item["snippet"]["publishedAt"],
                    "channel": item["snippet"]["channelTitle"]
                }
                
        except Exception as e:
            logger.error(f"[ContentSummarizer] Failed to get metadata for video {video_id}: {e}")
            return {}
    
    async def _summarize_content(self, video_data: List[Dict], news_summary: str) -> Dict[str, Any]:
        """
        Summarizes collected video data and extracts relevant additional context.
        """
        if not video_data:
            return {
                "summary": "No additional context found from YouTube videos.",
                "perspectives": []
            }
        
        # Prepare the input for the LLM
        processed_videos = []
        
        for video in video_data:
            video_info = f"Video: '{video['title']}' by {video['channel']}\n"
            
            if video.get('metadata'):
                metadata = video['metadata']
                video_info += f"Views: {metadata.get('viewCount', 'N/A')}, Likes: {metadata.get('likeCount', 'N/A')}\n"
                if metadata.get('description'):
                    # Truncate description if too long
                    desc = metadata['description']
                    if len(desc) > 300:
                        desc = desc[:297] + "..."
                    video_info += f"Description: {desc}\n"
            
            if video.get('captions'):
                video_info += f"Captions: {video['captions']}\n"
            
            if video.get('comments'):
                video_info += "Top Comments:\n"
                for i, comment in enumerate(video['comments'][:5], 1):  # Include up to 5 comments
                    video_info += f"{i}. {comment}\n"
            
            processed_videos.append(video_info)
        
        # LLM prompt for summarization
        prompt = f"""Based on the following news summary and related YouTube videos, provide:
1. A comprehensive summary of the additional context and perspectives found in the videos and comments
2. A list of 3-5 unique perspectives or insights gained from these sources

News Summary:
---
{news_summary}
---

YouTube Video Data:
---
{chr(10).join(processed_videos)}
---

Respond with:
1. ADDITIONAL CONTEXT: A concise paragraph summarizing the additional context and insights from the videos and comments
2. KEY PERSPECTIVES: A bulleted list of 3-5 unique perspectives or insights (not already mentioned in the original news summary)
"""
        
        try:
            response = await call_llm_api(prompt, max_tokens=500, temperature=0.3)
            
            # Extract the summary and perspectives from the response
            additional_context = ""
            perspectives = []
            
            # Parse response
            in_context = False
            in_perspectives = False
            
            for line in response.split('\n'):
                line = line.strip()
                
                if line.startswith("ADDITIONAL CONTEXT:"):
                    in_context = True
                    in_perspectives = False
                    # Skip the header
                    continue
                
                elif line.startswith("KEY PERSPECTIVES:"):
                    in_context = False
                    in_perspectives = True
                    # Skip the header
                    continue
                
                elif in_context and line:
                    additional_context += line + " "
                
                elif in_perspectives and line and (line.startswith('-') or line.startswith('*') or line[0].isdigit()):
                    perspectives.append(line.lstrip('-*0123456789. '))
            
            return {
                "summary": additional_context.strip(),
                "perspectives": perspectives
            }
            
        except Exception as e:
            logger.error(f"[ContentSummarizer] LLM summarization failed: {e}")
            return {
                "summary": f"Error generating summary: {str(e)}",
                "perspectives": []
            }
    
    async def run(self, state: ComicState) -> Dict[str, Optional[Any]]:
        """
        1. Search for YouTube videos related to the news topic
        2. Collect captions, comments, and metadata for each video
        3. Summarize additional context and perspectives
        4. Store results in state
        """
        logger.info("--- [Content Summarizer Agent] Starting execution ---")
        updates: Dict[str, Optional[Any]] = {}
        
        # Verify news summary
        if not state.final_summary and not state.initial_query:
            logger.warning("[ContentSummarizer] No news summary or query available for analysis.")
            updates["additional_context"] = None
            return updates
            
        try:
            # 1. Extract search query from summary or initial query
            search_query = state.initial_query
            if not search_query and state.final_summary:
                # Use first sentence of summary as query
                search_query = state.final_summary.split('.')[0]
            
            if not search_query:
                logger.warning("[ContentSummarizer] Cannot extract search query.")
                updates["additional_context"] = None
                return updates
            
            # 2. Search for YouTube videos
            videos = await self._fetch_youtube_videos(search_query, max_results=3)
            
            if not videos:
                logger.warning("[ContentSummarizer] No videos found for the query.")
                updates["additional_context"] = {
                    "summary": "No relevant YouTube videos found for additional context.",
                    "perspectives": []
                }
                return updates
            
            # 3. Collect data for each video
            video_data = []
            
            for video in videos:
                video_id = video["id"]
                
                # Get metadata, captions, and comments in parallel
                metadata = await self._get_video_metadata(video_id)
                captions = await self._fetch_video_captions(video_id)
                comments = await self._fetch_video_comments(video_id)
                
                video_data.append({
                    "id": video_id,
                    "title": video["title"],
                    "channel": video["channel"],
                    "metadata": metadata,
                    "captions": captions,
                    "comments": comments
                })
            
            # 4. Summarize the collected content
            logger.info(f"[ContentSummarizer] Processing data from {len(video_data)} videos")
            additional_context = await self._summarize_content(
                video_data, 
                state.final_summary or search_query
            )
            
            # 5. Store results
            updates["additional_context"] = additional_context
            logger.info(f"[ContentSummarizer] Content summarization complete")
            
        except Exception as e:
            logger.exception(f"[ContentSummarizer] Error during execution: {e}")
            updates["additional_context"] = {
                "summary": f"Error analyzing YouTube content: {str(e)}",
                "perspectives": []
            }
            
        logger.info("--- [Content Summarizer Agent] Execution complete ---")
        return updates