"""
Collectors Node for LangGraph News Processing Agent
"""
import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Clients for various search engines
class NaverNewsClient:
    """Client for Naver News API"""
    def __init__(self):
        self.client_id = os.getenv("NAVER_CLIENT_ID")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET")
        self.base_url = "https://openapi.naver.com/v1/search/news.json"
        
    async def search(self, query: str, display: int = 10) -> Dict[str, Any]:
        """Search news from Naver"""
        if not self.client_id or not self.client_secret:
            raise ValueError("Naver API credentials not found in environment variables")
        
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }
        params = {
            "query": query,
            "display": display,
            "sort": "sim"  # Sort by relevance
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(self.base_url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
    
    def process_results(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Process and normalize Naver search results"""
        if "items" not in results:
            return []
        
        processed_results = []
        for item in results["items"]:
            processed_results.append({
                "title": item.get("title", "").replace("<b>", "").replace("</b>", ""),
                "url": item.get("link", ""),
                "description": item.get("description", "").replace("<b>", "").replace("</b>", ""),
                "source": "Naver",
                "published_date": item.get("pubDate", "")
            })
        
        return processed_results


class DaumNewsClient:
    """Client for Daum News API (Kakao)"""
    def __init__(self):
        self.api_key = os.getenv("KAKAO_API_KEY")
        self.base_url = "https://dapi.kakao.com/v2/search/web"
        
    async def search(self, query: str, size: int = 10) -> Dict[str, Any]:
        """Search news from Daum/Kakao"""
        if not self.api_key:
            raise ValueError("Kakao API key not found in environment variables")
        
        headers = {
            "Authorization": f"KakaoAK {self.api_key}"
        }
        params = {
            "query": f"{query} 뉴스",  # Adding '뉴스' to focus on news content
            "size": size
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(self.base_url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
    
    def process_results(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Process and normalize Daum/Kakao search results"""
        if "documents" not in results:
            return []
        
        processed_results = []
        for item in results["documents"]:
            # Filter only news domains
            url = item.get("url", "")
            if any(domain in url for domain in [".co.kr/", ".com/", ".kr/", "news."]):
                processed_results.append({
                    "title": item.get("title", ""),
                    "url": url,
                    "description": item.get("contents", ""),
                    "source": "Daum",
                    "published_date": ""  # Daum API doesn't directly provide date
                })
        
        return processed_results


class GoogleNewsClient:
    """Client for Google Custom Search API"""
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.cx = os.getenv("GOOGLE_SEARCH_CX")  # Search engine ID
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
    async def search(self, query: str, num: int = 10) -> Dict[str, Any]:
        """Search news from Google"""
        if not self.api_key or not self.cx:
            raise ValueError("Google API credentials not found in environment variables")
        
        params = {
            "key": self.api_key,
            "cx": self.cx,
            "q": f"{query} news",
            "num": num
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
    
    def process_results(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Process and normalize Google search results"""
        if "items" not in results:
            return []
        
        processed_results = []
        for item in results["items"]:
            processed_results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "description": item.get("snippet", ""),
                "source": "Google",
                "published_date": ""  # Google API doesn't directly provide date
            })
        
        return processed_results


class TavilyNewsClient:
    """Client for Tavily Search API"""
    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.base_url = "https://api.tavily.com/search"
        
    async def search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Search news from Tavily"""
        if not self.api_key:
            raise ValueError("Tavily API key not found in environment variables")
        
        headers = {
            "Content-Type": "application/json"
        }
        params = {
            "api_key": self.api_key,
            "query": f"{query} news",
            "search_depth": "advanced",
            "max_results": max_results,
            "include_domains": ["news.com", "bbc.com", "nytimes.com", "cnn.com", "reuters.com", 
                               "chosun.com", "joongang.co.kr", "donga.com", "hani.co.kr", "yna.co.kr"],
            "filter_domains": [],
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(self.base_url, headers=headers, json=params)
            response.raise_for_status()
            return response.json()
    
    def process_results(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Process and normalize Tavily search results"""
        if "results" not in results:
            return []
        
        processed_results = []
        for item in results["results"]:
            processed_results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": item.get("content", ""),
                "source": "Tavily",
                "published_date": item.get("published_date", "")
            })
        
        return processed_results


class BingNewsClient:
    """Client for Bing Search API"""
    def __init__(self):
        self.api_key = os.getenv("BING_API_KEY")
        self.base_url = "https://api.bing.microsoft.com/v7.0/news/search"
        
    async def search(self, query: str, count: int = 10) -> Dict[str, Any]:
        """Search news from Bing"""
        if not self.api_key:
            raise ValueError("Bing API key not found in environment variables")
        
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key
        }
        params = {
            "q": query,
            "count": count,
            "mkt": "ko-KR",  # Market setting for Korean
            "freshness": "Day"  # Get recent news
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(self.base_url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
    
    def process_results(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Process and normalize Bing search results"""
        if "value" not in results:
            return []
        
        processed_results = []
        for item in results["value"]:
            processed_results.append({
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "description": item.get("description", ""),
                "source": "Bing",
                "published_date": item.get("datePublished", "")
            })
        
        return processed_results


class LLMClient:
    """Client for LLM API"""
    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("LLM_API_URL", "http://localhost:8000")
        self.endpoint = f"{self.base_url}/v1/llm/generate"
        
    async def generate(self, prompt: str, model: str = "llama-3.2-3b") -> Dict[str, Any]:
        """Generate text using LLM API"""
        if not self.api_key:
            print("Warning: LLM API key not found, proceeding without authentication")
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "prompt": prompt,
            "model": model,
            "max_tokens": 1024,
            "temperature": 0.7
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.endpoint, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                print(f"HTTP error occurred: {e}")
                return {"error": str(e), "generated_text": ""}
            except Exception as e:
                print(f"An error occurred: {e}")
                return {"error": str(e), "generated_text": ""}


class CollectorsNode:
    """News Collectors Node for LangGraph"""
    
    def __init__(self):
        # Initialize clients
        self.naver_client = NaverNewsClient()
        self.daum_client = DaumNewsClient()
        self.google_client = GoogleNewsClient()
        self.tavily_client = TavilyNewsClient()
        self.bing_client = BingNewsClient()
        self.llm_client = LLMClient()
        
        # Default primary search engines
        self.primary_search_engines = ["naver", "bing"]
    
    async def search_news(self, search_query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """Search news from multiple search engines"""
        results = []
        tasks = []
        
        for engine in self.primary_search_engines:
            if engine == "naver":
                tasks.append(self._search_naver(search_query, num_results))
            elif engine == "daum":
                tasks.append(self._search_daum(search_query, num_results))
            elif engine == "google":
                tasks.append(self._search_google(search_query, num_results))
            elif engine == "tavily":
                tasks.append(self._search_tavily(search_query, num_results))
            elif engine == "bing":
                tasks.append(self._search_bing(search_query, num_results))
        
        # Run searches in parallel
        search_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results (filtering out exceptions)
        for result in search_results:
            if isinstance(result, list):
                results.extend(result)
            else:
                print(f"Error in search: {result}")
        
        return results
    
    async def _search_naver(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search news from Naver"""
        try:
            results = await self.naver_client.search(query, display=num_results)
            return self.naver_client.process_results(results)
        except Exception as e:
            print(f"Error searching Naver news: {e}")
            return []
    
    async def _search_daum(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search news from Daum"""
        try:
            results = await self.daum_client.search(query, size=num_results)
            return self.daum_client.process_results(results)
        except Exception as e:
            print(f"Error searching Daum news: {e}")
            return []
    
    async def _search_google(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search news from Google"""
        try:
            results = await self.google_client.search(query, num=num_results)
            return self.google_client.process_results(results)
        except Exception as e:
            print(f"Error searching Google news: {e}")
            return []
    
    async def _search_tavily(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search news from Tavily"""
        try:
            results = await self.tavily_client.search(query, max_results=num_results)
            return self.tavily_client.process_results(results)
        except Exception as e:
            print(f"Error searching Tavily news: {e}")
            return []
    
    async def _search_bing(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search news from Bing"""
        try:
            results = await self.bing_client.search(query, count=num_results)
            return self.bing_client.process_results(results)
        except Exception as e:
            print(f"Error searching Bing news: {e}")
            return []
    
    async def select_news(self, search_results: List[Dict[str, Any]], max_urls: int = 5) -> Dict[str, Any]:
        """Select news URLs using LLM"""
        if not search_results:
            return {"news_urls": [], "selected_url": "", "error": "No search results found"}
        
        # Create a prompt for the LLM
        prompt = self._create_selection_prompt(search_results)
        
        # Call LLM to select news
        llm_response = await self.llm_client.generate(prompt)
        
        if "error" in llm_response and llm_response["error"]:
            return {"news_urls": [], "selected_url": "", "error": f"LLM error: {llm_response['error']}"}
        
        # Parse LLM response
        generated_text = llm_response.get("generated_text", "")
        selected_urls = self._parse_llm_selection(generated_text, search_results, max_urls)
        
        if not selected_urls:
            # Fallback: select top results based on relevance
            selected_urls = [result["url"] for result in search_results[:max_urls]]
        
        return {
            "news_urls": selected_urls,
            "selected_url": selected_urls[0] if selected_urls else "",
            "error": ""
        }
    
    def _create_selection_prompt(self, search_results: List[Dict[str, Any]]) -> str:
        """Create a prompt for the LLM to select relevant news"""
        items_text = ""
        for i, result in enumerate(search_results[:20]):  # Limit to top 20 for prompt size
            items_text += f"{i+1}. Title: {result['title']}\n"
            items_text += f"   URL: {result['url']}\n"
            items_text += f"   Description: {result['description']}\n"
            items_text += f"   Source: {result['source']}\n\n"
        
        prompt = f"""You are an AI assistant helping to select the most relevant news articles to create a 4-panel comic strip.

Given the following news search results, identify the top 5 most interesting, substantial, and newsworthy articles. 
Prefer articles that:
- Are current news (not opinion or analysis)
- Have humorous potential or ironic elements
- Contain clear events or statements that could be visualized
- Are substantive enough to create a 4-panel comic about
- Are preferably from reliable news sources

SEARCH RESULTS:
{items_text}

Please analyze the above search results and respond in the following JSON format:
```json
{{
  "selected_urls": [
    "URL1",
    "URL2",
    "URL3",
    "URL4",
    "URL5"
  ],
  "reasoning": "Brief explanation of why you selected these articles"
}}
```

Make sure to select the most appropriate articles for creating an interesting comic strip.
"""
        return prompt
    
    def _parse_llm_selection(self, llm_response: str, search_results: List[Dict[str, Any]], max_urls: int) -> List[str]:
        """Parse LLM response to extract selected URLs"""
        try:
            # Extract JSON part from the response
            json_start = llm_response.find('```json') + 7 if '```json' in llm_response else llm_response.find('{')
            json_end = llm_response.rfind('```') if '```' in llm_response else llm_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = llm_response[json_start:json_end].strip()
                parsed = json.loads(json_content)
                
                if "selected_urls" in parsed and isinstance(parsed["selected_urls"], list):
                    # Filter to ensure URLs are in the original search results
                    all_urls = [result["url"] for result in search_results]
                    valid_urls = [url for url in parsed["selected_urls"] if url in all_urls]
                    
                    # Limit to requested max
                    return valid_urls[:max_urls]
            
            # Fallback: check if URLs are directly in the text
            urls = []
            for result in search_results:
                if result["url"] in llm_response and len(urls) < max_urls:
                    urls.append(result["url"])
            
            if urls:
                return urls
                
        except Exception as e:
            print(f"Error parsing LLM selection response: {e}")
        
        # If parsing fails, return an empty list
        return []
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the Collectors node"""
        search_query = state.get("search_query", "")
        if not search_query:
            return {**state, "error_message": "No search query provided"}
        
        try:
            # Search news from multiple engines
            search_results = await self.search_news(search_query)
            
            # Update state with search results
            updated_state = {**state, "search_results": search_results}
            
            # If no results found
            if not search_results:
                return {**updated_state, "error_message": "No news found for the given query"}
            
            # Select news using LLM
            selection_result = await self.select_news(search_results)
            
            # Update state with selection results
            updated_state.update({
                "news_urls": selection_result.get("news_urls", []),
                "selected_url": selection_result.get("selected_url", ""),
                "error_message": selection_result.get("error", "")
            })
            
            return updated_state
            
        except Exception as e:
            return {**state, "error_message": f"Error in Collectors node: {str(e)}"}


# For LangGraph integration
def collectors_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Function to be used as a node in LangGraph"""
    collectors = CollectorsNode()
    return asyncio.run(collectors.run(state))
