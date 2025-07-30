#!/usr/bin/env python3
"""
Web Search Tool for Agents

Provides comprehensive web search capabilities using multiple search engines
and APIs, with result filtering, caching, and safety controls.
"""

import time
import json
import hashlib
import requests
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from bs4 import BeautifulSoup


class WebSearchTool(BaseTool):
    """Advanced web search tool with multiple providers and caching."""
    
    name = "web_search"
    description = """
    Search the web for information using multiple search engines.
    
    Input format: "query" or "query|options"
    Options: max_results=N, engine=google|duckduckgo|bing, safe=true/false
    
    Examples:
    - "artificial intelligence trends"
    - "python programming|max_results=5"
    - "weather forecast|engine=duckduckgo|safe=true"
    
    Returns: Formatted search results with titles, URLs, and snippets
    """
    
    def __init__(self, 
                 cache_ttl: int = 3600,  # 1 hour cache
                 max_results: int = 10,
                 timeout: int = 30,
                 safe_search: bool = True):
        super().__init__()
        self.cache_ttl = cache_ttl
        self.max_results = max_results
        self.timeout = timeout
        self.safe_search = safe_search
        self.cache = {}
        
        # Search engine configurations
        self.engines = {
            "duckduckgo": self._search_duckduckgo,
            "google": self._search_google_custom,
            "bing": self._search_bing,
            "wikipedia": self._search_wikipedia
        }
        
        # API keys (would be loaded from environment in production)
        self.api_keys = {
            "google_api_key": None,  # Set from env
            "google_cse_id": None,   # Set from env
            "bing_api_key": None     # Set from env
        }
    
    def _run(self, query_input: str) -> str:
        """Execute web search."""
        try:
            # Parse query and options
            query, options = self._parse_input(query_input)
            
            if not query.strip():
                return "Error: Empty search query provided"
            
            # Check cache first
            cache_key = self._get_cache_key(query, options)
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if datetime.now() - cached_result["timestamp"] < timedelta(seconds=self.cache_ttl):
                    return f"🔍 Cached search results for: {query}\\n\\n{cached_result['results']}"
            
            # Determine search engine
            engine = options.get("engine", "duckduckgo")
            if engine not in self.engines:
                engine = "duckduckgo"  # Fallback
            
            # Perform search
            results = self.engines[engine](query, options)
            
            # Format results
            formatted_results = self._format_results(query, results, options)
            
            # Cache results
            self.cache[cache_key] = {
                "timestamp": datetime.now(),
                "results": formatted_results
            }
            
            return formatted_results
        
        except Exception as e:
            return f"Error during web search: {str(e)}"
    
    def _parse_input(self, query_input: str) -> tuple:
        """Parse query input and extract options."""
        parts = query_input.split("|")
        query = parts[0].strip()
        
        options = {
            "max_results": self.max_results,
            "engine": "duckduckgo",
            "safe": self.safe_search
        }
        
        # Parse options
        for part in parts[1:]:
            if "=" in part:
                key, value = part.strip().split("=", 1)
                key = key.strip()
                value = value.strip()
                
                if key == "max_results":
                    try:
                        options[key] = int(value)
                    except ValueError:
                        pass
                elif key == "safe":
                    options[key] = value.lower() in ("true", "1", "yes")
                else:
                    options[key] = value
        
        return query, options
    
    def _get_cache_key(self, query: str, options: Dict) -> str:
        """Generate cache key for query and options."""
        cache_data = f"{query}|{json.dumps(options, sort_keys=True)}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def _search_duckduckgo(self, query: str, options: Dict) -> List[Dict]:
        """Search using DuckDuckGo (scraping-based)."""
        try:
            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Extract instant answer
            if data.get("Abstract"):
                results.append({
                    "title": data.get("Heading", "DuckDuckGo Instant Answer"),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data.get("Abstract", "")
                })
            
            # Extract related topics
            for topic in data.get("RelatedTopics", [])[:options["max_results"]]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("Text", "").split(" - ")[0],
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", "")
                    })
            
            # If no results, try web scraping (simplified)
            if not results:
                results = self._scrape_duckduckgo_web(query, options)
            
            return results[:options["max_results"]]
        
        except Exception as e:
            return [{"title": "Search Error", "url": "", "snippet": f"DuckDuckGo search failed: {str(e)}"}]
    
    def _scrape_duckduckgo_web(self, query: str, options: Dict) -> List[Dict]:
        """Scrape DuckDuckGo web results (simplified)."""
        try:
            # This is a simplified version - production code would need more robust scraping
            url = "https://html.duckduckgo.com/html/"
            params = {"q": query}
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Extract search results (simplified parsing)
            for result_div in soup.find_all('div', class_='result')[:options["max_results"]]:
                title_elem = result_div.find('a', class_='result__a')
                snippet_elem = result_div.find('a', class_='result__snippet')
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet
                    })
            
            return results
        
        except Exception:
            # Return mock results if scraping fails
            return self._get_mock_results(query, options["max_results"])
    
    def _search_google_custom(self, query: str, options: Dict) -> List[Dict]:
        """Search using Google Custom Search API."""
        try:
            if not self.api_keys["google_api_key"] or not self.api_keys["google_cse_id"]:
                return self._get_mock_results(query, options["max_results"], "Google Custom Search")
            
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.api_keys["google_api_key"],
                "cx": self.api_keys["google_cse_id"],
                "q": query,
                "num": min(options["max_results"], 10),
                "safe": "active" if options["safe"] else "off"
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("items", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                })
            
            return results
        
        except Exception as e:
            return [{"title": "Search Error", "url": "", "snippet": f"Google search failed: {str(e)}"}]
    
    def _search_bing(self, query: str, options: Dict) -> List[Dict]:
        """Search using Bing Search API."""
        try:
            if not self.api_keys["bing_api_key"]:
                return self._get_mock_results(query, options["max_results"], "Bing Search")
            
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": self.api_keys["bing_api_key"]}
            params = {
                "q": query,
                "count": options["max_results"],
                "safeSearch": "Strict" if options["safe"] else "Off"
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("webPages", {}).get("value", []):
                results.append({
                    "title": item.get("name", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("snippet", "")
                })
            
            return results
        
        except Exception as e:
            return [{"title": "Search Error", "url": "", "snippet": f"Bing search failed: {str(e)}"}]
    
    def _search_wikipedia(self, query: str, options: Dict) -> List[Dict]:
        """Search Wikipedia articles."""
        try:
            # Search for articles
            search_url = "https://en.wikipedia.org/api/rest_v1/page/search"
            params = {"q": query, "limit": options["max_results"]}
            
            response = requests.get(search_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for page in data.get("pages", []):
                title = page.get("title", "")
                page_id = page.get("id")
                description = page.get("description", "")
                
                # Get page URL
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                
                # Get extract if available
                extract = page.get("extract", description)
                
                results.append({
                    "title": f"Wikipedia: {title}",
                    "url": url,
                    "snippet": extract or description
                })
            
            return results
        
        except Exception as e:
            return [{"title": "Wikipedia Error", "url": "", "snippet": f"Wikipedia search failed: {str(e)}"}]
    
    def _get_mock_results(self, query: str, max_results: int, source: str = "Mock Search") -> List[Dict]:
        """Generate mock search results for testing."""
        results = []
        for i in range(min(max_results, 3)):
            results.append({
                "title": f"{source} Result {i+1} for '{query}'",
                "url": f"https://example.com/result-{i+1}",
                "snippet": f"This is a mock search result {i+1} for the query '{query}'. It contains relevant information about the topic."
            })
        
        return results
    
    def _format_results(self, query: str, results: List[Dict], options: Dict) -> str:
        """Format search results for display."""
        if not results:
            return f"🔍 No results found for: {query}"
        
        formatted = f"🔍 Search results for: {query}\\n"
        formatted += f"Engine: {options.get('engine', 'unknown')} | Results: {len(results)}\\n"
        formatted += "=" * 60 + "\\n\\n"
        
        for i, result in enumerate(results, 1):
            title = result.get("title", "No Title")
            url = result.get("url", "")
            snippet = result.get("snippet", "No description available")
            
            # Truncate long snippets
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."
            
            formatted += f"{i}. **{title}**\\n"
            if url:
                formatted += f"   🔗 {url}\\n"
            formatted += f"   📝 {snippet}\\n\\n"
        
        # Add search timestamp
        formatted += f"\\n🕒 Search performed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return formatted
    
    def clear_cache(self) -> str:
        """Clear search result cache."""
        cache_size = len(self.cache)
        self.cache.clear()
        return f"🗑️ Cache cleared. Removed {cache_size} cached results."
    
    def get_cache_stats(self) -> str:
        """Get cache statistics."""
        total_items = len(self.cache)
        expired_items = 0
        
        now = datetime.now()
        for cached_item in self.cache.values():
            if now - cached_item["timestamp"] > timedelta(seconds=self.cache_ttl):
                expired_items += 1
        
        return f"📊 Cache Stats: {total_items} items ({expired_items} expired)"
    
    async def _arun(self, query_input: str) -> str:
        """Async version of web search."""
        return self._run(query_input)


# Example usage and testing
def test_web_search():
    """Test the web search tool."""
    search = WebSearchTool()
    
    test_queries = [
        "artificial intelligence trends 2024",
        "python programming tutorial|max_results=3",
        "climate change solutions|engine=wikipedia|max_results=2",
        "machine learning basics|safe=true"
    ]
    
    print("🔍 Web Search Tool Test Results:")
    print("=" * 60)
    
    for query in test_queries:
        result = search._run(query)
        print(f"Query: {query}")
        print(f"Results:\\n{result}")
        print("-" * 60)
    
    # Test cache functionality
    print("\\n" + search.get_cache_stats())


if __name__ == "__main__":
    test_web_search()