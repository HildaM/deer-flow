# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import os
from typing import List, Optional

from langchain_community.tools import (
    BraveSearch,
    DuckDuckGoSearchResults,
    WikipediaQueryRun,
)
from langchain_community.tools.arxiv import ArxivQueryRun
from langchain_community.utilities import (
    ArxivAPIWrapper,
    BraveSearchWrapper,
    WikipediaAPIWrapper,
)

from src.config import SearchEngine, SELECTED_SEARCH_ENGINE
from src.config import load_yaml_config
from src.tools.tavily_search.tavily_search_results_with_images import (
    TavilySearchResultsWithImages,
)

from src.tools.decorators import create_logged_tool
from src.utils.token_utils import truncate_search_results, get_max_token_limit

logger = logging.getLogger(__name__)

# Create logged versions of the search tools
LoggedTavilySearch = create_logged_tool(TavilySearchResultsWithImages)
LoggedDuckDuckGoSearch = create_logged_tool(DuckDuckGoSearchResults)
LoggedBraveSearch = create_logged_tool(BraveSearch)
LoggedArxivSearch = create_logged_tool(ArxivQueryRun)
LoggedWikipediaSearch = create_logged_tool(WikipediaQueryRun)


def get_search_config():
    config = load_yaml_config("conf.yaml")
    search_config = config.get("SEARCH_ENGINE", {})
    return search_config


# Get the selected search tool
def get_web_search_tool(max_search_results: int, enable_token_truncation: bool = True):
    search_config = get_search_config()

    # Create the base search tool based on selected engine
    if SELECTED_SEARCH_ENGINE == SearchEngine.TAVILY.value:
        # Only get and apply include/exclude domains for Tavily
        include_domains: Optional[List[str]] = search_config.get("include_domains", [])
        exclude_domains: Optional[List[str]] = search_config.get("exclude_domains", [])

        logger.info(
            f"Tavily search configuration loaded: include_domains={include_domains}, exclude_domains={exclude_domains}"
        )

        search_tool = LoggedTavilySearch(
            name="web_search",
            max_results=max_search_results,
            include_raw_content=True,
            include_images=True,
            include_image_descriptions=True,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.DUCKDUCKGO.value:
        search_tool = LoggedDuckDuckGoSearch(
            name="web_search",
            num_results=max_search_results,
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.BRAVE_SEARCH.value:
        search_tool = LoggedBraveSearch(
            name="web_search",
            search_wrapper=BraveSearchWrapper(
                api_key=os.getenv("BRAVE_SEARCH_API_KEY", ""),
                search_kwargs={"count": max_search_results},
            ),
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.ARXIV.value:
        search_tool = LoggedArxivSearch(
            name="web_search",
            api_wrapper=ArxivAPIWrapper(
                top_k_results=max_search_results,
                load_max_docs=max_search_results,
                load_all_available_meta=True,
            ),
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.WIKIPEDIA.value:
        wiki_lang = search_config.get("wikipedia_lang", "en")
        wiki_doc_content_chars_max = search_config.get(
            "wikipedia_doc_content_chars_max", 4000
        )
        search_tool = LoggedWikipediaSearch(
            name="web_search",
            api_wrapper=WikipediaAPIWrapper(
                lang=wiki_lang,
                top_k_results=max_search_results,
                load_all_available_meta=True,
                doc_content_chars_max=wiki_doc_content_chars_max,
            ),
        )
    else:
        raise ValueError(f"Unsupported search engine: {SELECTED_SEARCH_ENGINE}")
    
    # Apply token truncation if enabled
    if enable_token_truncation:
        from langchain_core.tools import tool
        import json
        
        max_token_limit = get_max_token_limit()
        base_tool = search_tool
        
        @tool
        def truncated_web_search(query: str) -> str:
            """Search the web and return truncated results to avoid token limits.
            
            Args:
                query: The search query
                
            Returns:
                Truncated search results as a string
            """
            try:
                # Get search results from the base tool
                results = base_tool.invoke(query)
                
                # Convert results to string format
                if isinstance(results, list):
                    # For Tavily and similar structured results
                    results_text = "\n\n".join([
                        f"## {elem.get('title', 'No Title')}\n\n{elem.get('content', elem.get('snippet', 'No content'))}"
                        for elem in results
                    ])
                else:
                    # For other search engines that return string or dict
                    results_text = json.dumps(results, ensure_ascii=False) if isinstance(results, dict) else str(results)
                
                # Apply token truncation
                truncated_results = truncate_search_results(results_text, max_token_limit)
                return truncated_results
                
            except Exception as e:
                logger.error(f"Error in truncated web search: {e}")
                return f"Search error: {str(e)}"
        
        # Copy metadata from base tool
        truncated_web_search.name = "web_search"
        truncated_web_search.description = f"Search the web for information. Results are automatically truncated to stay within {max_token_limit} token limit."
        
        return truncated_web_search
    
    return search_tool
