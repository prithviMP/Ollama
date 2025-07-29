"""
Shared utilities for the LangChain Course

This package contains helper functions and utilities that can be used
across all lessons to reduce code duplication and provide common functionality.
"""

from .llm_setup import setup_llm_providers, get_available_providers
from .prompt_helpers import create_prompt_template, format_response
from .error_handling import safe_llm_call, handle_api_errors

__all__ = [
    "setup_llm_providers",
    "get_available_providers", 
    "create_prompt_template",
    "format_response",
    "safe_llm_call",
    "handle_api_errors"
] 