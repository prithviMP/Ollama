"""
Sample tools for agent development and testing.

This package provides pre-built tools that can be used with LangChain agents
for common tasks like web search, file operations, calculations, and more.
"""

from .calculator_tool import CalculatorTool
from .file_manager_tool import FileManagerTool
from .web_search_tool import WebSearchTool
from .weather_tool import WeatherTool
from .email_tool import EmailTool

__all__ = [
    "CalculatorTool",
    "FileManagerTool", 
    "WebSearchTool",
    "WeatherTool",
    "EmailTool"
]