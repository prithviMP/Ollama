#!/usr/bin/env python3
"""
Lesson 8: Agents & Tools with LangChain

This lesson covers:
1. Agent architectures and reasoning frameworks (ReAct, Plan-and-Execute)
2. Building autonomous agents with tool-calling capabilities
3. Creating custom tools for domain-specific tasks
4. Multi-agent coordination and collaboration
5. Production agents with monitoring and safety controls

Author: LangChain Course
"""

import os
import sys
import time
import json
import asyncio
from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path
from dotenv import load_dotenv
import logging
from datetime import datetime

# Add shared resources to path
sys.path.append('../shared-resources')

# Load environment variables
load_dotenv()

# LangChain imports
from langchain.agents import (
    create_react_agent, 
    create_openai_functions_agent,
    AgentExecutor,
    Tool
)
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.callbacks import get_openai_callback
from langchain.schema.runnable import RunnablePassthrough
from pydantic import BaseModel, Field
import structlog

# External tool imports
try:
    from langchain.tools import WikipediaQueryRun
    from langchain.utilities import WikipediaAPIWrapper
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False

try:
    from langchain.tools import PythonREPLTool
    PYTHON_REPL_AVAILABLE = True
except ImportError:
    PYTHON_REPL_AVAILABLE = False

# Shared utilities
from utils.llm_setup import setup_llm_providers, get_preferred_llm
from utils.error_handling import safe_llm_call

# Custom imports for this lesson
import requests
import re
import math
import statistics
from datetime import datetime, timedelta
import sqlite3
import csv
import json


class SafetyValidator:
    """Validates agent inputs and outputs for safety."""
    
    def __init__(self):
        self.blocked_keywords = os.getenv("BLOCKED_KEYWORDS", "").split(",")
        self.allowed_domains = os.getenv("ALLOWED_DOMAINS", "").split(",")
        self.enable_content_filtering = os.getenv("ENABLE_CONTENT_FILTERING", "True").lower() == "true"
    
    def validate_input(self, user_input: str) -> bool:
        """Validate user input for safety."""
        if not self.enable_content_filtering:
            return True
        
        # Check for blocked keywords
        for keyword in self.blocked_keywords:
            if keyword.lower() in user_input.lower():
                return False
        
        # Additional safety checks can be added here
        return True
    
    def validate_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and potentially filter agent output."""
        if not self.enable_content_filtering:
            return output
        
        # Filter output content
        if "output" in output:
            filtered_output = self._filter_content(output["output"])
            output["output"] = filtered_output
        
        return output
    
    def _filter_content(self, content: str) -> str:
        """Filter potentially harmful content."""
        for keyword in self.blocked_keywords:
            if keyword.lower() in content.lower():
                content = content.replace(keyword, "[FILTERED]")
        
        return content


class CostMonitor:
    """Monitors and controls agent execution costs."""
    
    def __init__(self, max_cost_per_session: float = 10.0):
        self.max_cost_per_session = max_cost_per_session
        self.session_cost = 0.0
        self.cost_history = []
    
    def would_exceed_limit(self, estimated_cost: float) -> bool:
        """Check if operation would exceed cost limit."""
        return (self.session_cost + estimated_cost) > self.max_cost_per_session
    
    def add_cost(self, cost: float):
        """Add cost to session total."""
        self.session_cost += cost
        self.cost_history.append({
            "cost": cost,
            "timestamp": datetime.now().isoformat(),
            "cumulative": self.session_cost
        })
    
    def get_remaining_budget(self) -> float:
        """Get remaining budget for session."""
        return max(0, self.max_cost_per_session - self.session_cost)


class AgentLogger:
    """Structured logging for agent interactions."""
    
    def __init__(self):
        self.logger = structlog.get_logger("agent_system")
    
    def log_interaction(self, user_id: str, input_text: str, output: Dict[str, Any]):
        """Log agent interaction."""
        self.logger.info(
            "agent_interaction",
            user_id=user_id,
            input=input_text[:200],  # Truncate for logging
            output_type=type(output.get("output", "")).__name__,
            success=True,
            timestamp=datetime.now().isoformat()
        )
    
    def log_error(self, user_id: str, input_text: str, error: str):
        """Log agent error."""
        self.logger.error(
            "agent_error",
            user_id=user_id,
            input=input_text[:200],
            error=error,
            timestamp=datetime.now().isoformat()
        )
    
    def log_tool_usage(self, tool_name: str, input_args: Dict, output: str, execution_time: float):
        """Log tool usage."""
        self.logger.info(
            "tool_usage",
            tool_name=tool_name,
            input_args=input_args,
            output_length=len(output),
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )


# Custom Tools Implementation

class CalculatorTool(BaseTool):
    """Enhanced calculator tool with advanced mathematical operations."""
    
    name = "calculator"
    description = """
    Advanced calculator that can perform:
    - Basic arithmetic: +, -, *, /, %, **
    - Math functions: sin, cos, tan, log, sqrt, abs
    - Statistics: mean, median, mode, stdev
    - Constants: pi, e
    
    Input should be a mathematical expression or function call.
    Examples: "2 + 3 * 4", "sqrt(16)", "mean([1, 2, 3, 4, 5])"
    """
    
    def _run(self, expression: str) -> str:
        """Execute mathematical calculation safely."""
        try:
            # Safe evaluation environment
            safe_dict = {
                "__builtins__": {},
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "len": len,
                "math": math, "statistics": statistics,
                "pi": math.pi, "e": math.e,
                "sin": math.sin, "cos": math.cos, "tan": math.tan,
                "log": math.log, "sqrt": math.sqrt,
                "mean": statistics.mean, "median": statistics.median,
                "stdev": statistics.stdev
            }
            
            # Handle special function formats
            if "mean(" in expression:
                # Extract list from mean([1,2,3]) format
                match = re.search(r'mean\(\[(.*?)\]\)', expression)
                if match:
                    numbers = [float(x.strip()) for x in match.group(1).split(',')]
                    return str(statistics.mean(numbers))
            
            # Standard evaluation
            result = eval(expression, safe_dict)
            return str(result)
            
        except Exception as e:
            return f"Calculator error: {str(e)}"
    
    async def _arun(self, expression: str) -> str:
        """Async version of calculator."""
        return self._run(expression)


class WebScrapingTool(BaseTool):
    """Tool for scraping and extracting information from web pages."""
    
    name = "web_scraper"
    description = """
    Scrape content from web pages and extract specific information.
    Input should be a URL. Returns cleaned text content from the page.
    Use this when you need to get current information from a specific website.
    """
    
    def _run(self, url: str) -> str:
        """Scrape content from a web page."""
        try:
            import requests
            from bs4 import BeautifulSoup
            import html2text
            
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                return "Error: URL must start with http:// or https://"
            
            # Check if domain is allowed (if restrictions are enabled)
            allowed_domains = os.getenv("ALLOWED_DOMAINS", "").split(",")
            if allowed_domains and allowed_domains[0]:  # If domains are specified
                domain_allowed = any(domain in url for domain in allowed_domains if domain)
                if not domain_allowed:
                    return "Error: Domain not in allowed list"
            
            # Make request with timeout
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Truncate if too long
            if len(text) > 2000:
                text = text[:2000] + "... [truncated]"
            
            return f"Content from {url}:\n{text}"
            
        except Exception as e:
            return f"Web scraping error: {str(e)}"
    
    async def _arun(self, url: str) -> str:
        """Async version of web scraper."""
        return self._run(url)


class FileManagerTool(BaseTool):
    """Tool for safe file operations within a restricted workspace."""
    
    name = "file_manager"
    description = """
    Manage files in the agent workspace. Operations include:
    - read_file: Read content from a file
    - write_file: Write content to a file
    - list_files: List files in directory
    - file_info: Get file information
    
    Input format: "operation:filename" or "operation:filename:content"
    Example: "read_file:data.txt" or "write_file:output.txt:Hello World"
    """
    
    def __init__(self):
        super().__init__()
        self.workspace_dir = Path(os.getenv("WORKSPACE_DIRECTORY", "./agent_workspace"))
        self.workspace_dir.mkdir(exist_ok=True)
        self.allowed_extensions = os.getenv("ALLOWED_FILE_EXTENSIONS", ".txt,.json,.csv,.md").split(",")
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
    
    def _run(self, operation_string: str) -> str:
        """Execute file operation safely."""
        try:
            parts = operation_string.split(":", 2)
            operation = parts[0]
            
            if operation == "list_files":
                files = list(self.workspace_dir.glob("*"))
                file_list = [f.name for f in files if f.is_file()]
                return f"Files in workspace: {', '.join(file_list)}"
            
            if len(parts) < 2:
                return "Error: Invalid operation format"
            
            filename = parts[1]
            filepath = self.workspace_dir / filename
            
            # Security check: ensure file is within workspace
            try:
                filepath.resolve().relative_to(self.workspace_dir.resolve())
            except ValueError:
                return "Error: File path outside workspace"
            
            # Check file extension
            if not any(filename.endswith(ext) for ext in self.allowed_extensions):
                return f"Error: File extension not allowed. Allowed: {', '.join(self.allowed_extensions)}"
            
            if operation == "read_file":
                if not filepath.exists():
                    return "Error: File does not exist"
                
                if filepath.stat().st_size > self.max_file_size:
                    return "Error: File too large"
                
                content = filepath.read_text(encoding='utf-8')
                return f"Content of {filename}:\n{content}"
            
            elif operation == "write_file":
                if len(parts) < 3:
                    return "Error: No content provided"
                
                content = parts[2]
                if len(content.encode('utf-8')) > self.max_file_size:
                    return "Error: Content too large"
                
                filepath.write_text(content, encoding='utf-8')
                return f"Successfully wrote {len(content)} characters to {filename}"
            
            elif operation == "file_info":
                if not filepath.exists():
                    return "Error: File does not exist"
                
                stat = filepath.stat()
                return f"File info for {filename}:\nSize: {stat.st_size} bytes\nModified: {datetime.fromtimestamp(stat.st_mtime)}"
            
            else:
                return f"Error: Unknown operation '{operation}'. Available: read_file, write_file, list_files, file_info"
        
        except Exception as e:
            return f"File operation error: {str(e)}"
    
    async def _arun(self, operation_string: str) -> str:
        """Async version of file manager."""
        return self._run(operation_string)


class DatabaseTool(BaseTool):
    """Tool for interacting with SQLite database."""
    
    name = "database"
    description = """
    Execute SQL queries on SQLite database. Operations include:
    - query: Execute SELECT queries
    - execute: Execute INSERT, UPDATE, DELETE queries
    - tables: List all tables
    - schema: Get table schema
    
    Input format: "operation:sql_query"
    Example: "query:SELECT * FROM users LIMIT 5"
    """
    
    def __init__(self):
        super().__init__()
        self.db_path = os.getenv("DATABASE_URL", "sqlite:///agents.db").replace("sqlite:///", "")
        self._init_database()
    
    def _init_database(self):
        """Initialize database with sample tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create sample table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    user_input TEXT,
                    agent_output TEXT,
                    success BOOLEAN
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sample_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    value REAL,
                    category TEXT,
                    created_at TEXT
                )
            """)
            
            # Insert sample data if table is empty
            cursor.execute("SELECT COUNT(*) FROM sample_data")
            if cursor.fetchone()[0] == 0:
                sample_records = [
                    ("Product A", 100.50, "Electronics", "2024-01-01"),
                    ("Product B", 75.25, "Clothing", "2024-01-02"),
                    ("Product C", 200.00, "Electronics", "2024-01-03"),
                    ("Product D", 50.75, "Books", "2024-01-04"),
                    ("Product E", 125.00, "Clothing", "2024-01-05")
                ]
                cursor.executemany(
                    "INSERT INTO sample_data (name, value, category, created_at) VALUES (?, ?, ?, ?)",
                    sample_records
                )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def _run(self, operation_string: str) -> str:
        """Execute database operation safely."""
        try:
            parts = operation_string.split(":", 1)
            if len(parts) != 2:
                return "Error: Invalid format. Use 'operation:sql_query'"
            
            operation, sql_query = parts
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if operation == "tables":
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                return f"Available tables: {', '.join(tables)}"
            
            elif operation == "schema":
                table_name = sql_query.strip()
                cursor.execute(f"PRAGMA table_info({table_name})")
                schema = cursor.fetchall()
                if not schema:
                    return f"Table '{table_name}' not found"
                
                schema_str = f"Schema for {table_name}:\n"
                for col in schema:
                    schema_str += f"  {col[1]} ({col[2]})\n"
                return schema_str
            
            elif operation == "query":
                # Only allow SELECT queries for safety
                if not sql_query.strip().upper().startswith("SELECT"):
                    return "Error: Only SELECT queries allowed with 'query' operation"
                
                cursor.execute(sql_query)
                results = cursor.fetchall()
                
                if not results:
                    return "No results found"
                
                # Get column names
                columns = [description[0] for description in cursor.description]
                
                # Format results
                result_str = f"Query results ({len(results)} rows):\n"
                result_str += " | ".join(columns) + "\n"
                result_str += "-" * (len(" | ".join(columns))) + "\n"
                
                for row in results[:10]:  # Limit to 10 rows
                    result_str += " | ".join(str(cell) for cell in row) + "\n"
                
                if len(results) > 10:
                    result_str += f"... and {len(results) - 10} more rows"
                
                return result_str
            
            elif operation == "execute":
                # Allow INSERT, UPDATE, DELETE for sample data table only
                allowed_prefixes = ["INSERT", "UPDATE", "DELETE"]
                if not any(sql_query.strip().upper().startswith(prefix) for prefix in allowed_prefixes):
                    return "Error: Only INSERT, UPDATE, DELETE queries allowed with 'execute' operation"
                
                cursor.execute(sql_query)
                conn.commit()
                
                return f"Query executed successfully. Rows affected: {cursor.rowcount}"
            
            else:
                return f"Error: Unknown operation '{operation}'. Available: query, execute, tables, schema"
        
        except Exception as e:
            return f"Database error: {str(e)}"
        
        finally:
            if 'conn' in locals():
                conn.close()
    
    async def _arun(self, operation_string: str) -> str:
        """Async version of database tool."""
        return self._run(operation_string)


# Agent Architectures

class BasicReActAgent:
    """Basic ReAct (Reasoning + Acting) agent implementation."""
    
    def __init__(self, llm, tools: List[BaseTool], verbose: bool = True):
        self.llm = llm
        self.tools = tools
        self.verbose = verbose
        self.tool_map = {tool.name: tool for tool in tools}
        
        # Create agent executor
        agent = create_react_agent(llm, tools, self._get_react_prompt())
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=verbose,
            max_iterations=int(os.getenv("MAX_ITERATIONS", "10")),
            max_execution_time=int(os.getenv("MAX_EXECUTION_TIME", "300"))
        )
    
    def _get_react_prompt(self) -> str:
        """Get ReAct prompt template."""
        return """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
    
    def run(self, input_text: str) -> Dict[str, Any]:
        """Run the agent with given input."""
        try:
            with get_openai_callback() as cb:
                result = self.agent_executor.invoke({"input": input_text})
            
            return {
                "output": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", []),
                "cost": {
                    "total_tokens": cb.total_tokens,
                    "total_cost": cb.total_cost
                }
            }
        except Exception as e:
            return {
                "error": str(e),
                "output": "Agent execution failed"
            }


class ConversationalAgent:
    """Conversational agent with memory."""
    
    def __init__(self, llm, tools: List[BaseTool], memory_type: str = "buffer"):
        self.llm = llm
        self.tools = tools
        
        # Set up memory
        if memory_type == "summary":
            self.memory = ConversationSummaryBufferMemory(
                llm=llm,
                memory_key="chat_history",
                return_messages=True,
                max_token_limit=2000
            )
        else:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        
        # Create conversational agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant with access to various tools. 
            Use the tools when needed to provide accurate and helpful responses.
            
            Available tools:
            {tools}
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_functions_agent(llm, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            return_intermediate_steps=True
        )
    
    def chat(self, user_input: str) -> Dict[str, Any]:
        """Have a conversation with the agent."""
        try:
            with get_openai_callback() as cb:
                result = self.agent_executor.invoke({"input": user_input})
            
            return {
                "output": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", []),
                "cost": {
                    "total_tokens": cb.total_tokens,
                    "total_cost": cb.total_cost
                }
            }
        except Exception as e:
            return {
                "error": str(e),
                "output": "Agent execution failed"
            }
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()


class MultiAgentOrchestrator:
    """Orchestrate multiple specialized agents."""
    
    def __init__(self, llm):
        self.llm = llm
        self.agents = {}
        self._setup_specialized_agents()
    
    def _setup_specialized_agents(self):
        """Set up specialized agents for different tasks."""
        
        # Research Agent
        research_tools = []
        if WIKIPEDIA_AVAILABLE:
            wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
            research_tools.append(wikipedia)
        
        try:
            search_tool = DuckDuckGoSearchRun()
            research_tools.append(search_tool)
        except:
            pass
        
        research_tools.append(WebScrapingTool())
        
        self.agents["researcher"] = BasicReActAgent(
            self.llm, 
            research_tools, 
            verbose=False
        )
        
        # Analysis Agent
        analysis_tools = [
            CalculatorTool(),
            DatabaseTool()
        ]
        
        self.agents["analyst"] = BasicReActAgent(
            self.llm,
            analysis_tools,
            verbose=False
        )
        
        # File Manager Agent
        file_tools = [FileManagerTool()]
        
        self.agents["file_manager"] = BasicReActAgent(
            self.llm,
            file_tools,
            verbose=False
        )
    
    def process_complex_request(self, request: str) -> Dict[str, Any]:
        """Process complex request using multiple agents."""
        results = {}
        
        try:
            # Step 1: Research phase
            if any(keyword in request.lower() for keyword in ["research", "find", "search", "information"]):
                print("🔍 Starting research phase...")
                research_result = self.agents["researcher"].run(
                    f"Research information about: {request}"
                )
                results["research"] = research_result
            
            # Step 2: Analysis phase
            if any(keyword in request.lower() for keyword in ["analyze", "calculate", "compute", "data"]):
                print("📊 Starting analysis phase...")
                analysis_input = request
                if "research" in results:
                    analysis_input += f"\n\nContext from research: {results['research']['output']}"
                
                analysis_result = self.agents["analyst"].run(
                    f"Analyze this information: {analysis_input}"
                )
                results["analysis"] = analysis_result
            
            # Step 3: File operations if needed
            if any(keyword in request.lower() for keyword in ["save", "file", "write", "store"]):
                print("💾 Starting file management phase...")
                file_result = self.agents["file_manager"].run(
                    f"Handle file operations for: {request}"
                )
                results["file_operations"] = file_result
            
            # Combine results
            final_output = "Multi-agent processing complete:\n\n"
            for phase, result in results.items():
                final_output += f"{phase.upper()}:\n{result['output']}\n\n"
            
            return {
                "output": final_output,
                "phase_results": results,
                "agents_used": list(results.keys())
            }
        
        except Exception as e:
            return {
                "error": str(e),
                "output": "Multi-agent processing failed",
                "phase_results": results
            }


class SafeProductionAgent:
    """Production agent with safety controls and monitoring."""
    
    def __init__(self, llm, tools: List[BaseTool], config: Dict[str, Any] = None):
        self.llm = llm
        self.tools = tools
        self.config = config or {}
        
        # Initialize safety and monitoring components
        self.safety_checker = SafetyValidator()
        self.cost_monitor = CostMonitor(
            max_cost_per_session=float(os.getenv("MAX_COST_PER_SESSION", "10.0"))
        )
        self.logger = AgentLogger()
        
        # Create base agent
        self.agent = BasicReActAgent(llm, tools, verbose=False)
    
    def execute(self, user_input: str, user_id: str = "anonymous") -> Dict[str, Any]:
        """Execute agent with full safety and monitoring."""
        start_time = time.time()
        
        try:
            # Pre-execution safety check
            if not self.safety_checker.validate_input(user_input):
                self.logger.log_error(user_id, user_input, "Input safety validation failed")
                return {
                    "error": "Input violates safety policies",
                    "output": "Request blocked for safety reasons"
                }
            
            # Cost check (simplified - would need better estimation in production)
            estimated_cost = len(user_input) * 0.0001  # Rough estimation
            if self.cost_monitor.would_exceed_limit(estimated_cost):
                self.logger.log_error(user_id, user_input, "Cost limit would be exceeded")
                return {
                    "error": "Cost limit would be exceeded",
                    "output": f"Remaining budget: ${self.cost_monitor.get_remaining_budget():.2f}"
                }
            
            # Execute with monitoring
            result = self.agent.run(user_input)
            
            # Post-execution validation
            validated_result = self.safety_checker.validate_output(result)
            
            # Update cost monitoring
            if "cost" in result:
                self.cost_monitor.add_cost(result["cost"].get("total_cost", 0))
            
            # Log successful interaction
            execution_time = time.time() - start_time
            validated_result["execution_time"] = execution_time
            
            self.logger.log_interaction(user_id, user_input, validated_result)
            
            return validated_result
        
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.log_error(user_id, user_input, str(e))
            return {
                "error": "Agent execution failed",
                "output": "An error occurred during processing",
                "execution_time": execution_time
            }
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        return {
            "cost_used": self.cost_monitor.session_cost,
            "cost_remaining": self.cost_monitor.get_remaining_budget(),
            "cost_limit": self.cost_monitor.max_cost_per_session,
            "cost_history": self.cost_monitor.cost_history
        }


def setup_lesson():
    """Set up the lesson environment."""
    print("🦜🔗 LangChain Course - Lesson 8: Agents & Tools")
    print("=" * 70)
    
    providers = setup_llm_providers()
    if not providers:
        print("❌ No LLM providers available. Please check your setup.")
        return None
    
    llm = get_preferred_llm(providers, prefer_chat=True)
    print(f"✅ Using LLM: {type(llm).__name__}")
    
    return llm


def demonstrate_basic_agent(llm):
    """Demonstrate basic ReAct agent functionality."""
    print("\n" + "="*60)
    print("🤖 BASIC REACT AGENT DEMONSTRATION")
    print("="*60)
    
    # Set up tools
    tools = [
        CalculatorTool(),
        DatabaseTool()
    ]
    
    # Add search if available
    try:
        search_tool = DuckDuckGoSearchRun()
        tools.append(search_tool)
        print("✅ Search tool added")
    except Exception as e:
        print(f"⚠️  Search tool not available: {e}")
    
    # Create agent
    agent = BasicReActAgent(llm, tools)
    
    # Test queries
    test_queries = [
        "What is 15 * 24 + 100?",
        "List the tables in the database and show me some sample data",
        "Calculate the mean of these numbers: 10, 20, 30, 40, 50"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 50)
        
        result = agent.run(query)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"🤖 Response: {result['output']}")
            if "cost" in result:
                print(f"💰 Cost: ${result['cost']['total_cost']:.4f}")
        
        print("-" * 50)


def demonstrate_conversational_agent(llm):
    """Demonstrate conversational agent with memory."""
    print("\n" + "="*60)
    print("💬 CONVERSATIONAL AGENT DEMONSTRATION")
    print("="*60)
    
    # Set up tools
    tools = [
        CalculatorTool(),
        FileManagerTool()
    ]
    
    # Create conversational agent
    agent = ConversationalAgent(llm, tools, memory_type="buffer")
    
    # Simulate conversation
    conversation = [
        "Hi, I'm working on a math project. Can you help me calculate 25 * 34?",
        "Great! Can you also calculate what 20% of that result would be?",
        "Perfect. Now can you save both results to a file called 'calculations.txt'?",
        "What was the first calculation I asked you to do?"
    ]
    
    print("🗣️  Starting conversation simulation...")
    
    for i, message in enumerate(conversation, 1):
        print(f"\n👤 User {i}: {message}")
        
        result = agent.chat(message)
        
        if "error" in result:
            print(f"❌ Agent: Error occurred - {result['error']}")
        else:
            print(f"🤖 Agent: {result['output']}")
        
        # Short pause for realism
        time.sleep(1)


def demonstrate_multi_agent_system(llm):
    """Demonstrate multi-agent orchestration."""
    print("\n" + "="*60)
    print("🏢 MULTI-AGENT ORCHESTRATION DEMONSTRATION")
    print("="*60)
    
    # Create orchestrator
    orchestrator = MultiAgentOrchestrator(llm)
    
    # Test complex requests
    complex_requests = [
        "Research information about artificial intelligence and analyze the current trends",
        "Find data about electric car sales and calculate the average growth rate",
        "Search for information about renewable energy and save a summary to a file"
    ]
    
    for request in complex_requests:
        print(f"\n🎯 Complex Request: {request}")
        print("-" * 60)
        
        result = orchestrator.process_complex_request(request)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"🏢 Multi-Agent Result:")
            print(f"Agents used: {result.get('agents_used', [])}")
            print(f"Final output: {result['output'][:300]}...")
        
        print("-" * 60)


def demonstrate_production_agent(llm):
    """Demonstrate production agent with safety controls."""
    print("\n" + "="*60)
    print("🏭 PRODUCTION AGENT DEMONSTRATION")
    print("="*60)
    
    # Set up tools
    tools = [
        CalculatorTool(),
        FileManagerTool(),
        DatabaseTool()
    ]
    
    # Create production agent
    agent = SafeProductionAgent(llm, tools)
    
    # Test queries including some that should be filtered
    test_queries = [
        "Calculate the square root of 144",
        "Create a file with some sample data",
        "Show me the database tables and their contents"
    ]
    
    for query in test_queries:
        print(f"\n🔒 Testing: {query}")
        
        result = agent.execute(query, user_id="test_user")
        
        if "error" in result:
            print(f"❌ Blocked/Error: {result['error']}")
        else:
            print(f"✅ Response: {result['output'][:200]}...")
            print(f"⏱️  Execution time: {result.get('execution_time', 0):.2f}s")
    
    # Show session stats
    stats = agent.get_session_stats()
    print(f"\n📊 Session Statistics:")
    print(f"   Cost used: ${stats['cost_used']:.4f}")
    print(f"   Cost remaining: ${stats['cost_remaining']:.2f}")


def interactive_agent_playground(llm):
    """Interactive agent playground for experimentation."""
    print("\n" + "="*60)
    print("🎮 INTERACTIVE AGENT PLAYGROUND")
    print("="*60)
    
    # Set up available agents
    agents = {}
    
    # Basic ReAct Agent
    basic_tools = [CalculatorTool(), DatabaseTool(), FileManagerTool()]
    try:
        basic_tools.append(DuckDuckGoSearchRun())
    except:
        pass
    
    agents["basic"] = BasicReActAgent(llm, basic_tools, verbose=False)
    
    # Conversational Agent
    agents["conversational"] = ConversationalAgent(llm, basic_tools)
    
    # Multi-Agent Orchestrator
    agents["multi"] = MultiAgentOrchestrator(llm)
    
    # Production Agent
    agents["production"] = SafeProductionAgent(llm, basic_tools)
    
    print("🤖 Available Agents:")
    for name in agents.keys():
        print(f"   • {name}")
    
    current_agent = "basic"
    
    while True:
        print(f"\n🎯 Current Agent: {current_agent}")
        print("Commands: 'switch <agent>', 'clear' (for conversational), 'quit'")
        
        user_input = input("\n💬 Your input: ").strip()
        
        if user_input.lower() == 'quit':
            print("👋 Thanks for using the Agent Playground!")
            break
        
        if user_input.lower().startswith('switch '):
            new_agent = user_input.split(' ', 1)[1]
            if new_agent in agents:
                current_agent = new_agent
                print(f"🔄 Switched to {current_agent} agent")
            else:
                print(f"❌ Unknown agent: {new_agent}")
            continue
        
        if user_input.lower() == 'clear' and current_agent == "conversational":
            agents["conversational"].clear_memory()
            print("🧹 Conversation memory cleared")
            continue
        
        if not user_input:
            continue
        
        # Execute with selected agent
        print(f"\n🔄 Processing with {current_agent} agent...")
        
        try:
            if current_agent == "basic":
                result = agents["basic"].run(user_input)
            elif current_agent == "conversational":
                result = agents["conversational"].chat(user_input)
            elif current_agent == "multi":
                result = agents["multi"].process_complex_request(user_input)
            elif current_agent == "production":
                result = agents["production"].execute(user_input, "playground_user")
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"🤖 Response: {result['output']}")
                
                if "cost" in result and result["cost"]:
                    print(f"💰 Cost: ${result['cost'].get('total_cost', 0):.4f}")
        
        except Exception as e:
            print(f"❌ Execution error: {e}")


def main():
    """Main function to run all agent demonstrations."""
    llm = setup_lesson()
    
    if not llm:
        print("❌ Cannot proceed without LLM. Please check your setup.")
        return
    
    try:
        # Run demonstrations
        demonstrate_basic_agent(llm)
        demonstrate_conversational_agent(llm)
        demonstrate_multi_agent_system(llm)
        demonstrate_production_agent(llm)
        
        # Interactive playground
        print("\n🎉 Core demonstrations completed!")
        
        run_playground = input("\nWould you like to try the Interactive Agent Playground? (y/n): ").strip().lower()
        if run_playground in ['y', 'yes']:
            interactive_agent_playground(llm)
        
        print("\n✨ Lesson 8 completed! You've mastered agents and tools.")
        print("\n📚 Key Skills Acquired:")
        print("   • Building ReAct agents with tool integration")
        print("   • Creating conversational agents with memory")
        print("   • Orchestrating multi-agent systems")
        print("   • Implementing production-ready agents with safety controls")
        print("   • Developing custom tools for specific use cases")
        
        print("\n🔗 Next: Advanced production patterns and deployment strategies")
        
    except KeyboardInterrupt:
        print("\n\n👋 Lesson interrupted. You can resume anytime!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    main()