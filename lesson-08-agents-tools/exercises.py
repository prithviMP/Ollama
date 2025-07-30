#!/usr/bin/env python3
"""
Lesson 8: Agents & Tools - Practice Exercises

Complete these exercises to master agent architectures and tool development.
Each exercise focuses on different aspects of building autonomous agent systems.

Instructions:
1. Implement each exercise function
2. Run individual exercises to test your implementations
3. Check solutions.py for reference implementations
4. Experiment with different agent configurations and tools
"""

import os
import sys
import time
import json
import asyncio
from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Add shared resources to path
sys.path.append('../shared-resources')

# Load environment variables
load_dotenv()

# LangChain imports
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.callbacks import get_openai_callback
from pydantic import BaseModel, Field

# Shared utilities
from utils.llm_setup import setup_llm_providers, get_preferred_llm
from utils.error_handling import safe_llm_call

# Set up providers
providers = setup_llm_providers()
llm = get_preferred_llm(providers, prefer_chat=True) if providers else None


def exercise_1_research_assistant_agent():
    """
    Exercise 1: Research Assistant Agent
    
    Task: Build an intelligent research assistant that can gather information
    from multiple sources, synthesize findings, and compile comprehensive reports.
    
    Requirements:
    1. Create custom tools for different research sources (web search, academic papers, news)
    2. Implement information synthesis and fact-checking capabilities
    3. Generate structured research reports with citations
    4. Handle multi-step research workflows with follow-up questions
    5. Provide confidence scores for information gathered
    
    Your implementation should:
    - Use multiple information sources effectively
    - Validate and cross-reference information
    - Generate well-structured, comprehensive reports
    - Handle ambiguous or incomplete queries intelligently
    """
    
    print("🔍 Exercise 1: Research Assistant Agent")
    print("-" * 60)
    
    # TODO: Implement your research assistant agent
    # Hints:
    # 1. Create specialized tools for different types of research
    # 2. Implement information validation and synthesis
    # 3. Design report generation with proper citations
    # 4. Add confidence scoring for research findings
    
    class ResearchTool(BaseTool):
        """Custom tool for research operations."""
        
        name = "research_tool"
        description = "Perform comprehensive research on a given topic"
        
        def _run(self, topic: str) -> str:
            """Perform research operation."""
            # TODO: Implement research logic
            pass
        
        async def _arun(self, topic: str) -> str:
            """Async version of research tool."""
            return self._run(topic)
    
    class FactCheckTool(BaseTool):
        """Tool for fact-checking information."""
        
        name = "fact_checker"
        description = "Verify facts and check information accuracy"
        
        def _run(self, statement: str) -> str:
            """Check fact accuracy."""
            # TODO: Implement fact-checking logic
            pass
        
        async def _arun(self, statement: str) -> str:
            """Async version of fact checker."""
            return self._run(statement)
    
    class ReportGeneratorTool(BaseTool):
        """Tool for generating structured research reports."""
        
        name = "report_generator"
        description = "Generate structured research reports with citations"
        
        def _run(self, research_data: str) -> str:
            """Generate research report."""
            # TODO: Implement report generation
            pass
        
        async def _arun(self, research_data: str) -> str:
            """Async version of report generator."""
            return self._run(research_data)
    
    class ResearchAssistantAgent:
        """Comprehensive research assistant agent."""
        
        def __init__(self, llm):
            self.llm = llm
            self.tools = [
                ResearchTool(),
                FactCheckTool(),
                ReportGeneratorTool()
            ]
            self.agent = None  # TODO: Initialize agent
        
        def conduct_research(self, topic: str, depth: str = "comprehensive") -> Dict[str, Any]:
            """Conduct comprehensive research on a topic."""
            # TODO: Implement research workflow
            pass
        
        def validate_findings(self, findings: List[Dict]) -> List[Dict]:
            """Validate and score research findings."""
            # TODO: Implement finding validation
            pass
        
        def generate_report(self, topic: str, findings: List[Dict]) -> str:
            """Generate final research report."""
            # TODO: Implement report generation
            pass
    
    # Your implementation here
    print("📝 TODO: Implement research assistant agent")
    print("💡 Consider: multi-source research, fact validation, report structuring")


def exercise_2_customer_service_agent():
    """
    Exercise 2: Intelligent Customer Service Agent
    
    Task: Create a sophisticated customer service agent with access to CRM data,
    knowledge base, and escalation capabilities.
    
    Requirements:
    1. Integrate with CRM system to access customer information
    2. Build knowledge base search for common issues and solutions
    3. Implement sentiment analysis and escalation logic
    4. Create ticket management and follow-up capabilities
    5. Handle multi-turn conversations with context awareness
    
    Your implementation should:
    - Provide personalized responses based on customer history
    - Escalate complex issues appropriately
    - Maintain conversation context across interactions
    - Track customer satisfaction and resolution metrics
    """
    
    print("📞 Exercise 2: Customer Service Agent")
    print("-" * 60)
    
    # TODO: Implement customer service agent system
    # Hints:
    # 1. Create CRM integration tools
    # 2. Build knowledge base search functionality
    # 3. Implement sentiment analysis and escalation logic
    # 4. Design conversation flow management
    
    class CRMTool(BaseTool):
        """Tool for accessing customer relationship management data."""
        
        name = "crm_lookup"
        description = "Look up customer information, history, and account details"
        
        def _run(self, customer_id: str) -> str:
            """Look up customer information."""
            # TODO: Implement CRM lookup
            pass
        
        async def _arun(self, customer_id: str) -> str:
            """Async version of CRM lookup."""
            return self._run(customer_id)
    
    class KnowledgeBaseTool(BaseTool):
        """Tool for searching company knowledge base."""
        
        name = "knowledge_search"
        description = "Search knowledge base for solutions and information"
        
        def _run(self, query: str) -> str:
            """Search knowledge base."""
            # TODO: Implement knowledge base search
            pass
        
        async def _arun(self, query: str) -> str:
            """Async version of knowledge search."""
            return self._run(query)
    
    class TicketManagementTool(BaseTool):
        """Tool for managing customer service tickets."""
        
        name = "ticket_manager"
        description = "Create, update, and manage customer service tickets"
        
        def _run(self, action: str) -> str:
            """Manage tickets."""
            # TODO: Implement ticket management
            pass
        
        async def _arun(self, action: str) -> str:
            """Async version of ticket manager."""
            return self._run(action)
    
    class SentimentAnalyzer:
        """Analyze customer sentiment and determine escalation needs."""
        
        def analyze_sentiment(self, text: str) -> Dict[str, Any]:
            """Analyze customer sentiment."""
            # TODO: Implement sentiment analysis
            pass
        
        def should_escalate(self, conversation_history: List[str]) -> bool:
            """Determine if conversation should be escalated."""
            # TODO: Implement escalation logic
            pass
    
    class CustomerServiceAgent:
        """Comprehensive customer service agent."""
        
        def __init__(self, llm):
            self.llm = llm
            self.tools = [
                CRMTool(),
                KnowledgeBaseTool(),
                TicketManagementTool()
            ]
            self.sentiment_analyzer = SentimentAnalyzer()
            self.conversation_memory = {}
        
        def handle_customer_inquiry(self, customer_id: str, inquiry: str) -> Dict[str, Any]:
            """Handle customer inquiry with full context."""
            # TODO: Implement customer inquiry handling
            pass
        
        def escalate_to_human(self, customer_id: str, reason: str) -> Dict[str, Any]:
            """Escalate conversation to human agent."""
            # TODO: Implement escalation logic
            pass
        
        def track_satisfaction(self, customer_id: str, interaction_id: str, rating: int):
            """Track customer satisfaction metrics."""
            # TODO: Implement satisfaction tracking
            pass
    
    # Your implementation here
    print("📝 TODO: Implement customer service agent")
    print("💡 Consider: CRM integration, knowledge base, sentiment analysis, escalation")


def exercise_3_code_analysis_agent():
    """
    Exercise 3: Code Analysis and Review Agent
    
    Task: Implement an agent that can analyze codebases, run tests,
    identify issues, and suggest improvements.
    
    Requirements:
    1. Create tools for code analysis (syntax, style, complexity)
    2. Implement test execution and result analysis
    3. Build security vulnerability scanning capabilities
    4. Generate comprehensive code review reports
    5. Suggest specific improvements with code examples
    
    Your implementation should:
    - Analyze multiple programming languages
    - Provide actionable feedback and suggestions
    - Integrate with development workflows
    - Maintain code quality standards and best practices
    """
    
    print("💻 Exercise 3: Code Analysis Agent")
    print("-" * 60)
    
    # TODO: Implement code analysis agent
    # Hints:
    # 1. Create code parsing and analysis tools
    # 2. Implement test execution and reporting
    # 3. Build security scanning capabilities
    # 4. Design improvement suggestion system
    
    class CodeAnalysisTool(BaseTool):
        """Tool for analyzing code quality and structure."""
        
        name = "code_analyzer"
        description = "Analyze code for quality, complexity, and potential issues"
        
        def _run(self, code_input: str) -> str:
            """Analyze code quality."""
            # TODO: Implement code analysis
            pass
        
        async def _arun(self, code_input: str) -> str:
            """Async version of code analyzer."""
            return self._run(code_input)
    
    class TestRunnerTool(BaseTool):
        """Tool for executing tests and analyzing results."""
        
        name = "test_runner"
        description = "Execute tests and analyze test results"
        
        def _run(self, test_command: str) -> str:
            """Run tests and analyze results."""
            # TODO: Implement test execution
            pass
        
        async def _arun(self, test_command: str) -> str:
            """Async version of test runner."""
            return self._run(test_command)
    
    class SecurityScannerTool(BaseTool):
        """Tool for scanning code for security vulnerabilities."""
        
        name = "security_scanner"
        description = "Scan code for security vulnerabilities and issues"
        
        def _run(self, code_path: str) -> str:
            """Scan for security issues."""
            # TODO: Implement security scanning
            pass
        
        async def _arun(self, code_path: str) -> str:
            """Async version of security scanner."""
            return self._run(code_path)
    
    class CodeImprovementSuggester:
        """Generate specific code improvement suggestions."""
        
        def suggest_improvements(self, analysis_results: Dict) -> List[Dict]:
            """Generate improvement suggestions."""
            # TODO: Implement improvement suggestions
            pass
        
        def generate_code_examples(self, suggestion: Dict) -> str:
            """Generate code examples for suggestions."""
            # TODO: Implement code example generation
            pass
    
    class CodeAnalysisAgent:
        """Comprehensive code analysis agent."""
        
        def __init__(self, llm):
            self.llm = llm
            self.tools = [
                CodeAnalysisTool(),
                TestRunnerTool(),
                SecurityScannerTool()
            ]
            self.improvement_suggester = CodeImprovementSuggester()
        
        def analyze_codebase(self, codebase_path: str) -> Dict[str, Any]:
            """Perform comprehensive codebase analysis."""
            # TODO: Implement codebase analysis
            pass
        
        def generate_review_report(self, analysis_results: Dict) -> str:
            """Generate comprehensive code review report."""
            # TODO: Implement report generation
            pass
        
        def suggest_refactoring(self, code_issues: List[Dict]) -> List[Dict]:
            """Suggest specific refactoring opportunities."""
            # TODO: Implement refactoring suggestions
            pass
    
    # Your implementation here
    print("📝 TODO: Implement code analysis agent")
    print("💡 Consider: multi-language analysis, test integration, security scanning")


def exercise_4_multi_agent_collaboration():
    """
    Exercise 4: Multi-Agent Collaboration System
    
    Task: Design a system where specialized agents collaborate on complex tasks
    that require different expertise areas.
    
    Requirements:
    1. Create multiple specialized agents (researcher, analyst, writer, reviewer)
    2. Implement agent communication and coordination protocols
    3. Design task decomposition and delegation strategies
    4. Build consensus and conflict resolution mechanisms
    5. Create monitoring and progress tracking for collaborative tasks
    
    Your implementation should:
    - Coordinate multiple agents effectively
    - Handle task dependencies and sequencing
    - Resolve conflicts between agent recommendations
    - Provide visibility into collaborative processes
    """
    
    print("🤝 Exercise 4: Multi-Agent Collaboration")
    print("-" * 60)
    
    # TODO: Implement multi-agent collaboration system
    # Hints:
    # 1. Design agent communication protocols
    # 2. Implement task decomposition and delegation
    # 3. Build consensus mechanisms
    # 4. Create collaboration monitoring
    
    class AgentMessage(BaseModel):
        """Standard message format for agent communication."""
        sender: str
        recipient: str
        message_type: str
        content: Dict[str, Any]
        timestamp: datetime
        correlation_id: str
    
    class CollaborationProtocol:
        """Protocol for managing agent collaboration."""
        
        def __init__(self):
            self.message_queue = []
            self.active_tasks = {}
        
        def send_message(self, message: AgentMessage):
            """Send message between agents."""
            # TODO: Implement message routing
            pass
        
        def coordinate_task(self, task: Dict, agents: List[str]) -> str:
            """Coordinate task across multiple agents."""
            # TODO: Implement task coordination
            pass
        
        def resolve_conflicts(self, conflicting_results: List[Dict]) -> Dict:
            """Resolve conflicts between agent results."""
            # TODO: Implement conflict resolution
            pass
    
    class SpecializedAgent:
        """Base class for specialized agents."""
        
        def __init__(self, agent_id: str, specialization: str, llm):
            self.agent_id = agent_id
            self.specialization = specialization
            self.llm = llm
            self.tools = []
            self.collaboration_protocol = None
        
        def process_task(self, task: Dict) -> Dict[str, Any]:
            """Process assigned task."""
            # TODO: Implement task processing
            pass
        
        def collaborate_with(self, other_agent_id: str, task: Dict) -> Dict[str, Any]:
            """Collaborate with another agent."""
            # TODO: Implement collaboration logic
            pass
        
        def request_assistance(self, task: Dict, expertise_needed: str) -> Dict[str, Any]:
            """Request assistance from other agents."""
            # TODO: Implement assistance request
            pass
    
    class MultiAgentOrchestrator:
        """Orchestrate collaboration between multiple agents."""
        
        def __init__(self):
            self.agents = {}
            self.collaboration_protocol = CollaborationProtocol()
            self.task_monitor = TaskMonitor()
        
        def register_agent(self, agent: SpecializedAgent):
            """Register agent in the system."""
            # TODO: Implement agent registration
            pass
        
        def decompose_complex_task(self, task: Dict) -> List[Dict]:
            """Break complex task into subtasks."""
            # TODO: Implement task decomposition
            pass
        
        def execute_collaborative_task(self, task: Dict) -> Dict[str, Any]:
            """Execute task requiring multiple agents."""
            # TODO: Implement collaborative execution
            pass
        
        def monitor_progress(self, task_id: str) -> Dict[str, Any]:
            """Monitor progress of collaborative task."""
            # TODO: Implement progress monitoring
            pass
    
    class TaskMonitor:
        """Monitor and track collaborative task progress."""
        
        def track_task_progress(self, task_id: str, agent_id: str, progress: Dict):
            """Track individual agent progress."""
            # TODO: Implement progress tracking
            pass
        
        def detect_bottlenecks(self, task_id: str) -> List[Dict]:
            """Detect bottlenecks in collaboration."""
            # TODO: Implement bottleneck detection
            pass
        
        def generate_progress_report(self, task_id: str) -> Dict[str, Any]:
            """Generate progress report."""
            # TODO: Implement report generation
            pass
    
    # Your implementation here
    print("📝 TODO: Implement multi-agent collaboration system")
    print("💡 Consider: communication protocols, task delegation, conflict resolution")


def exercise_5_production_agent_platform():
    """
    Exercise 5: Production Agent Platform
    
    Task: Build a complete agent platform with monitoring, safety controls,
    user management, and administrative interfaces.
    
    Requirements:
    1. Create user authentication and authorization system
    2. Implement comprehensive monitoring and logging
    3. Build safety controls and content filtering
    4. Design cost tracking and resource management
    5. Create administrative dashboard and controls
    6. Implement agent versioning and deployment management
    
    Your implementation should:
    - Handle multiple users and agent instances
    - Provide real-time monitoring and alerts
    - Ensure safe and compliant agent operations
    - Scale to handle production workloads
    """
    
    print("🏭 Exercise 5: Production Agent Platform")
    print("-" * 60)
    
    # TODO: Implement production agent platform
    # Hints:
    # 1. Design scalable architecture for multiple users
    # 2. Implement comprehensive monitoring and alerting
    # 3. Build safety and compliance controls
    # 4. Create resource management and cost tracking
    
    class UserManager:
        """Manage user authentication and authorization."""
        
        def __init__(self):
            self.users = {}
            self.sessions = {}
        
        def authenticate_user(self, username: str, password: str) -> Optional[str]:
            """Authenticate user and return session token."""
            # TODO: Implement user authentication
            pass
        
        def authorize_agent_access(self, user_id: str, agent_id: str) -> bool:
            """Check if user can access specific agent."""
            # TODO: Implement authorization logic
            pass
        
        def track_user_usage(self, user_id: str, usage_data: Dict):
            """Track user usage for billing and limits."""
            # TODO: Implement usage tracking
            pass
    
    class SafetyController:
        """Control safety and compliance for agent operations."""
        
        def __init__(self):
            self.safety_rules = []
            self.content_filters = []
        
        def validate_request(self, user_id: str, request: Dict) -> bool:
            """Validate request against safety rules."""
            # TODO: Implement request validation
            pass
        
        def filter_response(self, response: Dict) -> Dict:
            """Filter response content for safety."""
            # TODO: Implement response filtering
            pass
        
        def log_safety_incident(self, incident: Dict):
            """Log safety incidents for review."""
            # TODO: Implement incident logging
            pass
    
    class ResourceManager:
        """Manage computational resources and costs."""
        
        def __init__(self):
            self.resource_pools = {}
            self.cost_tracking = {}
        
        def allocate_resources(self, agent_id: str, requirements: Dict) -> bool:
            """Allocate resources for agent execution."""
            # TODO: Implement resource allocation
            pass
        
        def track_costs(self, agent_id: str, usage: Dict):
            """Track costs for agent operations."""
            # TODO: Implement cost tracking
            pass
        
        def enforce_limits(self, user_id: str, requested_resources: Dict) -> bool:
            """Enforce resource and cost limits."""
            # TODO: Implement limit enforcement
            pass
    
    class MonitoringSystem:
        """Monitor agent performance and health."""
        
        def __init__(self):
            self.metrics = {}
            self.alerts = []
        
        def collect_metrics(self, agent_id: str, metrics: Dict):
            """Collect performance metrics."""
            # TODO: Implement metrics collection
            pass
        
        def check_health(self, agent_id: str) -> Dict[str, Any]:
            """Check agent health status."""
            # TODO: Implement health checking
            pass
        
        def generate_alerts(self, conditions: List[Dict]) -> List[Dict]:
            """Generate alerts based on conditions."""
            # TODO: Implement alert generation
            pass
    
    class AgentDeploymentManager:
        """Manage agent versions and deployments."""
        
        def __init__(self):
            self.agent_versions = {}
            self.deployments = {}
        
        def deploy_agent(self, agent_config: Dict, environment: str) -> str:
            """Deploy agent to specified environment."""
            # TODO: Implement agent deployment
            pass
        
        def rollback_deployment(self, deployment_id: str) -> bool:
            """Rollback to previous agent version."""
            # TODO: Implement rollback logic
            pass
        
        def manage_versions(self, agent_id: str, version_config: Dict):
            """Manage agent versions."""
            # TODO: Implement version management
            pass
    
    class ProductionAgentPlatform:
        """Complete production agent platform."""
        
        def __init__(self):
            self.user_manager = UserManager()
            self.safety_controller = SafetyController()
            self.resource_manager = ResourceManager()
            self.monitoring_system = MonitoringSystem()
            self.deployment_manager = AgentDeploymentManager()
        
        def process_user_request(self, user_id: str, request: Dict) -> Dict[str, Any]:
            """Process user request through full platform."""
            # TODO: Implement request processing pipeline
            pass
        
        def admin_dashboard_data(self) -> Dict[str, Any]:
            """Generate data for administrative dashboard."""
            # TODO: Implement dashboard data generation
            pass
        
        def generate_platform_reports(self) -> Dict[str, Any]:
            """Generate comprehensive platform reports."""
            # TODO: Implement report generation
            pass
    
    # Your implementation here
    print("📝 TODO: Implement production agent platform")
    print("💡 Consider: scalability, monitoring, safety, cost management")


def run_exercise(exercise_number: int):
    """Run a specific exercise."""
    exercises = {
        1: exercise_1_research_assistant_agent,
        2: exercise_2_customer_service_agent,
        3: exercise_3_code_analysis_agent,
        4: exercise_4_multi_agent_collaboration,
        5: exercise_5_production_agent_platform,
    }
    
    if exercise_number in exercises:
        print(f"\n🏋️ Running Exercise {exercise_number}")
        print("=" * 80)
        exercises[exercise_number]()
    else:
        print(f"❌ Exercise {exercise_number} not found. Available exercises: 1-5")


def run_all_exercises():
    """Run all exercises in sequence."""
    print("🦜🔗 LangChain Course - Lesson 8: Agents & Tools Exercises")
    print("=" * 80)
    
    if not llm:
        print("❌ LLM not available. Please check your setup.")
        print("Required: At least one LLM provider (OpenAI, Anthropic, etc.)")
        return
    
    print(f"✅ Using LLM: {type(llm).__name__}")
    
    for i in range(1, 6):
        try:
            run_exercise(i)
            if i < 5:
                input(f"\nPress Enter to continue to Exercise {i+1} (or Ctrl+C to exit)...")
        except KeyboardInterrupt:
            print(f"\n👋 Stopped at Exercise {i}")
            break
        except Exception as e:
            print(f"❌ Error in Exercise {i}: {e}")
            continue
    
    print("\n🎉 All exercises completed!")
    print("\n💡 Next Steps:")
    print("   • Review your implementations")
    print("   • Compare with solutions.py")
    print("   • Experiment with different agent architectures")
    print("   • Test with real-world scenarios")
    print("   • Consider production deployment strategies")


# Helper functions for exercises

def create_sample_agent_config() -> Dict[str, Any]:
    """Create sample agent configuration."""
    return {
        "max_iterations": 10,
        "max_execution_time": 300,
        "memory_type": "buffer",
        "safety_enabled": True,
        "cost_limit": 5.0,
        "tools": ["calculator", "search", "file_manager"],
        "verbose": True
    }


def create_mock_crm_data() -> List[Dict[str, Any]]:
    """Create mock CRM data for testing."""
    return [
        {
            "customer_id": "CUST001",
            "name": "John Smith",
            "email": "john.smith@email.com",
            "status": "Premium",
            "last_contact": "2024-01-15",
            "issues": ["Login problems", "Billing question"],
            "satisfaction_score": 4.2
        },
        {
            "customer_id": "CUST002",
            "name": "Jane Doe",
            "email": "jane.doe@email.com",
            "status": "Standard",
            "last_contact": "2024-01-10",
            "issues": ["Feature request"],
            "satisfaction_score": 3.8
        }
    ]


def create_sample_code_for_analysis() -> str:
    """Create sample code for analysis exercises."""
    return '''
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance
    
    def deposit(self, amount):
        self.balance += amount
        return self.balance
    
    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
            return self.balance
        else:
            return "Insufficient funds"
'''


def simulate_agent_communication(agents: List[str], task: str) -> List[Dict]:
    """Simulate communication between agents."""
    messages = []
    for agent in agents:
        message = {
            "agent": agent,
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "status": "received"
        }
        messages.append(message)
    return messages


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Agents & Tools Exercises")
    parser.add_argument("--exercise", type=int, help="Run specific exercise (1-5)")
    parser.add_argument("--all", action="store_true", help="Run all exercises")
    
    args = parser.parse_args()
    
    if args.exercise:
        run_exercise(args.exercise)
    elif args.all:
        run_all_exercises()
    else:
        print("🏋️ Agents & Tools Practice Exercises")
        print("=" * 50)
        print("Usage:")
        print("  python exercises.py --exercise N  (run exercise N)")
        print("  python exercises.py --all        (run all exercises)")
        print("\nAvailable exercises:")
        print("  1. Research Assistant Agent")
        print("  2. Customer Service Agent")
        print("  3. Code Analysis Agent")
        print("  4. Multi-Agent Collaboration")
        print("  5. Production Agent Platform")