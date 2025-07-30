#!/usr/bin/env python3
"""
Lesson 8: Agents & Tools - Solution Implementations

Reference implementations for all agent and tool development exercises.
These solutions demonstrate best practices and production-ready patterns.

Study these implementations to understand optimal approaches to agent system design.
"""

import os
import sys
import time
import json
import asyncio
import logging
import hashlib
from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Add shared resources to path
sys.path.append('../shared-resources')

# Load environment variables
load_dotenv()

# LangChain imports
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks import get_openai_callback
from langchain.schema.runnable import RunnablePassthrough
from pydantic import BaseModel, Field, validator
import structlog

# External imports
import requests
import sqlite3
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import subprocess
import re

# Shared utilities
from utils.llm_setup import setup_llm_providers, get_preferred_llm
from utils.error_handling import safe_llm_call

# Set up providers
providers = setup_llm_providers()
llm = get_preferred_llm(providers, prefer_chat=True) if providers else None


@dataclass
class ResearchResult:
    """Container for research results with metadata."""
    content: str
    source: str
    confidence: float
    timestamp: datetime
    citations: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


@dataclass
class CustomerProfile:
    """Customer profile data structure."""
    customer_id: str
    name: str
    email: str
    status: str
    history: List[Dict] = field(default_factory=list)
    satisfaction_score: float = 0.0
    preferences: Dict = field(default_factory=dict)


@dataclass
class CodeAnalysisResult:
    """Container for code analysis results."""
    file_path: str
    issues: List[Dict]
    complexity_score: float
    test_coverage: float
    security_issues: List[Dict]
    suggestions: List[Dict]


def solution_1_research_assistant_agent():
    """
    Solution 1: Research Assistant Agent
    """
    print("🔍 Solution 1: Research Assistant Agent")
    print("-" * 70)
    
    class WebResearchTool(BaseTool):
        """Advanced web research tool with multiple sources."""
        
        name = "web_research"
        description = """
        Perform comprehensive web research on a topic using multiple sources.
        Input: topic to research
        Returns: structured research results with sources and confidence scores
        """
        
        def __init__(self):
            super().__init__()
            self.search_tool = None
            try:
                self.search_tool = DuckDuckGoSearchRun()
            except:
                pass
        
        def _run(self, topic: str) -> str:
            """Perform comprehensive web research."""
            try:
                results = []
                
                # Primary search
                if self.search_tool:
                    search_results = self.search_tool.run(topic)
                    results.append({
                        "source": "web_search",
                        "content": search_results,
                        "confidence": 0.8
                    })
                
                # Wikipedia search (mock implementation)
                wiki_result = self._search_wikipedia(topic)
                if wiki_result:
                    results.append(wiki_result)
                
                # News search (mock implementation)
                news_result = self._search_news(topic)
                if news_result:
                    results.append(news_result)
                
                # Format results
                formatted_results = self._format_research_results(results)
                return formatted_results
                
            except Exception as e:
                return f"Research error: {str(e)}"
        
        def _search_wikipedia(self, topic: str) -> Optional[Dict]:
            """Mock Wikipedia search."""
            return {
                "source": "wikipedia",
                "content": f"Wikipedia information about {topic}: [Mock encyclopedia content]",
                "confidence": 0.9
            }
        
        def _search_news(self, topic: str) -> Optional[Dict]:
            """Mock news search."""
            return {
                "source": "news",
                "content": f"Recent news about {topic}: [Mock news content]",
                "confidence": 0.7
            }
        
        def _format_research_results(self, results: List[Dict]) -> str:
            """Format research results into structured text."""
            formatted = f"Research Results for '{results[0].get('topic', 'Unknown')}':\n\n"
            
            for i, result in enumerate(results, 1):
                formatted += f"Source {i}: {result['source'].title()}\n"
                formatted += f"Confidence: {result['confidence']:.1%}\n"
                formatted += f"Content: {result['content'][:200]}...\n\n"
            
            return formatted
        
        async def _arun(self, topic: str) -> str:
            """Async version of research tool."""
            return self._run(topic)
    
    class FactCheckTool(BaseTool):
        """Advanced fact-checking tool."""
        
        name = "fact_checker"
        description = """
        Verify facts and check information accuracy using multiple validation methods.
        Input: statement to fact-check
        Returns: verification result with confidence score and sources
        """
        
        def _run(self, statement: str) -> str:
            """Perform fact checking on a statement."""
            try:
                # Multiple validation approaches
                validations = []
                
                # Cross-reference check
                cross_ref_score = self._cross_reference_check(statement)
                validations.append(("cross_reference", cross_ref_score))
                
                # Source reliability check
                source_score = self._check_source_reliability(statement)
                validations.append(("source_reliability", source_score))
                
                # Logical consistency check
                logic_score = self._check_logical_consistency(statement)
                validations.append(("logical_consistency", logic_score))
                
                # Calculate overall confidence
                overall_confidence = np.mean([score for _, score in validations])
                
                # Generate verification report
                report = self._generate_verification_report(statement, validations, overall_confidence)
                return report
                
            except Exception as e:
                return f"Fact checking error: {str(e)}"
        
        def _cross_reference_check(self, statement: str) -> float:
            """Check statement against multiple sources."""
            # Mock implementation - in practice, would check against databases
            return 0.85  # Mock confidence score
        
        def _check_source_reliability(self, statement: str) -> float:
            """Check reliability of sources."""
            # Mock implementation
            return 0.75
        
        def _check_logical_consistency(self, statement: str) -> float:
            """Check logical consistency of statement."""
            # Mock implementation
            return 0.90
        
        def _generate_verification_report(self, statement: str, validations: List, confidence: float) -> str:
            """Generate comprehensive verification report."""
            report = f"Fact Check Report for: '{statement[:100]}...'\n\n"
            report += f"Overall Confidence: {confidence:.1%}\n\n"
            report += "Validation Results:\n"
            
            for method, score in validations:
                status = "✓ VERIFIED" if score > 0.7 else "⚠ QUESTIONABLE" if score > 0.4 else "✗ DISPUTED"
                report += f"  {method.replace('_', ' ').title()}: {score:.1%} - {status}\n"
            
            if confidence > 0.8:
                report += "\n✅ Statement appears to be accurate based on available evidence."
            elif confidence > 0.6:
                report += "\n⚠️ Statement partially verified - some aspects may need clarification."
            else:
                report += "\n❌ Statement disputed or lacks sufficient evidence."
            
            return report
        
        async def _arun(self, statement: str) -> str:
            """Async version of fact checker."""
            return self._run(statement)
    
    class ReportGeneratorTool(BaseTool):
        """Generate structured research reports."""
        
        name = "report_generator"
        description = """
        Generate comprehensive research reports with proper citations and structure.
        Input: research data and topic
        Returns: formatted research report
        """
        
        def _run(self, research_data: str) -> str:
            """Generate structured research report."""
            try:
                # Parse research data
                sections = self._parse_research_data(research_data)
                
                # Generate report sections
                report = self._build_report_structure(sections)
                
                return report
                
            except Exception as e:
                return f"Report generation error: {str(e)}"
        
        def _parse_research_data(self, data: str) -> Dict[str, Any]:
            """Parse research data into structured sections."""
            # Mock parsing - in practice would use NLP techniques
            return {
                "title": "Research Report",
                "executive_summary": "Summary of key findings...",
                "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
                "sources": ["Source 1", "Source 2", "Source 3"],
                "methodology": "Research methodology used...",
                "confidence_scores": [0.85, 0.90, 0.75]
            }
        
        def _build_report_structure(self, sections: Dict) -> str:
            """Build formatted report structure."""
            report = f"# {sections['title']}\n\n"
            report += f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            report += "## Executive Summary\n"
            report += f"{sections['executive_summary']}\n\n"
            
            report += "## Key Findings\n"
            for i, finding in enumerate(sections['key_findings'], 1):
                confidence = sections['confidence_scores'][i-1] if i-1 < len(sections['confidence_scores']) else 0.8
                report += f"{i}. {finding} (Confidence: {confidence:.1%})\n"
            
            report += "\n## Methodology\n"
            report += f"{sections['methodology']}\n\n"
            
            report += "## Sources\n"
            for i, source in enumerate(sections['sources'], 1):
                report += f"[{i}] {source}\n"
            
            report += "\n---\n*Report generated using automated research assistant*"
            
            return report
        
        async def _arun(self, research_data: str) -> str:
            """Async version of report generator."""
            return self._run(research_data)
    
    class ResearchAssistantAgent:
        """Comprehensive research assistant agent."""
        
        def __init__(self, llm):
            self.llm = llm
            self.tools = [
                WebResearchTool(),
                FactCheckTool(),
                ReportGeneratorTool()
            ]
            self.research_history = []
            self.setup_agent()
        
        def setup_agent(self):
            """Set up the research agent."""
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert research assistant. Your role is to:
                1. Conduct comprehensive research on topics using available tools
                2. Verify information accuracy through fact-checking
                3. Generate well-structured research reports
                4. Provide confidence scores for all findings
                
                Always use multiple tools to gather comprehensive information."""),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            agent = create_react_agent(self.llm, self.tools, prompt)
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                max_iterations=10,
                return_intermediate_steps=True
            )
        
        def conduct_research(self, topic: str, depth: str = "comprehensive") -> Dict[str, Any]:
            """Conduct comprehensive research on a topic."""
            try:
                research_prompt = f"""
                Conduct {depth} research on: {topic}
                
                Please:
                1. Use web_research to gather information from multiple sources
                2. Fact-check key claims using fact_checker
                3. Generate a comprehensive report using report_generator
                
                Provide confidence scores for all major findings.
                """
                
                with get_openai_callback() as cb:
                    result = self.agent_executor.invoke({"input": research_prompt})
                
                # Store research in history
                research_record = {
                    "topic": topic,
                    "timestamp": datetime.now(),
                    "result": result,
                    "cost": cb.total_cost
                }
                self.research_history.append(research_record)
                
                return {
                    "research_output": result["output"],
                    "intermediate_steps": result["intermediate_steps"],
                    "cost": cb.total_cost,
                    "confidence_score": self._calculate_research_confidence(result)
                }
                
            except Exception as e:
                return {
                    "error": str(e),
                    "research_output": "Research failed to complete"
                }
        
        def validate_findings(self, findings: List[str]) -> List[Dict]:
            """Validate a list of research findings."""
            validated_findings = []
            
            for finding in findings:
                try:
                    fact_check_result = self.agent_executor.invoke({
                        "input": f"Use fact_checker to verify: {finding}"
                    })
                    
                    validated_findings.append({
                        "finding": finding,
                        "validation": fact_check_result["output"],
                        "verified": "VERIFIED" in fact_check_result["output"].upper()
                    })
                    
                except Exception as e:
                    validated_findings.append({
                        "finding": finding,
                        "validation": f"Validation failed: {e}",
                        "verified": False
                    })
            
            return validated_findings
        
        def generate_report(self, topic: str, findings: List[str]) -> str:
            """Generate final research report."""
            try:
                # Prepare research data
                research_data = {
                    "topic": topic,
                    "findings": findings,
                    "timestamp": datetime.now(),
                    "methodology": "Multi-source research with fact verification"
                }
                
                report_prompt = f"""
                Use report_generator to create a comprehensive research report for:
                Topic: {topic}
                
                Include the following findings:
                {chr(10).join(f"- {finding}" for finding in findings)}
                """
                
                result = self.agent_executor.invoke({"input": report_prompt})
                return result["output"]
                
            except Exception as e:
                return f"Report generation failed: {e}"
        
        def _calculate_research_confidence(self, result: Dict) -> float:
            """Calculate overall confidence score for research."""
            # Mock implementation - would analyze result quality
            return 0.85
        
        def get_research_summary(self) -> Dict[str, Any]:
            """Get summary of all research conducted."""
            return {
                "total_research_sessions": len(self.research_history),
                "topics_researched": [r["topic"] for r in self.research_history],
                "total_cost": sum(r.get("cost", 0) for r in self.research_history),
                "average_confidence": 0.85  # Mock average
            }
    
    # Demo the research assistant
    if llm:
        assistant = ResearchAssistantAgent(llm)
        
        # Test research topics
        test_topics = [
            "Impact of artificial intelligence on healthcare",
            "Climate change solutions for urban areas",
            "Future of renewable energy technology"
        ]
        
        print("🧪 Testing Research Assistant Agent:")
        
        for topic in test_topics[:1]:  # Test one topic for demo
            print(f"\n📚 Researching: {topic}")
            
            result = assistant.conduct_research(topic, depth="comprehensive")
            
            if "error" in result:
                print(f"❌ Research failed: {result['error']}")
            else:
                print(f"✅ Research completed")
                print(f"📊 Confidence: {result['confidence_score']:.1%}")
                print(f"💰 Cost: ${result['cost']:.4f}")
                print(f"📄 Output preview: {result['research_output'][:200]}...")
        
        # Show research summary
        summary = assistant.get_research_summary()
        print(f"\n📈 Research Summary:")
        print(f"   Sessions: {summary['total_research_sessions']}")
        print(f"   Total cost: ${summary['total_cost']:.4f}")


def solution_2_customer_service_agent():
    """
    Solution 2: Customer Service Agent
    """
    print("\n📞 Solution 2: Customer Service Agent")
    print("-" * 70)
    
    class CRMTool(BaseTool):
        """Advanced CRM integration tool."""
        
        name = "crm_lookup"
        description = """
        Look up customer information, history, and account details from CRM system.
        Input: customer_id or email
        Returns: comprehensive customer profile and interaction history
        """
        
        def __init__(self):
            super().__init__()
            self.customers_db = self._initialize_mock_crm()
        
        def _initialize_mock_crm(self) -> Dict[str, CustomerProfile]:
            """Initialize mock CRM database."""
            return {
                "CUST001": CustomerProfile(
                    customer_id="CUST001",
                    name="John Smith",
                    email="john.smith@email.com",
                    status="Premium",
                    history=[
                        {"date": "2024-01-15", "type": "support", "issue": "Login problems", "resolved": True},
                        {"date": "2024-01-10", "type": "billing", "issue": "Payment inquiry", "resolved": True}
                    ],
                    satisfaction_score=4.2,
                    preferences={"contact_method": "email", "language": "en"}
                ),
                "CUST002": CustomerProfile(
                    customer_id="CUST002",
                    name="Jane Doe", 
                    email="jane.doe@email.com",
                    status="Standard",
                    history=[
                        {"date": "2024-01-12", "type": "support", "issue": "Feature request", "resolved": False},
                        {"date": "2024-01-05", "type": "complaint", "issue": "Service issue", "resolved": True}
                    ],
                    satisfaction_score=3.8,
                    preferences={"contact_method": "phone", "language": "en"}
                )
            }
        
        def _run(self, customer_identifier: str) -> str:
            """Look up customer information."""
            try:
                # Find customer by ID or email
                customer = None
                for cust_id, profile in self.customers_db.items():
                    if (customer_identifier == cust_id or 
                        customer_identifier.lower() == profile.email.lower()):
                        customer = profile
                        break
                
                if not customer:
                    return f"Customer not found: {customer_identifier}"
                
                # Format customer information
                info = f"Customer Profile:\n"
                info += f"ID: {customer.customer_id}\n"
                info += f"Name: {customer.name}\n"
                info += f"Email: {customer.email}\n"
                info += f"Status: {customer.status}\n"
                info += f"Satisfaction Score: {customer.satisfaction_score}/5.0\n\n"
                
                info += "Recent History:\n"
                for item in customer.history[-3:]:  # Last 3 interactions
                    status = "✅ Resolved" if item["resolved"] else "⏳ Open"
                    info += f"- {item['date']}: {item['type'].title()} - {item['issue']} ({status})\n"
                
                info += f"\nPreferences: {customer.preferences}"
                
                return info
                
            except Exception as e:
                return f"CRM lookup error: {str(e)}"
        
        async def _arun(self, customer_identifier: str) -> str:
            """Async version of CRM lookup."""
            return self._run(customer_identifier)
    
    class KnowledgeBaseTool(BaseTool):
        """Knowledge base search tool."""
        
        name = "knowledge_search"
        description = """
        Search company knowledge base for solutions, policies, and information.
        Input: search query or issue description
        Returns: relevant knowledge base articles and solutions
        """
        
        def __init__(self):
            super().__init__()
            self.knowledge_base = self._initialize_knowledge_base()
        
        def _initialize_knowledge_base(self) -> List[Dict]:
            """Initialize mock knowledge base."""
            return [
                {
                    "id": "KB001",
                    "title": "Password Reset Procedure",
                    "category": "Account Issues",
                    "content": "To reset your password: 1. Go to login page 2. Click 'Forgot Password' 3. Enter email 4. Check email for reset link",
                    "tags": ["password", "reset", "login", "account"],
                    "rating": 4.5,
                    "updated": "2024-01-01"
                },
                {
                    "id": "KB002", 
                    "title": "Billing Cycle Information",
                    "category": "Billing",
                    "content": "Billing cycles run monthly from the signup date. Charges appear 2-3 days before renewal. Payment methods can be updated in account settings.",
                    "tags": ["billing", "payment", "cycle", "renewal"],
                    "rating": 4.2,
                    "updated": "2024-01-15"
                },
                {
                    "id": "KB003",
                    "title": "Feature Request Process",
                    "category": "General",
                    "content": "Feature requests can be submitted through the feedback portal. Popular requests are prioritized for development roadmap consideration.",
                    "tags": ["feature", "request", "feedback", "development"],
                    "rating": 3.8,
                    "updated": "2024-01-10"
                },
                {
                    "id": "KB004",
                    "title": "Refund Policy",
                    "category": "Billing",
                    "content": "Refunds are available within 30 days of purchase for annual plans. Monthly plans can be cancelled anytime without refund for current month.",
                    "tags": ["refund", "policy", "cancellation", "billing"],
                    "rating": 4.0,
                    "updated": "2024-01-05"
                }
            ]
        
        def _run(self, query: str) -> str:
            """Search knowledge base for relevant articles."""
            try:
                # Simple keyword-based search
                query_words = query.lower().split()
                scored_articles = []
                
                for article in self.knowledge_base:
                    score = 0
                    # Check title
                    if any(word in article["title"].lower() for word in query_words):
                        score += 3
                    # Check tags
                    for tag in article["tags"]:
                        if any(word in tag for word in query_words):
                            score += 2
                    # Check content
                    if any(word in article["content"].lower() for word in query_words):
                        score += 1
                    
                    if score > 0:
                        scored_articles.append((score, article))
                
                # Sort by score
                scored_articles.sort(key=lambda x: x[0], reverse=True)
                
                if not scored_articles:
                    return f"No relevant articles found for: {query}"
                
                # Format results
                results = f"Knowledge Base Results for: '{query}'\n\n"
                for score, article in scored_articles[:3]:  # Top 3 results
                    results += f"📄 {article['title']} (Score: {score})\n"
                    results += f"Category: {article['category']}\n"
                    results += f"Content: {article['content'][:150]}...\n"
                    results += f"Rating: {article['rating']}/5.0\n\n"
                
                return results
                
            except Exception as e:
                return f"Knowledge base search error: {str(e)}"
        
        async def _arun(self, query: str) -> str:
            """Async version of knowledge search."""
            return self._run(query)
    
    class TicketManagementTool(BaseTool):
        """Ticket management and tracking tool."""
        
        name = "ticket_manager"
        description = """
        Create, update, and manage customer service tickets.
        Input: action:ticket_id:details (e.g., "create::New issue description" or "update:TKT001:Resolution provided")
        Returns: ticket status and information
        """
        
        def __init__(self):
            super().__init__()
            self.tickets = {}
            self.ticket_counter = 1
        
        def _run(self, action_string: str) -> str:
            """Manage customer service tickets."""
            try:
                parts = action_string.split(":", 2)
                action = parts[0].lower()
                
                if action == "create":
                    return self._create_ticket(parts[2] if len(parts) > 2 else "")
                elif action == "update":
                    ticket_id = parts[1] if len(parts) > 1 else ""
                    details = parts[2] if len(parts) > 2 else ""
                    return self._update_ticket(ticket_id, details)
                elif action == "list":
                    return self._list_tickets()
                elif action == "get":
                    ticket_id = parts[1] if len(parts) > 1 else ""
                    return self._get_ticket(ticket_id)
                else:
                    return f"Unknown action: {action}. Available: create, update, list, get"
                
            except Exception as e:
                return f"Ticket management error: {str(e)}"
        
        def _create_ticket(self, description: str) -> str:
            """Create a new ticket."""
            ticket_id = f"TKT{self.ticket_counter:03d}"
            self.ticket_counter += 1
            
            ticket = {
                "id": ticket_id,
                "description": description,
                "status": "Open",
                "priority": "Medium",
                "created": datetime.now().isoformat(),
                "updated": datetime.now().isoformat(),
                "history": [f"Ticket created: {description}"]
            }
            
            self.tickets[ticket_id] = ticket
            
            return f"✅ Ticket created: {ticket_id}\nDescription: {description}\nStatus: Open"
        
        def _update_ticket(self, ticket_id: str, details: str) -> str:
            """Update an existing ticket."""
            if ticket_id not in self.tickets:
                return f"❌ Ticket not found: {ticket_id}"
            
            ticket = self.tickets[ticket_id]
            ticket["updated"] = datetime.now().isoformat()
            ticket["history"].append(f"Update: {details}")
            
            # Check if this is a resolution
            if any(word in details.lower() for word in ["resolved", "solved", "fixed", "completed"]):
                ticket["status"] = "Resolved"
            
            return f"✅ Ticket {ticket_id} updated\nLatest update: {details}\nStatus: {ticket['status']}"
        
        def _list_tickets(self) -> str:
            """List all tickets."""
            if not self.tickets:
                return "No tickets found."
            
            ticket_list = "Current Tickets:\n"
            for ticket_id, ticket in self.tickets.items():
                ticket_list += f"- {ticket_id}: {ticket['status']} - {ticket['description'][:50]}...\n"
            
            return ticket_list
        
        def _get_ticket(self, ticket_id: str) -> str:
            """Get detailed ticket information."""
            if ticket_id not in self.tickets:
                return f"❌ Ticket not found: {ticket_id}"
            
            ticket = self.tickets[ticket_id]
            info = f"Ticket Details: {ticket_id}\n"
            info += f"Status: {ticket['status']}\n"
            info += f"Priority: {ticket['priority']}\n"
            info += f"Created: {ticket['created']}\n"
            info += f"Updated: {ticket['updated']}\n"
            info += f"Description: {ticket['description']}\n\n"
            info += "History:\n"
            for entry in ticket['history']:
                info += f"- {entry}\n"
            
            return info
        
        async def _arun(self, action_string: str) -> str:
            """Async version of ticket manager."""
            return self._run(action_string)
    
    class SentimentAnalyzer:
        """Advanced sentiment analysis for customer interactions."""
        
        def __init__(self, llm):
            self.llm = llm
            self.escalation_threshold = -0.3  # Negative sentiment threshold
        
        def analyze_sentiment(self, text: str) -> Dict[str, Any]:
            """Analyze customer sentiment using LLM."""
            try:
                sentiment_prompt = f"""
                Analyze the sentiment of this customer message and provide:
                1. Overall sentiment (positive/neutral/negative)
                2. Sentiment score (-1.0 to 1.0)
                3. Emotional indicators (frustrated, angry, satisfied, etc.)
                4. Urgency level (low/medium/high)
                
                Customer message: "{text}"
                
                Respond in this format:
                Sentiment: [positive/neutral/negative]
                Score: [number between -1.0 and 1.0]
                Emotions: [list of emotions]
                Urgency: [low/medium/high]
                """
                
                response = safe_llm_call(self.llm, sentiment_prompt)
                
                if response:
                    # Parse response (simplified parsing)
                    lines = response.strip().split('\n')
                    result = {
                        "sentiment": "neutral",
                        "score": 0.0,
                        "emotions": [],
                        "urgency": "medium",
                        "raw_analysis": response
                    }
                    
                    for line in lines:
                        if line.startswith("Sentiment:"):
                            result["sentiment"] = line.split(":", 1)[1].strip()
                        elif line.startswith("Score:"):
                            try:
                                result["score"] = float(line.split(":", 1)[1].strip())
                            except:
                                pass
                        elif line.startswith("Urgency:"):
                            result["urgency"] = line.split(":", 1)[1].strip()
                    
                    return result
                
                return {"sentiment": "neutral", "score": 0.0, "emotions": [], "urgency": "medium"}
                
            except Exception as e:
                return {"error": str(e), "sentiment": "unknown", "score": 0.0}
        
        def should_escalate(self, conversation_history: List[str]) -> Dict[str, Any]:
            """Determine if conversation should be escalated."""
            try:
                # Analyze recent messages
                recent_sentiments = []
                for message in conversation_history[-3:]:  # Last 3 messages
                    sentiment = self.analyze_sentiment(message)
                    recent_sentiments.append(sentiment.get("score", 0.0))
                
                avg_sentiment = np.mean(recent_sentiments) if recent_sentiments else 0.0
                
                # Escalation criteria
                escalate = False
                reasons = []
                
                if avg_sentiment < self.escalation_threshold:
                    escalate = True
                    reasons.append("Negative sentiment detected")
                
                if len(conversation_history) > 5:
                    escalate = True
                    reasons.append("Long conversation without resolution")
                
                # Check for escalation keywords
                escalation_keywords = ["manager", "supervisor", "escalate", "complaint", "legal"]
                recent_text = " ".join(conversation_history[-2:]).lower()
                if any(keyword in recent_text for keyword in escalation_keywords):
                    escalate = True
                    reasons.append("Escalation keywords detected")
                
                return {
                    "escalate": escalate,
                    "reasons": reasons,
                    "sentiment_score": avg_sentiment,
                    "conversation_length": len(conversation_history)
                }
                
            except Exception as e:
                return {"escalate": False, "error": str(e)}
    
    class CustomerServiceAgent:
        """Comprehensive customer service agent."""
        
        def __init__(self, llm):
            self.llm = llm
            self.tools = [
                CRMTool(),
                KnowledgeBaseTool(),
                TicketManagementTool()
            ]
            self.sentiment_analyzer = SentimentAnalyzer(llm)
            self.conversation_memory = defaultdict(list)
            self.setup_agent()
        
        def setup_agent(self):
            """Set up the customer service agent."""
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a professional customer service representative. Your goals are to:
                1. Provide excellent customer service with empathy and professionalism
                2. Resolve customer issues efficiently using available tools
                3. Access customer information via CRM when needed
                4. Search knowledge base for solutions
                5. Create and manage support tickets
                6. Escalate to human agents when appropriate
                
                Always be helpful, patient, and solution-focused."""),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            agent = create_react_agent(self.llm, self.tools, prompt)
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                max_iterations=8,
                return_intermediate_steps=True
            )
        
        def handle_customer_inquiry(self, customer_id: str, inquiry: str, 
                                  conversation_id: str = None) -> Dict[str, Any]:
            """Handle customer inquiry with full context."""
            try:
                if not conversation_id:
                    conversation_id = f"conv_{int(time.time())}"
                
                # Add to conversation history
                self.conversation_memory[conversation_id].append(inquiry)
                
                # Analyze sentiment
                sentiment = self.sentiment_analyzer.analyze_sentiment(inquiry)
                
                # Check for escalation
                escalation_check = self.sentiment_analyzer.should_escalate(
                    self.conversation_memory[conversation_id]
                )
                
                # Prepare context-aware prompt
                context_prompt = f"""
                Customer ID: {customer_id}
                Customer inquiry: {inquiry}
                
                Sentiment analysis: {sentiment['sentiment']} (score: {sentiment['score']:.2f})
                
                Please:
                1. Look up customer information using crm_lookup
                2. Search for relevant solutions using knowledge_search if needed
                3. Provide a helpful, empathetic response
                4. Create a ticket if the issue requires follow-up
                
                Conversation history length: {len(self.conversation_memory[conversation_id])}
                """
                
                with get_openai_callback() as cb:
                    result = self.agent_executor.invoke({"input": context_prompt})
                
                response = {
                    "response": result["output"],
                    "conversation_id": conversation_id,
                    "sentiment": sentiment,
                    "escalation_needed": escalation_check["escalate"],
                    "escalation_reasons": escalation_check.get("reasons", []),
                    "cost": cb.total_cost,
                    "intermediate_steps": result["intermediate_steps"]
                }
                
                # Auto-escalate if needed
                if escalation_check["escalate"]:
                    escalation_result = self.escalate_to_human(
                        customer_id, 
                        f"Auto-escalation: {', '.join(escalation_check['reasons'])}"
                    )
                    response["escalation_result"] = escalation_result
                
                return response
                
            except Exception as e:
                return {
                    "error": str(e),
                    "response": "I apologize, but I'm experiencing technical difficulties. Please try again or contact support directly."
                }
        
        def escalate_to_human(self, customer_id: str, reason: str) -> Dict[str, Any]:
            """Escalate conversation to human agent."""
            escalation_id = f"ESC_{int(time.time())}"
            
            escalation_data = {
                "escalation_id": escalation_id,
                "customer_id": customer_id,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "priority": "High" if "angry" in reason.lower() or "legal" in reason.lower() else "Medium",
                "status": "Pending Human Review"
            }
            
            # In production, this would integrate with human agent routing system
            print(f"🚨 ESCALATION TRIGGERED: {escalation_id}")
            print(f"Customer: {customer_id}")
            print(f"Reason: {reason}")
            
            return escalation_data
        
        def track_satisfaction(self, customer_id: str, interaction_id: str, rating: int):
            """Track customer satisfaction metrics."""
            satisfaction_data = {
                "customer_id": customer_id,
                "interaction_id": interaction_id,
                "rating": rating,
                "timestamp": datetime.now().isoformat()
            }
            
            # In production, this would be stored in analytics database
            print(f"📊 Satisfaction tracked: {customer_id} rated {rating}/5")
            
            return satisfaction_data
        
        def get_performance_metrics(self) -> Dict[str, Any]:
            """Get agent performance metrics."""
            return {
                "total_conversations": len(self.conversation_memory),
                "average_conversation_length": np.mean([
                    len(conv) for conv in self.conversation_memory.values()
                ]) if self.conversation_memory else 0,
                "escalation_rate": 0.15,  # Mock rate
                "resolution_rate": 0.85,  # Mock rate
                "customer_satisfaction": 4.2  # Mock average
            }
    
    # Demo the customer service agent
    if llm:
        service_agent = CustomerServiceAgent(llm)
        
        # Test customer inquiries
        test_scenarios = [
            ("CUST001", "I can't log into my account and I'm getting frustrated!"),
            ("CUST002", "Hi, I'd like to know about your refund policy please."),
            ("CUST001", "This is ridiculous! I want to speak to a manager right now!")
        ]
        
        print("🧪 Testing Customer Service Agent:")
        
        for customer_id, inquiry in test_scenarios[:1]:  # Test one scenario for demo
            print(f"\n📞 Customer {customer_id}: {inquiry}")
            
            result = service_agent.handle_customer_inquiry(customer_id, inquiry)
            
            if "error" in result:
                print(f"❌ Service failed: {result['error']}")
            else:
                print(f"🤖 Agent response: {result['response'][:200]}...")
                print(f"😊 Sentiment: {result['sentiment']['sentiment']} ({result['sentiment']['score']:.2f})")
                
                if result['escalation_needed']:
                    print(f"🚨 Escalation triggered: {result['escalation_reasons']}")
                
                print(f"💰 Cost: ${result['cost']:.4f}")
        
        # Show performance metrics
        metrics = service_agent.get_performance_metrics()
        print(f"\n📈 Agent Performance:")
        print(f"   Resolution rate: {metrics['resolution_rate']:.1%}")
        print(f"   Customer satisfaction: {metrics['customer_satisfaction']}/5.0")
        print(f"   Escalation rate: {metrics['escalation_rate']:.1%}")


def solution_3_code_analysis_agent():
    """
    Solution 3: Code Analysis Agent
    """
    print("\n💻 Solution 3: Code Analysis Agent")
    print("-" * 70)
    
    class CodeAnalysisTool(BaseTool):
        """Advanced code analysis tool."""
        
        name = "code_analyzer"
        description = """
        Analyze code for quality, complexity, security issues, and potential improvements.
        Input: code content or file path
        Returns: comprehensive analysis report with scores and recommendations
        """
        
        def _run(self, code_input: str) -> str:
            """Analyze code quality and structure."""
            try:
                # Parse code (simplified Python analysis)
                analysis = self._analyze_python_code(code_input)
                return self._format_analysis_report(analysis)
                
            except Exception as e:
                return f"Code analysis error: {str(e)}"
        
        def _analyze_python_code(self, code: str) -> Dict[str, Any]:
            """Analyze Python code structure and quality."""
            try:
                tree = ast.parse(code)
                
                analysis = {
                    "lines_of_code": len(code.split('\n')),
                    "functions": [],
                    "classes": [],
                    "complexity_score": 0,
                    "issues": [],
                    "security_concerns": []
                }
                
                # Analyze AST
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        func_analysis = self._analyze_function(node, code)
                        analysis["functions"].append(func_analysis)
                    elif isinstance(node, ast.ClassDef):
                        class_analysis = self._analyze_class(node)
                        analysis["classes"].append(class_analysis)
                
                # Calculate complexity
                analysis["complexity_score"] = self._calculate_complexity(tree)
                
                # Check for common issues
                analysis["issues"] = self._find_code_issues(code, tree)
                
                # Security analysis
                analysis["security_concerns"] = self._check_security_issues(code, tree)
                
                return analysis
                
            except SyntaxError as e:
                return {
                    "error": f"Syntax error: {e}",
                    "issues": ["Code contains syntax errors"],
                    "complexity_score": 0
                }
        
        def _analyze_function(self, node: ast.FunctionDef, code: str) -> Dict:
            """Analyze individual function."""
            # Count parameters
            param_count = len(node.args.args)
            
            # Estimate function length
            func_lines = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 10
            
            return {
                "name": node.name,
                "parameters": param_count,
                "lines": func_lines,
                "complexity": min(param_count + func_lines // 10, 10),
                "has_docstring": ast.get_docstring(node) is not None
            }
        
        def _analyze_class(self, node: ast.ClassDef) -> Dict:
            """Analyze class structure."""
            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            
            return {
                "name": node.name,
                "methods": len(methods),
                "has_docstring": ast.get_docstring(node) is not None,
                "inheritance": len(node.bases)
            }
        
        def _calculate_complexity(self, tree: ast.AST) -> float:
            """Calculate cyclomatic complexity."""
            complexity = 1  # Base complexity
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            
            return min(complexity / 10.0, 1.0)  # Normalize to 0-1
        
        def _find_code_issues(self, code: str, tree: ast.AST) -> List[Dict]:
            """Find common code quality issues."""
            issues = []
            
            # Check for long lines
            for i, line in enumerate(code.split('\n'), 1):
                if len(line) > 100:
                    issues.append({
                        "type": "style",
                        "line": i,
                        "issue": "Line too long (>100 characters)",
                        "severity": "medium"
                    })
            
            # Check for missing docstrings
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not ast.get_docstring(node):
                        issues.append({
                            "type": "documentation",
                            "line": node.lineno,
                            "issue": f"Missing docstring for {node.name}",
                            "severity": "low"
                        })
            
            # Check for bare except clauses
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    issues.append({
                        "type": "best_practice",
                        "line": node.lineno,
                        "issue": "Bare except clause catches all exceptions",
                        "severity": "high"
                    })
            
            return issues
        
        def _check_security_issues(self, code: str, tree: ast.AST) -> List[Dict]:
            """Check for potential security issues."""
            security_issues = []
            
            # Check for dangerous functions
            dangerous_functions = ['eval', 'exec', 'input', '__import__']
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in dangerous_functions:
                        security_issues.append({
                            "type": "security",
                            "line": node.lineno,
                            "issue": f"Use of potentially dangerous function: {node.func.id}",
                            "severity": "high"
                        })
            
            # Check for hardcoded secrets (simple patterns)
            secret_patterns = ['password', 'api_key', 'secret', 'token']
            for i, line in enumerate(code.split('\n'), 1):
                line_lower = line.lower()
                if any(pattern in line_lower for pattern in secret_patterns):
                    if '=' in line and ('"' in line or "'" in line):
                        security_issues.append({
                            "type": "security",
                            "line": i,
                            "issue": "Potential hardcoded secret detected",
                            "severity": "medium"
                        })
            
            return security_issues
        
        def _format_analysis_report(self, analysis: Dict) -> str:
            """Format comprehensive analysis report."""
            if "error" in analysis:
                return f"❌ Analysis failed: {analysis['error']}"
            
            report = "📊 Code Analysis Report\n"
            report += "=" * 50 + "\n\n"
            
            # Overview
            report += f"📝 Lines of Code: {analysis['lines_of_code']}\n"
            report += f"🔧 Functions: {len(analysis['functions'])}\n"
            report += f"🏗️  Classes: {len(analysis['classes'])}\n"
            report += f"🌀 Complexity Score: {analysis['complexity_score']:.2f}/1.0\n\n"
            
            # Function analysis
            if analysis['functions']:
                report += "🔧 Function Analysis:\n"
                for func in analysis['functions']:
                    report += f"  • {func['name']}: {func['parameters']} params, {func['lines']} lines"
                    if not func['has_docstring']:
                        report += " ⚠️ No docstring"
                    report += "\n"
                report += "\n"
            
            # Issues summary
            if analysis['issues']:
                report += f"⚠️  Issues Found ({len(analysis['issues'])}):\n"
                for issue in analysis['issues'][:5]:  # Show first 5 issues
                    severity_icon = "🔴" if issue['severity'] == 'high' else "🟡" if issue['severity'] == 'medium' else "🟢"
                    report += f"  {severity_icon} Line {issue['line']}: {issue['issue']}\n"
                if len(analysis['issues']) > 5:
                    report += f"  ... and {len(analysis['issues']) - 5} more issues\n"
                report += "\n"
            
            # Security concerns
            if analysis['security_concerns']:
                report += f"🔒 Security Concerns ({len(analysis['security_concerns'])}):\n"
                for concern in analysis['security_concerns']:
                    report += f"  🚨 Line {concern['line']}: {concern['issue']}\n"
                report += "\n"
            
            # Overall assessment
            if analysis['complexity_score'] < 0.3 and len(analysis['issues']) < 5:
                report += "✅ Overall: Good code quality\n"
            elif analysis['complexity_score'] < 0.7 and len(analysis['issues']) < 10:
                report += "⚠️  Overall: Moderate code quality - consider refactoring\n"
            else:
                report += "❌ Overall: Code needs significant improvement\n"
            
            return report
        
        async def _arun(self, code_input: str) -> str:
            """Async version of code analyzer."""
            return self._run(code_input)
    
    class TestRunnerTool(BaseTool):
        """Test execution and analysis tool."""
        
        name = "test_runner"
        description = """
        Execute tests and analyze test results for code coverage and quality.
        Input: test command or test file path
        Returns: test execution results with coverage analysis
        """
        
        def _run(self, test_command: str) -> str:
            """Run tests and analyze results."""
            try:
                # Mock test execution (in practice would run actual tests)
                test_results = self._simulate_test_execution(test_command)
                return self._format_test_results(test_results)
                
            except Exception as e:
                return f"Test execution error: {str(e)}"
        
        def _simulate_test_execution(self, command: str) -> Dict[str, Any]:
            """Simulate test execution results."""
            # Mock test results
            return {
                "command": command,
                "total_tests": 15,
                "passed": 12,
                "failed": 2,
                "skipped": 1,
                "coverage": 78.5,
                "duration": 2.3,
                "failures": [
                    {"test": "test_user_login", "error": "AssertionError: Expected 200, got 404"},
                    {"test": "test_data_validation", "error": "ValueError: Invalid input format"}
                ]
            }
        
        def _format_test_results(self, results: Dict) -> str:
            """Format test execution results."""
            report = "🧪 Test Execution Report\n"
            report += "=" * 50 + "\n\n"
            
            # Summary
            report += f"📊 Test Summary:\n"
            report += f"  Total Tests: {results['total_tests']}\n"
            report += f"  ✅ Passed: {results['passed']}\n"
            report += f"  ❌ Failed: {results['failed']}\n"
            report += f"  ⏭️  Skipped: {results['skipped']}\n"
            report += f"  ⏱️  Duration: {results['duration']}s\n"
            report += f"  📈 Coverage: {results['coverage']:.1f}%\n\n"
            
            # Pass rate
            pass_rate = (results['passed'] / results['total_tests']) * 100
            if pass_rate >= 90:
                report += "✅ Excellent test pass rate!\n"
            elif pass_rate >= 75:
                report += "⚠️  Good test pass rate, some failures to address\n"
            else:
                report += "❌ Poor test pass rate, significant issues found\n"
            
            # Coverage assessment
            if results['coverage'] >= 80:
                report += "✅ Good test coverage\n"
            elif results['coverage'] >= 60:
                report += "⚠️  Moderate test coverage - consider adding more tests\n"
            else:
                report += "❌ Low test coverage - insufficient testing\n"
            
            # Failures detail
            if results['failures']:
                report += "\n❌ Test Failures:\n"
                for failure in results['failures']:
                    report += f"  • {failure['test']}: {failure['error']}\n"
            
            return report
        
        async def _arun(self, test_command: str) -> str:
            """Async version of test runner."""
            return self._run(test_command)
    
    class CodeAnalysisAgent:
        """Comprehensive code analysis agent."""
        
        def __init__(self, llm):
            self.llm = llm
            self.tools = [
                CodeAnalysisTool(),
                TestRunnerTool()
            ]
            self.analysis_history = []
            self.setup_agent()
        
        def setup_agent(self):
            """Set up the code analysis agent."""
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert code reviewer and quality analyst. Your role is to:
                1. Analyze code for quality, complexity, and maintainability
                2. Execute and analyze test results
                3. Identify security vulnerabilities and best practice violations
                4. Provide specific, actionable improvement recommendations
                5. Generate comprehensive code review reports
                
                Always provide constructive feedback with specific examples and solutions."""),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            agent = create_react_agent(self.llm, self.tools, prompt)
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                max_iterations=8
            )
        
        def analyze_codebase(self, code_content: str, run_tests: bool = True) -> Dict[str, Any]:
            """Perform comprehensive codebase analysis."""
            try:
                analysis_prompt = f"""
                Please perform a comprehensive analysis of this code:
                
                {code_content[:1000]}...
                
                Steps:
                1. Use code_analyzer to analyze code quality and structure
                2. {'Use test_runner to execute tests and check coverage' if run_tests else 'Skip test execution'}
                3. Provide overall assessment and recommendations
                
                Focus on code quality, security, and maintainability.
                """
                
                with get_openai_callback() as cb:
                    result = self.agent_executor.invoke({"input": analysis_prompt})
                
                analysis_record = {
                    "timestamp": datetime.now(),
                    "code_length": len(code_content),
                    "result": result,
                    "cost": cb.total_cost
                }
                
                self.analysis_history.append(analysis_record)
                
                return {
                    "analysis_output": result["output"],
                    "cost": cb.total_cost,
                    "recommendations": self._extract_recommendations(result["output"])
                }
                
            except Exception as e:
                return {"error": str(e), "analysis_output": "Analysis failed"}
        
        def _extract_recommendations(self, analysis_output: str) -> List[str]:
            """Extract actionable recommendations from analysis."""
            # Simple extraction (in practice would use more sophisticated NLP)
            recommendations = []
            
            lines = analysis_output.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'should', 'consider']):
                    recommendations.append(line.strip())
            
            return recommendations[:5]  # Return top 5 recommendations
    
    # Demo the code analysis agent
    sample_code = """
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
        self.password = "hardcoded_secret"  # Security issue
    
    def withdraw(self, amount):
        try:
            if amount <= self.balance:
                self.balance -= amount
                return self.balance
        except:
            pass  # Bare except clause
    """
    
    if llm:
        analyzer = CodeAnalysisAgent(llm)
        
        print("🧪 Testing Code Analysis Agent:")
        print(f"📝 Analyzing {len(sample_code)} characters of code...")
        
        result = analyzer.analyze_codebase(sample_code, run_tests=False)
        
        if "error" in result:
            print(f"❌ Analysis failed: {result['error']}")
        else:
            print(f"✅ Analysis completed")
            print(f"💰 Cost: ${result['cost']:.4f}")
            print(f"📄 Analysis preview: {result['analysis_output'][:300]}...")
            
            if result['recommendations']:
                print(f"\n💡 Key Recommendations:")
                for i, rec in enumerate(result['recommendations'][:3], 1):
                    print(f"   {i}. {rec}")


def solution_4_multi_agent_collaboration():
    """
    Solution 4: Multi-Agent Collaboration System
    """
    print("\n🤝 Solution 4: Multi-Agent Collaboration")
    print("-" * 70)
    
    @dataclass
    class AgentMessage:
        """Message structure for agent communication."""
        sender: str
        recipient: str
        message_type: str
        content: Dict[str, Any]
        timestamp: datetime = field(default_factory=datetime.now)
        correlation_id: str = field(default_factory=lambda: f"msg_{int(time.time() * 1000)}")
    
    class CollaborationProtocol:
        """Advanced protocol for managing agent collaboration."""
        
        def __init__(self):
            self.message_queue = deque()
            self.active_tasks = {}
            self.agent_capabilities = {}
            self.task_history = []
        
        def register_agent(self, agent_id: str, capabilities: List[str]):
            """Register agent with their capabilities."""
            self.agent_capabilities[agent_id] = capabilities
            print(f"🤖 Agent {agent_id} registered with capabilities: {capabilities}")
        
        def find_best_agent(self, task_requirements: List[str]) -> Optional[str]:
            """Find the best agent for a given task."""
            best_agent = None
            best_score = 0
            
            for agent_id, capabilities in self.agent_capabilities.items():
                score = len(set(task_requirements) & set(capabilities))
                if score > best_score:
                    best_score = score
                    best_agent = agent_id
            
            return best_agent
        
        def send_message(self, message: AgentMessage):
            """Send message between agents."""
            self.message_queue.append(message)
            print(f"📨 Message queued: {message.sender} → {message.recipient} ({message.message_type})")
        
        def get_messages_for_agent(self, agent_id: str) -> List[AgentMessage]:
            """Get pending messages for an agent."""
            messages = [msg for msg in self.message_queue if msg.recipient == agent_id]
            # Remove retrieved messages
            self.message_queue = deque([msg for msg in self.message_queue if msg.recipient != agent_id])
            return messages
        
        def coordinate_task(self, task: Dict, required_capabilities: List[str]) -> str:
            """Coordinate task across multiple agents."""
            task_id = f"task_{int(time.time())}"
            
            # Find suitable agents
            suitable_agents = []
            for capability in required_capabilities:
                best_agent = self.find_best_agent([capability])
                if best_agent and best_agent not in suitable_agents:
                    suitable_agents.append(best_agent)
            
            if not suitable_agents:
                return f"❌ No suitable agents found for task requirements: {required_capabilities}"
            
            # Create coordination plan
            self.active_tasks[task_id] = {
                "task": task,
                "agents": suitable_agents,
                "status": "assigned",
                "created": datetime.now(),
                "steps": []
            }
            
            # Send initial messages to agents
            for agent_id in suitable_agents:
                message = AgentMessage(
                    sender="coordinator",
                    recipient=agent_id,
                    message_type="task_assignment",
                    content={"task_id": task_id, "task": task}
                )
                self.send_message(message)
            
            return task_id
        
        def resolve_conflicts(self, task_id: str, conflicting_results: List[Dict]) -> Dict:
            """Resolve conflicts between agent results using voting."""
            if not conflicting_results:
                return {"resolution": "no_conflict", "result": None}
            
            # Simple majority voting for demonstration
            result_counts = defaultdict(int)
            for result in conflicting_results:
                result_key = str(result.get("conclusion", "unknown"))
                result_counts[result_key] += 1
            
            # Find majority result
            majority_result = max(result_counts.items(), key=lambda x: x[1])
            
            return {
                "resolution": "majority_vote",
                "result": majority_result[0],
                "confidence": majority_result[1] / len(conflicting_results),
                "vote_counts": dict(result_counts)
            }
        
        def get_task_status(self, task_id: str) -> Dict[str, Any]:
            """Get current status of a collaborative task."""
            if task_id not in self.active_tasks:
                return {"error": "Task not found"}
            
            task_info = self.active_tasks[task_id]
            return {
                "task_id": task_id,
                "status": task_info["status"],
                "agents": task_info["agents"],
                "created": task_info["created"],
                "steps_completed": len(task_info["steps"]),
                "duration": (datetime.now() - task_info["created"]).total_seconds()
            }
    
    class SpecializedAgent:
        """Base class for specialized collaborative agents."""
        
        def __init__(self, agent_id: str, specialization: str, capabilities: List[str], llm):
            self.agent_id = agent_id
            self.specialization = specialization
            self.capabilities = capabilities
            self.llm = llm
            self.collaboration_protocol = None
            self.task_queue = deque()
            self.results_cache = {}
        
        def set_collaboration_protocol(self, protocol: CollaborationProtocol):
            """Set the collaboration protocol."""
            self.collaboration_protocol = protocol
            protocol.register_agent(self.agent_id, self.capabilities)
        
        def process_task(self, task: Dict) -> Dict[str, Any]:
            """Process assigned task using agent's specialization."""
            try:
                task_type = task.get("type", "general")
                
                if task_type == "research":
                    return self._handle_research_task(task)
                elif task_type == "analysis":
                    return self._handle_analysis_task(task)
                elif task_type == "writing":
                    return self._handle_writing_task(task)
                elif task_type == "review":
                    return self._handle_review_task(task)
                else:
                    return self._handle_general_task(task)
                    
            except Exception as e:
                return {
                    "agent_id": self.agent_id,
                    "status": "error",
                    "error": str(e),
                    "task": task
                }
        
        def _handle_research_task(self, task: Dict) -> Dict[str, Any]:
            """Handle research-specific tasks."""
            query = task.get("query", "")
            
            # Mock research process
            research_results = {
                "sources": ["Source 1", "Source 2", "Source 3"],
                "findings": f"Research findings for: {query}",
                "confidence": 0.85,
                "methodology": "Multi-source analysis"
            }
            
            return {
                "agent_id": self.agent_id,
                "status": "completed",
                "result": research_results,
                "task_type": "research"
            }
        
        def _handle_analysis_task(self, task: Dict) -> Dict[str, Any]:
            """Handle analysis-specific tasks."""
            data = task.get("data", "")
            
            # Mock analysis process
            analysis_results = {
                "summary": f"Analysis summary for provided data",
                "key_insights": ["Insight 1", "Insight 2", "Insight 3"],
                "recommendations": ["Recommendation 1", "Recommendation 2"],
                "confidence": 0.78
            }
            
            return {
                "agent_id": self.agent_id,
                "status": "completed",
                "result": analysis_results,
                "task_type": "analysis"
            }
        
        def _handle_writing_task(self, task: Dict) -> Dict[str, Any]:
            """Handle writing-specific tasks."""
            content_type = task.get("content_type", "general")
            requirements = task.get("requirements", {})
            
            # Mock writing process
            writing_results = {
                "content": f"Generated {content_type} content based on requirements",
                "word_count": 500,
                "style": requirements.get("style", "professional"),
                "format": requirements.get("format", "article")
            }
            
            return {
                "agent_id": self.agent_id,
                "status": "completed",
                "result": writing_results,
                "task_type": "writing"
            }
        
        def _handle_review_task(self, task: Dict) -> Dict[str, Any]:
            """Handle review-specific tasks."""
            content = task.get("content", "")
            review_type = task.get("review_type", "general")
            
            # Mock review process
            review_results = {
                "quality_score": 0.82,
                "feedback": f"Review feedback for {review_type} content",
                "suggestions": ["Suggestion 1", "Suggestion 2"],
                "approval_status": "approved_with_suggestions"
            }
            
            return {
                "agent_id": self.agent_id,
                "status": "completed",
                "result": review_results,
                "task_type": "review"
            }
        
        def _handle_general_task(self, task: Dict) -> Dict[str, Any]:
            """Handle general tasks."""
            return {
                "agent_id": self.agent_id,
                "status": "completed",
                "result": {"message": f"Processed general task using {self.specialization} expertise"},
                "task_type": "general"
            }
        
        def collaborate_with(self, other_agent_id: str, task: Dict) -> Dict[str, Any]:
            """Initiate collaboration with another agent."""
            if not self.collaboration_protocol:
                return {"error": "No collaboration protocol set"}
            
            message = AgentMessage(
                sender=self.agent_id,
                recipient=other_agent_id,
                message_type="collaboration_request",
                content={"task": task, "requesting_agent": self.agent_id}
            )
            
            self.collaboration_protocol.send_message(message)
            
            return {
                "status": "collaboration_requested",
                "target_agent": other_agent_id,
                "task": task
            }
        
        def check_messages(self) -> List[AgentMessage]:
            """Check for pending messages."""
            if not self.collaboration_protocol:
                return []
            
            return self.collaboration_protocol.get_messages_for_agent(self.agent_id)
    
    class MultiAgentOrchestrator:
        """Advanced orchestrator for multi-agent collaboration."""
        
        def __init__(self):
            self.agents = {}
            self.collaboration_protocol = CollaborationProtocol()
            self.task_monitor = TaskMonitor()
            self.completed_tasks = []
        
        def register_agent(self, agent: SpecializedAgent):
            """Register agent in the collaboration system."""
            self.agents[agent.agent_id] = agent
            agent.set_collaboration_protocol(self.collaboration_protocol)
            print(f"✅ Agent {agent.agent_id} ({agent.specialization}) registered")
        
        def decompose_complex_task(self, task: Dict) -> List[Dict]:
            """Break complex task into manageable subtasks."""
            task_type = task.get("type", "general")
            complexity = task.get("complexity", "medium")
            
            if complexity == "high" or task_type == "comprehensive_analysis":
                # Break into research, analysis, writing, and review phases
                subtasks = [
                    {
                        "id": f"{task.get('id', 'task')}_research",
                        "type": "research",
                        "query": task.get("topic", "general topic"),
                        "requirements": ["research"]
                    },
                    {
                        "id": f"{task.get('id', 'task')}_analysis",
                        "type": "analysis",
                        "data": "research_output",
                        "requirements": ["analysis"],
                        "depends_on": [f"{task.get('id', 'task')}_research"]
                    },
                    {
                        "id": f"{task.get('id', 'task')}_writing",
                        "type": "writing",
                        "content_type": "report",
                        "requirements": ["writing"],
                        "depends_on": [f"{task.get('id', 'task')}_analysis"]
                    },
                    {
                        "id": f"{task.get('id', 'task')}_review",
                        "type": "review",
                        "content": "writing_output",
                        "requirements": ["review"],
                        "depends_on": [f"{task.get('id', 'task')}_writing"]
                    }
                ]
            else:
                # Simple task - no decomposition needed
                subtasks = [task]
            
            return subtasks
        
        def execute_collaborative_task(self, task: Dict) -> Dict[str, Any]:
            """Execute task requiring multiple agents."""
            try:
                start_time = datetime.now()
                
                # Decompose task if complex
                subtasks = self.decompose_complex_task(task)
                
                # Execute subtasks in dependency order
                results = {}
                for subtask in subtasks:
                    # Check dependencies
                    dependencies = subtask.get("depends_on", [])
                    if dependencies and not all(dep in results for dep in dependencies):
                        continue  # Skip for now - would implement proper dependency resolution
                    
                    # Find best agent for this subtask
                    required_capabilities = subtask.get("requirements", [])
                    best_agent_id = self.collaboration_protocol.find_best_agent(required_capabilities)
                    
                    if best_agent_id and best_agent_id in self.agents:
                        agent = self.agents[best_agent_id]
                        result = agent.process_task(subtask)
                        results[subtask["id"]] = result
                        
                        # Monitor progress
                        self.task_monitor.track_task_progress(
                            task.get("id", "unknown"),
                            best_agent_id,
                            result
                        )
                
                # Compile final result
                execution_time = (datetime.now() - start_time).total_seconds()
                
                final_result = {
                    "task_id": task.get("id", "unknown"),
                    "status": "completed",
                    "subtask_results": results,
                    "execution_time": execution_time,
                    "agents_involved": list(set(r.get("agent_id") for r in results.values() if "agent_id" in r)),
                    "completed_at": datetime.now()
                }
                
                self.completed_tasks.append(final_result)
                return final_result
                
            except Exception as e:
                return {
                    "error": str(e),
                    "task_id": task.get("id", "unknown"),
                    "status": "failed"
                }
        
        def monitor_progress(self, task_id: str) -> Dict[str, Any]:
            """Monitor progress of collaborative task."""
            return self.task_monitor.get_task_progress(task_id)
        
        def get_system_status(self) -> Dict[str, Any]:
            """Get overall system status and metrics."""
            return {
                "registered_agents": len(self.agents),
                "agent_capabilities": {aid: agent.capabilities for aid, agent in self.agents.items()},
                "completed_tasks": len(self.completed_tasks),
                "active_tasks": len(self.collaboration_protocol.active_tasks),
                "message_queue_size": len(self.collaboration_protocol.message_queue)
            }
    
    class TaskMonitor:
        """Monitor and track collaborative task progress."""
        
        def __init__(self):
            self.task_progress = defaultdict(list)
            self.performance_metrics = defaultdict(list)
        
        def track_task_progress(self, task_id: str, agent_id: str, progress: Dict):
            """Track individual agent progress on tasks."""
            progress_entry = {
                "timestamp": datetime.now(),
                "agent_id": agent_id,
                "progress": progress,
                "status": progress.get("status", "unknown")
            }
            
            self.task_progress[task_id].append(progress_entry)
            
            # Track performance metrics
            if progress.get("status") == "completed":
                self.performance_metrics[agent_id].append({
                    "task_id": task_id,
                    "completed_at": datetime.now(),
                    "success": True
                })
        
        def detect_bottlenecks(self, task_id: str) -> List[Dict]:
            """Detect bottlenecks in collaborative workflows."""
            bottlenecks = []
            
            if task_id in self.task_progress:
                progress_entries = self.task_progress[task_id]
                
                # Check for agents taking too long
                for entry in progress_entries:
                    if entry["status"] == "in_progress":
                        time_elapsed = (datetime.now() - entry["timestamp"]).total_seconds()
                        if time_elapsed > 300:  # 5 minutes threshold
                            bottlenecks.append({
                                "type": "slow_agent",
                                "agent_id": entry["agent_id"],
                                "time_elapsed": time_elapsed,
                                "task_id": task_id
                            })
            
            return bottlenecks
        
        def get_task_progress(self, task_id: str) -> Dict[str, Any]:
            """Get detailed progress information for a task."""
            if task_id not in self.task_progress:
                return {"error": "Task not found"}
            
            progress_entries = self.task_progress[task_id]
            
            return {
                "task_id": task_id,
                "total_steps": len(progress_entries),
                "completed_steps": len([e for e in progress_entries if e["status"] == "completed"]),
                "agents_involved": list(set(e["agent_id"] for e in progress_entries)),
                "latest_update": progress_entries[-1] if progress_entries else None,
                "bottlenecks": self.detect_bottlenecks(task_id)
            }
        
        def generate_performance_report(self) -> Dict[str, Any]:
            """Generate performance report for all agents."""
            report = {"agent_performance": {}}
            
            for agent_id, metrics in self.performance_metrics.items():
                completed_tasks = len(metrics)
                success_rate = len([m for m in metrics if m["success"]]) / completed_tasks if completed_tasks > 0 else 0
                
                report["agent_performance"][agent_id] = {
                    "completed_tasks": completed_tasks,
                    "success_rate": success_rate,
                    "avg_completion_time": 0  # Mock calculation
                }
            
            return report
    
    # Demo the multi-agent collaboration system
    if llm:
        # Create specialized agents
        research_agent = SpecializedAgent(
            "researcher_001", 
            "Research Specialist", 
            ["research", "data_gathering", "fact_checking"], 
            llm
        )
        
        analysis_agent = SpecializedAgent(
            "analyst_001", 
            "Data Analyst", 
            ["analysis", "data_processing", "statistics"], 
            llm
        )
        
        writer_agent = SpecializedAgent(
            "writer_001", 
            "Content Writer", 
            ["writing", "editing", "content_creation"], 
            llm
        )
        
        reviewer_agent = SpecializedAgent(
            "reviewer_001", 
            "Quality Reviewer", 
            ["review", "quality_assurance", "validation"], 
            llm
        )
        
        # Create orchestrator and register agents
        orchestrator = MultiAgentOrchestrator()
        for agent in [research_agent, analysis_agent, writer_agent, reviewer_agent]:
            orchestrator.register_agent(agent)
        
        print("\n🧪 Testing Multi-Agent Collaboration:")
        
        # Test complex collaborative task
        complex_task = {
            "id": "market_analysis_report",
            "type": "comprehensive_analysis",
            "topic": "AI market trends in healthcare",
            "complexity": "high",
            "requirements": ["research", "analysis", "writing", "review"]
        }
        
        print(f"📋 Executing complex task: {complex_task['topic']}")
        
        result = orchestrator.execute_collaborative_task(complex_task)
        
        if "error" in result:
            print(f"❌ Collaboration failed: {result['error']}")
        else:
            print(f"✅ Collaboration completed successfully")
            print(f"⏱️  Execution time: {result['execution_time']:.2f}s")
            print(f"🤖 Agents involved: {result['agents_involved']}")
            print(f"📊 Subtasks completed: {len(result['subtask_results'])}")
        
        # Show system status
        status = orchestrator.get_system_status()
        print(f"\n📈 System Status:")
        print(f"   Registered agents: {status['registered_agents']}")
        print(f"   Completed tasks: {status['completed_tasks']}")
        print(f"   Active tasks: {status['active_tasks']}")


def solution_5_production_agent_platform():
    """
    Solution 5: Production Agent Platform
    """
    print("\n🏭 Solution 5: Production Agent Platform")
    print("-" * 70)
    
    class UserManager:
        """Comprehensive user management system."""
        
        def __init__(self):
            self.users = {}
            self.sessions = {}
            self.usage_tracking = defaultdict(list)
            self.rate_limits = defaultdict(lambda: {"requests": 0, "window_start": datetime.now()})
        
        def authenticate_user(self, username: str, password: str) -> Optional[str]:
            """Authenticate user and return session token."""
            # Mock authentication - in production would use proper hashing
            if username in self.users and self.users[username]["password"] == password:
                session_token = f"session_{hashlib.md5(f'{username}{time.time()}'.encode()).hexdigest()}"
                self.sessions[session_token] = {
                    "username": username,
                    "created": datetime.now(),
                    "last_activity": datetime.now()
                }
                return session_token
            return None
        
        def create_user(self, username: str, password: str, role: str = "user") -> bool:
            """Create new user account."""
            if username in self.users:
                return False
            
            self.users[username] = {
                "password": password,  # In production: hash this
                "role": role,
                "created": datetime.now(),
                "active": True,
                "usage_limits": {
                    "daily_requests": 1000,
                    "monthly_cost": 100.0
                }
            }
            return True
        
        def authorize_agent_access(self, session_token: str, agent_id: str) -> bool:
            """Check if user can access specific agent."""
            if session_token not in self.sessions:
                return False
            
            session = self.sessions[session_token]
            username = session["username"]
            user = self.users[username]
            
            # Check rate limits
            if not self._check_rate_limit(username):
                return False
            
            # Check user role permissions
            if user["role"] == "admin":
                return True
            elif user["role"] == "premium" and not agent_id.startswith("restricted_"):
                return True
            elif user["role"] == "user" and agent_id.startswith("public_"):
                return True
            
            return False
        
        def _check_rate_limit(self, username: str) -> bool:
            """Check if user is within rate limits."""
            now = datetime.now()
            rate_info = self.rate_limits[username]
            
            # Reset window if expired (1 minute windows)
            if (now - rate_info["window_start"]).total_seconds() > 60:
                rate_info["requests"] = 0
                rate_info["window_start"] = now
            
            # Check limit (60 requests per minute)
            if rate_info["requests"] >= 60:
                return False
            
            rate_info["requests"] += 1
            return True
        
        def track_user_usage(self, session_token: str, usage_data: Dict):
            """Track user usage for billing and limits."""
            if session_token in self.sessions:
                username = self.sessions[session_token]["username"]
                usage_entry = {
                    "timestamp": datetime.now(),
                    "agent_id": usage_data.get("agent_id"),
                    "cost": usage_data.get("cost", 0),
                    "tokens": usage_data.get("tokens", 0),
                    "success": usage_data.get("success", True)
                }
                self.usage_tracking[username].append(usage_entry)
        
        def get_user_usage_summary(self, username: str, days: int = 30) -> Dict[str, Any]:
            """Get usage summary for a user."""
            if username not in self.usage_tracking:
                return {"error": "No usage data found"}
            
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_usage = [u for u in self.usage_tracking[username] if u["timestamp"] > cutoff_date]
            
            return {
                "username": username,
                "period_days": days,
                "total_requests": len(recent_usage),
                "total_cost": sum(u["cost"] for u in recent_usage),
                "total_tokens": sum(u["tokens"] for u in recent_usage),
                "success_rate": len([u for u in recent_usage if u["success"]]) / len(recent_usage) if recent_usage else 0,
                "most_used_agents": self._get_top_agents(recent_usage)
            }
        
        def _get_top_agents(self, usage_data: List[Dict]) -> List[Dict]:
            """Get most frequently used agents."""
            agent_counts = defaultdict(int)
            for usage in usage_data:
                if usage.get("agent_id"):
                    agent_counts[usage["agent_id"]] += 1
            
            return [
                {"agent_id": agent_id, "requests": count}
                for agent_id, count in sorted(agent_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            ]
    
    class SafetyController:
        """Advanced safety and compliance controller."""
        
        def __init__(self):
            self.safety_rules = self._initialize_safety_rules()
            self.content_filters = self._initialize_content_filters()
            self.incident_log = []
            self.blocked_patterns = ["execute", "delete", "sudo", "rm -rf", "drop table"]
        
        def _initialize_safety_rules(self) -> List[Dict]:
            """Initialize safety rules."""
            return [
                {
                    "id": "no_code_execution",
                    "description": "Prevent code execution requests",
                    "severity": "high",
                    "action": "block"
                },
                {
                    "id": "no_personal_info",
                    "description": "Block requests for personal information",
                    "severity": "medium",
                    "action": "filter"
                },
                {
                    "id": "content_policy",
                    "description": "Enforce content policy guidelines",
                    "severity": "medium",
                    "action": "review"
                }
            ]
        
        def _initialize_content_filters(self) -> List[Dict]:
            """Initialize content filtering rules."""
            return [
                {
                    "name": "profanity_filter",
                    "patterns": ["inappropriate", "offensive"],  # Simplified
                    "action": "replace",
                    "replacement": "[FILTERED]"
                },
                {
                    "name": "personal_info_filter",
                    "patterns": [r"\b\d{3}-\d{2}-\d{4}\b", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
                    "action": "redact",
                    "replacement": "[REDACTED]"
                }
            ]
        
        def validate_request(self, user_id: str, request: Dict) -> Dict[str, Any]:
            """Validate request against safety rules."""
            validation_result = {
                "allowed": True,
                "warnings": [],
                "blocked_reasons": [],
                "filtered_content": request.copy()
            }
            
            request_text = str(request.get("content", "")).lower()
            
            # Check against blocked patterns
            for pattern in self.blocked_patterns:
                if pattern in request_text:
                    validation_result["allowed"] = False
                    validation_result["blocked_reasons"].append(f"Contains blocked pattern: {pattern}")
            
            # Check safety rules
            for rule in self.safety_rules:
                if rule["id"] == "no_code_execution" and any(word in request_text for word in ["execute", "run", "eval"]):
                    if rule["action"] == "block":
                        validation_result["allowed"] = False
                        validation_result["blocked_reasons"].append(rule["description"])
                    elif rule["action"] == "review":
                        validation_result["warnings"].append(f"Flagged for review: {rule['description']}")
            
            # Apply content filters
            if validation_result["allowed"]:
                validation_result["filtered_content"] = self.filter_response(request)
            
            return validation_result
        
        def filter_response(self, response: Dict) -> Dict:
            """Filter response content for safety."""
            filtered_response = response.copy()
            
            if "content" in filtered_response:
                content = filtered_response["content"]
                
                # Apply content filters
                for filter_rule in self.content_filters:
                    for pattern in filter_rule["patterns"]:
                        if filter_rule["action"] == "replace":
                            content = re.sub(pattern, filter_rule["replacement"], content, flags=re.IGNORECASE)
                        elif filter_rule["action"] == "redact":
                            content = re.sub(pattern, filter_rule["replacement"], content)
                
                filtered_response["content"] = content
            
            return filtered_response
        
        def log_safety_incident(self, incident: Dict):
            """Log safety incidents for review."""
            incident_entry = {
                "id": f"incident_{len(self.incident_log) + 1}",
                "timestamp": datetime.now(),
                "user_id": incident.get("user_id"),
                "type": incident.get("type", "unknown"),
                "severity": incident.get("severity", "medium"),
                "description": incident.get("description", ""),
                "request_data": incident.get("request_data", {}),
                "action_taken": incident.get("action_taken", "logged")
            }
            
            self.incident_log.append(incident_entry)
            
            # Alert for high severity incidents
            if incident_entry["severity"] == "high":
                print(f"🚨 HIGH SEVERITY INCIDENT: {incident_entry['id']} - {incident_entry['description']}")
        
        def get_safety_metrics(self) -> Dict[str, Any]:
            """Get safety and compliance metrics."""
            total_incidents = len(self.incident_log)
            high_severity = len([i for i in self.incident_log if i["severity"] == "high"])
            recent_incidents = len([i for i in self.incident_log if 
                                 (datetime.now() - i["timestamp"]).days <= 7])
            
            return {
                "total_incidents": total_incidents,
                "high_severity_incidents": high_severity,
                "recent_incidents_7_days": recent_incidents,
                "safety_rules_active": len(self.safety_rules),
                "content_filters_active": len(self.content_filters),
                "blocked_patterns": len(self.blocked_patterns)
            }
    
    class ResourceManager:
        """Advanced resource and cost management."""
        
        def __init__(self):
            self.resource_pools = {
                "cpu": {"total": 100, "used": 0, "reserved": 0},
                "memory": {"total": 1000, "used": 0, "reserved": 0},  # GB
                "gpu": {"total": 8, "used": 0, "reserved": 0}
            }
            self.cost_tracking = defaultdict(list)
            self.resource_allocations = {}
        
        def allocate_resources(self, agent_id: str, requirements: Dict) -> Dict[str, Any]:
            """Allocate resources for agent execution."""
            allocation_id = f"alloc_{int(time.time())}_{agent_id}"
            
            # Check resource availability
            required_cpu = requirements.get("cpu", 1)
            required_memory = requirements.get("memory", 1)
            required_gpu = requirements.get("gpu", 0)
            
            allocation_result = {
                "allocation_id": allocation_id,
                "agent_id": agent_id,
                "success": True,
                "allocated_resources": {},
                "errors": []
            }
            
            # Check CPU availability
            if self.resource_pools["cpu"]["used"] + required_cpu > self.resource_pools["cpu"]["total"]:
                allocation_result["success"] = False
                allocation_result["errors"].append("Insufficient CPU resources")
            
            # Check memory availability
            if self.resource_pools["memory"]["used"] + required_memory > self.resource_pools["memory"]["total"]:
                allocation_result["success"] = False
                allocation_result["errors"].append("Insufficient memory resources")
            
            # Check GPU availability
            if required_gpu > 0 and self.resource_pools["gpu"]["used"] + required_gpu > self.resource_pools["gpu"]["total"]:
                allocation_result["success"] = False
                allocation_result["errors"].append("Insufficient GPU resources")
            
            # Allocate if resources are available
            if allocation_result["success"]:
                self.resource_pools["cpu"]["used"] += required_cpu
                self.resource_pools["memory"]["used"] += required_memory
                self.resource_pools["gpu"]["used"] += required_gpu
                
                allocation_result["allocated_resources"] = {
                    "cpu": required_cpu,
                    "memory": required_memory,
                    "gpu": required_gpu
                }
                
                self.resource_allocations[allocation_id] = {
                    "agent_id": agent_id,
                    "resources": allocation_result["allocated_resources"],
                    "allocated_at": datetime.now()
                }
            
            return allocation_result
        
        def release_resources(self, allocation_id: str) -> bool:
            """Release allocated resources."""
            if allocation_id not in self.resource_allocations:
                return False
            
            allocation = self.resource_allocations[allocation_id]
            resources = allocation["resources"]
            
            # Release resources back to pool
            self.resource_pools["cpu"]["used"] -= resources.get("cpu", 0)
            self.resource_pools["memory"]["used"] -= resources.get("memory", 0)
            self.resource_pools["gpu"]["used"] -= resources.get("gpu", 0)
            
            # Remove allocation
            del self.resource_allocations[allocation_id]
            
            return True
        
        def track_costs(self, agent_id: str, usage: Dict):
            """Track costs for agent operations."""
            cost_entry = {
                "timestamp": datetime.now(),
                "agent_id": agent_id,
                "cost_type": usage.get("type", "execution"),
                "amount": usage.get("amount", 0),
                "currency": usage.get("currency", "USD"),
                "resource_usage": usage.get("resources", {}),
                "duration": usage.get("duration", 0)
            }
            
            self.cost_tracking[agent_id].append(cost_entry)
        
        def enforce_limits(self, user_id: str, requested_resources: Dict) -> Dict[str, Any]:
            """Enforce resource and cost limits."""
            # Mock limit enforcement
            daily_cost_limit = 50.0  # $50 daily limit
            monthly_cost_limit = 1000.0  # $1000 monthly limit
            
            # Calculate current usage
            current_daily_cost = 15.0  # Mock calculation
            current_monthly_cost = 300.0  # Mock calculation
            
            enforcement_result = {
                "allowed": True,
                "reasons": [],
                "current_usage": {
                    "daily_cost": current_daily_cost,
                    "monthly_cost": current_monthly_cost
                },
                "limits": {
                    "daily_cost": daily_cost_limit,
                    "monthly_cost": monthly_cost_limit
                }
            }
            
            # Check limits
            estimated_cost = requested_resources.get("estimated_cost", 1.0)
            
            if current_daily_cost + estimated_cost > daily_cost_limit:
                enforcement_result["allowed"] = False
                enforcement_result["reasons"].append("Daily cost limit exceeded")
            
            if current_monthly_cost + estimated_cost > monthly_cost_limit:
                enforcement_result["allowed"] = False
                enforcement_result["reasons"].append("Monthly cost limit exceeded")
            
            return enforcement_result
        
        def get_resource_status(self) -> Dict[str, Any]:
            """Get current resource utilization status."""
            status = {}
            
            for resource_type, pool in self.resource_pools.items():
                utilization = (pool["used"] / pool["total"]) * 100 if pool["total"] > 0 else 0
                status[resource_type] = {
                    "total": pool["total"],
                    "used": pool["used"],
                    "available": pool["total"] - pool["used"],
                    "utilization_percent": round(utilization, 2)
                }
            
            status["active_allocations"] = len(self.resource_allocations)
            
            return status
    
    class ProductionAgentPlatform:
        """Complete production agent platform."""
        
        def __init__(self):
            self.user_manager = UserManager()
            self.safety_controller = SafetyController()
            self.resource_manager = ResourceManager()
            self.agents = {}
            self.request_history = []
            
            # Initialize demo users
            self._initialize_demo_data()
        
        def _initialize_demo_data(self):
            """Initialize demo users and agents."""
            # Create demo users
            self.user_manager.create_user("admin", "admin123", "admin")
            self.user_manager.create_user("premium_user", "premium123", "premium")
            self.user_manager.create_user("basic_user", "basic123", "user")
            
            # Register demo agents
            self.agents["public_chatbot"] = {
                "name": "Public Chatbot",
                "access_level": "public",
                "resource_requirements": {"cpu": 1, "memory": 2}
            }
            self.agents["premium_analyst"] = {
                "name": "Premium Data Analyst",
                "access_level": "premium",
                "resource_requirements": {"cpu": 4, "memory": 8, "gpu": 1}
            }
            self.agents["restricted_admin_bot"] = {
                "name": "Admin Management Bot", 
                "access_level": "admin",
                "resource_requirements": {"cpu": 2, "memory": 4}
            }
        
        def process_user_request(self, username: str, password: str, agent_id: str, request: Dict) -> Dict[str, Any]:
            """Process user request through full platform pipeline."""
            start_time = datetime.now()
            
            # 1. Authentication
            session_token = self.user_manager.authenticate_user(username, password)
            if not session_token:
                return {
                    "success": False,
                    "error": "Authentication failed",
                    "stage": "authentication"
                }
            
            # 2. Authorization
            if not self.user_manager.authorize_agent_access(session_token, agent_id):
                return {
                    "success": False,
                    "error": "Access denied to requested agent",
                    "stage": "authorization"
                }
            
            # 3. Safety validation
            safety_result = self.safety_controller.validate_request(username, request)
            if not safety_result["allowed"]:
                # Log safety incident
                self.safety_controller.log_safety_incident({
                    "user_id": username,
                    "type": "request_blocked",
                    "severity": "medium",
                    "description": f"Request blocked: {safety_result['blocked_reasons']}",
                    "request_data": request
                })
                
                return {
                    "success": False,
                    "error": "Request blocked by safety filters",
                    "reasons": safety_result["blocked_reasons"],
                    "stage": "safety_validation"
                }
            
            # 4. Resource allocation
            if agent_id in self.agents:
                resource_requirements = self.agents[agent_id]["resource_requirements"]
                allocation_result = self.resource_manager.allocate_resources(agent_id, resource_requirements)
                
                if not allocation_result["success"]:
                    return {
                        "success": False,
                        "error": "Insufficient resources",
                        "details": allocation_result["errors"],
                        "stage": "resource_allocation"
                    }
                
                allocation_id = allocation_result["allocation_id"]
            else:
                return {
                    "success": False,
                    "error": "Agent not found",
                    "stage": "agent_lookup"
                }
            
            try:
                # 5. Process request (mock agent execution)
                agent_response = self._execute_agent_request(agent_id, safety_result["filtered_content"])
                
                # 6. Filter response
                filtered_response = self.safety_controller.filter_response(agent_response)
                
                # 7. Track usage and costs
                execution_time = (datetime.now() - start_time).total_seconds()
                estimated_cost = execution_time * 0.01  # $0.01 per second
                
                self.user_manager.track_user_usage(session_token, {
                    "agent_id": agent_id,
                    "cost": estimated_cost,
                    "tokens": len(str(request)) + len(str(filtered_response)),
                    "success": True
                })
                
                self.resource_manager.track_costs(agent_id, {
                    "type": "execution",
                    "amount": estimated_cost,
                    "duration": execution_time,
                    "resources": resource_requirements
                })
                
                # 8. Release resources
                self.resource_manager.release_resources(allocation_id)
                
                # Store request history
                self.request_history.append({
                    "timestamp": start_time,
                    "username": username,
                    "agent_id": agent_id,
                    "success": True,
                    "execution_time": execution_time,
                    "cost": estimated_cost
                })
                
                return {
                    "success": True,
                    "response": filtered_response,
                    "metadata": {
                        "execution_time": execution_time,
                        "cost": estimated_cost,
                        "warnings": safety_result["warnings"]
                    }
                }
                
            except Exception as e:
                # Release resources on error
                self.resource_manager.release_resources(allocation_id)
                
                return {
                    "success": False,
                    "error": f"Agent execution failed: {str(e)}",
                    "stage": "agent_execution"
                }
        
        def _execute_agent_request(self, agent_id: str, request: Dict) -> Dict[str, Any]:
            """Mock agent execution."""
            agent_info = self.agents[agent_id]
            
            # Simulate processing time based on complexity
            time.sleep(0.1)  # Mock processing delay
            
            return {
                "agent_id": agent_id,
                "agent_name": agent_info["name"],
                "content": f"Response from {agent_info['name']} for request: {request.get('content', 'No content')}",
                "confidence": 0.85,
                "processing_time": 0.1
            }
        
        def admin_dashboard_data(self) -> Dict[str, Any]:
            """Generate comprehensive admin dashboard data."""
            return {
                "system_overview": {
                    "total_users": len(self.user_manager.users),
                    "active_sessions": len(self.user_manager.sessions),
                    "total_agents": len(self.agents),
                    "total_requests": len(self.request_history)
                },
                "resource_status": self.resource_manager.get_resource_status(),
                "safety_metrics": self.safety_controller.get_safety_metrics(),
                "recent_activity": self.request_history[-10:],  # Last 10 requests
                "top_users": self._get_top_users(),
                "agent_usage": self._get_agent_usage_stats()
            }
        
        def _get_top_users(self) -> List[Dict]:
            """Get top users by request count."""
            user_counts = defaultdict(int)
            for request in self.request_history:
                user_counts[request["username"]] += 1
            
            return [
                {"username": username, "requests": count}
                for username, count in sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            ]
        
        def _get_agent_usage_stats(self) -> List[Dict]:
            """Get agent usage statistics."""
            agent_counts = defaultdict(int)
            agent_costs = defaultdict(float)
            
            for request in self.request_history:
                agent_id = request["agent_id"]
                agent_counts[agent_id] += 1
                agent_costs[agent_id] += request.get("cost", 0)
            
            return [
                {
                    "agent_id": agent_id,
                    "requests": agent_counts[agent_id],
                    "total_cost": round(agent_costs[agent_id], 4)
                }
                for agent_id in self.agents.keys()
            ]
        
        def generate_platform_reports(self) -> Dict[str, Any]:
            """Generate comprehensive platform reports."""
            now = datetime.now()
            
            return {
                "executive_summary": {
                    "report_date": now.isoformat(),
                    "platform_health": "healthy",
                    "total_users": len(self.user_manager.users),
                    "requests_last_24h": len([r for r in self.request_history if 
                                             (now - r["timestamp"]).total_seconds() < 86400]),
                    "average_response_time": np.mean([r["execution_time"] for r in self.request_history]) if self.request_history else 0,
                    "success_rate": len([r for r in self.request_history if r["success"]]) / len(self.request_history) if self.request_history else 1.0
                },
                "resource_utilization": self.resource_manager.get_resource_status(),
                "safety_compliance": self.safety_controller.get_safety_metrics(),
                "financial_summary": {
                    "total_platform_cost": sum(r.get("cost", 0) for r in self.request_history),
                    "average_cost_per_request": np.mean([r.get("cost", 0) for r in self.request_history]) if self.request_history else 0
                },
                "user_analytics": {
                    "active_users_last_7_days": len(set(r["username"] for r in self.request_history if 
                                                       (now - r["timestamp"]).days <= 7)),
                    "top_users": self._get_top_users(),
                    "user_distribution": {
                        role: len([u for u in self.user_manager.users.values() if u["role"] == role])
                        for role in ["admin", "premium", "user"]
                    }
                }
            }
    
    # Demo the production platform
    platform = ProductionAgentPlatform()
    
    print("🧪 Testing Production Agent Platform:")
    
    # Test different user scenarios
    test_scenarios = [
        ("basic_user", "basic123", "public_chatbot", {"content": "Hello, how are you?"}),
        ("premium_user", "premium123", "premium_analyst", {"content": "Analyze sales data trends"}),
        ("basic_user", "basic123", "premium_analyst", {"content": "I want premium features"}),  # Should fail
        ("admin", "admin123", "restricted_admin_bot", {"content": "Execute system maintenance"}),  # Should be flagged
    ]
    
    for username, password, agent_id, request in test_scenarios[:2]:  # Test first 2 scenarios
        print(f"\n👤 Testing: {username} → {agent_id}")
        print(f"📝 Request: {request['content']}")
        
        result = platform.process_user_request(username, password, agent_id, request)
        
        if result["success"]:
            print(f"✅ Request successful")
            print(f"⏱️  Execution time: {result['metadata']['execution_time']:.2f}s")
            print(f"💰 Cost: ${result['metadata']['cost']:.4f}")
            print(f"🤖 Response: {result['response']['content'][:100]}...")
        else:
            print(f"❌ Request failed at {result['stage']}: {result['error']}")
    
    # Show admin dashboard
    dashboard = platform.admin_dashboard_data()
    print(f"\n📊 Admin Dashboard Summary:")
    print(f"   Total users: {dashboard['system_overview']['total_users']}")
    print(f"   Total requests: {dashboard['system_overview']['total_requests']}")
    print(f"   CPU utilization: {dashboard['resource_status']['cpu']['utilization_percent']:.1f}%")
    print(f"   Safety incidents: {dashboard['safety_metrics']['total_incidents']}")


def run_all_solutions():
    """Run all solution demonstrations."""
    print("🦜🔗 LangChain Course - Lesson 8: Agents & Tools Solutions")
    print("=" * 80)
    
    if not llm:
        print("❌ LLM not available. Please check your setup.")
        return
    
    print(f"✅ Using LLM: {type(llm).__name__}")
    
    # Run solutions
    solutions = [
        solution_1_research_assistant_agent,
        solution_2_customer_service_agent,
        solution_3_code_analysis_agent,
        solution_4_multi_agent_collaboration,
        solution_5_production_agent_platform
    ]
    
    for i, solution_func in enumerate(solutions, 1):
        try:
            solution_func()
            if i < len(solutions):
                input(f"\nPress Enter to continue to Solution {i+1} (or Ctrl+C to exit)...")
        except KeyboardInterrupt:
            print(f"\n👋 Stopped at Solution {i}")
            break
        except Exception as e:
            print(f"❌ Error in Solution {i}: {e}")
            continue
    
    print("\n🎉 Solutions demonstrated!")
    print("\n💡 Key Takeaways:")
    print("   • Agent architecture enables complex, multi-step reasoning")
    print("   • Custom tools extend agent capabilities to specific domains")
    print("   • Production agents require safety controls and monitoring")
    print("   • Multi-agent systems can tackle complex collaborative tasks")
    print("   • Proper error handling and fallbacks are essential")


if __name__ == "__main__":
    run_all_solutions()