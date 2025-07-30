#!/usr/bin/env python3
"""
Lesson 4: Memory & Conversation Management - Exercises
"""

import os
import sys
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add shared resources to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared-resources'))

# Import necessary LangChain components
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
)
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage, SystemMessage

# Import shared utilities if available
try:
    from utils.llm_setup import setup_llm_providers, get_preferred_llm
except ImportError:
    print("⚠️  Using basic provider setup")


def exercise_1_multi_user_chat():
    """
    Exercise 1: Multi-User Chat System
    Build a conversation system that maintains separate memory for multiple users.
    
    Requirements:
    - Each user should have isolated conversation history
    - Support at least 3 concurrent users
    - Implement session timeout (optional)
    - Display user-specific statistics
    """
    print("\n" + "="*60)
    print("🏋️ EXERCISE 1: Multi-User Chat System")
    print("="*60)
    
    print("""
📝 YOUR TASK:
Implement a ChatSystem class that:
1. Manages separate conversation memory for each user
2. Provides chat functionality with memory isolation
3. Tracks conversation statistics per user
4. Handles user session creation and management

💡 HINTS:
- Use ConversationBufferMemory for each user
- Store sessions in a dictionary with user_id as key
- Consider adding timestamps for session management
- Think about memory cleanup strategies
    """)
    
    # TODO: Implement your solution here
    class ChatSystem:
        def __init__(self):
            # TODO: Initialize your chat system
            pass
        
        def create_user_session(self, user_id: str):
            # TODO: Create a new conversation session for user
            pass
        
        def chat(self, user_id: str, message: str) -> str:
            # TODO: Process message and return response
            pass
        
        def get_user_stats(self, user_id: str) -> Dict:
            # TODO: Return conversation statistics for user
            pass
        
        def list_active_users(self) -> List[str]:
            # TODO: Return list of users with active sessions
            pass
    
    # Test your implementation
    print("\n🧪 Test your implementation:")
    chat_system = ChatSystem()
    
    # Test cases will be added here
    print("✅ Implement the ChatSystem class above and test it!")


def exercise_2_long_term_memory_assistant():
    """
    Exercise 2: Long-Term Memory Assistant
    Create an AI assistant that remembers user preferences across sessions.
    
    Requirements:
    - Persist user preferences to disk
    - Remember user information between program runs
    - Support different types of preferences (strings, lists, etc.)
    - Provide preference management commands
    """
    print("\n" + "="*60)
    print("🏋️ EXERCISE 2: Long-Term Memory Assistant")
    print("="*60)
    
    print("""
📝 YOUR TASK:
Create a LongTermMemoryAssistant that:
1. Stores user preferences persistently (JSON, SQLite, or pickle)
2. Remembers information across program restarts
3. Provides commands to set/get/delete preferences
4. Integrates with conversation memory

💡 HINTS:
- Use JSON files or SQLite for persistence
- Create commands like "remember that I like pizza"
- Parse user messages for memory operations
- Combine with conversation memory for context
    """)
    
    # TODO: Implement your solution here
    class LongTermMemoryAssistant:
        def __init__(self, user_id: str, storage_path: str = "user_memories.json"):
            # TODO: Initialize assistant with persistent storage
            pass
        
        def save_preference(self, key: str, value: Any):
            # TODO: Save a user preference
            pass
        
        def get_preference(self, key: str) -> Any:
            # TODO: Retrieve a user preference
            pass
        
        def delete_preference(self, key: str):
            # TODO: Delete a user preference
            pass
        
        def chat_with_memory(self, message: str) -> str:
            # TODO: Process message with long-term memory integration
            pass
        
        def list_preferences(self) -> Dict:
            # TODO: Return all user preferences
            pass
    
    print("\n🧪 Test your implementation:")
    print("✅ Implement the LongTermMemoryAssistant class above!")


def exercise_3_context_aware_support():
    """
    Exercise 3: Context-Aware Customer Support
    Implement a support bot that maintains conversation history and escalation context.
    
    Requirements:
    - Track support ticket context across conversation
    - Implement escalation levels with memory
    - Remember customer information and issue history
    - Provide conversation summaries for handoffs
    """
    print("\n" + "="*60)
    print("🏋️ EXERCISE 3: Context-Aware Customer Support")
    print("="*60)
    
    print("""
📝 YOUR TASK:
Build a SupportBot that:
1. Maintains ticket context throughout conversation
2. Tracks escalation levels and reasons
3. Stores customer information and issue details
4. Generates conversation summaries for agent handoffs

💡 HINTS:
- Use ConversationSummaryMemory for efficient context
- Create a ticket state system (new, in-progress, escalated, resolved)
- Store customer data separately from conversation memory
- Implement escalation triggers based on keywords or conversation length
    """)
    
    # TODO: Implement your solution here
    class SupportBot:
        def __init__(self):
            # TODO: Initialize support bot with memory systems
            pass
        
        def start_ticket(self, customer_id: str, issue_description: str) -> str:
            # TODO: Create new support ticket
            pass
        
        def handle_message(self, ticket_id: str, message: str) -> str:
            # TODO: Process customer message
            pass
        
        def escalate_ticket(self, ticket_id: str, reason: str):
            # TODO: Escalate ticket to human agent
            pass
        
        def get_conversation_summary(self, ticket_id: str) -> str:
            # TODO: Generate summary for agent handoff
            pass
        
        def get_ticket_status(self, ticket_id: str) -> Dict:
            # TODO: Return current ticket information
            pass
    
    print("\n🧪 Test your implementation:")
    print("✅ Implement the SupportBot class above!")


def exercise_4_memory_optimized_chatbot():
    """
    Exercise 4: Memory-Optimized Chatbot
    Design a chatbot that efficiently manages memory for long conversations.
    
    Requirements:
    - Automatically switch between memory types based on conversation length
    - Implement token-aware memory management
    - Preserve important context while optimizing memory usage
    - Provide memory usage statistics and controls
    """
    print("\n" + "="*60)
    print("🏋️ EXERCISE 4: Memory-Optimized Chatbot")
    print("="*60)
    
    print("""
📝 YOUR TASK:
Create an OptimizedChatbot that:
1. Dynamically switches between ConversationBufferMemory and ConversationSummaryMemory
2. Monitors token usage and optimizes automatically
3. Preserves important context during optimization
4. Provides memory statistics and manual controls

💡 HINTS:
- Start with buffer memory, switch to summary when token limit reached
- Implement token counting (rough: 4 characters ≈ 1 token)
- Consider hybrid approaches using ConversationSummaryBufferMemory
- Add importance scoring for context preservation
    """)
    
    # TODO: Implement your solution here
    class OptimizedChatbot:
        def __init__(self, max_tokens: int = 4000):
            # TODO: Initialize with memory optimization settings
            pass
        
        def estimate_tokens(self, text: str) -> int:
            # TODO: Estimate token count for text
            pass
        
        def get_memory_usage(self) -> Dict:
            # TODO: Return current memory usage statistics
            pass
        
        def optimize_memory(self):
            # TODO: Optimize memory when approaching limits
            pass
        
        def chat(self, message: str) -> str:
            # TODO: Process message with automatic optimization
            pass
        
        def force_optimization(self):
            # TODO: Manually trigger memory optimization
            pass
        
        def get_conversation_stats(self) -> Dict:
            # TODO: Return conversation and memory statistics
            pass
    
    print("\n🧪 Test your implementation:")
    print("✅ Implement the OptimizedChatbot class above!")


def exercise_5_business_process_memory():
    """
    Exercise 5: Business Process Memory
    Build a system that remembers complex business workflows and user decisions.
    
    Requirements:
    - Track multi-step business processes with decision points
    - Remember user choices and workflow state
    - Support workflow branching and conditional logic
    - Provide process history and rollback capabilities
    """
    print("\n" + "="*60)
    print("🏋️ EXERCISE 5: Business Process Memory")
    print("="*60)
    
    print("""
📝 YOUR TASK:
Implement a WorkflowMemorySystem that:
1. Tracks complex business workflows with multiple steps
2. Remembers user decisions at each decision point
3. Supports workflow branching based on conditions
4. Provides workflow history and rollback functionality

💡 HINTS:
- Design a workflow state machine with steps and transitions
- Store decision history with timestamps and reasoning
- Implement conditional branching logic
- Consider using a graph structure for complex workflows
    """)
    
    # TODO: Implement your solution here
    class WorkflowMemorySystem:
        def __init__(self):
            # TODO: Initialize workflow tracking system
            pass
        
        def define_workflow(self, workflow_id: str, steps: List[Dict]):
            # TODO: Define a new business workflow
            pass
        
        def start_workflow(self, workflow_id: str, user_id: str) -> str:
            # TODO: Start workflow instance for user
            pass
        
        def process_decision(self, instance_id: str, decision: str) -> str:
            # TODO: Process user decision and advance workflow
            pass
        
        def get_workflow_state(self, instance_id: str) -> Dict:
            # TODO: Return current workflow state
            pass
        
        def get_decision_history(self, instance_id: str) -> List[Dict]:
            # TODO: Return history of decisions made
            pass
        
        def rollback_decision(self, instance_id: str) -> str:
            # TODO: Rollback last decision in workflow
            pass
    
    print("\n🧪 Test your implementation:")
    print("✅ Implement the WorkflowMemorySystem class above!")


def run_all_exercises():
    """
    Run all exercises
    """
    print("🏋️ LangChain Lesson 4: Memory Management - Exercises")
    print("=" * 70)
    
    exercises = [
        exercise_1_multi_user_chat,
        exercise_2_long_term_memory_assistant,
        exercise_3_context_aware_support,
        exercise_4_memory_optimized_chatbot,
        exercise_5_business_process_memory
    ]
    
    for exercise in exercises:
        exercise()
        input("\nPress Enter to continue to next exercise...")


if __name__ == "__main__":
    run_all_exercises() 