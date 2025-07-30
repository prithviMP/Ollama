#!/usr/bin/env python3
"""
Lesson 4: Memory & Conversation Management - Exercise Solutions
"""

import os
import sys
import json
import sqlite3
import pickle
from datetime import datetime, timedelta
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
    ConversationSummaryBufferMemory
)
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage, SystemMessage

# Import shared utilities if available
try:
    from utils.llm_setup import setup_llm_providers, get_preferred_llm
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    SHARED_UTILS_AVAILABLE = False


# ===== SOLUTION 1: Multi-User Chat System =====

class ChatSystem:
    """Solution for Exercise 1: Multi-User Chat System"""
    
    def __init__(self, llm=None, session_timeout_minutes=30):
        self.llm = llm
        self.sessions = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        
    def create_user_session(self, user_id: str):
        """Create a new conversation session for user"""
        if not self.llm:
            raise ValueError("LLM not configured")
            
        memory = ConversationBufferMemory(return_messages=True)
        chain = ConversationChain(
            llm=self.llm,
            memory=memory,
            verbose=False
        )
        
        self.sessions[user_id] = {
            "memory": memory,
            "chain": chain,
            "created_at": datetime.now(),
            "last_active": datetime.now(),
            "message_count": 0
        }
        
        print(f"✅ Created session for user: {user_id}")
        return f"Session created for {user_id}"
    
    def chat(self, user_id: str, message: str) -> str:
        """Process message and return response"""
        # Clean up expired sessions first
        self._cleanup_expired_sessions()
        
        # Create session if doesn't exist
        if user_id not in self.sessions:
            self.create_user_session(user_id)
        
        # Update last active time
        self.sessions[user_id]["last_active"] = datetime.now()
        self.sessions[user_id]["message_count"] += 1
        
        # Get response from conversation chain
        response = self.sessions[user_id]["chain"].predict(input=message)
        
        return response
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Return conversation statistics for user"""
        if user_id not in self.sessions:
            return {"error": "User session not found"}
        
        session = self.sessions[user_id]
        memory_messages = len(session["memory"].chat_memory.messages)
        
        return {
            "user_id": user_id,
            "session_created": session["created_at"].isoformat(),
            "last_active": session["last_active"].isoformat(),
            "message_count": session["message_count"],
            "memory_messages": memory_messages,
            "session_duration_minutes": (datetime.now() - session["created_at"]).total_seconds() / 60
        }
    
    def list_active_users(self) -> List[str]:
        """Return list of users with active sessions"""
        self._cleanup_expired_sessions()
        return list(self.sessions.keys())
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = datetime.now()
        expired_users = []
        
        for user_id, session in self.sessions.items():
            if current_time - session["last_active"] > self.session_timeout:
                expired_users.append(user_id)
        
        for user_id in expired_users:
            del self.sessions[user_id]
            print(f"🗑️ Cleaned up expired session for user: {user_id}")


# ===== SOLUTION 2: Long-Term Memory Assistant =====

class LongTermMemoryAssistant:
    """Solution for Exercise 2: Long-Term Memory Assistant"""
    
    def __init__(self, user_id: str, storage_path: str = "user_memories.json", llm=None):
        self.user_id = user_id
        self.storage_path = storage_path
        self.llm = llm
        self.preferences = self._load_preferences()
        
        # Set up conversation memory
        if llm:
            self.conversation_memory = ConversationBufferMemory(return_messages=True)
            self.conversation_chain = ConversationChain(
                llm=llm,
                memory=self.conversation_memory,
                verbose=False
            )
    
    def _load_preferences(self) -> Dict:
        """Load preferences from storage"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    all_preferences = json.load(f)
                    return all_preferences.get(self.user_id, {})
            return {}
        except Exception as e:
            print(f"Error loading preferences: {e}")
            return {}
    
    def _save_preferences(self):
        """Save preferences to storage"""
        try:
            # Load all users' preferences
            all_preferences = {}
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    all_preferences = json.load(f)
            
            # Update this user's preferences
            all_preferences[self.user_id] = self.preferences
            
            # Save back to file
            with open(self.storage_path, 'w') as f:
                json.dump(all_preferences, f, indent=2)
        except Exception as e:
            print(f"Error saving preferences: {e}")
    
    def save_preference(self, key: str, value: Any):
        """Save a user preference"""
        self.preferences[key] = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "type": type(value).__name__
        }
        self._save_preferences()
        print(f"💾 Saved preference: {key} = {value}")
    
    def get_preference(self, key: str) -> Any:
        """Retrieve a user preference"""
        if key in self.preferences:
            return self.preferences[key]["value"]
        return None
    
    def delete_preference(self, key: str):
        """Delete a user preference"""
        if key in self.preferences:
            del self.preferences[key]
            self._save_preferences()
            print(f"🗑️ Deleted preference: {key}")
            return True
        return False
    
    def chat_with_memory(self, message: str) -> str:
        """Process message with long-term memory integration"""
        if not self.llm:
            return "LLM not configured"
        
        # Parse message for memory commands
        message_lower = message.lower()
        
        if "remember that" in message_lower:
            # Extract what to remember
            parts = message.split("remember that", 1)
            if len(parts) > 1:
                memory_content = parts[1].strip()
                # Simple key extraction (you could make this more sophisticated)
                key = f"note_{len(self.preferences) + 1}"
                self.save_preference(key, memory_content)
                return f"I'll remember that: {memory_content}"
        
        elif "what do you remember about" in message_lower:
            # Search preferences for relevant information
            search_term = message_lower.split("what do you remember about", 1)[1].strip()
            relevant_memories = []
            
            for key, pref_data in self.preferences.items():
                if search_term in str(pref_data["value"]).lower():
                    relevant_memories.append(pref_data["value"])
            
            if relevant_memories:
                return f"I remember: {'; '.join(relevant_memories)}"
            else:
                return f"I don't have any specific memories about {search_term}"
        
        # Add preference context to conversation
        if self.preferences:
            context = "My memories about you: " + "; ".join([
                f"{k}: {v['value']}" for k, v in list(self.preferences.items())[:3]
            ])
            enhanced_message = f"{message}\n\nContext: {context}"
        else:
            enhanced_message = message
        
        # Use conversation chain
        response = self.conversation_chain.predict(input=enhanced_message)
        return response
    
    def list_preferences(self) -> Dict:
        """Return all user preferences"""
        return {k: v["value"] for k, v in self.preferences.items()}


# ===== SOLUTION 3: Context-Aware Customer Support =====

class SupportBot:
    """Solution for Exercise 3: Context-Aware Customer Support"""
    
    def __init__(self, llm=None):
        self.llm = llm
        self.tickets = {}
        self.customers = {}
        self.next_ticket_id = 1
    
    def start_ticket(self, customer_id: str, issue_description: str) -> str:
        """Create new support ticket"""
        ticket_id = f"TICKET-{self.next_ticket_id:04d}"
        self.next_ticket_id += 1
        
        # Create conversation memory for this ticket
        memory = ConversationSummaryMemory(
            llm=self.llm,
            return_messages=True
        ) if self.llm else ConversationBufferMemory(return_messages=True)
        
        self.tickets[ticket_id] = {
            "customer_id": customer_id,
            "issue_description": issue_description,
            "status": "new",
            "created_at": datetime.now(),
            "last_updated": datetime.now(),
            "escalation_level": 0,
            "escalation_reasons": [],
            "memory": memory,
            "conversation_chain": ConversationChain(
                llm=self.llm,
                memory=memory,
                verbose=False
            ) if self.llm else None,
            "messages": []
        }
        
        # Initialize customer record if new
        if customer_id not in self.customers:
            self.customers[customer_id] = {
                "tickets": [ticket_id],
                "total_tickets": 1,
                "first_contact": datetime.now()
            }
        else:
            self.customers[customer_id]["tickets"].append(ticket_id)
            self.customers[customer_id]["total_tickets"] += 1
        
        # Add initial context to memory
        initial_context = f"Customer {customer_id} reported: {issue_description}"
        if self.llm:
            self.tickets[ticket_id]["memory"].save_context(
                {"input": "Issue reported"},
                {"output": initial_context}
            )
        
        return ticket_id
    
    def handle_message(self, ticket_id: str, message: str) -> str:
        """Process customer message"""
        if ticket_id not in self.tickets:
            return "Ticket not found"
        
        ticket = self.tickets[ticket_id]
        ticket["last_updated"] = datetime.now()
        ticket["messages"].append({
            "timestamp": datetime.now(),
            "sender": "customer",
            "message": message
        })
        
        # Check for escalation triggers
        self._check_escalation_triggers(ticket_id, message)
        
        # Generate response
        if ticket["conversation_chain"]:
            # Use conversation chain for context-aware response
            response = ticket["conversation_chain"].predict(input=message)
        else:
            # Fallback response
            response = f"Thank you for your message. We're working on ticket {ticket_id}."
        
        ticket["messages"].append({
            "timestamp": datetime.now(),
            "sender": "bot",
            "message": response
        })
        
        return response
    
    def escalate_ticket(self, ticket_id: str, reason: str):
        """Escalate ticket to human agent"""
        if ticket_id not in self.tickets:
            return False
        
        ticket = self.tickets[ticket_id]
        ticket["escalation_level"] += 1
        ticket["escalation_reasons"].append({
            "reason": reason,
            "timestamp": datetime.now(),
            "level": ticket["escalation_level"]
        })
        ticket["status"] = "escalated"
        
        print(f"🔺 Ticket {ticket_id} escalated to level {ticket['escalation_level']}: {reason}")
        return True
    
    def get_conversation_summary(self, ticket_id: str) -> str:
        """Generate summary for agent handoff"""
        if ticket_id not in self.tickets:
            return "Ticket not found"
        
        ticket = self.tickets[ticket_id]
        customer_id = ticket["customer_id"]
        
        summary = f"""
SUPPORT TICKET SUMMARY
======================
Ticket ID: {ticket_id}
Customer: {customer_id}
Status: {ticket["status"]}
Created: {ticket["created_at"].strftime("%Y-%m-%d %H:%M")}
Escalation Level: {ticket["escalation_level"]}

ISSUE DESCRIPTION:
{ticket["issue_description"]}

CUSTOMER HISTORY:
- Total tickets: {self.customers[customer_id]["total_tickets"]}
- First contact: {self.customers[customer_id]["first_contact"].strftime("%Y-%m-%d")}

CONVERSATION HIGHLIGHTS:
"""
        
        # Add recent messages
        recent_messages = ticket["messages"][-6:]  # Last 6 messages
        for msg in recent_messages:
            summary += f"[{msg['timestamp'].strftime('%H:%M')}] {msg['sender']}: {msg['message'][:100]}...\n"
        
        if ticket["escalation_reasons"]:
            summary += "\nESCALATION REASONS:\n"
            for escalation in ticket["escalation_reasons"]:
                summary += f"- Level {escalation['level']}: {escalation['reason']}\n"
        
        return summary
    
    def get_ticket_status(self, ticket_id: str) -> Dict:
        """Return current ticket information"""
        if ticket_id not in self.tickets:
            return {"error": "Ticket not found"}
        
        ticket = self.tickets[ticket_id]
        return {
            "ticket_id": ticket_id,
            "customer_id": ticket["customer_id"],
            "status": ticket["status"],
            "created_at": ticket["created_at"].isoformat(),
            "last_updated": ticket["last_updated"].isoformat(),
            "escalation_level": ticket["escalation_level"],
            "message_count": len(ticket["messages"]),
            "duration_hours": (datetime.now() - ticket["created_at"]).total_seconds() / 3600
        }
    
    def _check_escalation_triggers(self, ticket_id: str, message: str):
        """Check if message should trigger escalation"""
        escalation_keywords = [
            "frustrated", "angry", "cancel", "refund", "manager", 
            "unacceptable", "terrible", "awful", "lawsuit", "lawyer"
        ]
        
        message_lower = message.lower()
        for keyword in escalation_keywords:
            if keyword in message_lower:
                self.escalate_ticket(ticket_id, f"Customer used escalation keyword: {keyword}")
                break
        
        # Check message count
        ticket = self.tickets[ticket_id]
        if len(ticket["messages"]) > 10:
            self.escalate_ticket(ticket_id, "Long conversation - may need human intervention")


# ===== SOLUTION 4: Memory-Optimized Chatbot =====

class OptimizedChatbot:
    """Solution for Exercise 4: Memory-Optimized Chatbot"""
    
    def __init__(self, max_tokens: int = 4000, llm=None):
        self.max_tokens = max_tokens
        self.llm = llm
        self.buffer_memory = ConversationBufferMemory(return_messages=True)
        self.summary_memory = ConversationSummaryMemory(
            llm=llm, return_messages=True
        ) if llm else None
        self.current_mode = "buffer"
        self.optimization_count = 0
        
        self.buffer_chain = ConversationChain(
            llm=llm,
            memory=self.buffer_memory,
            verbose=False
        ) if llm else None
        
        self.summary_chain = ConversationChain(
            llm=llm,
            memory=self.summary_memory,
            verbose=False
        ) if llm and self.summary_memory else None
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough: 4 characters ≈ 1 token)"""
        return len(str(text)) // 4
    
    def get_memory_usage(self) -> Dict:
        """Return current memory usage statistics"""
        if self.current_mode == "buffer":
            messages = self.buffer_memory.chat_memory.messages
            total_chars = sum(len(str(msg.content)) for msg in messages)
            estimated_tokens = self.estimate_tokens(str(total_chars))
        else:
            # For summary mode, estimate based on summary content
            memory_vars = self.summary_memory.load_memory_variables({})
            total_chars = len(str(memory_vars))
            estimated_tokens = self.estimate_tokens(str(total_chars))
        
        return {
            "mode": self.current_mode,
            "estimated_tokens": estimated_tokens,
            "max_tokens": self.max_tokens,
            "usage_percentage": (estimated_tokens / self.max_tokens) * 100,
            "optimization_count": self.optimization_count,
            "message_count": len(self.buffer_memory.chat_memory.messages) if self.current_mode == "buffer" else "N/A"
        }
    
    def optimize_memory(self):
        """Optimize memory when approaching limits"""
        if not self.llm or not self.summary_memory:
            print("⚠️ Cannot optimize: LLM or summary memory not available")
            return
        
        if self.current_mode == "buffer":
            print("⚡ Optimizing memory: switching to summary mode")
            
            # Transfer messages to summary memory
            messages = self.buffer_memory.chat_memory.messages
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    human_msg = messages[i]
                    ai_msg = messages[i + 1]
                    self.summary_memory.save_context(
                        {"input": human_msg.content},
                        {"output": ai_msg.content}
                    )
            
            # Clear buffer memory
            self.buffer_memory.clear()
            self.current_mode = "summary"
            self.optimization_count += 1
            
            print(f"✅ Memory optimized (optimization #{self.optimization_count})")
        else:
            print("⚡ Already in summary mode - creating new summary")
            # Could implement more sophisticated summary compression here
            self.optimization_count += 1
    
    def chat(self, message: str) -> str:
        """Process message with automatic optimization"""
        if not self.llm:
            return "LLM not configured"
        
        # Check if we need to optimize before processing
        usage = self.get_memory_usage()
        if usage["estimated_tokens"] > self.max_tokens * 0.8:  # 80% threshold
            self.optimize_memory()
        
        # Process message with appropriate memory
        if self.current_mode == "buffer":
            response = self.buffer_chain.predict(input=message)
        else:
            response = self.summary_chain.predict(input=message)
        
        # Check again after processing
        usage_after = self.get_memory_usage()
        if usage_after["estimated_tokens"] > self.max_tokens:
            self.optimize_memory()
        
        return response
    
    def force_optimization(self):
        """Manually trigger memory optimization"""
        self.optimize_memory()
    
    def get_conversation_stats(self) -> Dict:
        """Return conversation and memory statistics"""
        usage = self.get_memory_usage()
        
        stats = {
            "memory_mode": self.current_mode,
            "token_usage": usage,
            "optimization_history": self.optimization_count,
            "memory_efficiency": "high" if usage["usage_percentage"] < 70 else "medium" if usage["usage_percentage"] < 90 else "low"
        }
        
        return stats


# ===== SOLUTION 5: Business Process Memory =====

class WorkflowMemorySystem:
    """Solution for Exercise 5: Business Process Memory"""
    
    def __init__(self):
        self.workflows = {}
        self.instances = {}
        self.next_instance_id = 1
    
    def define_workflow(self, workflow_id: str, steps: List[Dict]):
        """Define a new business workflow"""
        self.workflows[workflow_id] = {
            "steps": steps,
            "created_at": datetime.now(),
            "instances_created": 0
        }
        print(f"📋 Defined workflow: {workflow_id} with {len(steps)} steps")
    
    def start_workflow(self, workflow_id: str, user_id: str) -> str:
        """Start workflow instance for user"""
        if workflow_id not in self.workflows:
            return "Workflow not found"
        
        instance_id = f"WF-{self.next_instance_id:04d}"
        self.next_instance_id += 1
        
        self.instances[instance_id] = {
            "workflow_id": workflow_id,
            "user_id": user_id,
            "current_step": 0,
            "status": "active",
            "started_at": datetime.now(),
            "decision_history": [],
            "context_data": {}
        }
        
        self.workflows[workflow_id]["instances_created"] += 1
        
        # Return first step
        first_step = self.workflows[workflow_id]["steps"][0]
        return f"Started workflow {workflow_id} (Instance: {instance_id})\n\n{first_step.get('description', 'No description')}"
    
    def process_decision(self, instance_id: str, decision: str) -> str:
        """Process user decision and advance workflow"""
        if instance_id not in self.instances:
            return "Workflow instance not found"
        
        instance = self.instances[instance_id]
        workflow = self.workflows[instance["workflow_id"]]
        current_step_idx = instance["current_step"]
        
        if current_step_idx >= len(workflow["steps"]):
            return "Workflow already completed"
        
        current_step = workflow["steps"][current_step_idx]
        
        # Record decision
        decision_record = {
            "step_index": current_step_idx,
            "step_name": current_step.get("name", f"Step {current_step_idx + 1}"),
            "decision": decision,
            "timestamp": datetime.now(),
            "previous_context": instance["context_data"].copy()
        }
        instance["decision_history"].append(decision_record)
        
        # Process decision logic
        next_step_idx = self._determine_next_step(current_step, decision, instance)
        
        if next_step_idx is None:
            # Workflow completed
            instance["status"] = "completed"
            instance["completed_at"] = datetime.now()
            return f"Workflow completed! Final decision: {decision}"
        elif next_step_idx == -1:
            # Invalid decision
            return f"Invalid decision for current step. Please choose from: {current_step.get('options', [])}"
        else:
            # Advance to next step
            instance["current_step"] = next_step_idx
            next_step = workflow["steps"][next_step_idx]
            
            return f"Decision recorded: {decision}\n\nNext step: {next_step.get('description', 'No description')}"
    
    def get_workflow_state(self, instance_id: str) -> Dict:
        """Return current workflow state"""
        if instance_id not in self.instances:
            return {"error": "Instance not found"}
        
        instance = self.instances[instance_id]
        workflow = self.workflows[instance["workflow_id"]]
        
        current_step_info = None
        if instance["current_step"] < len(workflow["steps"]):
            current_step_info = workflow["steps"][instance["current_step"]]
        
        return {
            "instance_id": instance_id,
            "workflow_id": instance["workflow_id"],
            "user_id": instance["user_id"],
            "status": instance["status"],
            "current_step": instance["current_step"],
            "total_steps": len(workflow["steps"]),
            "current_step_info": current_step_info,
            "progress_percentage": (instance["current_step"] / len(workflow["steps"])) * 100,
            "decisions_made": len(instance["decision_history"]),
            "started_at": instance["started_at"].isoformat()
        }
    
    def get_decision_history(self, instance_id: str) -> List[Dict]:
        """Return history of decisions made"""
        if instance_id not in self.instances:
            return []
        
        history = []
        for decision in self.instances[instance_id]["decision_history"]:
            history.append({
                "step": decision["step_name"],
                "decision": decision["decision"],
                "timestamp": decision["timestamp"].isoformat(),
                "step_index": decision["step_index"]
            })
        
        return history
    
    def rollback_decision(self, instance_id: str) -> str:
        """Rollback last decision in workflow"""
        if instance_id not in self.instances:
            return "Instance not found"
        
        instance = self.instances[instance_id]
        
        if not instance["decision_history"]:
            return "No decisions to rollback"
        
        # Remove last decision
        last_decision = instance["decision_history"].pop()
        
        # Restore previous state
        instance["current_step"] = last_decision["step_index"]
        instance["context_data"] = last_decision["previous_context"]
        instance["status"] = "active"  # Reactivate if was completed
        
        return f"Rolled back decision: {last_decision['decision']} from step: {last_decision['step_name']}"
    
    def _determine_next_step(self, current_step: Dict, decision: str, instance: Dict) -> Optional[int]:
        """Determine next step based on decision and conditions"""
        # Handle simple linear progression
        if "options" in current_step:
            if decision not in current_step["options"]:
                return -1  # Invalid decision
        
        # Handle conditional branching
        if "branches" in current_step:
            for branch in current_step["branches"]:
                if branch["condition"] == decision:
                    # Update context if specified
                    if "context_updates" in branch:
                        instance["context_data"].update(branch["context_updates"])
                    return branch["next_step"]
        
        # Default: go to next step
        next_idx = instance["current_step"] + 1
        workflow = self.workflows[instance["workflow_id"]]
        
        if next_idx >= len(workflow["steps"]):
            return None  # Workflow completed
        
        return next_idx


# ===== DEMO FUNCTIONS =====

def demo_solution_1():
    """Demo Solution 1: Multi-User Chat System"""
    print("\n🎯 SOLUTION 1 DEMO: Multi-User Chat System")
    print("="*50)
    
    # Mock LLM for demo
    class MockLLM:
        def predict(self, text):
            return f"Mock response to: {text}"
    
    chat_system = ChatSystem(llm=MockLLM())
    
    # Demo multi-user conversations
    print("👤 Alice joins:")
    print(chat_system.chat("alice", "Hi, I'm working on a Python project"))
    
    print("\n👤 Bob joins:")
    print(chat_system.chat("bob", "Hello, I need help with JavaScript"))
    
    print("\n👤 Alice continues:")
    print(chat_system.chat("alice", "What did I mention I was working on?"))
    
    # Show statistics
    print(f"\n📊 Alice's stats: {chat_system.get_user_stats('alice')}")
    print(f"📊 Active users: {chat_system.list_active_users()}")


def demo_solution_2():
    """Demo Solution 2: Long-Term Memory Assistant"""
    print("\n🎯 SOLUTION 2 DEMO: Long-Term Memory Assistant")
    print("="*50)
    
    assistant = LongTermMemoryAssistant("demo_user")
    
    # Save some preferences
    assistant.save_preference("favorite_color", "blue")
    assistant.save_preference("programming_language", "Python")
    assistant.save_preference("hobby", "reading sci-fi")
    
    # Retrieve preferences
    print(f"Favorite color: {assistant.get_preference('favorite_color')}")
    print(f"All preferences: {assistant.list_preferences()}")


def run_all_solutions():
    """Run demonstrations of all solutions"""
    print("🏆 LangChain Lesson 4: Memory Management - SOLUTIONS")
    print("=" * 70)
    
    demo_solution_1()
    demo_solution_2()
    
    print("\n✅ All solution demos completed!")
    print("💡 These are complete implementations you can extend and customize!")


if __name__ == "__main__":
    run_all_solutions() 