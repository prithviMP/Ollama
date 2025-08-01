#!/usr/bin/env python3
"""
Lesson 4: Memory & Conversation Management with LangChain
"""

import os
import sys
import sqlite3
import pickle
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add shared resources to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared-resources'))

# LangChain core imports
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
    VectorStoreRetrieverMemory
)
from langchain.chains import ConversationChain
from langchain.callbacks import get_openai_callback

# Vector store and embedding imports
try:
    from langchain_community.vectorstores import Chroma, FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    VECTORSTORE_AVAILABLE = True
except ImportError:
    VECTORSTORE_AVAILABLE = False
    print("⚠️  Vector stores not available. Install with: pip install chromadb faiss-cpu sentence-transformers")

# Import shared LLM setup
try:
    from utils.llm_setup import setup_llm_providers, get_preferred_llm
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    SHARED_UTILS_AVAILABLE = False
    print("⚠️  Shared utilities not available. Setting up basic providers...")

# Provider imports with error handling
try:
    from langchain_openai import OpenAI, ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI not available. Install with: pip install langchain-openai")

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠️  Anthropic not available. Install with: pip install langchain-anthropic")

try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  Ollama not available. Install with: pip install langchain-ollama")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("⚠️  Google Gemini not available. Install with: pip install langchain-google-genai")

try:
    from langchain_deepseek import ChatDeepSeek
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    print("⚠️  DeepSeek not available. Install with: pip install langchain-deepseek")


def setup_llm_providers_local():
    """
    Local LLM provider setup (fallback if shared utils not available)
    """
    providers = {}
    
    # OpenAI Setup
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            providers["openai"] = OpenAI(
                temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.7)),
                max_tokens=int(os.getenv("MAX_TOKENS", 1500))
            )
            providers["chat_openai"] = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.7))
            )
            print("✅ OpenAI configured successfully")
        except Exception as e:
            print(f"❌ OpenAI configuration failed: {e}")
    
    # Anthropic Setup
    if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
        try:
            providers["anthropic"] = ChatAnthropic(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.7))
            )
            print("✅ Anthropic configured successfully")
        except Exception as e:
            print(f"❌ Anthropic configuration failed: {e}")
    
    # Google Gemini Setup
    if GOOGLE_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
        try:
            providers["google"] = ChatGoogleGenerativeAI(
                model=os.getenv("GOOGLE_MODEL", "gemini-pro"),
                temperature=float(os.getenv("GOOGLE_TEMPERATURE", 0.7)),
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            print("✅ Google Gemini configured successfully")
        except Exception as e:
            print(f"❌ Google Gemini configuration failed: {e}")
    
    # DeepSeek Setup
    if DEEPSEEK_AVAILABLE and os.getenv("DEEPSEEK_API_KEY"):
        try:
            providers["deepseek"] = ChatDeepSeek(
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", 0.7)),
                api_key=os.getenv("DEEPSEEK_API_KEY")
            )
            print("✅ DeepSeek configured successfully")
        except Exception as e:
            print(f"❌ DeepSeek configuration failed: {e}")
    
    # OpenRouter Setup
    if OPENAI_AVAILABLE and os.getenv("OPENROUTER_API_KEY"):
        try:
            providers["openrouter"] = ChatOpenAI(
                model=os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat"),
                temperature=float(os.getenv("OPENROUTER_TEMPERATURE", 0.7)),
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1"
            )
            print("✅ OpenRouter configured successfully")
        except Exception as e:
            print(f"❌ OpenRouter configuration failed: {e}")
    
    # Ollama Setup
    if OLLAMA_AVAILABLE:
        try:
            providers["ollama"] = ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "llama2"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
            print("✅ Ollama configured successfully")
        except Exception as e:
            print(f"❌ Ollama configuration failed: {e}")
    
    return providers


def setup_embeddings():
    """
    Set up embedding model for vector store memory
    """
    try:
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        print(f"✅ Embeddings configured: {model_name}")
        return embeddings
    except Exception as e:
        print(f"❌ Embeddings setup failed: {e}")
        return None


def conversation_buffer_memory_example(llm):
    """
    Demonstrate ConversationBufferMemory - stores full conversation history
    """
    print("\n" + "="*60)
    print("🧠 CONVERSATION BUFFER MEMORY EXAMPLE")
    print("="*60)
    print("📋 Stores complete conversation history in memory")
    
    try:
        # Create buffer memory
        memory = ConversationBufferMemory(return_messages=True)
        
        # Create conversation chain
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=True
        )
        
        print("\n💬 Starting conversation with buffer memory...")
        
        # First interaction
        response1 = conversation.predict(input="Hi, I'm learning about machine learning. Can you help me?")
        print(f"🤖 Response 1: {response1}")
        
        # Second interaction - should remember context
        response2 = conversation.predict(input="What did I just say I was learning about?")
        print(f"🤖 Response 2: {response2}")
        
        # Show memory contents
        print(f"\n📚 Memory Buffer Contents:")
        memory_vars = memory.load_memory_variables({})
        print(f"History: {memory_vars}")
        
        # Show memory statistics
        total_chars = sum(len(str(msg)) for msg in memory.chat_memory.messages)
        print(f"📊 Memory Stats: {len(memory.chat_memory.messages)} messages, ~{total_chars} characters")
        
    except Exception as e:
        print(f"❌ Error in buffer memory example: {e}")


def conversation_summary_memory_example(llm):
    """
    Demonstrate ConversationSummaryMemory - summarizes conversation to save tokens
    """
    print("\n" + "="*60)
    print("📝 CONVERSATION SUMMARY MEMORY EXAMPLE")
    print("="*60)
    print("🔄 Summarizes conversation history to reduce token usage")
    
    try:
        # Create summary memory
        memory = ConversationSummaryMemory(
            llm=llm,
            return_messages=True
        )
        
        # Create conversation chain
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=True
        )
        
        print("\n💬 Starting conversation with summary memory...")
        
        # Simulate a longer conversation
        conversations = [
            "I'm planning a trip to Japan. Can you help me?",
            "I'm interested in visiting Tokyo and Kyoto. What should I see?",
            "How long should I stay in each city?",
            "What's the best way to travel between Tokyo and Kyoto?",
            "Can you summarize our conversation so far?"
        ]
        
        for i, user_input in enumerate(conversations, 1):
            print(f"\n📨 User {i}: {user_input}")
            response = conversation.predict(input=user_input)
            print(f"🤖 Bot {i}: {response[:200]}..." if len(response) > 200 else f"🤖 Bot {i}: {response}")
        
        # Show memory summary
        print(f"\n📋 Conversation Summary:")
        memory_vars = memory.load_memory_variables({})
        print(f"Summary: {memory_vars}")
        
    except Exception as e:
        print(f"❌ Error in summary memory example: {e}")


def conversation_buffer_window_memory_example(llm):
    """
    Demonstrate ConversationBufferWindowMemory - keeps only last K interactions
    """
    print("\n" + "="*60)
    print("🪟 CONVERSATION BUFFER WINDOW MEMORY EXAMPLE")
    print("="*60)
    print("🔄 Keeps only the last K conversation turns in memory")
    
    try:
        # Create window memory (keep last 3 exchanges)
        memory = ConversationBufferWindowMemory(
            k=1,
            return_messages=True
        )
        
        # Create conversation chain
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=True
        )
        
        print("\n💬 Starting conversation with window memory (k=3)...")
        
        # Simulate multiple interactions to see window effect
        interactions = [
            "What's the capital of France?",
            "What about Germany?",
            "And Italy?",
            "What about Spain?",
            "Can you tell me the first capital I asked about?",  # Should forget France
        ]
        
        for i, user_input in enumerate(interactions, 1):
            print(f"\n📨 User {i}: {user_input}")
            response = conversation.predict(input=user_input)
            print(f"🤖 Bot {i}: {response}")
            
            # Show current window contents
            memory_vars = memory.load_memory_variables({})
            num_messages = len(memory.chat_memory.messages)
            print(f"📊 Window Status: {num_messages} messages in memory")
        
    except Exception as e:
        print(f"❌ Error in window memory example: {e}")


def vector_store_memory_example(llm, embeddings):
    """
    Demonstrate VectorStoreRetrieverMemory - semantic memory with similarity search
    """
    print("\n" + "="*60)
    print("🔍 VECTOR STORE MEMORY EXAMPLE")
    print("="*60)
    print("🧮 Uses embeddings for semantic similarity-based memory retrieval")
    
    if not embeddings or not VECTORSTORE_AVAILABLE:
        print("❌ Vector store or embeddings not available")
        return
    
    try:
        # Create vector store
        persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs=dict(k=3))
        
        # Create vector store memory
        memory = VectorStoreRetrieverMemory(retriever=retriever)
        
        print("\n📚 Adding information to vector memory...")
        
        # Add various pieces of information
        memory_items = [
            {"input": "favorite color", "output": "I love blue, it's calming and reminds me of the ocean"},
            {"input": "pet preferences", "output": "I prefer cats because they're independent but affectionate"},
            {"input": "hobby interests", "output": "I enjoy reading science fiction novels and playing chess"},
            {"input": "food preferences", "output": "I'm vegetarian and love Italian cuisine, especially pasta"},
            {"input": "travel dreams", "output": "I want to visit Japan to see the cherry blossoms and temples"},
        ]
        
        for item in memory_items:
            memory.save_context({"input": item["input"]}, {"output": item["output"]})
            print(f"  📝 Saved: {item['input']} -> {item['output'][:50]}...")
        
        print("\n🔍 Testing semantic retrieval...")
        
        # Test semantic queries
        test_queries = [
            "What colors do you like?",
            "Tell me about animals you prefer",
            "What do you like to eat?",
            "Where would you like to go?",
            "What activities do you enjoy?"
        ]
        
        for query in test_queries:
            print(f"\n❓ Query: {query}")
            relevant_docs = memory.load_memory_variables({"prompt": query})
            if "history" in relevant_docs:
                print(f"🎯 Retrieved: {relevant_docs['history']}")
            else:
                print("🔍 No relevant memories found")
        
    except Exception as e:
        print(f"❌ Error in vector store memory example: {e}")


def custom_persistent_memory_example():
    """
    Demonstrate custom memory implementation with persistence
    """
    print("\n" + "="*60)
    print("🛠️ CUSTOM PERSISTENT MEMORY EXAMPLE")
    print("="*60)
    print("💾 Custom memory implementation with SQLite persistence")
    
    try:
        # Create custom memory class
        class PersistentMemory:
            def __init__(self, user_id: str, db_path: str = "memory.db"):
                self.user_id = user_id
                self.db_path = db_path
                self._init_db()
            
            def _init_db(self):
                """Initialize SQLite database"""
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_memory (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
                conn.close()
            
            def save_memory(self, key: str, value: str):
                """Save a memory item"""
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO user_memory (user_id, key, value)
                    VALUES (?, ?, ?)
                ''', (self.user_id, key, value))
                conn.commit()
                conn.close()
                print(f"💾 Saved memory: {key} -> {value[:50]}...")
            
            def get_memory(self, key: str) -> Optional[str]:
                """Retrieve a specific memory"""
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT value FROM user_memory
                    WHERE user_id = ? AND key = ?
                    ORDER BY timestamp DESC LIMIT 1
                ''', (self.user_id, key))
                result = cursor.fetchone()
                conn.close()
                return result[0] if result else None
            
            def get_all_memories(self) -> List[Dict]:
                """Get all memories for user"""
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT key, value, timestamp FROM user_memory
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                ''', (self.user_id,))
                results = cursor.fetchall()
                conn.close()
                return [{"key": r[0], "value": r[1], "timestamp": r[2]} for r in results]
        
        # Demonstrate custom memory
        user_memory = PersistentMemory("user_123")
        
        print("\n📝 Saving user preferences...")
        user_memory.save_memory("name", "Alice Johnson")
        user_memory.save_memory("favorite_language", "Python")
        user_memory.save_memory("project_type", "Machine Learning")
        user_memory.save_memory("experience_level", "Intermediate")
        
        print("\n🔍 Retrieving specific memories...")
        name = user_memory.get_memory("name")
        language = user_memory.get_memory("favorite_language")
        print(f"👤 Name: {name}")
        print(f"💻 Favorite Language: {language}")
        
        print("\n📚 All user memories:")
        all_memories = user_memory.get_all_memories()
        for memory in all_memories:
            print(f"  🧠 {memory['key']}: {memory['value']} ({memory['timestamp']})")
        
    except Exception as e:
        print(f"❌ Error in custom memory example: {e}")


def multi_user_session_example(llm):
    """
    Demonstrate multi-user session management
    """
    print("\n" + "="*60)
    print("👥 MULTI-USER SESSION EXAMPLE")
    print("="*60)
    print("🔄 Managing separate memory contexts for multiple users")
    
    try:
        # Session manager
        class SessionManager:
            def __init__(self):
                self.sessions = {}
            
            def get_or_create_session(self, user_id: str):
                """Get existing session or create new one"""
                if user_id not in self.sessions:
                    memory = ConversationBufferMemory(return_messages=True)
                    chain = ConversationChain(llm=llm, memory=memory, verbose=False)
                    self.sessions[user_id] = {
                        "memory": memory,
                        "chain": chain,
                        "created_at": datetime.now(),
                        "last_active": datetime.now()
                    }
                    print(f"🆕 Created new session for user: {user_id}")
                else:
                    self.sessions[user_id]["last_active"] = datetime.now()
                    print(f"🔄 Retrieved existing session for user: {user_id}")
                
                return self.sessions[user_id]
            
            def chat(self, user_id: str, message: str) -> str:
                """Process user message in their session"""
                session = self.get_or_create_session(user_id)
                response = session["chain"].predict(input=message)
                return response
            
            def get_session_stats(self):
                """Get statistics about all sessions"""
                stats = {}
                for user_id, session in self.sessions.items():
                    stats[user_id] = {
                        "message_count": len(session["memory"].chat_memory.messages),
                        "created_at": session["created_at"].isoformat(),
                        "last_active": session["last_active"].isoformat()
                    }
                return stats
        
        # Demonstrate multi-user sessions
        session_mgr = SessionManager()
        
        print("\n💬 Simulating multi-user conversations...")
        
        # User 1 conversation
        print(f"\n👤 User Alice:")
        response1 = session_mgr.chat("alice", "Hi, I'm working on a Python project")
        print(f"🤖 Response: {response1[:100]}...")
        
        # User 2 conversation
        print(f"\n👤 User Bob:")
        response2 = session_mgr.chat("bob", "Hello, I need help with JavaScript")
        print(f"🤖 Response: {response2[:100]}...")
        
        # Continue User 1 conversation
        print(f"\n👤 User Alice (continued):")
        response3 = session_mgr.chat("alice", "What did I mention I was working on?")
        print(f"🤖 Response: {response3[:100]}...")
        
        # Continue User 2 conversation
        print(f"\n👤 User Bob (continued):")
        response4 = session_mgr.chat("bob", "What programming language did I ask about?")
        print(f"🤖 Response: {response4[:100]}...")
        
        # Show session statistics
        print(f"\n📊 Session Statistics:")
        stats = session_mgr.get_session_stats()
        for user_id, user_stats in stats.items():
            print(f"  👤 {user_id}: {user_stats['message_count']} messages, last active: {user_stats['last_active']}")
        
    except Exception as e:
        print(f"❌ Error in multi-user session example: {e}")


def memory_optimization_example(llm):
    """
    Demonstrate memory optimization techniques
    """
    print("\n" + "="*60)
    print("⚡ MEMORY OPTIMIZATION EXAMPLE")
    print("="*60)
    print("🔧 Techniques for optimizing memory usage and performance")
    
    try:
        # Token counting function
        def count_tokens(text: str) -> int:
            """Rough token estimation (4 chars ≈ 1 token)"""
            return len(text) // 4
        
        # Optimized memory class
        class OptimizedMemory:
            def __init__(self, max_tokens: int = 4000, summary_threshold: int = 3000):
                self.max_tokens = max_tokens
                self.summary_threshold = summary_threshold
                self.buffer_memory = ConversationBufferMemory(return_messages=True)
                self.summary_memory = ConversationSummaryMemory(llm=llm, return_messages=True)
                self.current_mode = "buffer"
            
            def add_message(self, input_text: str, output_text: str):
                """Add message with automatic optimization"""
                self.buffer_memory.save_context({"input": input_text}, {"output": output_text})
                
                # Check if we need to optimize
                current_tokens = self._estimate_tokens()
                print(f"📊 Current memory usage: ~{current_tokens} tokens")
                
                if current_tokens > self.max_tokens:
                    self._optimize_memory()
            
            def _estimate_tokens(self) -> int:
                """Estimate total tokens in memory"""
                messages = self.buffer_memory.chat_memory.messages
                total_chars = sum(len(str(msg.content)) for msg in messages)
                return count_tokens(str(total_chars))
            
            def _optimize_memory(self):
                """Optimize memory by summarizing or truncating"""
                print("⚡ Optimizing memory...")
                
                if self.current_mode == "buffer":
                    # Switch to summary mode
                    print("🔄 Switching to summary mode")
                    messages = self.buffer_memory.chat_memory.messages
                    
                    # Transfer to summary memory
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
                    print("✅ Memory optimized via summarization")
            
            def get_memory_variables(self):
                """Get current memory state"""
                if self.current_mode == "buffer":
                    return self.buffer_memory.load_memory_variables({})
                else:
                    return self.summary_memory.load_memory_variables({})
        
        # Demonstrate optimization
        optimized_memory = OptimizedMemory(max_tokens=500, summary_threshold=300)
        
        print("\n📝 Adding messages to test optimization...")
        
        # Add multiple messages to trigger optimization
        conversations = [
            ("Tell me about artificial intelligence", "AI is a broad field of computer science..."),
            ("What are neural networks?", "Neural networks are computing systems inspired by biological neural networks..."),
            ("How does machine learning work?", "Machine learning is a subset of AI that enables computers to learn..."),
            ("What is deep learning?", "Deep learning is a subset of machine learning that uses artificial neural networks..."),
            ("Can you explain computer vision?", "Computer vision is a field of AI that trains computers to interpret visual information..."),
        ]
        
        for i, (input_text, output_text) in enumerate(conversations, 1):
            print(f"\n💬 Conversation {i}:")
            print(f"👤 User: {input_text}")
            print(f"🤖 AI: {output_text[:50]}...")
            optimized_memory.add_message(input_text, output_text)
        
        print(f"\n📋 Final memory state:")
        final_memory = optimized_memory.get_memory_variables()
        print(f"Mode: {optimized_memory.current_mode}")
        print(f"Content: {str(final_memory)[:200]}...")
        
    except Exception as e:
        print(f"❌ Error in memory optimization example: {e}")


def interactive_memory_demo(providers):
    """
    Interactive demonstration of different memory types
    """
    print("\n" + "="*60)
    print("🎮 INTERACTIVE MEMORY DEMO")
    print("="*60)
    
    if not providers:
        print("❌ No providers available for interactive demo")
        return
    
    try:
        print("Available memory types:")
        memory_types = [
            "ConversationBufferMemory",
            "ConversationSummaryMemory", 
            "ConversationBufferWindowMemory",
            "Custom Persistent Memory"
        ]
        
        for i, memory_type in enumerate(memory_types, 1):
            print(f"  {i}. {memory_type}")
        
        print("  q. Quit")
        
        choice = input("\nSelect memory type (1-4): ").strip()
        
        if choice.lower() == 'q':
            return
        
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(memory_types):
                selected_memory = memory_types[choice_num - 1]
                print(f"\n🔍 Testing {selected_memory}...")
                
                # Get LLM for demo
                llm = list(providers.values())[0]  # Use first available
                
                if choice_num == 1:
                    conversation_buffer_memory_example(llm)
                elif choice_num == 2:
                    conversation_summary_memory_example(llm)
                elif choice_num == 3:
                    conversation_buffer_window_memory_example(llm)
                elif choice_num == 4:
                    custom_persistent_memory_example()
            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Please enter a number")
            
    except Exception as e:
        print(f"❌ Error in interactive demo: {e}")


def main():
    """
    Main function to run all memory examples
    """
    print("🧠 LangChain Course - Lesson 4: Memory & Conversation Management")
    print("=" * 80)
    
    # Set up providers
    if SHARED_UTILS_AVAILABLE:
        providers = setup_llm_providers()
    else:
        providers = setup_llm_providers_local()
    
    if not providers:
        print("❌ Cannot proceed without LLM providers. Please check your setup.")
        return
    
    # Set up embeddings for vector store examples
    embeddings = setup_embeddings()
    
    # Get preferred LLM
    if SHARED_UTILS_AVAILABLE:
        llm = get_preferred_llm(providers, prefer_chat=True)
    else:
        # Use first available chat-capable provider
        chat_providers = [name for name in providers.keys() 
                         if any(x in name for x in ["chat", "anthropic", "google", "deepseek", "openrouter"])]
        llm = providers[chat_providers[0]] if chat_providers else list(providers.values())[0]
    
    print(f"🤖 Using LLM: {type(llm).__name__}")
    
    # Run memory examples
    conversation_buffer_memory_example(llm)
    conversation_summary_memory_example(llm)
    conversation_buffer_window_memory_example(llm)
    
    if embeddings:
        vector_store_memory_example(llm, embeddings)
    
    custom_persistent_memory_example()
    multi_user_session_example(llm)
    memory_optimization_example(llm)
    
    # Interactive demo
    interactive_memory_demo(providers)
    
    print("\n🎉 Lesson 4 completed! Check out the exercises to practice more.")


if __name__ == "__main__":
    main() 