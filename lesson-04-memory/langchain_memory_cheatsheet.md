# LangChain Memory Components - Complete Cheat Sheet

## 🧠 Memory Fundamentals

### What is Memory in LangChain?
Memory allows LLM applications to maintain context across multiple interactions. It stores and retrieves conversation history, user preferences, and contextual information to create more coherent and personalized experiences.

### Key Concepts
- **Memory Variables**: Data that gets passed to the LLM (e.g., conversation history)
- **Memory Keys**: Identifiers for different types of stored information
- **Memory Persistence**: How long and where memory is stored
- **Token Management**: Balancing context retention with token limits

---

## 📚 Built-in Memory Types

### 1. ConversationBufferMemory
**Purpose**: Stores the complete conversation history as-is
**Best For**: Short conversations, debugging, when you need full context
**Limitations**: Can grow very large, may hit token limits

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Basic usage
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# With custom memory keys
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True  # Returns as Message objects instead of strings
)

# Save and load context manually
memory.save_context({"input": "Hi, I'm Alice"}, {"output": "Hello Alice!"})
memory.load_memory_variables({})
```

**When to Use**:
- ✅ Short conversations (< 10-15 exchanges)
- ✅ Debugging conversation flows
- ✅ When you need complete context
- ❌ Long conversations (will hit token limits)
- ❌ Production systems with many users

### 2. ConversationSummaryMemory
**Purpose**: Maintains a running summary of the conversation
**Best For**: Long conversations, production systems, when you need to remember key points
**Advantages**: Token-efficient, scales well

```python
from langchain.memory import ConversationSummaryMemory

# Basic usage
memory = ConversationSummaryMemory(llm=llm)

# With custom settings
memory = ConversationSummaryMemory(
    llm=llm,
    max_token_limit=2000,  # Maximum tokens for summary
    return_messages=True
)

# Manual summary management
memory.save_context({"input": "I like pizza"}, {"output": "That's great!"})
summary = memory.load_memory_variables({})
```

**When to Use**:
- ✅ Long conversations
- ✅ Production systems
- ✅ When you need to remember key points
- ✅ Token-constrained environments
- ❌ When you need exact conversation details
- ❌ Short, simple conversations

### 3. ConversationBufferWindowMemory
**Purpose**: Keeps only the last N interactions
**Best For**: Recent context matters most, sliding window approach
**Advantages**: Predictable memory size, good for recent context

```python
from langchain.memory import ConversationBufferWindowMemory

# Keep last 5 interactions
memory = ConversationBufferWindowMemory(k=5)

# With custom settings
memory = ConversationBufferWindowMemory(
    k=10,  # Number of interactions to keep
    return_messages=True,
    memory_key="recent_history"
)
```

**When to Use**:
- ✅ Recent context is most important
- ✅ Predictable memory usage
- ✅ Chat applications
- ✅ When older context becomes irrelevant
- ❌ When you need long-term memory
- ❌ When all context is equally important

### 4. ConversationSummaryBufferMemory
**Purpose**: Hybrid approach - keeps recent messages + summary of older ones
**Best For**: Best of both worlds - recent detail + long-term memory
**Advantages**: Combines benefits of buffer and summary approaches

```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=2000,  # When to start summarizing
    return_messages=True
)
```

**When to Use**:
- ✅ Complex conversations
- ✅ When both recent and historical context matter
- ✅ Production systems with varied conversation lengths
- ✅ When you want to balance detail and efficiency

---

## 🔍 Vector Store Memory

### VectorStoreRetrieverMemory
**Purpose**: Stores memories as embeddings for semantic retrieval
**Best For**: Long-term memory, finding relevant past conversations
**Advantages**: Semantic search, scalable, can find related memories

```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Setup vector store
embeddings = HuggingFaceEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs=dict(k=3))

# Create memory
memory = VectorStoreRetrieverMemory(
    retriever=retriever,
    memory_key="relevant_history"
)

# Save memories
memory.save_context(
    {"input": "My favorite color is blue"},
    {"output": "I'll remember you like blue"}
)

# Retrieve relevant memories
relevant = memory.load_memory_variables({"prompt": "What colors do I like?"})
```

**When to Use**:
- ✅ Long-term memory systems
- ✅ When you need semantic search
- ✅ Multi-user systems
- ✅ When memories need to be searchable
- ❌ Simple, short conversations
- ❌ When exact conversation order matters

---

## 🛠️ Custom Memory Implementation

### Building Custom Memory Classes

```python
from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import BaseMessage
from typing import Dict, Any

class CustomBusinessMemory(BaseChatMemory):
    """Custom memory for business applications"""
    
    def __init__(self, user_id: str, max_memories: int = 100):
        super().__init__()
        self.user_id = user_id
        self.max_memories = max_memories
        self.business_context = {}
        self.memory_key = "business_history"
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save business context"""
        # Extract business-relevant information
        if "business_decision" in inputs:
            self.business_context[inputs["business_decision"]] = {
                "timestamp": datetime.now(),
                "decision": inputs["business_decision"],
                "reasoning": outputs.get("output", "")
            }
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load relevant business context"""
        # Return business history as context
        return {
            self.memory_key: self.business_context
        }
    
    def clear(self) -> None:
        """Clear all business context"""
        self.business_context.clear()
```

### Multi-User Session Management

```python
class SessionManager:
    """Manages multiple user sessions with isolated memory"""
    
    def __init__(self, llm):
        self.llm = llm
        self.sessions = {}
        self.session_timeout = timedelta(minutes=30)
    
    def get_or_create_session(self, user_id: str):
        """Get existing session or create new one"""
        if user_id not in self.sessions:
            memory = ConversationBufferMemory(return_messages=True)
            chain = ConversationChain(
                llm=self.llm,
                memory=memory,
                verbose=False
            )
            self.sessions[user_id] = {
                "chain": chain,
                "created_at": datetime.now(),
                "last_active": datetime.now()
            }
        return self.sessions[user_id]
    
    def chat(self, user_id: str, message: str) -> str:
        """Process message for specific user"""
        session = self.get_or_create_session(user_id)
        session["last_active"] = datetime.now()
        return session["chain"].predict(input=message)
```

---

## 🎯 Advanced Memory Patterns

### 1. Memory Composition
Combine different memory types for complex applications:

```python
class CompositeMemory:
    """Combines multiple memory types"""
    
    def __init__(self, llm):
        self.recent_memory = ConversationBufferWindowMemory(k=5)
        self.summary_memory = ConversationSummaryMemory(llm=llm)
        self.vector_memory = VectorStoreRetrieverMemory(retriever=retriever)
    
    def get_combined_context(self, inputs):
        recent = self.recent_memory.load_memory_variables(inputs)
        summary = self.summary_memory.load_memory_variables(inputs)
        vector = self.vector_memory.load_memory_variables(inputs)
        
        return {
            "recent_context": recent,
            "summary_context": summary,
            "relevant_memories": vector
        }
```

### 2. Conditional Memory Activation
Activate different memory types based on context:

```python
class ConditionalMemory:
    """Activates memory based on conversation type"""
    
    def __init__(self, llm):
        self.short_term = ConversationBufferMemory()
        self.long_term = ConversationSummaryMemory(llm=llm)
        self.current_mode = "short_term"
    
    def switch_mode(self, conversation_length: int):
        """Switch memory mode based on conversation length"""
        if conversation_length > 10:
            self.current_mode = "long_term"
        else:
            self.current_mode = "short_term"
    
    def get_active_memory(self):
        if self.current_mode == "short_term":
            return self.short_term
        return self.long_term
```

### 3. Memory Compression Techniques

```python
class CompressedMemory:
    """Implements memory compression strategies"""
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.messages = []
        self.compression_threshold = 0.8  # Compress when 80% full
    
    def add_message(self, message: str):
        self.messages.append(message)
        if self._estimate_tokens() > self.max_tokens * self.compression_threshold:
            self._compress_memory()
    
    def _compress_memory(self):
        """Compress memory by summarizing older messages"""
        if len(self.messages) > 10:
            # Keep recent messages, summarize older ones
            recent = self.messages[-5:]
            older = self.messages[:-5]
            summary = self._summarize_messages(older)
            self.messages = [summary] + recent
    
    def _estimate_tokens(self) -> int:
        """Estimate token count"""
        return sum(len(msg.split()) * 1.3 for msg in self.messages)
```

---

## 🚀 Production Memory Management

### 1. Persistent Storage
Store memory across application restarts:

```python
import sqlite3
import json

class PersistentMemory:
    """Memory with SQLite persistence"""
    
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                user_id TEXT,
                memory_key TEXT,
                memory_value TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    def save_memory(self, user_id: str, key: str, value: str):
        """Save memory to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO memories (user_id, memory_key, memory_value) VALUES (?, ?, ?)",
            (user_id, key, value)
        )
        conn.commit()
        conn.close()
    
    def get_memory(self, user_id: str, key: str) -> Optional[str]:
        """Retrieve memory from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT memory_value FROM memories WHERE user_id = ? AND memory_key = ? ORDER BY timestamp DESC LIMIT 1",
            (user_id, key)
        )
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
```

### 2. Memory Optimization
Optimize memory usage for production:

```python
class OptimizedMemory:
    """Memory with automatic optimization"""
    
    def __init__(self, max_tokens: int = 4000, summary_threshold: int = 3000):
        self.max_tokens = max_tokens
        self.summary_threshold = summary_threshold
        self.messages = []
        self.summaries = []
    
    def add_message(self, input_text: str, output_text: str):
        """Add message with automatic optimization"""
        self.messages.append({
            "input": input_text,
            "output": output_text,
            "timestamp": datetime.now()
        })
        
        if self._estimate_tokens() > self.summary_threshold:
            self._optimize_memory()
    
    def _optimize_memory(self):
        """Optimize memory by summarizing older messages"""
        if len(self.messages) > 5:
            # Keep recent messages, summarize older ones
            recent = self.messages[-3:]
            older = self.messages[:-3]
            
            # Create summary of older messages
            summary_text = self._create_summary(older)
            self.summaries.append(summary_text)
            self.messages = recent
    
    def get_memory_variables(self):
        """Get optimized memory context"""
        context_parts = []
        
        # Add summaries
        if self.summaries:
            context_parts.append("Previous conversation summary: " + " ".join(self.summaries))
        
        # Add recent messages
        for msg in self.messages:
            context_parts.append(f"User: {msg['input']}")
            context_parts.append(f"Assistant: {msg['output']}")
        
        return {
            "memory": "\n".join(context_parts)
        }
```

---

## 📊 Memory Usage Patterns

### 1. Token Management
Monitor and manage token usage:

```python
def count_tokens(text: str) -> int:
    """Estimate token count (rough approximation)"""
    return len(text.split()) * 1.3  # Rough estimate

def monitor_memory_usage(memory):
    """Monitor memory token usage"""
    variables = memory.load_memory_variables({})
    total_tokens = sum(count_tokens(str(v)) for v in variables.values())
    return {
        "total_tokens": total_tokens,
        "memory_variables": list(variables.keys()),
        "usage_percentage": (total_tokens / 4000) * 100  # Assuming 4k limit
    }
```

### 2. Memory Statistics
Track memory performance:

```python
class MemoryStats:
    """Track memory performance metrics"""
    
    def __init__(self):
        self.stats = {
            "total_memories": 0,
            "memory_hits": 0,
            "memory_misses": 0,
            "compression_events": 0,
            "average_retrieval_time": 0
        }
    
    def record_memory_access(self, hit: bool, retrieval_time: float):
        """Record memory access statistics"""
        if hit:
            self.stats["memory_hits"] += 1
        else:
            self.stats["memory_misses"] += 1
        
        # Update average retrieval time
        total_accesses = self.stats["memory_hits"] + self.stats["memory_misses"]
        current_avg = self.stats["average_retrieval_time"]
        self.stats["average_retrieval_time"] = (
            (current_avg * (total_accesses - 1) + retrieval_time) / total_accesses
        )
    
    def get_stats(self):
        """Get current statistics"""
        total_accesses = self.stats["memory_hits"] + self.stats["memory_misses"]
        hit_rate = (self.stats["memory_hits"] / total_accesses * 100) if total_accesses > 0 else 0
        
        return {
            **self.stats,
            "hit_rate_percentage": hit_rate
        }
```

---

## 🎯 Best Practices

### 1. Memory Selection Guide

| Use Case | Recommended Memory | Why |
|----------|-------------------|-----|
| Short conversations (< 10 exchanges) | `ConversationBufferMemory` | Full context, simple |
| Long conversations | `ConversationSummaryMemory` | Token efficient, scalable |
| Recent context matters most | `ConversationBufferWindowMemory` | Predictable size |
| Complex applications | `ConversationSummaryBufferMemory` | Best of both worlds |
| Long-term memory | `VectorStoreRetrieverMemory` | Semantic search |
| Multi-user systems | Custom session management | User isolation |

### 2. Performance Optimization

```python
# ✅ DO: Use appropriate memory types
memory = ConversationSummaryMemory(llm=llm)  # For long conversations

# ❌ DON'T: Use buffer memory for long conversations
memory = ConversationBufferMemory()  # Will hit token limits

# ✅ DO: Monitor token usage
def check_memory_usage(memory):
    variables = memory.load_memory_variables({})
    token_count = sum(len(str(v).split()) for v in variables.values())
    if token_count > 3000:  # Near limit
        print("⚠️ Memory approaching token limit")

# ✅ DO: Implement memory cleanup
def cleanup_old_memories(memory, max_age_hours=24):
    # Remove memories older than 24 hours
    pass
```

### 3. Error Handling

```python
class RobustMemory:
    """Memory with error handling"""
    
    def __init__(self, fallback_memory=None):
        self.primary_memory = ConversationSummaryMemory(llm=llm)
        self.fallback_memory = fallback_memory or ConversationBufferMemory()
    
    def load_memory_variables(self, inputs):
        try:
            return self.primary_memory.load_memory_variables(inputs)
        except Exception as e:
            print(f"⚠️ Primary memory failed: {e}")
            return self.fallback_memory.load_memory_variables(inputs)
    
    def save_context(self, inputs, outputs):
        try:
            self.primary_memory.save_context(inputs, outputs)
        except Exception as e:
            print(f"⚠️ Failed to save to primary memory: {e}")
            self.fallback_memory.save_context(inputs, outputs)
```

---

## 🔧 Common Patterns and Solutions

### 1. Multi-User Chat System
```python
class MultiUserChatSystem:
    def __init__(self, llm):
        self.llm = llm
        self.user_sessions = {}
    
    def get_user_memory(self, user_id: str):
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = ConversationBufferMemory()
        return self.user_sessions[user_id]
    
    def chat(self, user_id: str, message: str) -> str:
        memory = self.get_user_memory(user_id)
        chain = ConversationChain(llm=self.llm, memory=memory)
        return chain.predict(input=message)
```

### 2. Long-Term Memory Assistant
```python
class LongTermMemoryAssistant:
    def __init__(self, user_id: str, llm):
        self.user_id = user_id
        self.llm = llm
        self.preferences = {}
        self.conversation_memory = ConversationSummaryMemory(llm=llm)
    
    def save_preference(self, key: str, value: str):
        self.preferences[key] = value
    
    def chat_with_memory(self, message: str) -> str:
        # Include preferences in context
        context = f"User preferences: {self.preferences}\nUser message: {message}"
        chain = ConversationChain(llm=self.llm, memory=self.conversation_memory)
        return chain.predict(input=context)
```

### 3. Context-Aware Support Bot
```python
class SupportBot:
    def __init__(self, llm):
        self.llm = llm
        self.tickets = {}
    
    def start_ticket(self, customer_id: str, issue: str) -> str:
        ticket_id = f"ticket_{customer_id}_{datetime.now().timestamp()}"
        self.tickets[ticket_id] = {
            "customer_id": customer_id,
            "issue": issue,
            "memory": ConversationSummaryMemory(llm=self.llm),
            "status": "open",
            "created_at": datetime.now()
        }
        return ticket_id
    
    def handle_message(self, ticket_id: str, message: str) -> str:
        if ticket_id not in self.tickets:
            return "Ticket not found"
        
        ticket = self.tickets[ticket_id]
        memory = ticket["memory"]
        chain = ConversationChain(llm=self.llm, memory=memory)
        return chain.predict(input=message)
```

---

## 📝 Quick Reference

### Memory Types Summary
- **ConversationBufferMemory**: Complete history, simple
- **ConversationSummaryMemory**: Summarized history, efficient
- **ConversationBufferWindowMemory**: Recent history only
- **ConversationSummaryBufferMemory**: Hybrid approach
- **VectorStoreRetrieverMemory**: Semantic search, long-term

### Key Methods
```python
# Save context
memory.save_context({"input": "user message"}, {"output": "ai response"})

# Load memory variables
variables = memory.load_memory_variables({"prompt": "current input"})

# Clear memory
memory.clear()

# Get memory as messages
messages = memory.chat_memory.messages
```

### Common Parameters
```python
memory = ConversationBufferMemory(
    return_messages=True,      # Return Message objects instead of strings
    memory_key="chat_history", # Custom key for memory variables
    input_key="user_input",    # Custom key for input
    output_key="ai_response"   # Custom key for output
)
```

---

## 🎓 Learning Path

1. **Start Simple**: Use `ConversationBufferMemory` for basic applications
2. **Scale Up**: Switch to `ConversationSummaryMemory` for longer conversations
3. **Add Complexity**: Implement custom memory for specific use cases
4. **Optimize**: Add token management and compression
5. **Production**: Implement persistence and multi-user support

This cheat sheet covers the essential LangChain memory components with practical examples and best practices for building robust conversational AI applications. 