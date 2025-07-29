# Lesson 9: Production & Advanced Patterns with LangChain

## 🎯 Learning Objectives

By the end of this lesson, you will be able to:
- Deploy LangChain applications to production environments
- Implement streaming responses and real-time processing
- Set up comprehensive monitoring and observability
- Optimize performance and manage costs at scale
- Apply advanced LangChain patterns for enterprise applications

## 📚 Concepts Covered

### 1. Production Deployment
- Containerization with Docker and Kubernetes
- Cloud deployment strategies (AWS, GCP, Azure)
- Environment configuration and secrets management
- Load balancing and auto-scaling
- CI/CD pipelines for LangChain applications

### 2. Streaming & Real-time Processing
- Streaming LLM responses for better UX
- WebSocket integration for real-time chat
- Server-sent events (SSE) implementation
- Async processing patterns
- Queue-based architectures

### 3. Monitoring & Observability
- LangSmith integration for tracing and debugging
- Custom callback handlers for logging
- Performance metrics and alerting
- Error tracking and debugging
- Usage analytics and reporting

### 4. Performance Optimization
- Response caching strategies
- Model selection and optimization
- Prompt optimization for production
- Batch processing patterns
- Resource management and scaling

### 5. Advanced Enterprise Patterns
- Multi-tenancy and user isolation
- Security and authentication integration
- Rate limiting and quota management
- Audit logging and compliance
- Disaster recovery and backup strategies

## 🚀 Getting Started

### Prerequisites
- Completed Lessons 1-8 (Complete LangChain Foundation)
- Understanding of web application deployment
- Experience with cloud platforms and containerization

### Setup
```bash
cd lesson-09-production
poetry install && poetry shell
cp env.example .env
python main.py
```

## 📝 Code Examples

### Streaming Response Implementation
```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import LLMResult
import asyncio

class CustomStreamingHandler(StreamingStdOutCallbackHandler):
    def __init__(self, websocket=None):
        self.websocket = websocket
        self.tokens = []
    
    async def on_llm_new_token(self, token: str, **kwargs):
        """Handle new token from LLM."""
        if self.websocket:
            await self.websocket.send_text(token)
        self.tokens.append(token)

# FastAPI streaming endpoint
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        # Receive message
        message = await websocket.receive_text()
        
        # Set up streaming handler
        handler = CustomStreamingHandler(websocket)
        llm = ChatOpenAI(callbacks=[handler], streaming=True)
        
        # Stream response
        await llm.apredict(message)
```

### Production Monitoring System
```python
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
import time
import logging

class ProductionCallbackHandler(BaseCallbackHandler):
    def __init__(self, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id
        self.start_time = None
        self.logger = logging.getLogger("langchain_production")
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Track LLM call start."""
        self.start_time = time.time()
        self.logger.info(f"LLM call started - User: {self.user_id}, Session: {self.session_id}")
    
    def on_llm_end(self, response: LLMResult, **kwargs):
        """Track LLM call completion."""
        duration = time.time() - self.start_time
        tokens_used = response.llm_output.get("token_usage", {})
        
        # Log metrics
        self.logger.info(f"LLM call completed - Duration: {duration:.2f}s, Tokens: {tokens_used}")
        
        # Send to monitoring system
        self._send_metrics({
            "user_id": self.user_id,
            "session_id": self.session_id,
            "duration": duration,
            "tokens": tokens_used,
            "timestamp": time.time()
        })
    
    def on_llm_error(self, error: Exception, **kwargs):
        """Track LLM errors."""
        self.logger.error(f"LLM error - User: {self.user_id}, Error: {str(error)}")

# Production chain with monitoring
class ProductionChain:
    def __init__(self, user_id: str, session_id: str):
        self.callback_handler = ProductionCallbackHandler(user_id, session_id)
        self.llm = ChatOpenAI(callbacks=[self.callback_handler])
        self.chain = self._setup_chain()
    
    def invoke(self, input_data):
        return self.chain.invoke(input_data)
```

### Caching and Performance Optimization
```python
from langchain.cache import InMemoryCache, RedisCache
from langchain.globals import set_llm_cache
import redis

# Set up caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)
set_llm_cache(RedisCache(redis_client))

class OptimizedRAGSystem:
    def __init__(self):
        self.cache = RedisCache(redis_client)
        self.llm = ChatOpenAI(cache=True)
        self.vectorstore = self._setup_vectorstore()
        self.metrics = ProductionMetrics()
    
    def query(self, question: str, user_id: str):
        # Check cache first
        cache_key = f"rag:{hash(question)}"
        cached_result = self.cache.lookup(question, cache_key)
        
        if cached_result:
            self.metrics.record_cache_hit(user_id)
            return cached_result
        
        # Process query
        start_time = time.time()
        
        # Retrieve documents
        docs = self.vectorstore.similarity_search(question, k=3)
        
        # Generate response
        response = self.llm.predict(
            f"Context: {docs}\n\nQuestion: {question}"
        )
        
        # Cache result
        self.cache.update(question, cache_key, response)
        
        # Record metrics
        self.metrics.record_query(
            user_id=user_id,
            duration=time.time() - start_time,
            cache_miss=True
        )
        
        return response
```

### Multi-tenant Production Architecture
```python
class MultiTenantLangChainService:
    def __init__(self):
        self.tenant_configs = {}
        self.rate_limiters = {}
        self.usage_trackers = {}
    
    def get_tenant_chain(self, tenant_id: str):
        """Get or create chain for specific tenant."""
        if tenant_id not in self.tenant_configs:
            config = self._load_tenant_config(tenant_id)
            self.tenant_configs[tenant_id] = self._create_chain(config)
        
        return self.tenant_configs[tenant_id]
    
    async def process_request(self, tenant_id: str, user_id: str, request: dict):
        """Process request with tenant isolation."""
        # Rate limiting
        if not await self._check_rate_limit(tenant_id, user_id):
            raise HTTPException(429, "Rate limit exceeded")
        
        # Usage tracking
        if not self._check_usage_quota(tenant_id):
            raise HTTPException(429, "Usage quota exceeded")
        
        # Get tenant-specific chain
        chain = self.get_tenant_chain(tenant_id)
        
        # Process with monitoring
        try:
            result = await chain.ainvoke(request)
            self._track_usage(tenant_id, user_id, result)
            return result
            
        except Exception as e:
            self._log_error(tenant_id, user_id, str(e))
            raise
```

### Complete Production Deployment
```python
# docker-compose.yml for production deployment
"""
version: '3.8'
services:
  langchain-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/langchain
    depends_on:
      - redis
      - postgres
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: langchain
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
"""

# Kubernetes deployment manifest
"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langchain-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langchain-app
  template:
    metadata:
      labels:
        app: langchain-app
    spec:
      containers:
      - name: langchain-app
        image: langchain-course:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: langchain-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
"""
```

## 🏋️ Exercises

### Exercise 1: Streaming Chat Application
Build a complete streaming chat application with WebSocket support.

### Exercise 2: Production Monitoring Dashboard
Create a comprehensive monitoring system with metrics, alerts, and dashboards.

### Exercise 3: Multi-tenant RAG Service
Implement a multi-tenant RAG service with isolated data and configurations.

### Exercise 4: Performance Optimization Suite
Build tools for caching, rate limiting, and performance optimization.

### Exercise 5: Enterprise Deployment Pipeline
Design a complete CI/CD pipeline for LangChain application deployment.

## 💡 Key Takeaways

1. **Scalability**: Design for scale from the beginning with proper architecture patterns
2. **Monitoring**: Comprehensive observability is crucial for production LLM applications
3. **Performance**: Caching, optimization, and resource management directly impact user experience
4. **Security**: Implement proper authentication, authorization, and data isolation
5. **Reliability**: Error handling, fallbacks, and disaster recovery ensure system stability

## 🎓 Course Completion

**Congratulations!** You have completed the comprehensive 9-hour LangChain course. You now have the knowledge and practical experience to:

- Build sophisticated LLM applications from basic prompts to complex agent systems
- Implement production-ready RAG systems with document processing and vector search
- Deploy and scale LangChain applications in enterprise environments
- Monitor, optimize, and maintain LLM applications in production

### Next Steps
- Apply these concepts to your own projects
- Contribute to the LangChain community
- Explore advanced topics like LangGraph and multi-modal applications
- Stay updated with the latest LangChain developments

---

**Duration:** ~1.5 hours  
**Difficulty:** Advanced  
**Prerequisites:** Lessons 1-8 completed  
**Certification:** Complete LangChain Mastery 