#!/usr/bin/env python3
"""
Lesson 6: Vector Stores & Embeddings - Practice Exercises

Complete these exercises to master vector stores and embedding techniques.
Each exercise focuses on specific aspects of vector database operations and optimization.

Instructions:
1. Implement each exercise function
2. Run individual exercises to test your implementations
3. Check solutions.py for reference implementations
4. Experiment with different configurations and parameters
"""

import os
import sys
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Add shared resources to path
sys.path.append('../shared-resources')

# Load environment variables
load_dotenv()

# LangChain imports
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity

# Shared utilities
from utils.llm_setup import setup_llm_providers, get_preferred_llm
from utils.error_handling import safe_llm_call

# Set up providers
providers = setup_llm_providers()
llm = get_preferred_llm(providers, prefer_chat=True) if providers else None

# Set up embedding model
embedding_model = None
if os.getenv("OPENAI_API_KEY"):
    try:
        embedding_model = OpenAIEmbeddings()
    except:
        pass

if not embedding_model:
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except:
        pass


def exercise_1_embedding_comparison():
    """
    Exercise 1: Multi-Vector Store Implementation and Comparison
    
    Task: Implement a system that can work with multiple vector store backends
    and compare their performance characteristics.
    
    Requirements:
    1. Set up at least 3 different vector stores (Chroma, FAISS, and one cloud-based)
    2. Load the same dataset into all vector stores
    3. Implement performance benchmarking for search operations
    4. Compare memory usage, search speed, and result quality
    5. Generate a comprehensive comparison report
    
    Your implementation should:
    - Handle different vector store APIs uniformly
    - Measure and compare key performance metrics
    - Provide recommendations based on use case requirements
    - Include error handling for unavailable services
    """
    
    print("🏗️ Exercise 1: Multi-Vector Store Implementation and Comparison")
    print("-" * 70)
    
    # TODO: Implement your multi-vector store comparison system
    # Hints:
    # 1. Create a unified interface for different vector stores
    # 2. Implement performance monitoring and benchmarking
    # 3. Compare search accuracy, speed, and resource usage
    # 4. Generate detailed comparison reports
    
    class VectorStoreComparator:
        """Compare multiple vector store implementations."""
        
        def __init__(self, embedding_model):
            self.embedding_model = embedding_model
            self.stores = {}
            self.benchmarks = {}
        
        def setup_stores(self, documents: List[Document]):
            """Set up multiple vector stores with the same documents."""
            # TODO: Implement setup for multiple vector stores
            pass
        
        def benchmark_search_performance(self, queries: List[str], k: int = 5):
            """Benchmark search performance across all stores."""
            # TODO: Implement comprehensive benchmarking
            pass
        
        def compare_result_quality(self, test_queries: List[Dict]):
            """Compare result quality using ground truth data."""
            # TODO: Implement quality comparison
            pass
        
        def generate_comparison_report(self):
            """Generate detailed comparison report."""
            # TODO: Implement report generation
            pass
    
    # Sample documents for testing
    sample_docs = [
        Document(
            page_content="Vector databases store high-dimensional vectors for similarity search.",
            metadata={"category": "database", "difficulty": "beginner"}
        ),
        Document(
            page_content="Embeddings represent text as dense vectors in semantic space.",
            metadata={"category": "embeddings", "difficulty": "intermediate"}
        ),
        # Add more test documents
    ]
    
    # Your implementation here
    print("📝 TODO: Implement multi-vector store comparison")
    print("💡 Consider: performance metrics, scalability, ease of use")


def exercise_2_hybrid_search_system():
    """
    Exercise 2: Hybrid Search System with Ranking
    
    Task: Build a hybrid search system that combines semantic vector search
    with keyword-based search and implements intelligent result ranking.
    
    Requirements:
    1. Implement both dense vector search and sparse keyword search
    2. Create a result fusion algorithm to combine results from both methods
    3. Implement custom ranking that considers multiple relevance signals
    4. Add query expansion and reformulation capabilities
    5. Include evaluation metrics for hybrid search quality
    
    Your implementation should:
    - Balance semantic similarity with keyword relevance
    - Handle different query types effectively
    - Provide tunable parameters for different use cases
    - Demonstrate improved performance over single-method search
    """
    
    print("🔀 Exercise 2: Hybrid Search System with Ranking")
    print("-" * 70)
    
    # TODO: Implement hybrid search system
    # Hints:
    # 1. Combine vector similarity with TF-IDF or BM25 scoring
    # 2. Implement result fusion algorithms (RRF, weighted combination)
    # 3. Add query preprocessing and expansion
    # 4. Create custom ranking functions
    
    class HybridSearchSystem:
        """Advanced hybrid search combining vector and keyword search."""
        
        def __init__(self, embedding_model):
            self.embedding_model = embedding_model
            self.vector_store = None
            self.keyword_index = None
        
        def build_indexes(self, documents: List[Document]):
            """Build both vector and keyword indexes."""
            # TODO: Implement dual indexing
            pass
        
        def vector_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
            """Perform semantic vector search."""
            # TODO: Implement vector search with scores
            pass
        
        def keyword_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
            """Perform keyword-based search."""
            # TODO: Implement keyword search (TF-IDF, BM25, etc.)
            pass
        
        def fuse_results(self, vector_results: List, keyword_results: List, 
                        alpha: float = 0.5) -> List[Tuple[Document, float]]:
            """Fuse results from vector and keyword search."""
            # TODO: Implement result fusion algorithm
            pass
        
        def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5):
            """Perform hybrid search with configurable weighting."""
            # TODO: Implement complete hybrid search pipeline
            pass
        
        def evaluate_search_quality(self, test_queries: List[Dict]):
            """Evaluate hybrid search quality against baselines."""
            # TODO: Implement evaluation metrics
            pass
    
    # Your implementation here
    print("📝 TODO: Implement hybrid search system")
    print("💡 Consider: result fusion algorithms, query preprocessing, evaluation metrics")


def exercise_3_document_recommendation_engine():
    """
    Exercise 3: Document Recommendation Engine with User Profiles
    
    Task: Create an intelligent document recommendation system that learns
    from user interactions and provides personalized recommendations.
    
    Requirements:
    1. Build user profile vectors based on interaction history
    2. Implement content-based and collaborative filtering approaches
    3. Create a feedback loop to improve recommendations over time
    4. Handle cold-start problems for new users
    5. Implement A/B testing framework for recommendation algorithms
    
    Your implementation should:
    - Learn user preferences from implicit and explicit feedback
    - Provide diverse and relevant recommendations
    - Handle concept drift and changing user interests
    - Scale to large document collections and user bases
    """
    
    print("📚 Exercise 3: Document Recommendation Engine")
    print("-" * 70)
    
    # TODO: Implement recommendation engine
    # Hints:
    # 1. Create user embedding profiles from interaction history
    # 2. Implement multiple recommendation strategies
    # 3. Add diversity and novelty considerations
    # 4. Build evaluation framework with multiple metrics
    
    class DocumentRecommendationEngine:
        """Intelligent document recommendation system."""
        
        def __init__(self, embedding_model):
            self.embedding_model = embedding_model
            self.vector_store = None
            self.user_profiles = {}
            self.interaction_history = {}
        
        def build_document_index(self, documents: List[Document]):
            """Build searchable document index."""
            # TODO: Implement document indexing
            pass
        
        def update_user_profile(self, user_id: str, interactions: List[Dict]):
            """Update user profile based on interactions."""
            # TODO: Implement user profile learning
            pass
        
        def content_based_recommendations(self, user_id: str, k: int = 10):
            """Generate content-based recommendations."""
            # TODO: Implement content-based filtering
            pass
        
        def collaborative_recommendations(self, user_id: str, k: int = 10):
            """Generate collaborative filtering recommendations."""
            # TODO: Implement collaborative filtering
            pass
        
        def hybrid_recommendations(self, user_id: str, k: int = 10, 
                                 content_weight: float = 0.7):
            """Generate hybrid recommendations."""
            # TODO: Implement hybrid recommendation strategy
            pass
        
        def handle_cold_start(self, user_id: str, initial_preferences: Dict):
            """Handle cold-start problem for new users."""
            # TODO: Implement cold-start handling
            pass
        
        def evaluate_recommendations(self, test_users: List[str], 
                                   ground_truth: Dict):
            """Evaluate recommendation quality."""
            # TODO: Implement evaluation metrics (precision@k, recall@k, NDCG, etc.)
            pass
    
    # Your implementation here
    print("📝 TODO: Implement document recommendation engine")
    print("💡 Consider: user modeling, recommendation strategies, evaluation metrics")


def exercise_4_realtime_vector_updates():
    """
    Exercise 4: Real-time Vector Store Updates and Consistency
    
    Task: Implement a system for efficiently updating vector stores with
    new content while maintaining search performance and consistency.
    
    Requirements:
    1. Design incremental update strategies for large vector stores
    2. Implement conflict resolution for concurrent updates
    3. Create indexing strategies that minimize search disruption
    4. Handle document versioning and deduplication
    5. Implement rollback and recovery mechanisms
    
    Your implementation should:
    - Support high-frequency updates without performance degradation
    - Maintain search consistency during updates
    - Handle partial failures gracefully
    - Provide monitoring and alerting for update operations
    """
    
    print("⚡ Exercise 4: Real-time Vector Updates System")
    print("-" * 70)
    
    # TODO: Implement real-time update system
    # Hints:
    # 1. Design update batching and scheduling strategies
    # 2. Implement versioning and conflict resolution
    # 3. Create monitoring and health check systems
    # 4. Add rollback and recovery capabilities
    
    class RealTimeVectorUpdateSystem:
        """System for real-time vector store updates."""
        
        def __init__(self, embedding_model, vector_store):
            self.embedding_model = embedding_model
            self.vector_store = vector_store
            self.update_queue = []
            self.version_history = {}
            self.update_stats = {}
        
        def queue_update(self, operation: str, document: Document, 
                        priority: int = 1):
            """Queue a document update operation."""
            # TODO: Implement update queuing system
            pass
        
        def process_update_batch(self, batch_size: int = 100):
            """Process a batch of queued updates."""
            # TODO: Implement batch processing
            pass
        
        def handle_document_conflict(self, existing_doc: Document, 
                                   new_doc: Document):
            """Handle conflicting document updates."""
            # TODO: Implement conflict resolution
            pass
        
        def incremental_index_update(self, updates: List[Dict]):
            """Perform incremental index updates."""
            # TODO: Implement incremental indexing
            pass
        
        def rollback_to_version(self, version_id: str):
            """Rollback to previous version."""
            # TODO: Implement rollback mechanism
            pass
        
        def monitor_update_performance(self):
            """Monitor update system performance."""
            # TODO: Implement performance monitoring
            pass
        
        def health_check(self):
            """Perform system health check."""
            # TODO: Implement health monitoring
            pass
    
    # Your implementation here
    print("📝 TODO: Implement real-time update system")
    print("💡 Consider: update batching, consistency, rollback mechanisms")


def exercise_5_production_vector_architecture():
    """
    Exercise 5: Production-Scale Vector Database Architecture
    
    Task: Design and implement a production-ready vector database architecture
    that can handle enterprise-scale workloads with high availability.
    
    Requirements:
    1. Design horizontal scaling strategies for vector search
    2. Implement load balancing and failover mechanisms
    3. Create comprehensive monitoring and alerting systems
    4. Design backup and disaster recovery procedures
    5. Implement security and access control measures
    6. Optimize for cost-effectiveness and performance
    
    Your implementation should:
    - Handle millions of vectors with sub-second search times
    - Provide 99.9% uptime with automatic failover
    - Include comprehensive observability and diagnostics
    - Support multiple tenants with isolation guarantees
    """
    
    print("🏢 Exercise 5: Production Vector Architecture")
    print("-" * 70)
    
    # TODO: Implement production architecture
    # Hints:
    # 1. Design distributed indexing and search strategies
    # 2. Implement circuit breakers and fallback mechanisms
    # 3. Create comprehensive logging and metrics collection
    # 4. Add security and multi-tenancy support
    
    class ProductionVectorArchitecture:
        """Production-ready vector database architecture."""
        
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.nodes = []
            self.load_balancer = None
            self.monitoring_system = None
            self.backup_system = None
        
        def setup_cluster(self, node_configs: List[Dict]):
            """Set up distributed vector store cluster."""
            # TODO: Implement cluster setup
            pass
        
        def setup_load_balancing(self):
            """Configure load balancing across nodes."""
            # TODO: Implement load balancing
            pass
        
        def implement_failover(self):
            """Implement automatic failover mechanisms."""
            # TODO: Implement failover logic
            pass
        
        def setup_monitoring(self):
            """Set up comprehensive monitoring and alerting."""
            # TODO: Implement monitoring system
            pass
        
        def implement_security(self):
            """Implement security and access control."""
            # TODO: Implement security measures
            pass
        
        def setup_backup_recovery(self):
            """Set up backup and disaster recovery."""
            # TODO: Implement backup/recovery system
            pass
        
        def optimize_performance(self):
            """Optimize system performance and costs."""
            # TODO: Implement performance optimization
            pass
        
        def multi_tenant_isolation(self):
            """Implement multi-tenant isolation."""
            # TODO: Implement tenant isolation
            pass
    
    # Your implementation here
    print("📝 TODO: Implement production architecture")
    print("💡 Consider: scalability, reliability, security, cost optimization")


def run_exercise(exercise_number: int):
    """Run a specific exercise."""
    exercises = {
        1: exercise_1_embedding_comparison,
        2: exercise_2_hybrid_search_system,
        3: exercise_3_document_recommendation_engine,
        4: exercise_4_realtime_vector_updates,
        5: exercise_5_production_vector_architecture,
    }
    
    if exercise_number in exercises:
        print(f"\n🏋️ Running Exercise {exercise_number}")
        print("=" * 80)
        exercises[exercise_number]()
    else:
        print(f"❌ Exercise {exercise_number} not found. Available exercises: 1-5")


def run_all_exercises():
    """Run all exercises in sequence."""
    print("🦜🔗 LangChain Course - Lesson 6: Vector Stores & Embeddings Exercises")
    print("=" * 80)
    
    if not embedding_model:
        print("❌ Embedding model not available. Please check your setup.")
        print("Required: OPENAI_API_KEY or local Sentence Transformers")
        return
    
    print(f"✅ Using Embedding Model: {type(embedding_model).__name__}")
    if llm:
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
    print("   • Experiment with different vector stores")
    print("   • Test with real-world datasets")
    print("   • Consider production deployment strategies")


# Additional helper functions for exercises

def create_test_documents(num_docs: int = 50) -> List[Document]:
    """Create test documents for exercises."""
    topics = [
        "machine learning", "artificial intelligence", "data science",
        "vector databases", "natural language processing", "computer vision",
        "deep learning", "neural networks", "embeddings", "similarity search"
    ]
    
    documents = []
    for i in range(num_docs):
        topic = topics[i % len(topics)]
        content = f"This is a test document about {topic}. " * 10
        
        documents.append(Document(
            page_content=content,
            metadata={
                "doc_id": f"doc_{i}",
                "topic": topic,
                "length": len(content),
                "created_at": time.time()
            }
        ))
    
    return documents


def measure_search_latency(vector_store, queries: List[str], k: int = 5) -> Dict[str, float]:
    """Measure search latency for a vector store."""
    latencies = []
    
    for query in queries:
        start_time = time.time()
        results = vector_store.similarity_search(query, k=k)
        latency = time.time() - start_time
        latencies.append(latency)
    
    return {
        "mean_latency": np.mean(latencies),
        "median_latency": np.median(latencies),
        "p95_latency": np.percentile(latencies, 95),
        "p99_latency": np.percentile(latencies, 99),
        "min_latency": np.min(latencies),
        "max_latency": np.max(latencies)
    }


def calculate_recall_at_k(retrieved_docs: List[Document], 
                         relevant_doc_ids: List[str], k: int) -> float:
    """Calculate recall@k metric."""
    retrieved_ids = [doc.metadata.get('doc_id', '') for doc in retrieved_docs[:k]]
    relevant_retrieved = set(retrieved_ids) & set(relevant_doc_ids)
    
    return len(relevant_retrieved) / len(relevant_doc_ids) if relevant_doc_ids else 0.0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vector Stores & Embeddings Exercises")
    parser.add_argument("--exercise", type=int, help="Run specific exercise (1-5)")
    parser.add_argument("--all", action="store_true", help="Run all exercises")
    
    args = parser.parse_args()
    
    if args.exercise:
        run_exercise(args.exercise)
    elif args.all:
        run_all_exercises()
    else:
        print("🏋️ Vector Stores & Embeddings Practice Exercises")
        print("=" * 50)
        print("Usage:")
        print("  python exercises.py --exercise N  (run exercise N)")
        print("  python exercises.py --all        (run all exercises)")
        print("\nAvailable exercises:")
        print("  1. Multi-Vector Store Implementation and Comparison")
        print("  2. Hybrid Search System with Ranking")
        print("  3. Document Recommendation Engine")
        print("  4. Real-time Vector Updates System")
        print("  5. Production Vector Architecture")