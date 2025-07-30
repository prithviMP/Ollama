#!/usr/bin/env python3
"""
Lesson 6: Vector Stores & Embeddings - Solution Implementations

Reference implementations for all vector store and embedding exercises.
These solutions demonstrate best practices and advanced techniques.

Study these implementations to understand optimal approaches to vector database operations.
"""

import os
import sys
import time
import json
import logging
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
from dotenv import load_dotenv

# Add shared resources to path
sys.path.append('../shared-resources')

# Load environment variables
load_dotenv()

# LangChain imports
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.callbacks import get_openai_callback

# ML and analysis imports
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, ndcg_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Shared utilities
from utils.llm_setup import setup_llm_providers, get_preferred_llm
from utils.error_handling import safe_llm_call

# Set up providers
providers = setup_llm_providers()
llm = get_preferred_llm(providers, prefer_chat=True) if providers else None
embedding_model = OpenAIEmbeddings() if os.getenv("OPENAI_API_KEY") else HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    latency_mean: float
    latency_p95: float
    throughput: float
    memory_usage: float
    accuracy: float


@dataclass
class UserInteraction:
    """User interaction data structure."""
    user_id: str
    document_id: str
    interaction_type: str  # 'view', 'like', 'share', 'bookmark'
    duration: float
    timestamp: float
    rating: Optional[float] = None


def solution_1_embedding_comparison():
    """
    Solution 1: Multi-Vector Store Implementation and Comparison
    """
    print("🏗️ Solution 1: Multi-Vector Store Implementation and Comparison")
    print("-" * 70)
    
    class UnifiedVectorStoreInterface:
        """Unified interface for different vector store implementations."""
        
        def __init__(self, store_type: str, embedding_model, config: Dict = None):
            self.store_type = store_type
            self.embedding_model = embedding_model
            self.config = config or {}
            self.store = None
            self.documents = []
        
        def setup_store(self, documents: List[Document]):
            """Set up the vector store with documents."""
            self.documents = documents
            
            if self.store_type == "chroma":
                self.store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embedding_model,
                    persist_directory=self.config.get("persist_dir", "./chroma_test")
                )
            
            elif self.store_type == "faiss":
                self.store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embedding_model
                )
            
            else:
                raise ValueError(f"Unsupported store type: {self.store_type}")
        
        def search(self, query: str, k: int = 5) -> List[Document]:
            """Perform similarity search."""
            return self.store.similarity_search(query, k=k)
        
        def search_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
            """Perform similarity search with scores."""
            return self.store.similarity_search_with_score(query, k=k)
        
        def get_memory_usage(self) -> float:
            """Estimate memory usage in MB."""
            if self.store_type == "faiss" and hasattr(self.store, 'index'):
                # FAISS index memory estimation
                return self.store.index.ntotal * self.store.index.d * 4 / (1024 * 1024)  # 4 bytes per float
            else:
                # Rough estimation for other stores
                return len(self.documents) * 1000 * 4 / (1024 * 1024)  # Assume 1000 dim embeddings
    
    class VectorStoreComparator:
        """Comprehensive vector store comparison system."""
        
        def __init__(self, embedding_model):
            self.embedding_model = embedding_model
            self.stores = {}
            self.benchmarks = {}
        
        def setup_stores(self, documents: List[Document]):
            """Set up multiple vector stores with the same documents."""
            store_configs = {
                "chroma": {"persist_dir": "./chroma_comparison"},
                "faiss": {}
            }
            
            for store_type, config in store_configs.items():
                try:
                    print(f"Setting up {store_type}...")
                    store = UnifiedVectorStoreInterface(store_type, self.embedding_model, config)
                    store.setup_store(documents)
                    self.stores[store_type] = store
                    print(f"✅ {store_type} setup complete")
                except Exception as e:
                    print(f"❌ {store_type} setup failed: {e}")
        
        def benchmark_search_performance(self, queries: List[str], k: int = 5) -> Dict[str, Dict]:
            """Comprehensive search performance benchmarking."""
            results = {}
            
            for store_name, store in self.stores.items():
                print(f"\n📊 Benchmarking {store_name}...")
                
                latencies = []
                accuracies = []
                memory_usage = store.get_memory_usage()
                
                # Warmup
                for _ in range(5):
                    store.search(queries[0], k=k)
                
                # Actual benchmarking
                start_time = time.time()
                for query in queries:
                    query_start = time.time()
                    results_docs = store.search(query, k=k)
                    query_latency = time.time() - query_start
                    latencies.append(query_latency)
                
                total_time = time.time() - start_time
                throughput = len(queries) / total_time
                
                # Calculate performance metrics
                metrics = PerformanceMetrics(
                    latency_mean=np.mean(latencies),
                    latency_p95=np.percentile(latencies, 95),
                    throughput=throughput,
                    memory_usage=memory_usage,
                    accuracy=0.85  # Placeholder - would need ground truth for real accuracy
                )
                
                results[store_name] = {
                    "metrics": metrics,
                    "latencies": latencies,
                    "raw_stats": {
                        "total_queries": len(queries),
                        "total_time": total_time,
                        "avg_latency": metrics.latency_mean,
                        "p95_latency": metrics.latency_p95,
                        "throughput_qps": throughput,
                        "memory_mb": memory_usage
                    }
                }
                
                print(f"  Avg Latency: {metrics.latency_mean:.4f}s")
                print(f"  P95 Latency: {metrics.latency_p95:.4f}s")
                print(f"  Throughput: {throughput:.2f} queries/sec")
                print(f"  Memory Usage: {memory_usage:.2f} MB")
            
            return results
        
        def compare_result_quality(self, test_queries: List[Dict]) -> Dict[str, float]:
            """Compare result quality using overlap analysis."""
            quality_scores = {}
            
            for store_name, store in self.stores.items():
                overlaps = []
                
                for query_data in test_queries:
                    query = query_data["query"]
                    
                    # Get results from current store
                    results = store.search(query, k=5)
                    result_ids = [doc.metadata.get("doc_id", hash(doc.page_content)) for doc in results]
                    
                    # Compare with other stores
                    other_stores = {k: v for k, v in self.stores.items() if k != store_name}
                    for other_name, other_store in other_stores.items():
                        other_results = other_store.search(query, k=5)
                        other_ids = [doc.metadata.get("doc_id", hash(doc.page_content)) for doc in other_results]
                        
                        # Calculate overlap
                        overlap = len(set(result_ids) & set(other_ids)) / len(set(result_ids) | set(other_ids))
                        overlaps.append(overlap)
                
                quality_scores[store_name] = np.mean(overlaps) if overlaps else 0.0
            
            return quality_scores
        
        def generate_comparison_report(self, benchmark_results: Dict, quality_scores: Dict) -> str:
            """Generate comprehensive comparison report."""
            report = "# Vector Store Comparison Report\n\n"
            
            # Performance comparison table
            report += "## Performance Comparison\n\n"
            report += "| Store | Avg Latency | P95 Latency | Throughput | Memory (MB) |\n"
            report += "|-------|-------------|-------------|------------|-------------|\n"
            
            for store_name, data in benchmark_results.items():
                stats = data["raw_stats"]
                report += f"| {store_name} | {stats['avg_latency']:.4f}s | {stats['p95_latency']:.4f}s | {stats['throughput_qps']:.2f} | {stats['memory_mb']:.2f} |\n"
            
            # Quality scores
            report += "\n## Result Quality Scores\n\n"
            for store_name, score in quality_scores.items():
                report += f"- **{store_name}**: {score:.3f}\n"
            
            # Recommendations
            report += "\n## Recommendations\n\n"
            
            # Find best performers in each category
            best_latency = min(benchmark_results.items(), key=lambda x: x[1]["raw_stats"]["avg_latency"])
            best_throughput = max(benchmark_results.items(), key=lambda x: x[1]["raw_stats"]["throughput_qps"])
            best_memory = min(benchmark_results.items(), key=lambda x: x[1]["raw_stats"]["memory_mb"])
            
            report += f"- **Best Latency**: {best_latency[0]} ({best_latency[1]['raw_stats']['avg_latency']:.4f}s)\n"
            report += f"- **Best Throughput**: {best_throughput[0]} ({best_throughput[1]['raw_stats']['throughput_qps']:.2f} QPS)\n"
            report += f"- **Best Memory Efficiency**: {best_memory[0]} ({best_memory[1]['raw_stats']['memory_mb']:.2f} MB)\n"
            
            # Use case recommendations
            report += "\n### Use Case Recommendations\n\n"
            report += "- **Chroma**: Best for development and prototyping with persistence needs\n"
            report += "- **FAISS**: Best for high-performance production deployments\n"
            report += "- **Cloud Services**: Best for managed, scalable solutions\n"
            
            return report
    
    # Demo the comparison system
    if embedding_model:
        # Create test documents
        test_docs = [
            Document(
                page_content=f"This is test document {i} about machine learning and artificial intelligence. " * 5,
                metadata={"doc_id": f"doc_{i}", "category": "ML", "length": 200}
            )
            for i in range(20)
        ] + [
            Document(
                page_content=f"This is test document {i} about databases and data storage systems. " * 5,
                metadata={"doc_id": f"doc_{i}", "category": "DB", "length": 200}
            )
            for i in range(20, 40)
        ]
        
        # Run comparison
        comparator = VectorStoreComparator(embedding_model)
        comparator.setup_stores(test_docs)
        
        # Benchmark performance
        test_queries = [
            "machine learning algorithms",
            "database optimization",
            "artificial intelligence applications",
            "data storage solutions",
            "performance benchmarking"
        ]
        
        benchmark_results = comparator.benchmark_search_performance(test_queries)
        
        # Compare quality
        test_query_data = [{"query": q} for q in test_queries]
        quality_scores = comparator.compare_result_quality(test_query_data)
        
        # Generate report
        report = comparator.generate_comparison_report(benchmark_results, quality_scores)
        
        print(f"\n📄 Comparison Report:")
        print(report[:1000] + "..." if len(report) > 1000 else report)
        
        # Save full report
        with open("vector_store_comparison_report.md", "w") as f:
            f.write(report)
        print("\n💾 Full report saved to 'vector_store_comparison_report.md'")


def solution_2_hybrid_search_system():
    """
    Solution 2: Hybrid Search System with Ranking
    """
    print("\n🔀 Solution 2: Hybrid Search System with Ranking")
    print("-" * 70)
    
    class HybridSearchSystem:
        """Advanced hybrid search combining vector and keyword search."""
        
        def __init__(self, embedding_model):
            self.embedding_model = embedding_model
            self.vector_store = None
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            self.documents = []
            self.doc_texts = []
        
        def build_indexes(self, documents: List[Document]):
            """Build both vector and keyword indexes."""
            self.documents = documents
            self.doc_texts = [doc.page_content for doc in documents]
            
            print("🔧 Building vector index...")
            self.vector_store = FAISS.from_documents(documents, self.embedding_model)
            
            print("🔧 Building TF-IDF index...")
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2),
                max_df=0.8,
                min_df=2
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.doc_texts)
            
            print("✅ Both indexes built successfully")
        
        def vector_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
            """Perform semantic vector search with scores."""
            results = self.vector_store.similarity_search_with_score(query, k=k)
            # Convert distance to similarity (assuming cosine distance)
            normalized_results = []
            for doc, distance in results:
                similarity = 1 / (1 + distance)  # Convert distance to similarity
                normalized_results.append((doc, similarity))
            return normalized_results
        
        def keyword_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
            """Perform TF-IDF based keyword search."""
            query_vector = self.tfidf_vectorizer.transform([query])
            scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top k results
            top_indices = np.argsort(scores)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include non-zero scores
                    results.append((self.documents[idx], scores[idx]))
            
            return results
        
        def reciprocal_rank_fusion(self, rankings: List[List[Tuple]], k: int = 60) -> List[Tuple]:
            """Implement Reciprocal Rank Fusion (RRF) algorithm."""
            doc_scores = defaultdict(float)
            
            for ranking in rankings:
                for rank, (doc, score) in enumerate(ranking, 1):
                    doc_id = hash(doc.page_content)  # Use content hash as ID
                    doc_scores[doc_id] += 1.0 / (k + rank)
            
            # Get documents for the best scores
            result_docs = {}
            for ranking in rankings:
                for doc, score in ranking:
                    doc_id = hash(doc.page_content)
                    if doc_id not in result_docs:
                        result_docs[doc_id] = doc
            
            # Sort by RRF score
            sorted_scores = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            
            return [(result_docs[doc_id], score) for doc_id, score in sorted_scores 
                   if doc_id in result_docs]
        
        def weighted_fusion(self, vector_results: List[Tuple], keyword_results: List[Tuple], 
                          alpha: float = 0.7) -> List[Tuple[Document, float]]:
            """Weighted combination of vector and keyword results."""
            doc_scores = defaultdict(float)
            all_docs = {}
            
            # Add vector search scores
            for doc, score in vector_results:
                doc_id = hash(doc.page_content)
                doc_scores[doc_id] += alpha * score
                all_docs[doc_id] = doc
            
            # Add keyword search scores
            for doc, score in keyword_results:
                doc_id = hash(doc.page_content)
                doc_scores[doc_id] += (1 - alpha) * score
                all_docs[doc_id] = doc
            
            # Sort by combined score
            sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            
            return [(all_docs[doc_id], score) for doc_id, score in sorted_results]
        
        def expand_query(self, query: str) -> str:
            """Simple query expansion using synonyms and related terms."""
            # This is a simplified implementation - in practice, you might use
            # word embeddings, WordNet, or domain-specific thesauri
            expansion_terms = {
                "machine learning": ["ML", "artificial intelligence", "AI", "algorithms"],
                "database": ["DB", "data storage", "DBMS", "SQL"],
                "search": ["retrieval", "find", "lookup", "query"],
                "vector": ["embedding", "representation", "feature"],
            }
            
            expanded_query = query
            for term, synonyms in expansion_terms.items():
                if term.lower() in query.lower():
                    expanded_query += " " + " ".join(synonyms)
            
            return expanded_query
        
        def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.7, 
                         use_expansion: bool = True, fusion_method: str = "weighted") -> List[Tuple[Document, float]]:
            """Perform hybrid search with configurable parameters."""
            
            # Optionally expand query
            search_query = self.expand_query(query) if use_expansion else query
            
            # Get results from both methods
            vector_results = self.vector_search(search_query, k=k*2)
            keyword_results = self.keyword_search(search_query, k=k*2)
            
            print(f"🔍 Vector search found {len(vector_results)} results")
            print(f"🔍 Keyword search found {len(keyword_results)} results")
            
            # Fuse results
            if fusion_method == "weighted":
                fused_results = self.weighted_fusion(vector_results, keyword_results, alpha)
            elif fusion_method == "rrf":
                fused_results = self.reciprocal_rank_fusion([vector_results, keyword_results])
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")
            
            return fused_results[:k]
        
        def evaluate_search_quality(self, test_queries: List[Dict]) -> Dict[str, float]:
            """Evaluate hybrid search against individual methods."""
            results = {
                "vector_only": [],
                "keyword_only": [],
                "hybrid_weighted": [],
                "hybrid_rrf": []
            }
            
            for query_data in test_queries:
                query = query_data["query"]
                
                # Vector only
                vector_results = self.vector_search(query, k=5)
                results["vector_only"].append(len(vector_results))
                
                # Keyword only
                keyword_results = self.keyword_search(query, k=5)
                results["keyword_only"].append(len(keyword_results))
                
                # Hybrid weighted
                hybrid_weighted = self.hybrid_search(query, k=5, fusion_method="weighted")
                results["hybrid_weighted"].append(len(hybrid_weighted))
                
                # Hybrid RRF
                hybrid_rrf = self.hybrid_search(query, k=5, fusion_method="rrf")
                results["hybrid_rrf"].append(len(hybrid_rrf))
            
            # Calculate average results returned
            evaluation = {}
            for method, counts in results.items():
                evaluation[method] = {
                    "avg_results": np.mean(counts),
                    "coverage": sum(1 for c in counts if c > 0) / len(counts)
                }
            
            return evaluation
    
    # Demo the hybrid search system
    if embedding_model:
        # Create diverse test documents
        test_docs = []
        
        # ML/AI documents
        for i in range(15):
            content = f"Machine learning document {i}. This covers algorithms, neural networks, deep learning, and artificial intelligence applications. AI and ML are transforming technology."
            test_docs.append(Document(
                page_content=content,
                metadata={"doc_id": f"ml_{i}", "category": "ML"}
            ))
        
        # Database documents
        for i in range(15):
            content = f"Database document {i}. This covers SQL, NoSQL, data storage, indexing, query optimization, and database management systems. DBMS and data warehousing."
            test_docs.append(Document(
                page_content=content,
                metadata={"doc_id": f"db_{i}", "category": "DB"}
            ))
        
        # Create and test hybrid search system
        hybrid_system = HybridSearchSystem(embedding_model)
        hybrid_system.build_indexes(test_docs)
        
        # Test different search methods
        test_queries = [
            "machine learning algorithms and AI",
            "database optimization techniques",
            "neural networks and deep learning",
            "SQL query performance"
        ]
        
        print("\n🧪 Testing Hybrid Search Methods:")
        
        for query in test_queries:
            print(f"\n❓ Query: '{query}'")
            
            # Test different fusion methods
            weighted_results = hybrid_system.hybrid_search(query, k=3, fusion_method="weighted")
            rrf_results = hybrid_system.hybrid_search(query, k=3, fusion_method="rrf")
            
            print(f"🔀 Weighted Fusion (top 3):")
            for i, (doc, score) in enumerate(weighted_results[:3], 1):
                print(f"   {i}. Score: {score:.4f} - {doc.metadata.get('category', 'Unknown')}")
            
            print(f"🔀 RRF Fusion (top 3):")
            for i, (doc, score) in enumerate(rrf_results[:3], 1):
                print(f"   {i}. Score: {score:.4f} - {doc.metadata.get('category', 'Unknown')}")
        
        # Evaluate search quality
        test_query_data = [{"query": q} for q in test_queries]
        evaluation = hybrid_system.evaluate_search_quality(test_query_data)
        
        print(f"\n📊 Search Quality Evaluation:")
        for method, metrics in evaluation.items():
            print(f"  {method}: Avg Results: {metrics['avg_results']:.1f}, Coverage: {metrics['coverage']:.2%}")


def solution_3_document_recommendation_engine():
    """
    Solution 3: Document Recommendation Engine with User Profiles
    """
    print("\n📚 Solution 3: Document Recommendation Engine")
    print("-" * 70)
    
    class DocumentRecommendationEngine:
        """Intelligent document recommendation system with user profiling."""
        
        def __init__(self, embedding_model):
            self.embedding_model = embedding_model
            self.vector_store = None
            self.user_profiles = {}
            self.interaction_history = defaultdict(list)
            self.document_embeddings = {}
            self.documents = []
            self.item_similarity_matrix = None
        
        def build_document_index(self, documents: List[Document]):
            """Build searchable document index with embeddings."""
            self.documents = documents
            
            print("🔧 Building document index...")
            self.vector_store = FAISS.from_documents(documents, self.embedding_model)
            
            # Store document embeddings for user profile calculation
            print("🔧 Computing document embeddings...")
            for i, doc in enumerate(documents):
                doc_id = doc.metadata.get("doc_id", f"doc_{i}")
                embedding = self.embedding_model.embed_query(doc.page_content)
                self.document_embeddings[doc_id] = np.array(embedding)
            
            # Compute item-item similarity matrix
            print("🔧 Computing item similarity matrix...")
            embeddings_matrix = np.array(list(self.document_embeddings.values()))
            self.item_similarity_matrix = cosine_similarity(embeddings_matrix)
            
            print("✅ Document index built successfully")
        
        def update_user_profile(self, user_id: str, interactions: List[UserInteraction]):
            """Update user profile based on interaction history."""
            self.interaction_history[user_id].extend(interactions)
            
            # Compute weighted average of document embeddings
            user_vector = np.zeros(len(list(self.document_embeddings.values())[0]))
            total_weight = 0
            
            # Weight interactions by type and recency
            interaction_weights = {
                'view': 1.0,
                'like': 3.0,
                'share': 5.0,
                'bookmark': 4.0
            }
            
            current_time = time.time()
            
            for interaction in self.interaction_history[user_id]:
                doc_id = interaction.document_id
                if doc_id in self.document_embeddings:
                    # Time decay (older interactions have less weight)
                    time_weight = np.exp(-(current_time - interaction.timestamp) / (30 * 24 * 3600))  # 30-day half-life
                    
                    # Interaction type weight
                    type_weight = interaction_weights.get(interaction.interaction_type, 1.0)
                    
                    # Duration weight (longer engagement = higher weight)
                    duration_weight = min(interaction.duration / 300, 2.0)  # Cap at 5 minutes
                    
                    total_interaction_weight = time_weight * type_weight * duration_weight
                    
                    doc_embedding = self.document_embeddings[doc_id]
                    user_vector += doc_embedding * total_interaction_weight
                    total_weight += total_interaction_weight
            
            if total_weight > 0:
                user_vector /= total_weight
                self.user_profiles[user_id] = user_vector
            else:
                # Default profile for users with no interactions
                self.user_profiles[user_id] = np.mean(list(self.document_embeddings.values()), axis=0)
        
        def content_based_recommendations(self, user_id: str, k: int = 10, 
                                        exclude_seen: bool = True) -> List[Tuple[str, float]]:
            """Generate content-based recommendations using user profile."""
            if user_id not in self.user_profiles:
                return []
            
            user_profile = self.user_profiles[user_id]
            
            # Calculate similarity to all documents
            doc_scores = []
            seen_docs = {interaction.document_id for interaction in self.interaction_history[user_id]} if exclude_seen else set()
            
            for doc_id, doc_embedding in self.document_embeddings.items():
                if doc_id not in seen_docs:
                    similarity = cosine_similarity([user_profile], [doc_embedding])[0][0]
                    doc_scores.append((doc_id, similarity))
            
            # Sort by similarity and return top k
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            return doc_scores[:k]
        
        def collaborative_recommendations(self, user_id: str, k: int = 10) -> List[Tuple[str, float]]:
            """Generate collaborative filtering recommendations."""
            if user_id not in self.user_profiles:
                return []
            
            user_profile = self.user_profiles[user_id]
            
            # Find similar users
            similar_users = []
            for other_user_id, other_profile in self.user_profiles.items():
                if other_user_id != user_id:
                    similarity = cosine_similarity([user_profile], [other_profile])[0][0]
                    similar_users.append((other_user_id, similarity))
            
            # Sort by similarity and take top users
            similar_users.sort(key=lambda x: x[1], reverse=True)
            top_similar_users = similar_users[:10]  # Top 10 similar users
            
            # Aggregate recommendations from similar users
            doc_scores = defaultdict(float)
            user_interactions = {interaction.document_id for interaction in self.interaction_history[user_id]}
            
            for similar_user_id, user_similarity in top_similar_users:
                for interaction in self.interaction_history[similar_user_id]:
                    if interaction.document_id not in user_interactions:
                        # Weight by user similarity and interaction strength
                        interaction_weight = {'view': 1, 'like': 3, 'share': 5, 'bookmark': 4}.get(
                            interaction.interaction_type, 1
                        )
                        doc_scores[interaction.document_id] += user_similarity * interaction_weight
            
            # Sort and return top k
            recommendations = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            return recommendations[:k]
        
        def item_based_collaborative(self, user_id: str, k: int = 10) -> List[Tuple[str, float]]:
            """Item-based collaborative filtering using item similarity matrix."""
            user_interactions = self.interaction_history[user_id]
            if not user_interactions:
                return []
            
            doc_scores = defaultdict(float)
            user_seen_docs = {interaction.document_id for interaction in user_interactions}
            
            # Get document indices
            doc_id_to_index = {doc.metadata.get("doc_id", f"doc_{i}"): i 
                             for i, doc in enumerate(self.documents)}
            
            for interaction in user_interactions:
                doc_id = interaction.document_id
                if doc_id in doc_id_to_index:
                    doc_index = doc_id_to_index[doc_id]
                    
                    # Find similar items
                    similarities = self.item_similarity_matrix[doc_index]
                    
                    interaction_weight = {'view': 1, 'like': 3, 'share': 5, 'bookmark': 4}.get(
                        interaction.interaction_type, 1
                    )
                    
                    for other_doc_id, other_index in doc_id_to_index.items():
                        if other_doc_id not in user_seen_docs:
                            item_similarity = similarities[other_index]
                            doc_scores[other_doc_id] += item_similarity * interaction_weight
            
            recommendations = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            return recommendations[:k]
        
        def hybrid_recommendations(self, user_id: str, k: int = 10, 
                                 content_weight: float = 0.4, 
                                 collaborative_weight: float = 0.3,
                                 item_weight: float = 0.3) -> List[Tuple[str, float]]:
            """Generate hybrid recommendations combining multiple approaches."""
            
            # Get recommendations from each method
            content_recs = self.content_based_recommendations(user_id, k*2)
            collab_recs = self.collaborative_recommendations(user_id, k*2)
            item_recs = self.item_based_collaborative(user_id, k*2)
            
            # Combine scores
            doc_scores = defaultdict(float)
            
            # Content-based scores
            for doc_id, score in content_recs:
                doc_scores[doc_id] += content_weight * score
            
            # Collaborative scores
            max_collab_score = max([score for _, score in collab_recs], default=1)
            for doc_id, score in collab_recs:
                normalized_score = score / max_collab_score if max_collab_score > 0 else 0
                doc_scores[doc_id] += collaborative_weight * normalized_score
            
            # Item-based scores
            max_item_score = max([score for _, score in item_recs], default=1)
            for doc_id, score in item_recs:
                normalized_score = score / max_item_score if max_item_score > 0 else 0
                doc_scores[doc_id] += item_weight * normalized_score
            
            # Sort and return top k
            recommendations = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            return recommendations[:k]
        
        def handle_cold_start(self, user_id: str, initial_preferences: Dict) -> List[Tuple[str, float]]:
            """Handle cold-start problem using initial preferences."""
            
            # Create initial profile based on preferences
            category_embeddings = defaultdict(list)
            
            # Group documents by category and compute average embeddings
            for doc in self.documents:
                category = doc.metadata.get("category", "general")
                doc_id = doc.metadata.get("doc_id", hash(doc.page_content))
                if doc_id in self.document_embeddings:
                    category_embeddings[category].append(self.document_embeddings[doc_id])
            
            # Compute category centroids
            category_centroids = {}
            for category, embeddings in category_embeddings.items():
                category_centroids[category] = np.mean(embeddings, axis=0)
            
            # Create initial user profile based on preferences
            initial_profile = np.zeros(len(list(self.document_embeddings.values())[0]))
            total_weight = 0
            
            for category, preference_score in initial_preferences.items():
                if category in category_centroids:
                    initial_profile += category_centroids[category] * preference_score
                    total_weight += preference_score
            
            if total_weight > 0:
                initial_profile /= total_weight
                self.user_profiles[user_id] = initial_profile
                
                # Generate content-based recommendations
                return self.content_based_recommendations(user_id, k=10)
            
            return []
        
        def evaluate_recommendations(self, test_users: List[str], 
                                   ground_truth: Dict[str, List[str]]) -> Dict[str, float]:
            """Evaluate recommendation quality using precision@k and other metrics."""
            
            metrics = {
                "precision_at_5": [],
                "precision_at_10": [],
                "recall_at_10": [],
                "coverage": set()
            }
            
            for user_id in test_users:
                if user_id in ground_truth:
                    relevant_docs = set(ground_truth[user_id])
                    
                    # Get recommendations
                    recommendations = self.hybrid_recommendations(user_id, k=10)
                    recommended_docs = [doc_id for doc_id, _ in recommendations]
                    
                    # Calculate precision@5
                    precision_5 = len(set(recommended_docs[:5]) & relevant_docs) / 5
                    metrics["precision_at_5"].append(precision_5)
                    
                    # Calculate precision@10
                    precision_10 = len(set(recommended_docs[:10]) & relevant_docs) / 10
                    metrics["precision_at_10"].append(precision_10)
                    
                    # Calculate recall@10
                    recall_10 = len(set(recommended_docs[:10]) & relevant_docs) / len(relevant_docs)
                    metrics["recall_at_10"].append(recall_10)
                    
                    # Update coverage
                    metrics["coverage"].update(recommended_docs)
            
            # Compute averages
            evaluation_results = {
                "avg_precision_at_5": np.mean(metrics["precision_at_5"]),
                "avg_precision_at_10": np.mean(metrics["precision_at_10"]),
                "avg_recall_at_10": np.mean(metrics["recall_at_10"]),
                "catalog_coverage": len(metrics["coverage"]) / len(self.documents)
            }
            
            return evaluation_results
    
    # Demo the recommendation engine
    if embedding_model:
        # Create diverse test documents
        categories = ["technology", "science", "business", "health", "education"]
        test_docs = []
        
        for category in categories:
            for i in range(8):  # 8 docs per category
                content = f"This is a {category} document {i}. " * 20  # Make it substantial
                test_docs.append(Document(
                    page_content=content,
                    metadata={"doc_id": f"{category}_{i}", "category": category}
                ))
        
        # Initialize recommendation engine
        rec_engine = DocumentRecommendationEngine(embedding_model)
        rec_engine.build_document_index(test_docs)
        
        # Simulate user interactions
        users = ["user_1", "user_2", "user_3"]
        
        for user_id in users:
            # Create simulated interactions
            interactions = []
            
            # User 1 likes technology and science
            # User 2 likes business and health  
            # User 3 likes education and technology
            
            preferred_categories = {
                "user_1": ["technology", "science"],
                "user_2": ["business", "health"],
                "user_3": ["education", "technology"]
            }
            
            current_time = time.time()
            
            for category in preferred_categories[user_id]:
                for i in range(3):  # 3 interactions per preferred category
                    interactions.append(UserInteraction(
                        user_id=user_id,
                        document_id=f"{category}_{i}",
                        interaction_type=np.random.choice(["view", "like", "bookmark"], p=[0.5, 0.3, 0.2]),
                        duration=np.random.uniform(60, 300),  # 1-5 minutes
                        timestamp=current_time - np.random.uniform(0, 86400)  # Last 24 hours
                    ))
            
            rec_engine.update_user_profile(user_id, interactions)
        
        # Test recommendations
        print("\n🧪 Testing Recommendation Methods:")
        
        for user_id in users:
            print(f"\n👤 Recommendations for {user_id}:")
            
            # Content-based
            content_recs = rec_engine.content_based_recommendations(user_id, k=5)
            print(f"   📄 Content-based: {[doc_id for doc_id, _ in content_recs]}")
            
            # Collaborative
            collab_recs = rec_engine.collaborative_recommendations(user_id, k=5)
            print(f"   👥 Collaborative: {[doc_id for doc_id, _ in collab_recs]}")
            
            # Hybrid
            hybrid_recs = rec_engine.hybrid_recommendations(user_id, k=5)
            print(f"   🔀 Hybrid: {[doc_id for doc_id, _ in hybrid_recs]}")
        
        # Test cold-start handling
        print(f"\n🆕 Cold-start recommendations for new user:")
        cold_start_prefs = {"technology": 0.8, "science": 0.6, "business": 0.3}
        cold_start_recs = rec_engine.handle_cold_start("new_user", cold_start_prefs)
        print(f"   Recommendations: {[doc_id for doc_id, _ in cold_start_recs[:5]]}")


def run_all_solutions():
    """Run all solution demonstrations."""
    print("🦜🔗 LangChain Course - Lesson 6: Vector Stores & Embeddings Solutions")
    print("=" * 80)
    
    if not embedding_model:
        print("❌ Embedding model not available. Please check your setup.")
        return
    
    print(f"✅ Using Embedding Model: {type(embedding_model).__name__}")
    
    # Run solutions
    solutions = [
        solution_1_embedding_comparison,
        solution_2_hybrid_search_system,
        solution_3_document_recommendation_engine,
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
    
    print("\n🎉 All solutions demonstrated!")
    print("\n💡 Key Takeaways:")
    print("   • Vector store selection depends on specific requirements")
    print("   • Hybrid search often outperforms single-method approaches")
    print("   • User profiling enables personalized recommendations")
    print("   • Production systems require careful performance optimization")
    print("   • Real-time updates need sophisticated conflict resolution")


if __name__ == "__main__":
    run_all_solutions()