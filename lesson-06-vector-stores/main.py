#!/usr/bin/env python3
"""
Lesson 6: Vector Stores & Embeddings with LangChain

This lesson covers:
1. Understanding embedding models and vector representations
2. Setting up and managing different vector databases
3. Implementing efficient similarity search operations
4. Optimizing vector store performance
5. Building semantic search systems with metadata filtering

Author: LangChain Course
"""

import os
import sys
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from dotenv import load_dotenv

# Add shared resources to path
sys.path.append('../shared-resources')

# Load environment variables
load_dotenv()

# LangChain imports
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.schema import Document
from langchain.callbacks import get_openai_callback

# Optional vector store imports (install as needed)
try:
    from langchain.vectorstores import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    from langchain.vectorstores import Weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

try:
    from langchain.vectorstores import Qdrant
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Shared utilities
from utils.llm_setup import setup_llm_providers, get_preferred_llm
from utils.error_handling import safe_llm_call

# Performance and analysis imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


class EmbeddingModelComparator:
    """Compare different embedding models for performance and quality."""
    
    def __init__(self):
        self.models = {}
        self.test_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are inspired by biological neural networks.",
            "Deep learning uses multiple layers to learn representations.",
            "Natural language processing helps computers understand text.",
            "Computer vision enables machines to interpret visual information.",
            "Reinforcement learning learns through trial and error.",
            "Supervised learning uses labeled training data.",
            "Unsupervised learning finds patterns without labels."
        ]
    
    def setup_embedding_models(self):
        """Set up different embedding models for comparison."""
        print("🔧 Setting up embedding models...")
        
        # OpenAI Embeddings
        if os.getenv("OPENAI_API_KEY"):
            try:
                self.models["openai_small"] = OpenAIEmbeddings(
                    model="text-embedding-3-small"
                )
                self.models["openai_large"] = OpenAIEmbeddings(
                    model="text-embedding-3-large"
                )
                print("✅ OpenAI embedding models loaded")
            except Exception as e:
                print(f"❌ OpenAI embeddings failed: {e}")
        
        # Sentence Transformers (HuggingFace)
        try:
            self.models["sentence_transformer"] = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            print("✅ Sentence Transformer model loaded")
        except Exception as e:
            print(f"❌ Sentence Transformer failed: {e}")
        
        # Add more models as available
        try:
            self.models["mpnet"] = HuggingFaceEmbeddings(
                model_name="all-mpnet-base-v2"
            )
            print("✅ MPNet model loaded")
        except Exception as e:
            print(f"❌ MPNet failed: {e}")
    
    def compare_embedding_quality(self):
        """Compare embedding quality across models."""
        print("\n📊 Comparing Embedding Quality")
        print("-" * 50)
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\nTesting {model_name}...")
            
            start_time = time.time()
            embeddings = []
            
            # Handle different model types
            if hasattr(model, 'embed_documents'):
                embeddings = model.embed_documents(self.test_texts)
            else:
                for text in self.test_texts:
                    embedding = model.embed_query(text)
                    embeddings.append(embedding)
            
            embedding_time = time.time() - start_time
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Calculate metrics
            dimensions = len(embeddings[0]) if embeddings else 0
            avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, 1)])
            
            results[model_name] = {
                "dimensions": dimensions,
                "embedding_time": embedding_time,
                "avg_similarity": avg_similarity,
                "embeddings": embeddings
            }
            
            print(f"  Dimensions: {dimensions}")
            print(f"  Time: {embedding_time:.3f}s")
            print(f"  Avg similarity: {avg_similarity:.3f}")
        
        return results
    
    def visualize_embeddings(self, results: Dict[str, Any]):
        """Visualize embeddings using PCA."""
        print("\n📈 Visualizing Embeddings with PCA")
        print("-" * 50)
        
        fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
        if len(results) == 1:
            axes = [axes]
        
        for idx, (model_name, data) in enumerate(results.items()):
            embeddings = np.array(data["embeddings"])
            
            # Apply PCA for 2D visualization
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
            
            # Plot
            axes[idx].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=range(len(embeddings_2d)), cmap='viridis')
            axes[idx].set_title(f'{model_name}\n({data["dimensions"]}D)')
            axes[idx].set_xlabel('PC1')
            axes[idx].set_ylabel('PC2')
            
            # Add text labels
            for i, txt in enumerate(range(len(self.test_texts))):
                axes[idx].annotate(f'T{txt+1}', 
                                 (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                                 xytext=(5, 5), textcoords='offset points',
                                 fontsize=8)
        
        plt.tight_layout()
        plt.savefig('embedding_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("📊 Visualization saved as 'embedding_comparison.png'")


class VectorStoreManager:
    """Manage different vector store implementations."""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.stores = {}
        self.documents = []
    
    def create_sample_documents(self) -> List[Document]:
        """Create sample documents for vector store demonstrations."""
        sample_docs = [
            Document(
                page_content="""
                Vector databases are specialized databases designed to store and query 
                high-dimensional vectors efficiently. They are essential for applications 
                involving machine learning, recommendation systems, and semantic search.
                """,
                metadata={"source": "vector_db_intro.txt", "category": "database", "difficulty": "beginner"}
            ),
            Document(
                page_content="""
                Chroma is an open-source embedding database that makes it easy to build 
                LLM applications. It offers a simple API for storing documents and their 
                embeddings, with built-in support for metadata filtering and persistence.
                """,
                metadata={"source": "chroma_overview.txt", "category": "vector_store", "difficulty": "beginner"}
            ),
            Document(
                page_content="""
                FAISS (Facebook AI Similarity Search) is a library for efficient similarity 
                search and clustering of dense vectors. It provides GPU acceleration and 
                supports various indexing methods for different use cases and performance requirements.
                """,
                metadata={"source": "faiss_guide.txt", "category": "vector_store", "difficulty": "intermediate"}
            ),
            Document(
                page_content="""
                Pinecone is a managed vector database service that provides low-latency 
                vector search at scale. It handles infrastructure management and offers 
                features like real-time updates, metadata filtering, and hybrid search.
                """,
                metadata={"source": "pinecone_features.txt", "category": "managed_service", "difficulty": "intermediate"}
            ),
            Document(
                page_content="""
                Similarity search algorithms find the most similar vectors to a query vector. 
                Common metrics include cosine similarity, Euclidean distance, and dot product. 
                The choice of metric depends on the nature of your embeddings and use case.
                """,
                metadata={"source": "similarity_metrics.txt", "category": "algorithms", "difficulty": "advanced"}
            ),
            Document(
                page_content="""
                Vector indexing techniques like HNSW (Hierarchical Navigable Small World) 
                and IVF (Inverted File) trade off between search accuracy and speed. 
                Understanding these trade-offs is crucial for production deployments.
                """,
                metadata={"source": "indexing_techniques.txt", "category": "algorithms", "difficulty": "advanced"}
            ),
            Document(
                page_content="""
                Embedding dimensionality affects both storage requirements and search performance. 
                Higher dimensions can capture more nuanced relationships but require more 
                computational resources and may suffer from the curse of dimensionality.
                """,
                metadata={"source": "dimensionality_analysis.txt", "category": "theory", "difficulty": "advanced"}
            ),
            Document(
                page_content="""
                Hybrid search combines dense vector search with sparse keyword search 
                to improve retrieval quality. This approach leverages the strengths of 
                both semantic similarity and exact keyword matching.
                """,
                metadata={"source": "hybrid_search.txt", "category": "advanced_techniques", "difficulty": "advanced"}
            )
        ]
        
        self.documents = sample_docs
        print(f"📚 Created {len(sample_docs)} sample documents")
        return sample_docs
    
    def setup_chroma_store(self):
        """Set up Chroma vector store."""
        print("\n🔵 Setting up Chroma Vector Store")
        print("-" * 40)
        
        try:
            persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
            collection_name = os.getenv("CHROMA_COLLECTION_NAME", "documents")
            
            # Create Chroma store
            chroma_store = Chroma.from_documents(
                documents=self.documents,
                embedding=self.embedding_model,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
            
            self.stores["chroma"] = chroma_store
            print(f"✅ Chroma store created with {len(self.documents)} documents")
            print(f"📁 Persisted to: {persist_directory}")
            
            return chroma_store
            
        except Exception as e:
            print(f"❌ Chroma setup failed: {e}")
            return None
    
    def setup_faiss_store(self):
        """Set up FAISS vector store."""
        print("\n🟠 Setting up FAISS Vector Store")
        print("-" * 40)
        
        try:
            # Create FAISS store
            faiss_store = FAISS.from_documents(
                documents=self.documents,
                embedding=self.embedding_model
            )
            
            self.stores["faiss"] = faiss_store
            print(f"✅ FAISS store created with {len(self.documents)} documents")
            
            # Save index
            index_path = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
            faiss_store.save_local(index_path)
            print(f"💾 Index saved to: {index_path}")
            
            return faiss_store
            
        except Exception as e:
            print(f"❌ FAISS setup failed: {e}")
            return None
    
    def setup_pinecone_store(self):
        """Set up Pinecone vector store (if available)."""
        if not PINECONE_AVAILABLE or not os.getenv("PINECONE_API_KEY"):
            print("\n🟡 Pinecone not available (missing package or API key)")
            return None
        
        print("\n🟡 Setting up Pinecone Vector Store")
        print("-" * 40)
        
        try:
            import pinecone
            
            # Initialize Pinecone
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENVIRONMENT")
            )
            
            index_name = os.getenv("PINECONE_INDEX_NAME", "langchain-demo")
            
            # Check if index exists, create if not
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=1536,  # Adjust based on embedding model
                    metric="cosine"
                )
                print(f"📝 Created new Pinecone index: {index_name}")
            
            # Create Pinecone store
            pinecone_store = Pinecone.from_documents(
                documents=self.documents,
                embedding=self.embedding_model,
                index_name=index_name
            )
            
            self.stores["pinecone"] = pinecone_store
            print(f"✅ Pinecone store created with {len(self.documents)} documents")
            
            return pinecone_store
            
        except Exception as e:
            print(f"❌ Pinecone setup failed: {e}")
            return None
    
    def compare_vector_stores(self):
        """Compare performance across different vector stores."""
        print("\n⚡ Comparing Vector Store Performance")
        print("-" * 50)
        
        test_queries = [
            "What is a vector database?",
            "How does similarity search work?",
            "Explain indexing techniques",
            "Advanced search methods"
        ]
        
        results = {}
        
        for store_name, store in self.stores.items():
            print(f"\n🔍 Testing {store_name.upper()}:")
            
            store_results = {
                "search_times": [],
                "result_counts": [],
                "total_docs": len(self.documents)
            }
            
            for query in test_queries:
                start_time = time.time()
                
                # Perform similarity search
                docs = store.similarity_search(query, k=3)
                
                search_time = time.time() - start_time
                store_results["search_times"].append(search_time)
                store_results["result_counts"].append(len(docs))
                
                print(f"  Query: '{query[:30]}...' -> {len(docs)} results ({search_time:.4f}s)")
            
            # Calculate averages
            store_results["avg_search_time"] = np.mean(store_results["search_times"])
            store_results["avg_results"] = np.mean(store_results["result_counts"])
            
            results[store_name] = store_results
            
            print(f"  Average search time: {store_results['avg_search_time']:.4f}s")
        
        return results
    
    def demonstrate_metadata_filtering(self):
        """Demonstrate metadata filtering capabilities."""
        print("\n🔎 Demonstrating Metadata Filtering")
        print("-" * 50)
        
        # Test with Chroma (has good metadata support)
        if "chroma" in self.stores:
            chroma_store = self.stores["chroma"]
            
            # Different filter scenarios
            filter_scenarios = [
                {"category": "vector_store"},
                {"difficulty": "beginner"},
                {"difficulty": "advanced"},
                {"category": "algorithms", "difficulty": "advanced"}
            ]
            
            for filter_dict in filter_scenarios:
                print(f"\n🎯 Filter: {filter_dict}")
                
                try:
                    # Search with metadata filter
                    results = chroma_store.similarity_search(
                        query="vector database information",
                        k=5,
                        filter=filter_dict
                    )
                    
                    print(f"  Found {len(results)} documents")
                    for i, doc in enumerate(results, 1):
                        source = doc.metadata.get("source", "Unknown")
                        category = doc.metadata.get("category", "Unknown")
                        difficulty = doc.metadata.get("difficulty", "Unknown")
                        print(f"    {i}. {source} ({category}, {difficulty})")
                        
                except Exception as e:
                    print(f"  ❌ Filter failed: {e}")
        
        else:
            print("❌ Chroma store not available for metadata filtering demo")
    
    def demonstrate_search_types(self):
        """Demonstrate different search types and parameters."""
        print("\n🔍 Demonstrating Different Search Types")
        print("-" * 50)
        
        if "chroma" in self.stores:
            chroma_store = self.stores["chroma"]
            query = "machine learning and AI techniques"
            
            print(f"Query: '{query}'")
            
            # Similarity search
            print("\n1. Standard Similarity Search (k=3)")
            docs = chroma_store.similarity_search(query, k=3)
            for i, doc in enumerate(docs, 1):
                print(f"   {i}. {doc.metadata['source']}: {doc.page_content[:100]}...")
            
            # Similarity search with scores
            print("\n2. Similarity Search with Scores")
            docs_with_scores = chroma_store.similarity_search_with_score(query, k=3)
            for i, (doc, score) in enumerate(docs_with_scores, 1):
                print(f"   {i}. Score: {score:.4f} - {doc.metadata['source']}")
            
            # Maximum Marginal Relevance (MMR) search
            print("\n3. Maximum Marginal Relevance (MMR) Search")
            try:
                mmr_docs = chroma_store.max_marginal_relevance_search(
                    query, k=3, fetch_k=6, lambda_mult=0.5
                )
                for i, doc in enumerate(mmr_docs, 1):
                    print(f"   {i}. {doc.metadata['source']}: {doc.page_content[:100]}...")
            except Exception as e:
                print(f"   ❌ MMR search failed: {e}")


class VectorStoreOptimizer:
    """Optimize vector store performance and configuration."""
    
    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
    
    def benchmark_search_performance(self, num_queries: int = 100):
        """Benchmark search performance with different parameters."""
        print(f"\n⚡ Benchmarking Search Performance ({num_queries} queries)")
        print("-" * 60)
        
        # Generate test queries
        base_queries = [
            "vector database operations",
            "machine learning algorithms",
            "similarity search methods",
            "indexing techniques",
            "embedding models"
        ]
        
        test_queries = (base_queries * (num_queries // len(base_queries) + 1))[:num_queries]
        
        # Test different k values
        k_values = [1, 3, 5, 10]
        results = {}
        
        for k in k_values:
            print(f"\nTesting k={k}:")
            
            start_time = time.time()
            total_results = 0
            
            for query in test_queries:
                docs = self.vector_store.similarity_search(query, k=k)
                total_results += len(docs)
            
            total_time = time.time() - start_time
            avg_time_per_query = total_time / num_queries
            
            results[k] = {
                "total_time": total_time,
                "avg_time_per_query": avg_time_per_query,
                "total_results": total_results,
                "avg_results_per_query": total_results / num_queries
            }
            
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Avg per query: {avg_time_per_query:.4f}s")
            print(f"  Avg results: {total_results / num_queries:.1f}")
        
        return results
    
    def analyze_embedding_distribution(self):
        """Analyze the distribution of embeddings in the vector space."""
        print("\n📊 Analyzing Embedding Distribution")
        print("-" * 50)
        
        # Get all documents from vector store
        if hasattr(self.vector_store, '_collection'):
            # Chroma-specific
            collection = self.vector_store._collection
            results = collection.get(include=['embeddings', 'metadatas'])
            embeddings = np.array(results['embeddings'])
            
        elif hasattr(self.vector_store, 'index'):
            # FAISS-specific
            embeddings = self.vector_store.index.reconstruct_n(0, self.vector_store.index.ntotal)
            
        else:
            print("❌ Cannot access embeddings from this vector store type")
            return
        
        # Calculate statistics
        mean_embedding = np.mean(embeddings, axis=0)
        std_embedding = np.std(embeddings, axis=0)
        
        print(f"📏 Embedding dimensions: {embeddings.shape[1]}")
        print(f"📊 Number of vectors: {embeddings.shape[0]}")
        print(f"📈 Mean magnitude: {np.mean(np.linalg.norm(embeddings, axis=1)):.4f}")
        print(f"📉 Std magnitude: {np.std(np.linalg.norm(embeddings, axis=1)):.4f}")
        
        # Calculate pairwise similarities
        similarity_matrix = cosine_similarity(embeddings)
        
        # Remove diagonal (self-similarity)
        mask = np.eye(similarity_matrix.shape[0], dtype=bool)
        similarities = similarity_matrix[~mask]
        
        print(f"🎯 Avg pairwise similarity: {np.mean(similarities):.4f}")
        print(f"📊 Similarity std: {np.std(similarities):.4f}")
        print(f"🔺 Max similarity: {np.max(similarities):.4f}")
        print(f"🔻 Min similarity: {np.min(similarities):.4f}")
        
        return {
            "embeddings": embeddings,
            "similarities": similarities,
            "stats": {
                "dimensions": embeddings.shape[1],
                "num_vectors": embeddings.shape[0],
                "mean_magnitude": np.mean(np.linalg.norm(embeddings, axis=1)),
                "avg_similarity": np.mean(similarities)
            }
        }


def setup_lesson():
    """Set up the lesson environment."""
    print("🦜🔗 LangChain Course - Lesson 6: Vector Stores & Embeddings")
    print("=" * 70)
    
    # Set up LLM providers (for any LLM-based operations)
    providers = setup_llm_providers()
    llm = get_preferred_llm(providers, prefer_chat=True) if providers else None
    
    # Set up embedding model
    embedding_model = None
    
    if os.getenv("OPENAI_API_KEY"):
        try:
            embedding_model = OpenAIEmbeddings(
                model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            )
            print("✅ OpenAI embeddings configured")
        except Exception as e:
            print(f"❌ OpenAI embeddings failed: {e}")
    
    if not embedding_model:
        try:
            embedding_model = HuggingFaceEmbeddings(
                model_name=os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
            )
            print("✅ HuggingFace embeddings configured")
        except Exception as e:
            print(f"❌ HuggingFace embeddings failed: {e}")
    
    if not embedding_model:
        print("❌ No embedding model available. Please check your configuration.")
        return None, None
    
    return llm, embedding_model


def interactive_vector_store_explorer(embedding_model):
    """Interactive vector store exploration."""
    print("\n🚀 Interactive Vector Store Explorer")
    print("=" * 50)
    
    # Set up vector store manager
    manager = VectorStoreManager(embedding_model)
    manager.create_sample_documents()
    
    # Set up available stores
    available_stores = []
    
    chroma_store = manager.setup_chroma_store()
    if chroma_store:
        available_stores.append("chroma")
    
    faiss_store = manager.setup_faiss_store()
    if faiss_store:
        available_stores.append("faiss")
    
    if not available_stores:
        print("❌ No vector stores available")
        return
    
    print(f"\n📚 Available stores: {', '.join(available_stores)}")
    
    while True:
        print("\n🔍 Vector Store Explorer Options:")
        print("1. Search documents")
        print("2. Compare search methods")
        print("3. Filter by metadata")
        print("4. Performance benchmark")
        print("5. Analyze embeddings")
        print("6. Exit")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            query = input("Enter search query: ").strip()
            if query:
                k = int(input("Number of results (1-10, default 3): ") or "3")
                
                for store_name in available_stores:
                    store = manager.stores[store_name]
                    print(f"\n🔍 Results from {store_name.upper()}:")
                    
                    start_time = time.time()
                    docs = store.similarity_search(query, k=k)
                    search_time = time.time() - start_time
                    
                    print(f"   Found {len(docs)} results in {search_time:.4f}s")
                    
                    for i, doc in enumerate(docs, 1):
                        source = doc.metadata.get("source", "Unknown")
                        print(f"   {i}. {source}")
                        print(f"      {doc.page_content[:150]}...")
                        print()
        
        elif choice == "2":
            manager.demonstrate_search_types()
        
        elif choice == "3":
            manager.demonstrate_metadata_filtering()
        
        elif choice == "4":
            if available_stores:
                optimizer = VectorStoreOptimizer(
                    manager.stores[available_stores[0]], 
                    embedding_model
                )
                optimizer.benchmark_search_performance(num_queries=50)
        
        elif choice == "5":
            if available_stores:
                optimizer = VectorStoreOptimizer(
                    manager.stores[available_stores[0]], 
                    embedding_model
                )
                optimizer.analyze_embedding_distribution()
        
        elif choice == "6":
            print("👋 Thanks for exploring vector stores!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-6.")


def main():
    """Main function to run all vector store demonstrations."""
    llm, embedding_model = setup_lesson()
    
    if not embedding_model:
        print("❌ Cannot proceed without embedding model. Please check your setup.")
        return
    
    print(f"\n🔧 Using Embedding Model: {type(embedding_model).__name__}")
    if llm:
        print(f"🔧 Using LLM: {type(llm).__name__}")
    
    try:
        # Embedding model comparison
        print("\n" + "="*70)
        print("PART 1: EMBEDDING MODEL COMPARISON")
        print("="*70)
        
        comparator = EmbeddingModelComparator()
        comparator.setup_embedding_models()
        
        if comparator.models:
            embedding_results = comparator.compare_embedding_quality()
            
            # Visualization (optional, comment out if matplotlib issues)
            try:
                comparator.visualize_embeddings(embedding_results)
            except Exception as e:
                print(f"❌ Visualization failed: {e}")
        
        # Vector store setup and comparison
        print("\n" + "="*70)
        print("PART 2: VECTOR STORE SETUP AND COMPARISON")
        print("="*70)
        
        manager = VectorStoreManager(embedding_model)
        manager.create_sample_documents()
        
        # Set up different vector stores
        manager.setup_chroma_store()
        manager.setup_faiss_store()
        manager.setup_pinecone_store()  # If available
        
        if manager.stores:
            # Compare performance
            performance_results = manager.compare_vector_stores()
            
            # Demonstrate features
            manager.demonstrate_metadata_filtering()
            manager.demonstrate_search_types()
        
        # Interactive exploration
        print("\n" + "="*70)
        print("PART 3: INTERACTIVE EXPLORATION")
        print("="*70)
        
        run_interactive = input("\nWould you like to try the Interactive Vector Store Explorer? (y/n): ").strip().lower()
        if run_interactive in ['y', 'yes']:
            interactive_vector_store_explorer(embedding_model)
        
        print("\n✨ Lesson 6 completed! You've mastered vector stores and embeddings.")
        print("\n📚 Key Skills Acquired:")
        print("   • Understanding different embedding models")
        print("   • Setting up various vector databases")
        print("   • Implementing efficient similarity search")
        print("   • Optimizing vector store performance")
        print("   • Using metadata filtering for precise search")
        
        print("\n🔗 Next: Lesson 7 - RAG (Retrieval Augmented Generation) Systems")
        
    except KeyboardInterrupt:
        print("\n\n👋 Lesson interrupted. You can resume anytime!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    main()