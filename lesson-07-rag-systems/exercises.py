#!/usr/bin/env python3
"""
Lesson 7: RAG Systems - Practice Exercises

Complete these exercises to master RAG (Retrieval Augmented Generation) systems.
Each exercise focuses on specific aspects of RAG implementation and optimization.

Instructions:
1. Implement each exercise function
2. Run individual exercises to test your implementations
3. Check solutions.py for reference implementations
4. Experiment with different configurations and approaches
"""

import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Add shared resources to path
sys.path.append('../shared-resources')

# Load environment variables
load_dotenv()

# LangChain imports
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from pydantic import BaseModel, Field

# Shared utilities
from utils.llm_setup import setup_llm_providers, get_preferred_llm
from utils.error_handling import safe_llm_call

# Set up LLM providers
providers = setup_llm_providers()
llm = get_preferred_llm(providers, prefer_chat=True) if providers else None
embeddings = OpenAIEmbeddings() if os.getenv("OPENAI_API_KEY") else None


def exercise_1_custom_rag_system():
    """
    Exercise 1: Build a Custom RAG System for Technical Documentation
    
    Task: Create a RAG system specifically designed for technical documentation.
    Requirements:
    1. Implement custom document processing for technical content
    2. Use appropriate chunking strategy for code examples and explanations
    3. Create a custom prompt template for technical Q&A
    4. Include source attribution in responses
    5. Handle code-related queries effectively
    
    Your implementation should:
    - Process documents with mixed text and code content
    - Preserve code formatting in chunks
    - Generate responses with proper technical accuracy
    - Cite sources clearly
    """
    
    print("🏗️ Exercise 1: Custom RAG System for Technical Documentation")
    print("-" * 60)
    
    # TODO: Implement your custom RAG system
    # Hints:
    # 1. Create custom text splitter that preserves code blocks
    # 2. Design prompt template for technical Q&A
    # 3. Implement document processing pipeline
    # 4. Test with technical documentation samples
    
    # Sample technical documents to work with
    tech_docs = [
        Document(
            page_content="""
            # Python List Comprehensions
            
            List comprehensions provide a concise way to create lists in Python.
            
            ## Basic Syntax
            ```python
            [expression for item in iterable]
            ```
            
            ## Examples
            ```python
            # Square numbers from 0 to 9
            squares = [x**2 for x in range(10)]
            print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
            
            # Filter even numbers
            evens = [x for x in range(20) if x % 2 == 0]
            ```
            
            ## Performance Benefits
            List comprehensions are generally faster than equivalent for loops
            because they're optimized at the C level in CPython.
            """,
            metadata={"source": "python_comprehensions.md", "type": "tutorial"}
        ),
        Document(
            page_content="""
            # Database Connection Pooling
            
            Connection pooling is a technique used to maintain a cache of database
            connections that can be reused across multiple requests.
            
            ## Why Use Connection Pooling?
            - Reduces connection overhead
            - Improves application performance
            - Controls resource usage
            
            ## Implementation Example
            ```python
            from sqlalchemy import create_engine
            from sqlalchemy.pool import QueuePool
            
            engine = create_engine(
                'postgresql://user:password@localhost/db',
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20
            )
            ```
            
            ## Best Practices
            1. Set appropriate pool size based on concurrent users
            2. Configure connection timeout values
            3. Monitor pool utilization metrics
            """,
            metadata={"source": "database_pooling.md", "type": "guide"}
        )
    ]
    
    # Your implementation here
    pass


def exercise_2_multi_document_synthesis():
    """
    Exercise 2: Multi-Document Information Synthesis
    
    Task: Create a system that can synthesize information from multiple documents
    to answer complex questions that require information from different sources.
    
    Requirements:
    1. Retrieve relevant information from multiple documents
    2. Synthesize and combine information coherently
    3. Handle conflicting information appropriately
    4. Provide clear attribution for different pieces of information
    5. Generate comprehensive answers that address all aspects of complex queries
    
    Your implementation should:
    - Use advanced retrieval strategies
    - Implement result ranking and filtering
    - Create synthesis-focused prompt templates
    - Handle multi-faceted questions effectively
    """
    
    print("🔄 Exercise 2: Multi-Document Information Synthesis")
    print("-" * 60)
    
    # TODO: Implement multi-document synthesis system
    # Hints:
    # 1. Retrieve more documents than usual (k=8-10)
    # 2. Implement document ranking by relevance
    # 3. Create prompts that encourage synthesis
    # 4. Handle potential conflicts in information
    
    # Sample documents with potentially overlapping/conflicting information
    diverse_docs = [
        Document(
            page_content="Machine learning is primarily a statistical approach to pattern recognition, focusing on algorithms that can learn from data without being explicitly programmed for every scenario.",
            metadata={"source": "ml_statistical_view.txt", "perspective": "statistical"}
        ),
        Document(
            page_content="Machine learning is fundamentally about creating systems that can automatically improve their performance through experience, much like how humans learn from trial and error.",
            metadata={"source": "ml_learning_view.txt", "perspective": "cognitive"}
        ),
        Document(
            page_content="From a computer science perspective, machine learning involves designing algorithms and data structures that can efficiently process large datasets and make predictions or decisions.",
            metadata={"source": "ml_cs_view.txt", "perspective": "computational"}
        )
    ]
    
    # Your implementation here
    pass


def exercise_3_conversational_knowledge_assistant():
    """
    Exercise 3: Advanced Conversational Knowledge Assistant
    
    Task: Build a sophisticated conversational RAG system that can handle
    complex multi-turn conversations with context awareness.
    
    Requirements:
    1. Maintain conversation context across multiple exchanges
    2. Handle follow-up questions and references to previous responses
    3. Distinguish between questions that need new retrieval vs. clarifications
    4. Implement conversation summarization for long interactions
    5. Provide contextually appropriate responses based on conversation flow
    
    Your implementation should:
    - Use advanced memory management
    - Implement context-aware retrieval
    - Handle conversational patterns naturally
    - Maintain coherence across long conversations
    """
    
    print("💬 Exercise 3: Advanced Conversational Knowledge Assistant")
    print("-" * 60)
    
    # TODO: Implement advanced conversational RAG system
    # Hints:
    # 1. Implement custom memory management
    # 2. Create context-aware retrieval logic
    # 3. Design prompts for conversational flow
    # 4. Handle question classification (new vs follow-up)
    
    # Your implementation here
    pass


def exercise_4_rag_evaluation_framework():
    """
    Exercise 4: Comprehensive RAG Evaluation Framework
    
    Task: Develop a framework to evaluate RAG system performance across
    multiple dimensions including retrieval quality and generation accuracy.
    
    Requirements:
    1. Implement retrieval evaluation metrics (precision, recall, MRR)
    2. Create generation quality assessments (faithfulness, relevance)
    3. Design automated evaluation pipelines
    4. Include both quantitative and qualitative evaluation methods
    5. Generate comprehensive evaluation reports
    
    Your implementation should:
    - Define clear evaluation metrics
    - Create test datasets with ground truth
    - Implement automated scoring
    - Provide detailed analysis and insights
    """
    
    print("📊 Exercise 4: RAG Evaluation Framework")
    print("-" * 60)
    
    # TODO: Implement RAG evaluation framework
    # Hints:
    # 1. Define evaluation metrics clearly
    # 2. Create test questions with expected answers
    # 3. Implement scoring algorithms
    # 4. Generate evaluation reports
    
    class RAGEvaluator:
        """Framework for evaluating RAG systems."""
        
        def __init__(self):
            self.metrics = {}
        
        def evaluate_retrieval(self, query: str, retrieved_docs: List[Document], relevant_docs: List[str]):
            """Evaluate retrieval quality."""
            # TODO: Implement retrieval evaluation
            pass
        
        def evaluate_generation(self, question: str, answer: str, context: List[Document]):
            """Evaluate generation quality."""
            # TODO: Implement generation evaluation
            pass
        
        def comprehensive_evaluation(self, test_cases: List[Dict]):
            """Run comprehensive evaluation on test cases."""
            # TODO: Implement comprehensive evaluation
            pass
    
    # Your implementation here
    pass


def exercise_5_production_rag_pipeline():
    """
    Exercise 5: Production-Ready RAG Pipeline
    
    Task: Design and implement a production-ready RAG system with proper
    error handling, monitoring, caching, and scalability considerations.
    
    Requirements:
    1. Implement robust error handling and fallback mechanisms
    2. Add caching for embeddings and frequent queries
    3. Include monitoring and logging capabilities
    4. Design for horizontal scalability
    5. Implement proper configuration management
    6. Add health checks and system diagnostics
    
    Your implementation should:
    - Handle various failure scenarios gracefully
    - Optimize for performance and cost
    - Include comprehensive logging
    - Support easy deployment and scaling
    """
    
    print("🚀 Exercise 5: Production RAG Pipeline")
    print("-" * 60)
    
    # TODO: Implement production-ready RAG pipeline
    # Hints:
    # 1. Add comprehensive error handling
    # 2. Implement caching mechanisms
    # 3. Add monitoring and metrics
    # 4. Design for scalability
    
    class ProductionRAGPipeline:
        """Production-ready RAG pipeline with monitoring and caching."""
        
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.cache = {}
            self.metrics = {}
            self.logger = self._setup_logging()
        
        def _setup_logging(self):
            """Set up comprehensive logging."""
            # TODO: Implement logging setup
            pass
        
        def _setup_monitoring(self):
            """Set up monitoring and metrics collection."""
            # TODO: Implement monitoring setup
            pass
        
        def _setup_caching(self):
            """Set up caching for embeddings and responses."""
            # TODO: Implement caching setup
            pass
        
        def health_check(self) -> Dict[str, Any]:
            """Perform system health check."""
            # TODO: Implement health check
            pass
        
        def process_query(self, query: str, user_id: str = None) -> Dict[str, Any]:
            """Process query with full production pipeline."""
            # TODO: Implement production query processing
            pass
    
    # Your implementation here
    pass


def exercise_6_domain_specific_rag():
    """
    Exercise 6: Domain-Specific RAG System
    
    Task: Build a specialized RAG system for a specific domain (choose one:
    legal documents, medical literature, financial reports, or scientific papers).
    
    Requirements:
    1. Implement domain-specific document preprocessing
    2. Use domain-appropriate chunking strategies
    3. Create specialized prompt templates for the domain
    4. Handle domain-specific terminology and concepts
    5. Implement domain-relevant evaluation metrics
    
    Your implementation should:
    - Understand domain-specific requirements
    - Process domain documents appropriately
    - Generate domain-expert-level responses
    - Handle specialized terminology correctly
    """
    
    print("⚖️ Exercise 6: Domain-Specific RAG System")
    print("-" * 60)
    
    # TODO: Choose a domain and implement specialized RAG system
    # Domains to choose from:
    # - Legal: contracts, case law, regulations
    # - Medical: research papers, clinical guidelines
    # - Financial: earnings reports, market analysis
    # - Scientific: research papers, technical specifications
    
    # Your implementation here
    pass


def exercise_7_hybrid_search_implementation():
    """
    Exercise 7: Hybrid Search RAG System
    
    Task: Implement a RAG system that combines dense vector search with
    sparse keyword search for improved retrieval performance.
    
    Requirements:
    1. Implement both dense (vector) and sparse (keyword) retrieval
    2. Create a fusion mechanism to combine results from both methods
    3. Implement result re-ranking based on multiple signals
    4. Optimize the balance between dense and sparse retrieval
    5. Evaluate performance improvements over single-method approaches
    
    Your implementation should:
    - Use both embedding-based and keyword-based search
    - Implement intelligent result fusion
    - Provide tunable parameters for optimization
    - Demonstrate improved retrieval performance
    """
    
    print("🔀 Exercise 7: Hybrid Search RAG Implementation")
    print("-" * 60)
    
    # TODO: Implement hybrid search RAG system
    # Hints:
    # 1. Use both vector similarity and keyword matching
    # 2. Implement result fusion algorithms
    # 3. Add re-ranking mechanisms
    # 4. Compare performance with single-method approaches
    
    class HybridSearchRAG:
        """RAG system with hybrid dense + sparse retrieval."""
        
        def __init__(self, llm, embeddings):
            self.llm = llm
            self.embeddings = embeddings
            self.dense_retriever = None
            self.sparse_retriever = None
        
        def setup_dense_retrieval(self, documents: List[Document]):
            """Set up dense vector retrieval."""
            # TODO: Implement dense retrieval setup
            pass
        
        def setup_sparse_retrieval(self, documents: List[Document]):
            """Set up sparse keyword retrieval."""
            # TODO: Implement sparse retrieval setup
            pass
        
        def fusion_retrieval(self, query: str, alpha: float = 0.5) -> List[Document]:
            """Combine dense and sparse retrieval results."""
            # TODO: Implement result fusion
            pass
        
        def rerank_results(self, query: str, documents: List[Document]) -> List[Document]:
            """Re-rank combined results."""
            # TODO: Implement re-ranking
            pass
    
    # Your implementation here
    pass


def run_exercise(exercise_number: int):
    """Run a specific exercise."""
    exercises = {
        1: exercise_1_custom_rag_system,
        2: exercise_2_multi_document_synthesis,
        3: exercise_3_conversational_knowledge_assistant,
        4: exercise_4_rag_evaluation_framework,
        5: exercise_5_production_rag_pipeline,
        6: exercise_6_domain_specific_rag,
        7: exercise_7_hybrid_search_implementation,
    }
    
    if exercise_number in exercises:
        print(f"\n🏋️ Running Exercise {exercise_number}")
        print("=" * 70)
        exercises[exercise_number]()
    else:
        print(f"❌ Exercise {exercise_number} not found. Available exercises: 1-7")


def run_all_exercises():
    """Run all exercises in sequence."""
    print("🦜🔗 LangChain Course - Lesson 7: RAG Systems Exercises")
    print("=" * 70)
    
    if not llm or not embeddings:
        print("❌ LLM or embeddings not available. Please check your setup.")
        print("Required environment variables:")
        print("  - OPENAI_API_KEY (for embeddings)")
        print("  - At least one of: ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.")
        return
    
    print(f"✅ Using LLM: {type(llm).__name__}")
    print(f"✅ Using Embeddings: {type(embeddings).__name__}")
    
    for i in range(1, 8):
        try:
            run_exercise(i)
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
    print("   • Experiment with different configurations")
    print("   • Try your systems with real data")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Systems Exercises")
    parser.add_argument("--exercise", type=int, help="Run specific exercise (1-7)")
    parser.add_argument("--all", action="store_true", help="Run all exercises")
    
    args = parser.parse_args()
    
    if args.exercise:
        run_exercise(args.exercise)
    elif args.all:
        run_all_exercises()
    else:
        print("🏋️ RAG Systems Practice Exercises")
        print("=" * 40)
        print("Usage:")
        print("  python exercises.py --exercise N  (run exercise N)")
        print("  python exercises.py --all        (run all exercises)")
        print("\nAvailable exercises:")
        print("  1. Custom RAG System for Technical Documentation")
        print("  2. Multi-Document Information Synthesis")
        print("  3. Advanced Conversational Knowledge Assistant")
        print("  4. RAG Evaluation Framework")
        print("  5. Production RAG Pipeline")
        print("  6. Domain-Specific RAG System")
        print("  7. Hybrid Search RAG Implementation")