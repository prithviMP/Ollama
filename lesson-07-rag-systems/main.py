#!/usr/bin/env python3
"""
Lesson 7: RAG (Retrieval Augmented Generation) Systems

This lesson covers:
1. Basic RAG implementation with document loading and vector stores
2. Advanced retrieval strategies (hybrid search, re-ranking)
3. Conversational RAG with memory management
4. Multi-modal RAG systems
5. Production RAG patterns and evaluation

Author: LangChain Course
"""

import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from pydantic import BaseModel, Field

# Shared utilities
from utils.llm_setup import setup_llm_providers, get_preferred_llm
from utils.error_handling import safe_llm_call


class RAGConfig(BaseModel):
    """Configuration for RAG systems."""
    chunk_size: int = Field(default=1000, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    max_retrieved_docs: int = Field(default=4, description="Number of docs to retrieve")
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold")
    enable_compression: bool = Field(default=True, description="Enable retrieval compression")
    enable_reranking: bool = Field(default=False, description="Enable result re-ranking")


class BasicRAGSystem:
    """
    Basic RAG system implementation with document loading and querying.
    """
    
    def __init__(self, llm, embeddings, config: RAGConfig = None):
        self.llm = llm
        self.embeddings = embeddings
        self.config = config or RAGConfig()
        self.vectorstore = None
        self.qa_chain = None
        
    def load_documents(self, data_path: str):
        """Load documents from a directory or file."""
        print(f"📚 Loading documents from: {data_path}")
        
        if not os.path.exists(data_path):
            print(f"❌ Path does not exist: {data_path}")
            return []
        
        documents = []
        
        if os.path.isfile(data_path):
            # Single file
            if data_path.endswith('.pdf'):
                loader = PyPDFLoader(data_path)
            else:
                loader = TextLoader(data_path)
            documents = loader.load()
        else:
            # Directory
            loader = DirectoryLoader(
                data_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True
            )
            documents = loader.load()
        
        print(f"✅ Loaded {len(documents)} documents")
        return documents
    
    def create_sample_documents(self):
        """Create sample documents for demonstration."""
        sample_docs = [
            Document(
                page_content="""
                Retrieval Augmented Generation (RAG) is a technique that combines the power of 
                large language models with external knowledge retrieval. RAG systems work by 
                first retrieving relevant documents from a knowledge base, then using those 
                documents as context for generating answers. This approach allows LLMs to access 
                up-to-date information and domain-specific knowledge that wasn't part of their 
                training data.
                """,
                metadata={"source": "rag_intro.txt", "topic": "RAG fundamentals"}
            ),
            Document(
                page_content="""
                Vector databases are specialized databases designed to store and query high-dimensional 
                vectors efficiently. In RAG systems, vector databases store document embeddings that 
                represent the semantic meaning of text chunks. Popular vector databases include Chroma, 
                Pinecone, Weaviate, and FAISS. These databases support similarity search operations 
                that allow retrieving the most relevant documents for a given query.
                """,
                metadata={"source": "vector_db.txt", "topic": "Vector databases"}
            ),
            Document(
                page_content="""
                Embedding models convert text into high-dimensional vectors that capture semantic 
                meaning. Popular embedding models include OpenAI's text-embedding-ada-002, 
                Sentence-BERT, and various models from Hugging Face. The choice of embedding model 
                significantly impacts retrieval quality. Factors to consider include embedding 
                dimensions, computational cost, and domain-specific performance.
                """,
                metadata={"source": "embeddings.txt", "topic": "Embedding models"}
            ),
            Document(
                page_content="""
                Chunking strategies determine how documents are split into smaller pieces for 
                embedding and retrieval. Good chunking preserves semantic coherence while ensuring 
                chunks fit within context windows. Common strategies include fixed-size chunking, 
                sentence-based splitting, and semantic chunking. Chunk size and overlap parameters 
                need to be tuned based on document type and retrieval requirements.
                """,
                metadata={"source": "chunking.txt", "topic": "Document chunking"}
            ),
            Document(
                page_content="""
                Production RAG systems require careful attention to performance, scalability, and 
                monitoring. Key considerations include caching strategies, load balancing, error 
                handling, and evaluation metrics. Monitoring should track retrieval quality, 
                generation accuracy, response times, and user satisfaction. A/B testing different 
                configurations helps optimize system performance over time.
                """,
                metadata={"source": "production_rag.txt", "topic": "Production systems"}
            )
        ]
        
        print(f"📝 Created {len(sample_docs)} sample documents")
        return sample_docs
    
    def process_documents(self, documents: List[Document]):
        """Process documents into chunks and create vector store."""
        print("🔄 Processing documents...")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"📄 Created {len(chunks)} document chunks")
        
        # Create vector store
        persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        
        print(f"💾 Vector store created with {len(chunks)} chunks")
        return chunks
    
    def setup_qa_chain(self):
        """Set up the QA chain with retriever."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call process_documents first.")
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.max_retrieved_docs}
        )
        
        # Add compression if enabled
        if self.config.enable_compression:
            compressor = LLMChainExtractor.from_llm(self.llm)
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=retriever
            )
            print("🗜️ Retrieval compression enabled")
        
        # Custom prompt template
        prompt_template = """
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        print("🔗 QA chain setup complete")
    
    def query(self, question: str):
        """Query the RAG system."""
        if not self.qa_chain:
            self.setup_qa_chain()
        
        print(f"\n❓ Question: {question}")
        response = self.qa_chain({"query": question})
        
        answer = response["result"]
        sources = response["source_documents"]
        
        print(f"🤖 Answer: {answer}")
        print(f"\n📚 Sources ({len(sources)} documents):")
        for i, doc in enumerate(sources, 1):
            source = doc.metadata.get("source", "Unknown")
            topic = doc.metadata.get("topic", "General")
            content_preview = doc.page_content[:100] + "..."
            print(f"  {i}. {source} ({topic}): {content_preview}")
        
        return {"answer": answer, "sources": sources}


class ConversationalRAGSystem(BasicRAGSystem):
    """
    RAG system with conversation memory for multi-turn interactions.
    """
    
    def __init__(self, llm, embeddings, config: RAGConfig = None):
        super().__init__(llm, embeddings, config)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.conversation_chain = None
    
    def setup_conversation_chain(self):
        """Set up conversational retrieval chain."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call process_documents first.")
        
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.max_retrieved_docs}
        )
        
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=os.getenv("VERBOSE", "False").lower() == "true"
        )
        
        print("💬 Conversational RAG chain setup complete")
    
    def chat(self, question: str):
        """Have a conversation with the RAG system."""
        if not self.conversation_chain:
            self.setup_conversation_chain()
        
        print(f"\n👤 You: {question}")
        response = self.conversation_chain({"question": question})
        
        answer = response["answer"]
        sources = response["source_documents"]
        
        print(f"🤖 Assistant: {answer}")
        
        if sources:
            print(f"\n📚 Referenced {len(sources)} sources")
        
        return {"answer": answer, "sources": sources}
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
        print("🧹 Conversation memory cleared")


class HybridRAGSystem:
    """
    Advanced RAG system with hybrid search capabilities.
    """
    
    def __init__(self, llm, embeddings, config: RAGConfig = None):
        self.llm = llm
        self.embeddings = embeddings
        self.config = config or RAGConfig()
        self.dense_retriever = None
        self.sparse_retriever = None
        self.hybrid_retriever = None
    
    def setup_hybrid_retrieval(self, documents: List[Document]):
        """Set up hybrid retrieval combining dense and sparse methods."""
        print("🔀 Setting up hybrid retrieval...")
        
        # This is a simplified example - in practice you might use
        # BM25 retriever from langchain-community for sparse retrieval
        
        # For now, just use dense retrieval with different configurations
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        self.dense_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.config.max_retrieved_docs}
        )
        
        print("✅ Hybrid retrieval setup complete")
    
    def query_hybrid(self, question: str):
        """Query using hybrid retrieval approach."""
        if not self.dense_retriever:
            raise ValueError("Hybrid retrieval not set up. Call setup_hybrid_retrieval first.")
        
        # For this example, we'll just use dense retrieval
        # In a real implementation, you'd combine dense and sparse results
        docs = self.dense_retriever.get_relevant_documents(question)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""
        Based on the following context, answer the question:
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        response = safe_llm_call(self.llm, prompt)
        
        return {"answer": response, "sources": docs}


def demonstrate_basic_rag(llm, embeddings):
    """Demonstrate basic RAG functionality."""
    print("\n" + "="*50)
    print("🏗️  BASIC RAG SYSTEM DEMONSTRATION")
    print("="*50)
    
    config = RAGConfig(
        chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
        max_retrieved_docs=int(os.getenv("MAX_RETRIEVED_DOCS", 4))
    )
    
    rag_system = BasicRAGSystem(llm, embeddings, config)
    
    # Use sample documents
    documents = rag_system.create_sample_documents()
    
    # Process documents
    rag_system.process_documents(documents)
    
    # Test queries
    test_queries = [
        "What is RAG and how does it work?",
        "What are vector databases and why are they important?",
        "How do you choose the right embedding model?",
        "What are best practices for document chunking?",
        "What should I consider when deploying RAG in production?"
    ]
    
    print("\n🧪 Testing basic RAG queries:")
    
    for query in test_queries:
        result = rag_system.query(query)
        print("\n" + "-"*60)


def demonstrate_conversational_rag(llm, embeddings):
    """Demonstrate conversational RAG with memory."""
    print("\n" + "="*50)
    print("💬 CONVERSATIONAL RAG DEMONSTRATION")
    print("="*50)
    
    conv_rag = ConversationalRAGSystem(llm, embeddings)
    documents = conv_rag.create_sample_documents()
    conv_rag.process_documents(documents)
    
    # Simulate a conversation
    conversation_flow = [
        "What is RAG?",
        "How does it differ from fine-tuning?",
        "What are the main components I mentioned earlier?",
        "Can you give me specific examples of vector databases?",
        "Which embedding model would you recommend for beginners?"
    ]
    
    print("\n🗣️  Simulating conversation flow:")
    
    for question in conversation_flow:
        response = conv_rag.chat(question)
        print("\n" + "-"*60)


def demonstrate_rag_evaluation():
    """Demonstrate RAG evaluation techniques."""
    print("\n" + "="*50)
    print("📊 RAG EVALUATION DEMONSTRATION")
    print("="*50)
    
    evaluation_metrics = {
        "Retrieval Quality": [
            "Relevance: Are retrieved documents relevant to the query?",
            "Coverage: Do retrieved documents contain necessary information?",
            "Diversity: Are different aspects of the topic covered?"
        ],
        "Generation Quality": [
            "Faithfulness: Is the answer grounded in retrieved documents?",
            "Completeness: Does the answer address all parts of the question?",
            "Clarity: Is the answer clear and well-structured?"
        ],
        "System Performance": [
            "Response Time: How fast does the system respond?",
            "Cost: What are the API and computational costs?",
            "Scalability: How does the system handle increased load?"
        ]
    }
    
    print("🎯 Key RAG Evaluation Areas:")
    
    for category, metrics in evaluation_metrics.items():
        print(f"\n{category}:")
        for metric in metrics:
            print(f"  • {metric}")
    
    print("\n📈 Evaluation Best Practices:")
    best_practices = [
        "Create diverse test datasets covering various query types",
        "Use both automated metrics and human evaluation",
        "Test edge cases and failure scenarios",
        "Monitor performance over time with real user queries",
        "A/B test different configurations and approaches"
    ]
    
    for practice in best_practices:
        print(f"  ✓ {practice}")


def interactive_rag_demo(llm, embeddings):
    """Interactive RAG demonstration."""
    print("\n" + "="*50)
    print("🚀 INTERACTIVE RAG DEMONSTRATION")
    print("="*50)
    
    print("Welcome to the Interactive RAG Demo!")
    print("Choose a RAG system type to try:")
    print("1. Basic RAG")
    print("2. Conversational RAG")
    print("3. Exit")
    
    while True:
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            print("\n🏗️  Starting Basic RAG...")
            rag_system = BasicRAGSystem(llm, embeddings)
            documents = rag_system.create_sample_documents()
            rag_system.process_documents(documents)
            
            while True:
                question = input("\nEnter your question (or 'back' to return): ").strip()
                if question.lower() == 'back':
                    break
                if question:
                    rag_system.query(question)
        
        elif choice == "2":
            print("\n💬 Starting Conversational RAG...")
            conv_rag = ConversationalRAGSystem(llm, embeddings)
            documents = conv_rag.create_sample_documents()
            conv_rag.process_documents(documents)
            
            print("(Type 'clear' to reset conversation, 'back' to return)")
            
            while True:
                question = input("\n👤 You: ").strip()
                if question.lower() == 'back':
                    break
                elif question.lower() == 'clear':
                    conv_rag.clear_memory()
                elif question:
                    conv_rag.chat(question)
        
        elif choice == "3":
            print("Thanks for trying the RAG demo!")
            break
        
        else:
            print("Invalid choice. Please select 1-3.")


def setup_lesson():
    """Set up the lesson environment."""
    print("🦜🔗 LangChain Course - Lesson 7: RAG Systems")
    print("=" * 60)
    
    providers = setup_llm_providers()
    if not providers:
        print("❌ No LLM providers available. Please check your setup.")
        return None, None
    
    llm = get_preferred_llm(providers, prefer_chat=True)
    
    # Set up embeddings
    try:
        embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        )
        print("✅ OpenAI embeddings configured")
    except Exception as e:
        print(f"❌ Failed to setup embeddings: {e}")
        return None, None
    
    return llm, embeddings


def main():
    """Main function to run RAG demonstrations."""
    llm, embeddings = setup_lesson()
    
    if not llm or not embeddings:
        print("❌ Cannot proceed without LLM and embeddings. Please check your setup.")
        return
    
    print(f"\n🔧 Using LLM: {type(llm).__name__}")
    print(f"🔧 Using Embeddings: {type(embeddings).__name__}")
    
    try:
        # Run demonstrations
        demonstrate_basic_rag(llm, embeddings)
        demonstrate_conversational_rag(llm, embeddings)
        demonstrate_rag_evaluation()
        
        # Interactive demo
        print("\n🎉 Core demonstrations completed!")
        
        run_demo = input("\nWould you like to try the Interactive RAG Demo? (y/n): ").strip().lower()
        if run_demo in ['y', 'yes']:
            interactive_rag_demo(llm, embeddings)
        
        print("\n✨ Lesson 7 completed! You've mastered RAG systems.")
        print("\n📚 Key Skills Acquired:")
        print("   • Basic RAG implementation")
        print("   • Vector store management")
        print("   • Conversational RAG with memory")
        print("   • RAG evaluation techniques")
        print("   • Production considerations")
        
        print("\n🔗 Next: Advanced RAG patterns and specialized applications")
        
    except KeyboardInterrupt:
        print("\n\n👋 Lesson interrupted. You can resume anytime!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    main()