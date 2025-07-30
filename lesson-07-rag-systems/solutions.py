#!/usr/bin/env python3
"""
Lesson 7: RAG Systems - Solution Implementations

Reference implementations for all RAG system exercises.
These solutions demonstrate best practices and advanced techniques.

Study these implementations to understand optimal approaches to RAG system design.
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import hashlib
from dotenv import load_dotenv

# Add shared resources to path
sys.path.append('../shared-resources')

# Load environment variables
load_dotenv()

# LangChain imports
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.retrievers import ContextualCompressionRetriever, MultiQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, DocumentCompressorPipeline
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document, BaseRetriever
from langchain.callbacks import get_openai_callback
from pydantic import BaseModel, Field

# Shared utilities
from utils.llm_setup import setup_llm_providers, get_preferred_llm
from utils.error_handling import safe_llm_call

# Set up providers
providers = setup_llm_providers()
llm = get_preferred_llm(providers, prefer_chat=True) if providers else None
embeddings = OpenAIEmbeddings() if os.getenv("OPENAI_API_KEY") else None


class CodeAwareTextSplitter(RecursiveCharacterTextSplitter):
    """Text splitter that preserves code blocks and technical formatting."""
    
    def __init__(self, **kwargs):
        # Add code-aware separators
        separators = [
            "\n\n",  # Paragraph breaks
            "\n```\n",  # Code block endings
            "\n```",    # Code block endings (alternative)
            "```\n",    # Code block beginnings
            "\n## ",    # Markdown headers
            "\n# ",     # Markdown headers
            "\n\n",     # Double newlines
            "\n",       # Single newlines
            " ",        # Spaces
            ""          # Characters
        ]
        super().__init__(separators=separators, **kwargs)
    
    def split_text(self, text: str) -> List[str]:
        """Split text while preserving code blocks."""
        # First, identify and protect code blocks
        code_blocks = []
        protected_text = text
        
        # Find code blocks and replace with placeholders
        import re
        code_pattern = r'```[\s\S]*?```'
        for i, match in enumerate(re.finditer(code_pattern, text)):
            placeholder = f"__CODE_BLOCK_{i}__"
            code_blocks.append(match.group())
            protected_text = protected_text.replace(match.group(), placeholder, 1)
        
        # Split the protected text
        chunks = super().split_text(protected_text)
        
        # Restore code blocks
        final_chunks = []
        for chunk in chunks:
            restored_chunk = chunk
            for i, code_block in enumerate(code_blocks):
                placeholder = f"__CODE_BLOCK_{i}__"
                restored_chunk = restored_chunk.replace(placeholder, code_block)
            final_chunks.append(restored_chunk)
        
        return final_chunks


def solution_1_custom_rag_system():
    """
    Solution 1: Custom RAG System for Technical Documentation
    """
    print("🏗️ Solution 1: Custom RAG System for Technical Documentation")
    print("-" * 60)
    
    class TechnicalRAGSystem:
        """RAG system optimized for technical documentation."""
        
        def __init__(self, llm, embeddings):
            self.llm = llm
            self.embeddings = embeddings
            self.vectorstore = None
            self.qa_chain = None
        
        def create_technical_prompt(self):
            """Create a prompt template optimized for technical content."""
            template = """
            You are a technical documentation expert. Use the provided context to answer the question accurately and comprehensively.
            
            Guidelines:
            - Provide precise technical information
            - Include code examples when relevant
            - Explain complex concepts clearly
            - Cite specific sources when possible
            - If information is insufficient, say so clearly
            
            Context:
            {context}
            
            Question: {question}
            
            Technical Answer:"""
            
            return PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
        
        def process_documents(self, documents: List[Document]):
            """Process technical documents with code-aware splitting."""
            # Use code-aware text splitter
            text_splitter = CodeAwareTextSplitter(
                chunk_size=1200,
                chunk_overlap=200,
                length_function=len
            )
            
            # Split documents
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory="./tech_docs_db"
            )
            
            print(f"✅ Processed {len(splits)} technical document chunks")
            return splits
        
        def setup_qa_chain(self):
            """Set up QA chain with technical prompt."""
            if not self.vectorstore:
                raise ValueError("Process documents first")
            
            # Create retriever with metadata filtering
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance
                search_kwargs={"k": 6, "fetch_k": 20}
            )
            
            # Add compression for better context
            compressor = LLMChainExtractor.from_llm(self.llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=retriever
            )
            
            # Create QA chain with custom prompt
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=compression_retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.create_technical_prompt()}
            )
        
        def query(self, question: str):
            """Query the technical documentation system."""
            if not self.qa_chain:
                self.setup_qa_chain()
            
            with get_openai_callback() as cb:
                response = self.qa_chain({"query": question})
            
            # Format response with sources
            answer = response["result"]
            sources = response["source_documents"]
            
            result = {
                "answer": answer,
                "sources": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in sources
                ],
                "cost": {
                    "total_tokens": cb.total_tokens,
                    "total_cost": cb.total_cost
                }
            }
            
            return result
    
    # Demo the technical RAG system
    if llm and embeddings:
        tech_docs = [
            Document(
                page_content="""
                # FastAPI Request Validation
                
                FastAPI provides automatic request validation using Pydantic models.
                
                ## Basic Example
                ```python
                from fastapi import FastAPI
                from pydantic import BaseModel
                
                class UserCreate(BaseModel):
                    name: str
                    email: str
                    age: int = None
                
                app = FastAPI()
                
                @app.post("/users/")
                def create_user(user: UserCreate):
                    return {"message": f"User {user.name} created"}
                ```
                
                ## Validation Features
                - Automatic type checking
                - Custom validators
                - Error handling with detailed messages
                - OpenAPI schema generation
                """,
                metadata={"source": "fastapi_validation.md", "type": "tutorial", "topic": "validation"}
            )
        ]
        
        system = TechnicalRAGSystem(llm, embeddings)
        system.process_documents(tech_docs)
        
        # Test technical queries
        test_queries = [
            "How do I implement request validation in FastAPI?",
            "Show me a code example of Pydantic model validation",
            "What are the benefits of automatic validation?"
        ]
        
        for query in test_queries:
            print(f"\n❓ Query: {query}")
            result = system.query(query)
            print(f"🤖 Answer: {result['answer'][:300]}...")
            print(f"📊 Cost: ${result['cost']['total_cost']:.4f}")


def solution_2_multi_document_synthesis():
    """
    Solution 2: Multi-Document Information Synthesis
    """
    print("\n🔄 Solution 2: Multi-Document Information Synthesis")
    print("-" * 60)
    
    class SynthesisRAGSystem:
        """RAG system that synthesizes information from multiple documents."""
        
        def __init__(self, llm, embeddings):
            self.llm = llm
            self.embeddings = embeddings
            self.vectorstore = None
        
        def create_synthesis_prompt(self):
            """Create prompt template for information synthesis."""
            template = """
            You are an expert information synthesizer. Your task is to combine information from multiple sources to provide a comprehensive answer.
            
            Instructions:
            1. Analyze all provided context from different sources
            2. Identify common themes and unique perspectives
            3. Synthesize information into a coherent response
            4. Note any conflicting information and explain differences
            5. Cite specific sources for different claims
            6. Provide a balanced, comprehensive answer
            
            Context from multiple sources:
            {context}
            
            Question: {question}
            
            Synthesized Response:"""
            
            return PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
        
        def enhanced_retrieval(self, query: str, k: int = 10) -> List[Document]:
            """Retrieve documents with enhanced strategy for synthesis."""
            if not self.vectorstore:
                raise ValueError("Vector store not initialized")
            
            # Get more documents for synthesis
            docs = self.vectorstore.similarity_search(query, k=k)
            
            # Group by source/perspective if available
            grouped_docs = defaultdict(list)
            for doc in docs:
                source = doc.metadata.get('source', 'unknown')
                perspective = doc.metadata.get('perspective', 'general')
                key = f"{source}_{perspective}"
                grouped_docs[key].append(doc)
            
            # Ensure diversity by taking from different groups
            diverse_docs = []
            max_per_group = max(1, k // len(grouped_docs)) if grouped_docs else k
            
            for group_docs in grouped_docs.values():
                diverse_docs.extend(group_docs[:max_per_group])
            
            return diverse_docs[:k]
        
        def synthesize_query(self, question: str):
            """Perform multi-document synthesis."""
            # Enhanced retrieval
            docs = self.enhanced_retrieval(question, k=8)
            
            # Create context with source attribution
            context_parts = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', f'Document {i}')
                perspective = doc.metadata.get('perspective', '')
                
                context_part = f"Source {i} ({source}"
                if perspective:
                    context_part += f" - {perspective} perspective"
                context_part += f"):\n{doc.page_content}\n"
                
                context_parts.append(context_part)
            
            context = "\n---\n".join(context_parts)
            
            # Generate synthesis
            prompt = self.create_synthesis_prompt()
            formatted_prompt = prompt.format(context=context, question=question)
            
            response = safe_llm_call(self.llm, formatted_prompt)
            
            return {
                "synthesis": response,
                "sources_used": len(docs),
                "perspectives": list(set(doc.metadata.get('perspective', 'general') for doc in docs))
            }
    
    # Demo the synthesis system
    if llm and embeddings:
        # Create documents with different perspectives
        synthesis_docs = [
            Document(
                page_content="Artificial Intelligence is fundamentally about creating machines that can perform tasks that typically require human intelligence, such as visual perception, speech recognition, and decision-making.",
                metadata={"source": "ai_overview.txt", "perspective": "functional"}
            ),
            Document(
                page_content="AI systems learn from data through statistical methods and pattern recognition, using algorithms like neural networks to identify complex relationships in large datasets.",
                metadata={"source": "ai_technical.txt", "perspective": "technical"}
            ),
            Document(
                page_content="From a philosophical standpoint, AI raises questions about consciousness, intelligence, and what it means to think, challenging our understanding of human uniqueness.",
                metadata={"source": "ai_philosophy.txt", "perspective": "philosophical"}
            )
        ]
        
        system = SynthesisRAGSystem(llm, embeddings)
        system.vectorstore = Chroma.from_documents(synthesis_docs, embeddings)
        
        result = system.synthesize_query("What is artificial intelligence?")
        print(f"🔗 Synthesis: {result['synthesis'][:400]}...")
        print(f"📊 Sources used: {result['sources_used']}")
        print(f"🎭 Perspectives: {result['perspectives']}")


def solution_3_conversational_knowledge_assistant():
    """
    Solution 3: Advanced Conversational Knowledge Assistant
    """
    print("\n💬 Solution 3: Advanced Conversational Knowledge Assistant")
    print("-" * 60)
    
    class AdvancedConversationalRAG:
        """Advanced conversational RAG with context management."""
        
        def __init__(self, llm, embeddings, max_tokens: int = 4000):
            self.llm = llm
            self.embeddings = embeddings
            self.max_tokens = max_tokens
            self.vectorstore = None
            
            # Use summary buffer memory for long conversations
            self.memory = ConversationSummaryBufferMemory(
                llm=llm,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                max_token_limit=max_tokens // 2  # Reserve half for context
            )
            
            self.conversation_chain = None
        
        def classify_question_type(self, question: str, chat_history: str) -> str:
            """Classify whether question needs new retrieval or is follow-up."""
            classification_prompt = f"""
            Analyze the following question and conversation history to determine if this is:
            A) A new question requiring fresh information retrieval
            B) A follow-up question about previously discussed topics
            C) A clarification request about recent responses
            
            Chat History:
            {chat_history[-1000:]}  # Last 1000 chars
            
            Current Question: {question}
            
            Classification (A/B/C):"""
            
            response = safe_llm_call(self.llm, classification_prompt)
            return response.strip() if response else "A"
        
        def context_aware_retrieval(self, question: str, chat_history: str) -> List[Document]:
            """Perform context-aware document retrieval."""
            if not self.vectorstore:
                return []
            
            # Expand query with conversation context for better retrieval
            expansion_prompt = f"""
            Given the conversation history, expand the current question to include relevant context that would help retrieve better information.
            
            Chat History:
            {chat_history[-800:]}
            
            Current Question: {question}
            
            Expanded Query:"""
            
            expanded_query = safe_llm_call(self.llm, expansion_prompt)
            search_query = expanded_query if expanded_query else question
            
            # Retrieve documents
            docs = self.vectorstore.similarity_search(search_query, k=4)
            return docs
        
        def setup_conversation_chain(self):
            """Set up the conversational chain."""
            if not self.vectorstore:
                raise ValueError("Vector store not initialized")
            
            # Custom retrieval chain that considers conversation context
            def context_aware_retriever_func(question: str) -> List[Document]:
                chat_history = self.memory.buffer_as_str
                question_type = self.classify_question_type(question, chat_history)
                
                if question_type.startswith('A'):  # New question
                    return self.context_aware_retrieval(question, chat_history)
                else:  # Follow-up or clarification
                    # For follow-ups, retrieve fewer documents or none
                    return self.vectorstore.similarity_search(question, k=2)
            
            # Create custom retriever
            class ContextAwareRetriever(BaseRetriever):
                def __init__(self, retriever_func):
                    self.retriever_func = retriever_func
                
                def get_relevant_documents(self, query: str) -> List[Document]:
                    return self.retriever_func(query)
            
            retriever = ContextAwareRetriever(context_aware_retriever_func)
            
            # Create conversational chain
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=True
            )
        
        def chat(self, question: str):
            """Have an intelligent conversation."""
            if not self.conversation_chain:
                self.setup_conversation_chain()
            
            start_time = time.time()
            response = self.conversation_chain({"question": question})
            response_time = time.time() - start_time
            
            return {
                "answer": response["answer"],
                "sources": response.get("source_documents", []),
                "response_time": response_time,
                "memory_size": len(self.memory.buffer_as_str)
            }
        
        def get_conversation_summary(self):
            """Get a summary of the current conversation."""
            return self.memory.moving_summary_buffer if hasattr(self.memory, 'moving_summary_buffer') else "No summary available"
    
    # Demo the advanced conversational system
    if llm and embeddings:
        knowledge_docs = [
            Document(
                page_content="Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
                metadata={"source": "python_intro.txt", "topic": "programming"}
            ),
            Document(
                page_content="Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every scenario.",
                metadata={"source": "ml_basics.txt", "topic": "AI/ML"}
            )
        ]
        
        assistant = AdvancedConversationalRAG(llm, embeddings)
        assistant.vectorstore = Chroma.from_documents(knowledge_docs, embeddings)
        
        # Simulate conversation
        conversation_flow = [
            "What is Python?",
            "What makes it different from other languages?",
            "Can you give me an example?",
            "Now tell me about machine learning",
            "How does it relate to what we discussed about Python?"
        ]
        
        print("🗣️ Conversation Simulation:")
        for question in conversation_flow:
            print(f"\n👤 User: {question}")
            result = assistant.chat(question)
            print(f"🤖 Assistant: {result['answer'][:200]}...")
            print(f"⏱️ Response time: {result['response_time']:.2f}s")
            print(f"🧠 Memory size: {result['memory_size']} chars")
            
            # Short pause for realism
            time.sleep(1)


def solution_4_rag_evaluation_framework():
    """
    Solution 4: Comprehensive RAG Evaluation Framework
    """
    print("\n📊 Solution 4: RAG Evaluation Framework")
    print("-" * 60)
    
    class ComprehensiveRAGEvaluator:
        """Complete framework for evaluating RAG systems."""
        
        def __init__(self, llm):
            self.llm = llm
            self.evaluation_results = {}
        
        def calculate_retrieval_metrics(self, retrieved_docs: List[Document], 
                                      relevant_doc_ids: List[str], 
                                      query: str) -> Dict[str, float]:
            """Calculate retrieval quality metrics."""
            retrieved_ids = [doc.metadata.get('id', str(hash(doc.page_content))) for doc in retrieved_docs]
            
            # Calculate metrics
            relevant_retrieved = set(retrieved_ids) & set(relevant_doc_ids)
            
            precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0
            recall = len(relevant_retrieved) / len(relevant_doc_ids) if relevant_doc_ids else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Mean Reciprocal Rank (MRR)
            mrr = 0
            for i, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_doc_ids:
                    mrr = 1.0 / (i + 1)
                    break
            
            return {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mrr": mrr,
                "retrieved_count": len(retrieved_ids),
                "relevant_count": len(relevant_doc_ids)
            }
        
        def evaluate_answer_faithfulness(self, question: str, answer: str, 
                                       context_docs: List[Document]) -> float:
            """Evaluate if the answer is faithful to the retrieved context."""
            context = "\n".join([doc.page_content for doc in context_docs])
            
            faithfulness_prompt = f"""
            Evaluate whether the given answer is faithful to the provided context. 
            An answer is faithful if all information in the answer can be found in or reasonably inferred from the context.
            
            Context:
            {context}
            
            Question: {question}
            Answer: {answer}
            
            Rate faithfulness on a scale of 0-1 where:
            0 = Completely unfaithful (answer contains information not in context)
            1 = Completely faithful (all answer information comes from context)
            
            Provide only the numerical score:"""
            
            response = safe_llm_call(self.llm, faithfulness_prompt)
            try:
                score = float(response.strip())
                return max(0, min(1, score))  # Clamp between 0 and 1
            except ValueError:
                return 0.5  # Default score if parsing fails
        
        def evaluate_answer_relevance(self, question: str, answer: str) -> float:
            """Evaluate how relevant the answer is to the question."""
            relevance_prompt = f"""
            Evaluate how relevant the answer is to the question.
            
            Question: {question}
            Answer: {answer}
            
            Rate relevance on a scale of 0-1 where:
            0 = Completely irrelevant
            1 = Perfectly relevant and addresses the question completely
            
            Provide only the numerical score:"""
            
            response = safe_llm_call(self.llm, relevance_prompt)
            try:
                score = float(response.strip())
                return max(0, min(1, score))
            except ValueError:
                return 0.5
        
        def evaluate_answer_completeness(self, question: str, answer: str, 
                                       expected_aspects: List[str] = None) -> float:
            """Evaluate how complete the answer is."""
            if expected_aspects:
                aspects_text = "Expected aspects to cover: " + ", ".join(expected_aspects)
            else:
                aspects_text = "Consider what aspects a complete answer should cover."
            
            completeness_prompt = f"""
            Evaluate how complete the answer is for the given question.
            {aspects_text}
            
            Question: {question}
            Answer: {answer}
            
            Rate completeness on a scale of 0-1 where:
            0 = Very incomplete, missing major aspects
            1 = Very complete, covers all important aspects
            
            Provide only the numerical score:"""
            
            response = safe_llm_call(self.llm, completeness_prompt)
            try:
                score = float(response.strip())
                return max(0, min(1, score))
            except ValueError:
                return 0.5
        
        def comprehensive_evaluation(self, test_cases: List[Dict]) -> Dict[str, Any]:
            """Run comprehensive evaluation on test cases."""
            results = {
                "retrieval_metrics": [],
                "generation_metrics": [],
                "overall_scores": {}
            }
            
            for i, test_case in enumerate(test_cases):
                print(f"Evaluating test case {i+1}/{len(test_cases)}")
                
                # Extract test case components
                question = test_case["question"]
                retrieved_docs = test_case["retrieved_docs"]
                answer = test_case["answer"]
                relevant_doc_ids = test_case.get("relevant_doc_ids", [])
                expected_aspects = test_case.get("expected_aspects", [])
                
                # Retrieval evaluation
                retrieval_metrics = self.calculate_retrieval_metrics(
                    retrieved_docs, relevant_doc_ids, question
                )
                results["retrieval_metrics"].append(retrieval_metrics)
                
                # Generation evaluation
                faithfulness = self.evaluate_answer_faithfulness(question, answer, retrieved_docs)
                relevance = self.evaluate_answer_relevance(question, answer)
                completeness = self.evaluate_answer_completeness(question, answer, expected_aspects)
                
                generation_metrics = {
                    "faithfulness": faithfulness,
                    "relevance": relevance,
                    "completeness": completeness,
                    "overall_generation": (faithfulness + relevance + completeness) / 3
                }
                results["generation_metrics"].append(generation_metrics)
            
            # Calculate overall scores
            if results["retrieval_metrics"]:
                avg_retrieval = {
                    metric: sum(r[metric] for r in results["retrieval_metrics"]) / len(results["retrieval_metrics"])
                    for metric in results["retrieval_metrics"][0].keys()
                }
                results["overall_scores"]["retrieval"] = avg_retrieval
            
            if results["generation_metrics"]:
                avg_generation = {
                    metric: sum(r[metric] for r in results["generation_metrics"]) / len(results["generation_metrics"])
                    for metric in results["generation_metrics"][0].keys()
                }
                results["overall_scores"]["generation"] = avg_generation
            
            return results
        
        def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
            """Generate a comprehensive evaluation report."""
            report = "# RAG System Evaluation Report\n\n"
            
            # Overall scores
            if "overall_scores" in results:
                report += "## Overall Performance\n\n"
                
                if "retrieval" in results["overall_scores"]:
                    retrieval = results["overall_scores"]["retrieval"]
                    report += f"### Retrieval Performance\n"
                    report += f"- Precision: {retrieval['precision']:.3f}\n"
                    report += f"- Recall: {retrieval['recall']:.3f}\n"
                    report += f"- F1 Score: {retrieval['f1']:.3f}\n"
                    report += f"- MRR: {retrieval['mrr']:.3f}\n\n"
                
                if "generation" in results["overall_scores"]:
                    generation = results["overall_scores"]["generation"]
                    report += f"### Generation Performance\n"
                    report += f"- Faithfulness: {generation['faithfulness']:.3f}\n"
                    report += f"- Relevance: {generation['relevance']:.3f}\n"
                    report += f"- Completeness: {generation['completeness']:.3f}\n"
                    report += f"- Overall Generation: {generation['overall_generation']:.3f}\n\n"
            
            # Recommendations
            report += "## Recommendations\n\n"
            if results["overall_scores"]["retrieval"]["precision"] < 0.7:
                report += "- Consider improving retrieval precision through better chunking or re-ranking\n"
            if results["overall_scores"]["retrieval"]["recall"] < 0.7:
                report += "- Consider retrieving more documents or improving embedding quality\n"
            if results["overall_scores"]["generation"]["faithfulness"] < 0.8:
                report += "- Improve prompt engineering to encourage faithfulness to context\n"
            if results["overall_scores"]["generation"]["relevance"] < 0.8:
                report += "- Review prompt templates to improve answer relevance\n"
            
            return report
    
    # Demo the evaluation framework
    if llm:
        evaluator = ComprehensiveRAGEvaluator(llm)
        
        # Sample test case
        sample_test_cases = [
            {
                "question": "What is machine learning?",
                "retrieved_docs": [
                    Document(
                        page_content="Machine learning is a subset of AI that enables computers to learn from data.",
                        metadata={"id": "ml_doc_1"}
                    )
                ],
                "answer": "Machine learning is a subset of artificial intelligence that allows computers to learn patterns from data without being explicitly programmed.",
                "relevant_doc_ids": ["ml_doc_1"],
                "expected_aspects": ["definition", "relationship to AI", "learning from data"]
            }
        ]
        
        print("🧪 Running evaluation on sample test case...")
        results = evaluator.comprehensive_evaluation(sample_test_cases)
        
        print("📋 Evaluation Results:")
        print(f"Retrieval F1: {results['overall_scores']['retrieval']['f1']:.3f}")
        print(f"Generation Overall: {results['overall_scores']['generation']['overall_generation']:.3f}")
        
        # Generate and display report
        report = evaluator.generate_evaluation_report(results)
        print("\n📄 Evaluation Report Preview:")
        print(report[:500] + "...")


def solution_5_production_rag_pipeline():
    """
    Solution 5: Production-Ready RAG Pipeline
    """
    print("\n🚀 Solution 5: Production RAG Pipeline")
    print("-" * 60)
    
    class ProductionRAGPipeline:
        """Production-ready RAG pipeline with comprehensive features."""
        
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.cache = {}
            self.metrics = defaultdict(list)
            self.logger = self._setup_logging()
            self.llm = None
            self.embeddings = None
            self.vectorstore = None
            self.qa_chain = None
            
            self._initialize_components()
        
        def _setup_logging(self):
            """Set up comprehensive logging."""
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('rag_system.log'),
                    logging.StreamHandler()
                ]
            )
            return logging.getLogger('ProductionRAG')
        
        def _initialize_components(self):
            """Initialize LLM and embedding components."""
            try:
                providers = setup_llm_providers()
                self.llm = get_preferred_llm(providers, prefer_chat=True)
                self.embeddings = OpenAIEmbeddings()
                self.logger.info("Components initialized successfully")
            except Exception as e:
                self.logger.error(f"Component initialization failed: {e}")
                raise
        
        def _cache_key(self, text: str) -> str:
            """Generate cache key for text."""
            return hashlib.md5(text.encode()).hexdigest()
        
        def _get_cached_embedding(self, text: str):
            """Get cached embedding or compute new one."""
            cache_key = self._cache_key(text)
            
            if cache_key in self.cache:
                self.metrics['cache_hits'].append(time.time())
                return self.cache[cache_key]
            
            # Compute embedding
            embedding = self.embeddings.embed_query(text)
            
            # Cache with size limit
            if len(self.cache) < self.config.get('max_cache_size', 1000):
                self.cache[cache_key] = embedding
            
            self.metrics['cache_misses'].append(time.time())
            return embedding
        
        def health_check(self) -> Dict[str, Any]:
            """Perform comprehensive health check."""
            health_status = {
                "status": "healthy",
                "timestamp": time.time(),
                "components": {},
                "metrics": {}
            }
            
            # Check LLM
            try:
                test_response = safe_llm_call(self.llm, "Test query")
                health_status["components"]["llm"] = "healthy" if test_response else "unhealthy"
            except Exception as e:
                health_status["components"]["llm"] = f"error: {e}"
                health_status["status"] = "degraded"
            
            # Check embeddings
            try:
                test_embedding = self.embeddings.embed_query("test")
                health_status["components"]["embeddings"] = "healthy" if test_embedding else "unhealthy"
            except Exception as e:
                health_status["components"]["embeddings"] = f"error: {e}"
                health_status["status"] = "degraded"
            
            # Check vector store
            if self.vectorstore:
                try:
                    test_results = self.vectorstore.similarity_search("test", k=1)
                    health_status["components"]["vectorstore"] = "healthy"
                except Exception as e:
                    health_status["components"]["vectorstore"] = f"error: {e}"
                    health_status["status"] = "degraded"
            else:
                health_status["components"]["vectorstore"] = "not_initialized"
            
            # Add metrics
            health_status["metrics"] = {
                "cache_size": len(self.cache),
                "cache_hits": len(self.metrics['cache_hits']),
                "cache_misses": len(self.metrics['cache_misses']),
                "queries_processed": len(self.metrics.get('queries', [])),
                "average_response_time": self._calculate_avg_response_time()
            }
            
            return health_status
        
        def _calculate_avg_response_time(self) -> float:
            """Calculate average response time from recent queries."""
            response_times = self.metrics.get('response_times', [])
            if not response_times:
                return 0.0
            
            # Use last 100 response times
            recent_times = response_times[-100:]
            return sum(recent_times) / len(recent_times)
        
        def process_query(self, query: str, user_id: str = None) -> Dict[str, Any]:
            """Process query with full production pipeline."""
            start_time = time.time()
            query_id = f"{int(start_time)}_{hash(query) % 10000}"
            
            self.logger.info(f"Processing query {query_id}: {query[:100]}")
            
            try:
                # Input validation
                if not query or len(query.strip()) == 0:
                    raise ValueError("Empty query")
                
                if len(query) > self.config.get('max_query_length', 1000):
                    raise ValueError("Query too long")
                
                # Check cache for complete response
                response_cache_key = self._cache_key(f"response_{query}")
                if response_cache_key in self.cache and self.config.get('enable_response_cache', True):
                    cached_response = self.cache[response_cache_key]
                    cached_response['cached'] = True
                    cached_response['query_id'] = query_id
                    
                    response_time = time.time() - start_time
                    self.metrics['response_times'].append(response_time)
                    self.metrics['queries'].append(query_id)
                    
                    return cached_response
                
                # Process with QA chain
                if not self.qa_chain:
                    raise RuntimeError("QA chain not initialized")
                
                with get_openai_callback() as cb:
                    response = self.qa_chain({"query": query})
                
                # Format response
                result = {
                    "query_id": query_id,
                    "answer": response["result"],
                    "sources": [
                        {
                            "content": doc.page_content[:200] + "...",
                            "metadata": doc.metadata,
                            "relevance_score": 1.0  # Could be enhanced with actual scoring
                        }
                        for doc in response.get("source_documents", [])
                    ],
                    "cached": False,
                    "processing_time": time.time() - start_time,
                    "cost": {
                        "total_tokens": cb.total_tokens,
                        "total_cost": cb.total_cost
                    },
                    "user_id": user_id,
                    "timestamp": time.time()
                }
                
                # Cache response if enabled
                if self.config.get('enable_response_cache', True):
                    self.cache[response_cache_key] = result.copy()
                
                # Record metrics
                response_time = time.time() - start_time
                self.metrics['response_times'].append(response_time)
                self.metrics['queries'].append(query_id)
                self.metrics['token_usage'].append(cb.total_tokens)
                self.metrics['costs'].append(cb.total_cost)
                
                self.logger.info(f"Query {query_id} completed in {response_time:.2f}s")
                
                return result
                
            except Exception as e:
                error_response = {
                    "query_id": query_id,
                    "error": str(e),
                    "processing_time": time.time() - start_time,
                    "user_id": user_id,
                    "timestamp": time.time()
                }
                
                self.logger.error(f"Query {query_id} failed: {e}")
                self.metrics['errors'].append(error_response)
                
                return error_response
        
        def initialize_vectorstore(self, documents: List[Document]):
            """Initialize vector store with documents."""
            try:
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.config.get('persist_directory', './prod_chroma_db')
                )
                
                # Set up QA chain
                retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": self.config.get('retrieval_k', 4)}
                )
                
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )
                
                self.logger.info(f"Vector store initialized with {len(documents)} documents")
                
            except Exception as e:
                self.logger.error(f"Vector store initialization failed: {e}")
                raise
        
        def get_system_metrics(self) -> Dict[str, Any]:
            """Get comprehensive system metrics."""
            return {
                "cache_performance": {
                    "size": len(self.cache),
                    "hit_rate": len(self.metrics['cache_hits']) / 
                              (len(self.metrics['cache_hits']) + len(self.metrics['cache_misses']))
                              if (len(self.metrics['cache_hits']) + len(self.metrics['cache_misses'])) > 0 else 0
                },
                "query_performance": {
                    "total_queries": len(self.metrics.get('queries', [])),
                    "average_response_time": self._calculate_avg_response_time(),
                    "error_rate": len(self.metrics.get('errors', [])) / 
                                 max(len(self.metrics.get('queries', [])), 1)
                },
                "cost_analysis": {
                    "total_cost": sum(self.metrics.get('costs', [])),
                    "average_cost_per_query": sum(self.metrics.get('costs', [])) / 
                                            max(len(self.metrics.get('costs', [])), 1),
                    "total_tokens": sum(self.metrics.get('token_usage', []))
                }
            }
    
    # Demo the production pipeline
    if llm and embeddings:
        config = {
            'max_cache_size': 500,
            'max_query_length': 1000,
            'enable_response_cache': True,
            'retrieval_k': 4,
            'persist_directory': './prod_demo_db'
        }
        
        pipeline = ProductionRAGPipeline(config)
        
        # Initialize with sample documents
        sample_docs = [
            Document(
                page_content="Production RAG systems require careful monitoring, caching, and error handling to ensure reliable performance at scale.",
                metadata={"source": "prod_best_practices.txt", "id": "prod_1"}
            )
        ]
        
        pipeline.initialize_vectorstore(sample_docs)
        
        # Test queries
        test_queries = [
            "What are best practices for production RAG systems?",
            "How do you monitor RAG performance?",
            "What are best practices for production RAG systems?"  # Duplicate to test caching
        ]
        
        print("🧪 Testing production pipeline:")
        for query in test_queries:
            result = pipeline.process_query(query, user_id="test_user")
            print(f"Query: {query[:50]}...")
            print(f"Cached: {result.get('cached', False)}")
            print(f"Response time: {result.get('processing_time', 0):.3f}s")
            if 'cost' in result:
                print(f"Cost: ${result['cost']['total_cost']:.4f}")
            print("-" * 40)
        
        # Health check
        health = pipeline.health_check()
        print(f"\n🔍 System Health: {health['status']}")
        
        # Metrics
        metrics = pipeline.get_system_metrics()
        print(f"📊 Cache hit rate: {metrics['cache_performance']['hit_rate']:.2%}")
        print(f"📊 Average response time: {metrics['query_performance']['average_response_time']:.3f}s")


def run_all_solutions():
    """Run all solution demonstrations."""
    print("🦜🔗 LangChain Course - Lesson 7: RAG Systems Solutions")
    print("=" * 70)
    
    if not llm or not embeddings:
        print("❌ LLM or embeddings not available. Please check your setup.")
        return
    
    print(f"✅ Using LLM: {type(llm).__name__}")
    print(f"✅ Using Embeddings: {type(embeddings).__name__}")
    
    solutions = [
        solution_1_custom_rag_system,
        solution_2_multi_document_synthesis,
        solution_3_conversational_knowledge_assistant,
        solution_4_rag_evaluation_framework,
        solution_5_production_rag_pipeline,
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
    print("   • Custom RAG systems can be optimized for specific domains")
    print("   • Multi-document synthesis provides comprehensive answers")
    print("   • Conversational RAG requires sophisticated memory management")
    print("   • Evaluation frameworks are essential for system improvement")
    print("   • Production systems need monitoring, caching, and error handling")


if __name__ == "__main__":
    run_all_solutions()