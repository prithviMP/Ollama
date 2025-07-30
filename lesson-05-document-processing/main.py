#!/usr/bin/env python3
"""
Lesson 5: Document Processing & Text Splitters with LangChain
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add shared resources to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared-resources'))

# Document loaders
try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        CSVLoader,
        JSONLoader,
        UnstructuredWordDocumentLoader,
        UnstructuredPowerPointLoader,
        UnstructuredExcelLoader,
        WebBaseLoader,
        DirectoryLoader,
        UnstructuredHTMLLoader
    )
    DOCUMENT_LOADERS_AVAILABLE = True
except ImportError:
    DOCUMENT_LOADERS_AVAILABLE = False
    print("⚠️  Document loaders not available. Install with: pip install langchain-community unstructured")

# Text splitters
try:
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
        CharacterTextSplitter,
        TokenTextSplitter,
        SpacyTextSplitter,
        NLTKTextSplitter,
        MarkdownHeaderTextSplitter,
        HTMLHeaderTextSplitter
    )
    TEXT_SPLITTERS_AVAILABLE = True
except ImportError:
    TEXT_SPLITTERS_AVAILABLE = False
    print("⚠️  Text splitters not available. Install with: pip install langchain")

# Document transformers
try:
    from langchain_community.document_transformers import Html2TextTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Document transformers not available. Install with: pip install langchain-community")

# Web scraping and file processing
try:
    import requests
    from bs4 import BeautifulSoup
    import chardet
    import filetype
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False
    print("⚠️  Web scraping tools not available. Install with: pip install requests beautifulsoup4 chardet filetype")

# NLP tools
try:
    import nltk
    import spacy
    NLP_TOOLS_AVAILABLE = True
except ImportError:
    NLP_TOOLS_AVAILABLE = False
    print("⚠️  NLP tools not available. Install with: pip install nltk spacy")

# Import shared utilities if available
try:
    from utils.llm_setup import setup_llm_providers, get_preferred_llm
    SHARED_UTILS_AVAILABLE = True
except ImportError:
    SHARED_UTILS_AVAILABLE = False
    print("⚠️  Shared utilities not available. Using basic provider setup...")

# LangChain core imports
from langchain.schema import Document


def create_sample_documents():
    """Create sample documents for demonstration"""
    documents_dir = Path(os.getenv("DOCUMENTS_DIRECTORY", "./documents"))
    documents_dir.mkdir(exist_ok=True)
    
    # Sample text document
    sample_txt = documents_dir / "sample.txt"
    if not sample_txt.exists():
        with open(sample_txt, "w") as f:
            f.write("""
# Document Processing with LangChain

This is a sample document to demonstrate document processing capabilities.

## Introduction
Document processing is a crucial part of building RAG (Retrieval Augmented Generation) systems. 
LangChain provides various document loaders and text splitters to handle different file formats.

## Key Concepts
1. Document Loaders: Extract text from various file formats
2. Text Splitters: Break down large documents into manageable chunks
3. Document Transformers: Clean and enrich document content
4. Processing Pipelines: Automate the entire document processing workflow

## Best Practices
- Choose appropriate chunk sizes based on your use case
- Maintain semantic coherence when splitting documents
- Add metadata to improve retrieval accuracy
- Validate document quality before processing
            """.strip())
    
    # Sample JSON document
    sample_json = documents_dir / "sample.json"
    if not sample_json.exists():
        with open(sample_json, "w") as f:
            json.dump({
                "title": "LangChain Document Processing",
                "content": "This JSON document demonstrates how to process structured data.",
                "metadata": {
                    "author": "LangChain Course",
                    "category": "Education",
                    "tags": ["nlp", "document-processing", "langchain"]
                },
                "sections": [
                    {
                        "title": "Getting Started",
                        "content": "Learn the basics of document processing with practical examples."
                    },
                    {
                        "title": "Advanced Techniques",
                        "content": "Explore advanced document processing patterns and optimizations."
                    }
                ]
            }, f, indent=2)
    
    # Sample CSV document
    sample_csv = documents_dir / "sample.csv"
    if not sample_csv.exists():
        with open(sample_csv, "w") as f:
            f.write("""name,description,category
PDF Loader,Extracts text from PDF files,Document Loader
Text Splitter,Splits text into manageable chunks,Text Processing
Vector Store,Stores document embeddings for retrieval,Storage
RAG System,Retrieval Augmented Generation system,Application
Embeddings,Vector representations of text,ML Model""")
    
    print(f"✅ Sample documents created in: {documents_dir}")
    return documents_dir


def document_loader_examples():
    """Demonstrate various document loaders"""
    print("\n" + "="*60)
    print("📁 DOCUMENT LOADER EXAMPLES")
    print("="*60)
    
    if not DOCUMENT_LOADERS_AVAILABLE:
        print("❌ Document loaders not available")
        return
    
    # Create sample documents
    docs_dir = create_sample_documents()
    
    # Text file loading
    print("\n📄 Loading Text File:")
    try:
        txt_file = docs_dir / "sample.txt"
        loader = TextLoader(str(txt_file))
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} document(s)")
        print(f"Content preview: {documents[0].page_content[:200]}...")
        print(f"Metadata: {documents[0].metadata}")
    except Exception as e:
        print(f"❌ Error loading text file: {e}")
    
    # JSON file loading
    print("\n📋 Loading JSON File:")
    try:
        json_file = docs_dir / "sample.json"
        loader = JSONLoader(
            file_path=str(json_file),
            jq_schema=".",
            text_content=False
        )
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} document(s)")
        for i, doc in enumerate(documents[:2]):  # Show first 2
            print(f"Document {i+1}: {doc.page_content[:100]}...")
    except Exception as e:
        print(f"❌ Error loading JSON file: {e}")
    
    # CSV file loading
    print("\n📊 Loading CSV File:")
    try:
        csv_file = docs_dir / "sample.csv"
        loader = CSVLoader(str(csv_file))
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} document(s)")
        for i, doc in enumerate(documents[:3]):  # Show first 3
            print(f"Row {i+1}: {doc.page_content}")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
    
    # Directory loading
    print("\n📂 Loading Directory:")
    try:
        loader = DirectoryLoader(
            str(docs_dir),
            glob="*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} document(s) from directory")
        print(f"Files processed: {[doc.metadata.get('source', 'Unknown') for doc in documents]}")
    except Exception as e:
        print(f"❌ Error loading directory: {e}")


def web_document_examples():
    """Demonstrate web document loading"""
    print("\n" + "="*60)
    print("🌐 WEB DOCUMENT EXAMPLES")
    print("="*60)
    
    if not WEB_SCRAPING_AVAILABLE:
        print("❌ Web scraping tools not available")
        return
    
    # Web page loading
    print("\n🔗 Loading Web Page:")
    try:
        # Using a reliable example URL
        url = "https://python.langchain.com/docs/introduction/"
        loader = WebBaseLoader(url)
        loader.requests_kwargs = {
            'timeout': int(os.getenv("WEB_REQUEST_TIMEOUT", 30))
        }
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} document(s) from web")
        print(f"Content preview: {documents[0].page_content[:300]}...")
        print(f"Source: {documents[0].metadata.get('source', 'Unknown')}")
    except Exception as e:
        print(f"❌ Error loading web page: {e}")
    
    # Multiple URLs
    print("\n🔗 Loading Multiple URLs:")
    try:
        urls = [
            "https://python.langchain.com/docs/introduction/",
            "https://python.langchain.com/docs/concepts/"
        ]
        loader = WebBaseLoader(urls)
        loader.requests_kwargs = {'timeout': 30}
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} document(s) from {len(urls)} URLs")
        for doc in documents:
            print(f"  - {doc.metadata.get('source', 'Unknown')}: {len(doc.page_content)} chars")
    except Exception as e:
        print(f"❌ Error loading multiple URLs: {e}")


def text_splitter_examples():
    """Demonstrate various text splitting strategies"""
    print("\n" + "="*60)
    print("✂️  TEXT SPLITTER EXAMPLES")
    print("="*60)
    
    if not TEXT_SPLITTERS_AVAILABLE:
        print("❌ Text splitters not available")
        return
    
    # Sample text for splitting
    sample_text = """
Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. These machines can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.

Machine Learning is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.

Deep Learning is a subset of machine learning that structures algorithms in layers to create an "artificial neural network" that can learn and make intelligent decisions on its own. Deep learning is a key technology behind driverless cars, enabling them to recognize a stop sign or distinguish a pedestrian from a lamppost.

Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding.
    """.strip()
    
    # Create a document
    doc = Document(page_content=sample_text, metadata={"source": "AI_Overview.txt"})
    
    # Recursive Character Text Splitter
    print("\n🔄 Recursive Character Text Splitter:")
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("RECURSIVE_CHUNK_SIZE", 500)),
            chunk_overlap=int(os.getenv("RECURSIVE_CHUNK_OVERLAP", 100)),
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents([doc])
        print(f"✅ Split into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: {len(chunk.page_content)} chars - {chunk.page_content[:100]}...")
    except Exception as e:
        print(f"❌ Error with recursive splitter: {e}")
    
    # Character Text Splitter
    print("\n📏 Character Text Splitter:")
    try:
        splitter = CharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separator="\n\n"
        )
        chunks = splitter.split_documents([doc])
        print(f"✅ Split into {len(chunks)} chunks using paragraph separator")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: {len(chunk.page_content)} chars")
    except Exception as e:
        print(f"❌ Error with character splitter: {e}")
    
    # Token Text Splitter
    print("\n🎯 Token Text Splitter:")
    try:
        splitter = TokenTextSplitter(
            chunk_size=int(os.getenv("TOKEN_CHUNK_SIZE", 100)),
            chunk_overlap=20
        )
        chunks = splitter.split_documents([doc])
        print(f"✅ Split into {len(chunks)} chunks by tokens")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: ~{len(chunk.page_content)//4} tokens - {chunk.page_content[:80]}...")
    except Exception as e:
        print(f"❌ Error with token splitter: {e}")


def document_transformer_examples():
    """Demonstrate document transformation"""
    print("\n" + "="*60)
    print("🔄 DOCUMENT TRANSFORMER EXAMPLES")
    print("="*60)
    
    # HTML to text transformation
    print("\n🌐 HTML to Text Transformation:")
    try:
        html_content = """
        <html>
        <head><title>LangChain Document Processing</title></head>
        <body>
            <h1>Welcome to Document Processing</h1>
            <p>This is a <strong>sample HTML document</strong> that demonstrates 
            how to extract clean text from HTML content.</p>
            <ul>
                <li>Remove HTML tags</li>
                <li>Preserve text structure</li>
                <li>Extract meaningful content</li>
            </ul>
            <div class="footer">Footer content that might be noise</div>
        </body>
        </html>
        """
        
        html_doc = Document(page_content=html_content, metadata={"source": "sample.html"})
        
        if TRANSFORMERS_AVAILABLE:
            transformer = Html2TextTransformer()
            transformed_docs = transformer.transform_documents([html_doc])
            print(f"✅ Transformed HTML document")
            print(f"Original length: {len(html_content)} chars")
            print(f"Cleaned length: {len(transformed_docs[0].page_content)} chars")
            print(f"Cleaned content: {transformed_docs[0].page_content[:300]}...")
        else:
            # Fallback: simple HTML tag removal
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            clean_text = soup.get_text(separator=' ', strip=True)
            print(f"✅ Transformed HTML using BeautifulSoup fallback")
            print(f"Cleaned content: {clean_text[:300]}...")
            
    except Exception as e:
        print(f"❌ Error with HTML transformation: {e}")
    
    # Custom metadata enrichment
    print("\n📋 Metadata Enrichment:")
    try:
        def enrich_metadata(docs: List[Document]) -> List[Document]:
            """Add custom metadata to documents"""
            enriched_docs = []
            for doc in docs:
                # Add word count
                word_count = len(doc.page_content.split())
                char_count = len(doc.page_content)
                
                # Add content analysis
                has_code = "```" in doc.page_content or "def " in doc.page_content
                has_urls = "http" in doc.page_content
                
                # Create enriched document
                enriched_metadata = {
                    **doc.metadata,
                    "word_count": word_count,
                    "char_count": char_count,
                    "has_code": has_code,
                    "has_urls": has_urls,
                    "processed_at": "2024-01-01T12:00:00"
                }
                
                enriched_docs.append(Document(
                    page_content=doc.page_content,
                    metadata=enriched_metadata
                ))
            
            return enriched_docs
        
        # Test with sample document
        sample_doc = Document(
            page_content="This is a sample document with some code: ```python\nprint('hello')\n```",
            metadata={"source": "test.md"}
        )
        
        enriched = enrich_metadata([sample_doc])
        print(f"✅ Enriched document metadata")
        print(f"Metadata: {enriched[0].metadata}")
        
    except Exception as e:
        print(f"❌ Error with metadata enrichment: {e}")


def document_processing_pipeline():
    """Demonstrate end-to-end document processing pipeline"""
    print("\n" + "="*60)
    print("⚙️ DOCUMENT PROCESSING PIPELINE")
    print("="*60)
    
    class DocumentProcessor:
        def __init__(self):
            self.chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
            self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
            self.max_doc_size = int(os.getenv("MAX_DOCUMENT_SIZE_MB", 50)) * 1024 * 1024
            self.min_content_length = int(os.getenv("MIN_CONTENT_LENGTH", 50))
            
        def validate_document(self, file_path: Path) -> bool:
            """Validate document before processing"""
            try:
                # Check file size
                if file_path.stat().st_size > self.max_doc_size:
                    print(f"⚠️ File too large: {file_path.name}")
                    return False
                
                # Check file type
                supported_types = os.getenv("SUPPORTED_FILE_TYPES", "pdf,txt,docx,csv,json").split(",")
                if file_path.suffix.lower().replace('.', '') not in supported_types:
                    print(f"⚠️ Unsupported file type: {file_path.name}")
                    return False
                
                return True
            except Exception as e:
                print(f"❌ Error validating {file_path}: {e}")
                return False
        
        def load_document(self, file_path: Path) -> List[Document]:
            """Load document using appropriate loader"""
            try:
                suffix = file_path.suffix.lower()
                
                if suffix == '.txt':
                    loader = TextLoader(str(file_path))
                elif suffix == '.csv':
                    loader = CSVLoader(str(file_path))
                elif suffix == '.json':
                    loader = JSONLoader(str(file_path), jq_schema=".", text_content=False)
                elif suffix == '.pdf' and DOCUMENT_LOADERS_AVAILABLE:
                    loader = PyPDFLoader(str(file_path))
                else:
                    # Fallback to text loader
                    loader = TextLoader(str(file_path))
                
                documents = loader.load()
                print(f"✅ Loaded {len(documents)} document(s) from {file_path.name}")
                return documents
                
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                return []
        
        def split_documents(self, documents: List[Document]) -> List[Document]:
            """Split documents into chunks"""
            try:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    length_function=len
                )
                
                chunks = splitter.split_documents(documents)
                
                # Filter out very short chunks
                filtered_chunks = [
                    chunk for chunk in chunks 
                    if len(chunk.page_content.strip()) >= self.min_content_length
                ]
                
                print(f"✅ Split into {len(filtered_chunks)} valid chunks (filtered from {len(chunks)})")
                return filtered_chunks
                
            except Exception as e:
                print(f"❌ Error splitting documents: {e}")
                return documents
        
        def enrich_documents(self, documents: List[Document]) -> List[Document]:
            """Add metadata and clean content"""
            enriched = []
            
            for i, doc in enumerate(documents):
                # Clean content
                cleaned_content = doc.page_content.strip()
                cleaned_content = ' '.join(cleaned_content.split())  # Normalize whitespace
                
                # Skip if too short after cleaning
                if len(cleaned_content) < self.min_content_length:
                    continue
                
                # Enrich metadata
                enriched_metadata = {
                    **doc.metadata,
                    "chunk_id": i,
                    "word_count": len(cleaned_content.split()),
                    "char_count": len(cleaned_content),
                    "language": "en",  # Could use language detection
                    "processed_by": "DocumentProcessor"
                }
                
                enriched.append(Document(
                    page_content=cleaned_content,
                    metadata=enriched_metadata
                ))
            
            print(f"✅ Enriched {len(enriched)} documents")
            return enriched
        
        def process_file(self, file_path: Path) -> List[Document]:
            """Process a single file through the complete pipeline"""
            print(f"\n📄 Processing: {file_path.name}")
            
            # Validate
            if not self.validate_document(file_path):
                return []
            
            # Load
            documents = self.load_document(file_path)
            if not documents:
                return []
            
            # Split
            chunks = self.split_documents(documents)
            
            # Enrich
            enriched = self.enrich_documents(chunks)
            
            return enriched
        
        def process_directory(self, directory: Path) -> List[Document]:
            """Process all files in a directory"""
            all_documents = []
            
            for file_path in directory.iterdir():
                if file_path.is_file():
                    documents = self.process_file(file_path)
                    all_documents.extend(documents)
            
            return all_documents
    
    # Demonstrate the pipeline
    try:
        processor = DocumentProcessor()
        docs_dir = create_sample_documents()
        
        print(f"\n🔄 Processing directory: {docs_dir}")
        all_docs = processor.process_directory(docs_dir)
        
        print(f"\n📊 Pipeline Results:")
        print(f"Total processed documents: {len(all_docs)}")
        
        # Group by source
        by_source = {}
        for doc in all_docs:
            source = doc.metadata.get("source", "Unknown")
            by_source.setdefault(source, []).append(doc)
        
        for source, docs in by_source.items():
            word_count = sum(doc.metadata.get("word_count", 0) for doc in docs)
            print(f"  - {Path(source).name}: {len(docs)} chunks, {word_count} words")
        
    except Exception as e:
        print(f"❌ Error in processing pipeline: {e}")


def batch_processing_example():
    """Demonstrate batch processing with parallel execution"""
    print("\n" + "="*60)
    print("⚡ BATCH PROCESSING EXAMPLE")
    print("="*60)
    
    def process_single_document(file_info: Tuple[Path, int]) -> Dict[str, Any]:
        """Process a single document (for parallel execution)"""
        file_path, doc_id = file_info
        
        try:
            # Simulate processing
            content_length = file_path.stat().st_size if file_path.exists() else 0
            
            # Mock processing time based on file size
            import time
            processing_time = min(content_length / 10000, 0.1)  # Max 0.1 seconds
            time.sleep(processing_time)
            
            return {
                "doc_id": doc_id,
                "file_path": str(file_path),
                "status": "success",
                "content_length": content_length,
                "processing_time": processing_time
            }
        except Exception as e:
            return {
                "doc_id": doc_id,
                "file_path": str(file_path),
                "status": "error",
                "error": str(e)
            }
    
    # Create more sample files for batch processing
    docs_dir = create_sample_documents()
    
    # Create additional sample files
    for i in range(5):
        sample_file = docs_dir / f"batch_sample_{i}.txt"
        with open(sample_file, "w") as f:
            f.write(f"This is batch sample document #{i}. " * 50)
    
    # Get all files for processing
    files_to_process = [
        (file_path, i) for i, file_path in enumerate(docs_dir.iterdir()) 
        if file_path.is_file()
    ]
    
    print(f"📁 Processing {len(files_to_process)} files...")
    
    # Sequential processing
    print(f"\n⏳ Sequential Processing:")
    import time
    start_time = time.time()
    
    sequential_results = []
    for file_info in files_to_process:
        result = process_single_document(file_info)
        sequential_results.append(result)
    
    sequential_time = time.time() - start_time
    print(f"✅ Sequential processing completed in {sequential_time:.2f} seconds")
    
    # Parallel processing
    print(f"\n⚡ Parallel Processing:")
    start_time = time.time()
    
    max_workers = int(os.getenv("MAX_WORKERS", 4))
    parallel_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_single_document, file_info): file_info 
            for file_info in files_to_process
        }
        
        for future in as_completed(future_to_file):
            result = future.result()
            parallel_results.append(result)
    
    parallel_time = time.time() - start_time
    speedup = sequential_time / parallel_time if parallel_time > 0 else 1
    
    print(f"✅ Parallel processing completed in {parallel_time:.2f} seconds")
    print(f"🚀 Speedup: {speedup:.2f}x")
    
    # Show results summary
    print(f"\n📊 Processing Results:")
    successful = len([r for r in parallel_results if r["status"] == "success"])
    failed = len([r for r in parallel_results if r["status"] == "error"])
    
    print(f"  - Successful: {successful}")
    print(f"  - Failed: {failed}")
    print(f"  - Total size: {sum(r.get('content_length', 0) for r in parallel_results)} bytes")


def interactive_document_demo():
    """Interactive demonstration of document processing"""
    print("\n" + "="*60)
    print("🎮 INTERACTIVE DOCUMENT PROCESSING DEMO")
    print("="*60)
    
    demos = [
        ("Document Loaders", document_loader_examples),
        ("Web Document Loading", web_document_examples),
        ("Text Splitters", text_splitter_examples),
        ("Document Transformers", document_transformer_examples),
        ("Processing Pipeline", document_processing_pipeline),
        ("Batch Processing", batch_processing_example)
    ]
    
    print("Available demonstrations:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print("  q. Quit")
    
    choice = input("\nSelect demonstration (1-6): ").strip()
    
    if choice.lower() == 'q':
        return
    
    try:
        choice_num = int(choice)
        if 1 <= choice_num <= len(demos):
            name, demo_func = demos[choice_num - 1]
            print(f"\n🔍 Running: {name}")
            demo_func()
        else:
            print("❌ Invalid selection")
    except ValueError:
        print("❌ Please enter a number")
    except Exception as e:
        print(f"❌ Error running demo: {e}")


def main():
    """
    Main function to run all document processing examples
    """
    print("📄 LangChain Course - Lesson 5: Document Processing & Text Splitters")
    print("=" * 80)
    
    # Check available features
    print("🔍 Feature Availability:")
    print(f"  - Document Loaders: {'✅' if DOCUMENT_LOADERS_AVAILABLE else '❌'}")
    print(f"  - Text Splitters: {'✅' if TEXT_SPLITTERS_AVAILABLE else '❌'}")
    print(f"  - Document Transformers: {'✅' if TRANSFORMERS_AVAILABLE else '❌'}")
    print(f"  - Web Scraping: {'✅' if WEB_SCRAPING_AVAILABLE else '❌'}")
    print(f"  - NLP Tools: {'✅' if NLP_TOOLS_AVAILABLE else '❌'}")
    print(f"  - Shared Utilities: {'✅' if SHARED_UTILS_AVAILABLE else '❌'}")
    
    # Run examples
    document_loader_examples()
    web_document_examples()
    text_splitter_examples()
    document_transformer_examples()
    document_processing_pipeline()
    batch_processing_example()
    
    # Interactive demo
    interactive_document_demo()
    
    print("\n🎉 Lesson 5 completed! Check out the exercises to practice more.")


if __name__ == "__main__":
    main() 