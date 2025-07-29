# Lesson 5: Document Processing & Text Splitters with LangChain

## 🎯 Learning Objectives

By the end of this lesson, you will be able to:
- Load and process documents from multiple sources (PDF, TXT, web, APIs)
- Implement intelligent text splitting strategies for optimal chunk sizes
- Transform and enrich documents with metadata
- Build efficient document processing pipelines
- Optimize document preparation for vector storage and retrieval

## 📚 Concepts Covered

### 1. Document Loaders
- File-based loaders (PDF, TXT, CSV, JSON)
- Web scraping and URL content extraction
- API-based content ingestion
- Database document loading
- Custom loader implementation

### 2. Text Splitting Strategies
- Character-based splitting with overlap
- Recursive character text splitter optimization
- Token-aware splitting for LLM limits
- Semantic splitting based on content structure
- Custom splitting logic for domain-specific content

### 3. Document Transformers
- Metadata extraction and enrichment
- Content cleaning and normalization
- Language detection and processing
- Document structure preservation
- Multi-format handling

### 4. Processing Pipelines
- Batch document processing
- Streaming document ingestion
- Error handling and recovery
- Progress monitoring and logging
- Performance optimization

### 5. Production Patterns
- Scalable document processing architectures
- Memory-efficient streaming
- Parallel processing strategies
- Document versioning and updates
- Quality validation and filtering

## 🚀 Getting Started

### Prerequisites
- Completed Lessons 1-4 (Basic through Memory Management)
- Understanding of text processing and NLP basics
- Familiarity with file I/O operations

### Setup
```bash
cd lesson-05-document-processing
poetry install && poetry shell
cp env.example .env
python main.py
```

## 📝 Code Examples

### Basic Document Loading
```python
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load PDF document
loader = PyPDFLoader("example.pdf")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
texts = text_splitter.split_documents(documents)
```

### Web Content Processing
```python
from langchain.document_loaders import WebBaseLoader
from langchain.document_transformers import Html2TextTransformer

# Load web content
loader = WebBaseLoader("https://example.com/article")
docs = loader.load()

# Transform HTML to clean text
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)
```

### Custom Document Pipeline
```python
class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
    
    def process_document(self, file_path):
        # Load -> Clean -> Split -> Enrich
        loader = self._get_loader(file_path)
        docs = loader.load()
        cleaned_docs = self._clean_documents(docs)
        chunks = self.text_splitter.split_documents(cleaned_docs)
        enriched_chunks = self._add_metadata(chunks)
        return enriched_chunks
```

## 🏋️ Exercises

### Exercise 1: Multi-Format Document Processor
Build a system that handles PDF, TXT, CSV, and web documents uniformly.

### Exercise 2: Intelligent Chunking System
Create adaptive text splitting based on document type and content structure.

### Exercise 3: Document Metadata Enrichment
Implement a pipeline that extracts and adds rich metadata to document chunks.

### Exercise 4: Batch Processing System
Design a system for processing large document collections efficiently.

### Exercise 5: Quality Control Pipeline
Build validation and filtering for document processing quality assurance.

## 💡 Key Takeaways

1. **Chunk Size Optimization**: Balance information density with retrieval precision
2. **Metadata Preservation**: Rich metadata improves retrieval and context understanding
3. **Error Resilience**: Robust processing handles various document formats and quality
4. **Performance**: Efficient processing is crucial for production document volumes
5. **Content Quality**: Clean, well-structured chunks improve downstream LLM performance

## 🔗 Next Lesson

[Lesson 6: Vector Stores & Embeddings](../lesson-06-vector-stores/) - Learn to create and manage vector databases for semantic search and retrieval.

---

**Duration:** ~1 hour  
**Difficulty:** Intermediate  
**Prerequisites:** Lessons 1-4 completed 