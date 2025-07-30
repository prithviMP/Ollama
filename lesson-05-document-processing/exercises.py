#!/usr/bin/env python3
"""
Lesson 5: Document Processing & Text Splitters - Exercises
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain imports
from langchain.schema import Document


def exercise_1_multi_format_processor():
    """
    Exercise 1: Multi-Format Document Processor
    Build a system that handles PDF, TXT, CSV, and web documents uniformly.
    
    Requirements:
    - Support at least 4 different file formats
    - Uniform output format regardless of input type
    - Error handling for unsupported formats
    - Metadata preservation and enrichment
    """
    print("\n" + "="*60)
    print("🏋️ EXERCISE 1: Multi-Format Document Processor")
    print("="*60)
    
    print("""
📝 YOUR TASK:
Create a UniversalDocumentProcessor that:
1. Detects file formats automatically
2. Uses appropriate loaders for each format
3. Produces consistent output structure
4. Handles errors gracefully
5. Enriches metadata with format-specific information

💡 HINTS:
- Use file extensions or magic numbers for format detection
- Create a registry of format handlers
- Implement a common Document interface
- Add format-specific metadata (page numbers for PDFs, row counts for CSV)
- Consider using factory pattern for loader selection
    """)
    
    # TODO: Implement your solution here
    class UniversalDocumentProcessor:
        def __init__(self):
            # TODO: Initialize processor with format handlers
            pass
        
        def detect_format(self, file_path: Path) -> str:
            # TODO: Detect document format
            pass
        
        def get_loader(self, format_type: str, file_path: Path):
            # TODO: Return appropriate loader for format
            pass
        
        def process_document(self, file_path: Path) -> List[Document]:
            # TODO: Process document with format-specific handling
            pass
        
        def enrich_metadata(self, documents: List[Document], format_type: str) -> List[Document]:
            # TODO: Add format-specific metadata
            pass
        
        def get_supported_formats(self) -> List[str]:
            # TODO: Return list of supported formats
            pass
    
    print("\n🧪 Test your implementation:")
    print("✅ Implement the UniversalDocumentProcessor class above!")


def exercise_2_intelligent_chunking():
    """
    Exercise 2: Intelligent Chunking System
    Create adaptive text splitting based on document type and content structure.
    
    Requirements:
    - Different strategies for different document types
    - Preserve semantic coherence
    - Adaptive chunk sizes based on content complexity
    - Maintain document structure (headers, sections)
    """
    print("\n" + "="*60)
    print("🏋️ EXERCISE 2: Intelligent Chunking System")
    print("="*60)
    
    print("""
📝 YOUR TASK:
Build an IntelligentChunker that:
1. Analyzes document structure and content type
2. Chooses optimal splitting strategy
3. Preserves semantic boundaries
4. Adapts chunk sizes based on content complexity
5. Maintains hierarchical structure information

💡 HINTS:
- Use different splitters for code vs prose vs structured data
- Detect document structure (headers, lists, code blocks)
- Consider content density and complexity
- Preserve relationships between chunks
- Add chunk sequence and hierarchy metadata
    """)
    
    # TODO: Implement your solution here
    class IntelligentChunker:
        def __init__(self):
            # TODO: Initialize with multiple splitting strategies
            pass
        
        def analyze_content(self, document: Document) -> Dict[str, Any]:
            # TODO: Analyze document content and structure
            pass
        
        def select_strategy(self, content_analysis: Dict[str, Any]) -> str:
            # TODO: Choose optimal chunking strategy
            pass
        
        def chunk_document(self, document: Document) -> List[Document]:
            # TODO: Apply intelligent chunking
            pass
        
        def preserve_structure(self, chunks: List[Document]) -> List[Document]:
            # TODO: Add structural metadata to chunks
            pass
        
        def optimize_chunk_sizes(self, chunks: List[Document]) -> List[Document]:
            # TODO: Adjust chunk sizes based on content
            pass
    
    print("\n🧪 Test your implementation:")
    print("✅ Implement the IntelligentChunker class above!")


def exercise_3_metadata_enrichment():
    """
    Exercise 3: Document Metadata Enrichment
    Implement a pipeline that extracts and adds rich metadata to document chunks.
    
    Requirements:
    - Extract semantic metadata (topics, entities, sentiment)
    - Add structural metadata (document hierarchy, chunk position)
    - Include quality metrics (readability, completeness)
    - Support custom metadata extractors
    """
    print("\n" + "="*60)
    print("🏋️ EXERCISE 3: Document Metadata Enrichment")
    print("="*60)
    
    print("""
📝 YOUR TASK:
Create a MetadataEnricher that:
1. Extracts multiple types of metadata
2. Analyzes content for semantic information
3. Calculates quality and readability metrics
4. Supports pluggable metadata extractors
5. Provides metadata validation and cleanup

💡 HINTS:
- Use NLP libraries for entity extraction and sentiment analysis
- Calculate readability scores (Flesch-Kincaid, etc.)
- Extract keywords and topics
- Add document structure metadata
- Consider using pipeline pattern for extractors
    """)
    
    # TODO: Implement your solution here
    class MetadataEnricher:
        def __init__(self):
            # TODO: Initialize with metadata extractors
            pass
        
        def register_extractor(self, name: str, extractor_func):
            # TODO: Register custom metadata extractor
            pass
        
        def extract_semantic_metadata(self, document: Document) -> Dict[str, Any]:
            # TODO: Extract semantic information
            pass
        
        def extract_structural_metadata(self, document: Document) -> Dict[str, Any]:
            # TODO: Extract structural information
            pass
        
        def calculate_quality_metrics(self, document: Document) -> Dict[str, Any]:
            # TODO: Calculate content quality metrics
            pass
        
        def enrich_document(self, document: Document) -> Document:
            # TODO: Apply all metadata enrichment
            pass
        
        def validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
            # TODO: Validate and clean metadata
            pass
    
    print("\n🧪 Test your implementation:")
    print("✅ Implement the MetadataEnricher class above!")


def exercise_4_batch_processing():
    """
    Exercise 4: Batch Processing System
    Design a system for processing large document collections efficiently.
    
    Requirements:
    - Parallel processing of multiple documents
    - Progress tracking and error handling
    - Memory-efficient streaming for large files
    - Resumable processing for interrupted jobs
    """
    print("\n" + "="*60)
    print("🏋️ EXERCISE 4: Batch Processing System")
    print("="*60)
    
    print("""
📝 YOUR TASK:
Build a BatchDocumentProcessor that:
1. Processes multiple documents in parallel
2. Handles memory constraints with streaming
3. Provides progress tracking and error recovery
4. Supports job resumption after interruption
5. Implements configurable processing pipelines

💡 HINTS:
- Use ThreadPoolExecutor or ProcessPoolExecutor for parallelization
- Implement streaming for large files to avoid memory issues
- Create job state persistence for resumability
- Add progress callbacks and error aggregation
- Consider using queue-based processing for scalability
    """)
    
    # TODO: Implement your solution here
    class BatchDocumentProcessor:
        def __init__(self, max_workers: int = 4):
            # TODO: Initialize batch processor
            pass
        
        def create_job(self, document_paths: List[Path], job_config: Dict[str, Any]) -> str:
            # TODO: Create processing job
            pass
        
        def process_job(self, job_id: str, progress_callback=None) -> Dict[str, Any]:
            # TODO: Execute batch processing job
            pass
        
        def resume_job(self, job_id: str) -> Dict[str, Any]:
            # TODO: Resume interrupted job
            pass
        
        def get_job_status(self, job_id: str) -> Dict[str, Any]:
            # TODO: Get current job status
            pass
        
        def stream_process_large_file(self, file_path: Path) -> List[Document]:
            # TODO: Stream process large files
            pass
        
        def aggregate_results(self, job_results: List[Dict[str, Any]]) -> Dict[str, Any]:
            # TODO: Aggregate processing results
            pass
    
    print("\n🧪 Test your implementation:")
    print("✅ Implement the BatchDocumentProcessor class above!")


def exercise_5_quality_control():
    """
    Exercise 5: Quality Control Pipeline
    Build validation and filtering for document processing quality assurance.
    
    Requirements:
    - Content quality validation
    - Automatic error detection and correction
    - Filtering of low-quality or irrelevant content
    - Quality scoring and reporting
    """
    print("\n" + "="*60)
    print("🏋️ EXERCISE 5: Quality Control Pipeline")
    print("="*60)
    
    print("""
📝 YOUR TASK:
Create a DocumentQualityController that:
1. Validates document content quality
2. Detects and corrects common issues
3. Filters out low-quality content
4. Provides quality scoring and detailed reports
5. Supports configurable quality thresholds

💡 HINTS:
- Check for minimum content length and readability
- Detect encoding issues and corrupted text
- Validate document structure and completeness
- Implement content deduplication
- Add quality scoring based on multiple criteria
    """)
    
    # TODO: Implement your solution here
    class DocumentQualityController:
        def __init__(self, quality_thresholds: Dict[str, float] = None):
            # TODO: Initialize quality controller
            pass
        
        def validate_content(self, document: Document) -> Dict[str, Any]:
            # TODO: Validate document content
            pass
        
        def detect_issues(self, document: Document) -> List[Dict[str, Any]]:
            # TODO: Detect quality issues
            pass
        
        def correct_issues(self, document: Document, issues: List[Dict[str, Any]]) -> Document:
            # TODO: Automatically correct detected issues
            pass
        
        def calculate_quality_score(self, document: Document) -> float:
            # TODO: Calculate overall quality score
            pass
        
        def filter_documents(self, documents: List[Document]) -> List[Document]:
            # TODO: Filter documents based on quality
            pass
        
        def generate_quality_report(self, documents: List[Document]) -> Dict[str, Any]:
            # TODO: Generate detailed quality report
            pass
    
    print("\n🧪 Test your implementation:")
    print("✅ Implement the DocumentQualityController class above!")


def run_all_exercises():
    """
    Run all exercises
    """
    print("🏋️ LangChain Lesson 5: Document Processing - Exercises")
    print("=" * 70)
    
    exercises = [
        exercise_1_multi_format_processor,
        exercise_2_intelligent_chunking,
        exercise_3_metadata_enrichment,
        exercise_4_batch_processing,
        exercise_5_quality_control
    ]
    
    for exercise in exercises:
        exercise()
        input("\nPress Enter to continue to next exercise...")


if __name__ == "__main__":
    run_all_exercises() 