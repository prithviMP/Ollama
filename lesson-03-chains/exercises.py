#!/usr/bin/env python3
"""
Lesson 3: Chains & Sequential Processing - Exercises

This file contains practical exercises to reinforce the concepts learned in Lesson 3.
Each exercise builds upon the previous ones and covers different aspects of LangChain chains.

Exercises:
1. Basic LLMChain - Content Generation
2. Sequential Chain - Multi-step Processing
3. Router Chain - Conditional Logic
4. Transform Chain - Data Processing
5. Custom Chain - Business Logic
6. Advanced Chain - Error Handling & Optimization

Author: LangChain Course
"""

import os
import sys
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Add shared resources to path
sys.path.append('../shared-resources')

# Load environment variables
load_dotenv()

# LangChain imports
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain.chains import TransformChain, LLMMathChain
from langchain.chains.router import MultiPromptChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import BaseOutputParser
import json
import time

# Shared utilities
from utils.llm_setup import setup_llm_providers, get_preferred_llm
from utils.error_handling import safe_llm_call


def setup_exercises():
    """Set up the exercise environment."""
    print("���� LangChain Course - Lesson 3: Chain Exercises")
    print("=" * 60)
    
    providers = setup_llm_providers()
    if not providers:
        print("❌ No LLM providers available. Please check your setup.")
        return None, None
    
    llm = get_preferred_llm(providers, prefer_chat=False)
    chat_llm = get_preferred_llm(providers, prefer_chat=True)
    
    return llm, chat_llm


def exercise_1_basic_llm_chain(llm):
    """
    Exercise 1: Basic LLMChain - Content Generation
    
    Task: Create a content generation chain that can:
    1. Generate blog post titles
    2. Create content outlines
    3. Write introductions
    
    Requirements:
    - Use LLMChain with appropriate prompts
    - Handle different content types
    - Implement error handling
    """
    print("\n" + "="*60)
    print("🏋️  EXERCISE 1: Basic LLMChain - Content Generation")
    print("="*60)
    
    # TODO: Implement the following chain components:
    
    # 1. Create a title generation chain
    # - Prompt should take topic and tone as input
    # - Generate engaging, SEO-friendly titles
    
    # 2. Create an outline generation chain  
    # - Prompt should take topic and target audience
    # - Generate structured content outlines
    
    # 3. Create an introduction writing chain
    # - Prompt should take topic and outline
    # - Write compelling introductions
    
    # 4. Test all chains with sample data
    # - Use different topics and tones
    # - Handle errors gracefully
    
    print("📝 TODO: Implement content generation chains")
    print("   - Title generation chain")
    print("   - Outline generation chain") 
    print("   - Introduction writing chain")
    print("   - Error handling and testing")
    
    return {
        "title_chain": None,  # TODO: Implement
        "outline_chain": None,  # TODO: Implement
        "intro_chain": None,  # TODO: Implement
        "test_results": []  # TODO: Add test results
    }


def exercise_2_sequential_chain(llm):
    """
    Exercise 2: Sequential Chain - Multi-step Processing
    
    Task: Create a content marketing pipeline that:
    1. Researches a topic
    2. Creates an outline
    3. Writes the content
    4. Optimizes for SEO
    
    Requirements:
    - Use SequentialChain for multi-step processing
    - Pass data between chains correctly
    - Handle multiple inputs/outputs
    """
    print("\n" + "="*60)
    print("⛓️  EXERCISE 2: Sequential Chain - Content Marketing Pipeline")
    print("="*60)
    
    # TODO: Implement the following sequential chain:
    
    # 1. Research Chain
    # - Input: topic, target_audience
    # - Output: research_points, key_insights
    
    # 2. Outline Chain
    # - Input: topic, research_points
    # - Output: content_outline, structure
    
    # 3. Content Chain
    # - Input: topic, content_outline, target_audience
    # - Output: draft_content, word_count
    
    # 4. SEO Chain
    # - Input: draft_content, topic
    # - Output: seo_optimized_content, keywords
    
    # 5. Combine into SequentialChain
    # - Handle all input/output variables correctly
    # - Test with sample topics
    
    print("📝 TODO: Implement content marketing pipeline")
    print("   - Research chain")
    print("   - Outline chain")
    print("   - Content writing chain")
    print("   - SEO optimization chain")
    print("   - Sequential chain combination")
    
    return {
        "research_chain": None,  # TODO: Implement
        "outline_chain": None,  # TODO: Implement
        "content_chain": None,  # TODO: Implement
        "seo_chain": None,  # TODO: Implement
        "pipeline": None,  # TODO: Implement
        "test_results": []  # TODO: Add test results
    }


def exercise_3_router_chain(llm):
    """
    Exercise 3: Router Chain - Conditional Logic
    
    Task: Create an intelligent content router that:
    1. Analyzes user queries
    2. Routes to appropriate specialized chains
    3. Handles different content types (technical, creative, business)
    
    Requirements:
    - Use MultiPromptChain for routing
    - Create specialized chains for different content types
    - Implement fallback handling
    """
    print("\n" + "="*60)
    print("�� EXERCISE 3: Router Chain - Intelligent Content Routing")
    print("="*60)
    
    # TODO: Implement the following router system:
    
    # 1. Technical Content Chain
    # - For technical documentation, tutorials, code explanations
    # - Specialized prompts for technical writing
    
    # 2. Creative Content Chain
    # - For storytelling, creative writing, marketing copy
    # - Engaging and persuasive prompts
    
    # 3. Business Content Chain
    # - For reports, analysis, professional communication
    # - Formal and structured prompts
    
    # 4. Router Chain
    # - Analyze input and route to appropriate chain
    # - Handle edge cases and fallbacks
    
    # 5. Test with different query types
    # - Technical queries
    # - Creative requests
    # - Business questions
    
    print("📝 TODO: Implement intelligent content router")
    print("   - Technical content chain")
    print("   - Creative content chain")
    print("   - Business content chain")
    print("   - Router chain with fallback")
    print("   - Query analysis and routing")
    
    return {
        "technical_chain": None,  # TODO: Implement
        "creative_chain": None,  # TODO: Implement
        "business_chain": None,  # TODO: Implement
        "router_chain": None,  # TODO: Implement
        "test_results": []  # TODO: Add test results
    }


def exercise_4_transform_chain(llm):
    """
    Exercise 4: Transform Chain - Data Processing
    
    Task: Create a data processing pipeline that:
    1. Cleans and preprocesses text data
    2. Extracts key information
    3. Formats output for different use cases
    
    Requirements:
    - Use TransformChain for data preprocessing
    - Implement custom transformation functions
    - Handle different data formats
    """
    print("\n" + "="*60)
    print("🔄 EXERCISE 4: Transform Chain - Data Processing Pipeline")
    print("="*60)
    
    # TODO: Implement the following transform chains:
    
    # 1. Text Cleaning Transform
    # - Remove extra whitespace, normalize text
    # - Count words, characters, sentences
    
    # 2. Information Extraction Transform
    # - Extract key phrases, entities
    # - Identify sentiment, tone
    
    # 3. Formatting Transform
    # - Convert to different formats (JSON, CSV, markdown)
    # - Structure data for specific use cases
    
    # 4. Combine transforms into pipeline
    # - Chain multiple transforms together
    # - Handle errors and edge cases
    
    print("📝 TODO: Implement data processing pipeline")
    print("   - Text cleaning transform")
    print("   - Information extraction transform")
    print("   - Formatting transform")
    print("   - Transform pipeline combination")
    
    return {
        "clean_transform": None,  # TODO: Implement
        "extract_transform": None,  # TODO: Implement
        "format_transform": None,  # TODO: Implement
        "pipeline": None,  # TODO: Implement
        "test_results": []  # TODO: Add test results
    }


def exercise_5_custom_chain(llm):
    """
    Exercise 5: Custom Chain - Business Logic
    
    Task: Create a customer service automation chain that:
    1. Classifies customer inquiries
    2. Generates appropriate responses
    3. Escalates complex issues
    4. Tracks interaction history
    
    Requirements:
    - Create custom chain classes
    - Implement business logic
    - Handle state management
    """
    print("\n" + "="*60)
    print("🛠️  EXERCISE 5: Custom Chain - Customer Service Automation")
    print("="*60)
    
    # TODO: Implement the following custom chains:
    
    # 1. Inquiry Classification Chain
    # - Classify inquiries into categories (support, sales, complaint, etc.)
    # - Use LLMChain with classification prompt
    
    # 2. Response Generation Chain
    # - Generate appropriate responses based on classification
    # - Include tone and style variations
    
    # 3. Escalation Chain
    # - Determine if issue needs human intervention
    # - Create escalation summaries
    
    # 4. Custom Service Chain
    # - Combine all chains into custom class
    # - Handle state and history tracking
    
    print("📝 TODO: Implement customer service automation")
    print("   - Inquiry classification chain")
    print("   - Response generation chain")
    print("   - Escalation chain")
    print("   - Custom service chain class")
    
    return {
        "classification_chain": None,  # TODO: Implement
        "response_chain": None,  # TODO: Implement
        "escalation_chain": None,  # TODO: Implement
        "service_chain": None,  # TODO: Implement
        "test_results": []  # TODO: Add test results
    }


def exercise_6_advanced_chain(llm):
    """
    Exercise 6: Advanced Chain - Error Handling & Optimization
    
    Task: Create a robust chain system with:
    1. Comprehensive error handling
    2. Performance monitoring
    3. Fallback mechanisms
    4. Caching and optimization
    
    Requirements:
    - Implement error handling patterns
    - Add performance monitoring
    - Create fallback chains
    - Optimize for production use
    """
    print("\n" + "="*60)
    print("🚀 EXERCISE 6: Advanced Chain - Error Handling & Optimization")
    print("="*60)
    
    # TODO: Implement the following advanced features:
    
    # 1. Error Handling Chain
    # - Catch and handle different types of errors
    # - Implement retry logic with exponential backoff
    
    # 2. Performance Monitoring Chain
    # - Track execution time, token usage
    # - Monitor success/failure rates
    
    # 3. Fallback Chain System
    # - Primary chain with multiple fallbacks
    # - Graceful degradation of functionality
    
    # 4. Caching and Optimization
    # - Implement result caching
    # - Optimize for repeated queries
    
    # 5. Production-Ready Chain
    # - Combine all advanced features
    # - Test with various scenarios
    
    print("📝 TODO: Implement advanced chain features")
    print("   - Error handling with retry logic")
    print("   - Performance monitoring")
    print("   - Fallback chain system")
    print("   - Caching and optimization")
    print("   - Production-ready implementation")
    
    return {
        "error_handler": None,  # TODO: Implement
        "performance_monitor": None,  # TODO: Implement
        "fallback_system": None,  # TODO: Implement
        "optimized_chain": None,  # TODO: Implement
        "test_results": []  # TODO: Add test results
    }


def run_all_exercises():
    """
    Run all exercises in sequence.
    """
    print("🎯 Starting Lesson 3 Chain Exercises")
    print("=" * 60)
    
    # Setup
    llm, chat_llm = setup_exercises()
    if not llm and not chat_llm:
        print("❌ Cannot run exercises without LLM providers.")
        return
    
    demo_llm = llm if llm else chat_llm
    print(f"�� Using LLM: {type(demo_llm).__name__}")
    
    # Run exercises
    results = {}
    
    try:
        print("\n🏃‍♂️ Running Exercise 1: Basic LLMChain")
        results["exercise_1"] = exercise_1_basic_llm_chain(demo_llm)
        
        print("\n🏃‍♂️ Running Exercise 2: Sequential Chain")
        results["exercise_2"] = exercise_2_sequential_chain(demo_llm)
        
        print("\n🏃‍♂️ Running Exercise 3: Router Chain")
        results["exercise_3"] = exercise_3_router_chain(demo_llm)
        
        print("\n🏃‍♂️ Running Exercise 4: Transform Chain")
        results["exercise_4"] = exercise_4_transform_chain(demo_llm)
        
        print("\n🏃‍♂️ Running Exercise 5: Custom Chain")
        results["exercise_5"] = exercise_5_custom_chain(demo_llm)
        
        print("\n🏃‍♂️ Running Exercise 6: Advanced Chain")
        results["exercise_6"] = exercise_6_advanced_chain(demo_llm)
        
        print("\n🎉 All exercises completed!")
        print("\n📊 Exercise Summary:")
        for exercise_name, result in results.items():
            completed = any(result.values()) if result else False
            status = "✅ Completed" if completed else "⏳ TODO"
            print(f"   {exercise_name}: {status}")
        
        print("\n💡 Next Steps:")
        print("   1. Implement each exercise step by step")
        print("   2. Test with different inputs and scenarios")
        print("   3. Check solutions.py for reference implementations")
        print("   4. Experiment with your own chain ideas!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Exercises interrupted. You can resume anytime!")
    except Exception as e:
        print(f"\n❌ Error running exercises: {e}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    run_all_exercises() 