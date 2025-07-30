#!/usr/bin/env python3
"""
Lesson 3: Chains & Sequential Processing - Solutions

This file contains complete solutions for all exercises in exercises.py.
Each solution demonstrates best practices and production-ready implementations.

Solutions:
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
import time
import json
import re

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
from langchain.chains.base import Chain
from langchain.schema import BaseMemory
import requests

# Shared utilities
from utils.llm_setup import setup_llm_providers, get_preferred_llm
from utils.error_handling import safe_llm_call


def setup_solutions():
    """Set up the solutions environment."""
    print("���� LangChain Course - Lesson 3: Chain Solutions")
    print("=" * 60)
    
    providers = setup_llm_providers()
    if not providers:
        print("❌ No LLM providers available. Please check your setup.")
        return None, None
    
    llm = get_preferred_llm(providers, prefer_chat=False)
    chat_llm = get_preferred_llm(providers, prefer_chat=True)
    
    return llm, chat_llm


def solution_1_basic_llm_chain(llm):
    """
    Solution 1: Basic LLMChain - Content Generation
    
    Complete implementation of content generation chains with error handling.
    """
    print("\n" + "="*60)
    print("✅ SOLUTION 1: Basic LLMChain - Content Generation")
    print("="*60)
    
    # 1. Title Generation Chain
    title_prompt = PromptTemplate(
        input_variables=["topic", "tone"],
        template="""Create an engaging, SEO-friendly blog post title for the topic: {topic}
Tone: {tone}

Requirements:
- Make it compelling and click-worthy
- Include relevant keywords
- Keep it under 60 characters
- Match the specified tone

Title:"""
    )
    
    title_chain = LLMChain(
        llm=llm,
        prompt=title_prompt,
        output_key="title"
    )
    
    # 2. Outline Generation Chain
    outline_prompt = PromptTemplate(
        input_variables=["topic", "target_audience"],
        template="""Create a comprehensive content outline for the topic: {topic}
Target Audience: {target_audience}

Structure:
1. Introduction
2. Main Points (3-5 key sections)
3. Conclusion
4. Call-to-Action

Include:
- Key talking points for each section
- Subheadings and bullet points
- Estimated word count per section

Outline:"""
    )
    
    outline_chain = LLMChain(
        llm=llm,
        prompt=outline_prompt,
        output_key="outline"
    )
    
    # 3. Introduction Writing Chain
    intro_prompt = PromptTemplate(
        input_variables=["topic", "outline"],
        template="""Write a compelling introduction for this content:

Topic: {topic}
Outline: {outline}

Requirements:
- Hook the reader in the first sentence
- Establish credibility and relevance
- Preview what the reader will learn
- Keep it engaging and professional
- 2-3 paragraphs maximum

Introduction:"""
    )
    
    intro_chain = LLMChain(
        llm=llm,
        prompt=intro_prompt,
        output_key="introduction"
    )
    
    # Test the chains
    test_cases = [
        {"topic": "machine learning basics", "tone": "educational", "audience": "beginners"},
        {"topic": "sustainable energy solutions", "tone": "professional", "audience": "business owners"},
        {"topic": "creative writing techniques", "tone": "inspirational", "audience": "writers"}
    ]
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['topic']}")
        
        try:
            # Generate title
            title_result = title_chain.invoke({
                "topic": test_case["topic"],
                "tone": test_case["tone"]
            })
            
            # Generate outline
            outline_result = outline_chain.invoke({
                "topic": test_case["topic"],
                "target_audience": test_case["audience"]
            })
            
            # Generate introduction
            intro_result = intro_chain.invoke({
                "topic": test_case["topic"],
                "outline": outline_result["outline"]
            })
            
            result = {
                "topic": test_case["topic"],
                "title": title_result["title"],
                "outline": outline_result["outline"],
                "introduction": intro_result["introduction"]
            }
            results.append(result)
            
            print(f"   ✅ Title: {title_result['title']}")
            print(f"   ✅ Outline: {len(outline_result['outline'])} characters")
            print(f"   ✅ Introduction: {len(intro_result['introduction'])} characters")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return {
        "title_chain": title_chain,
        "outline_chain": outline_chain,
        "intro_chain": intro_chain,
        "test_results": results
    }


def solution_2_sequential_chain(llm):
    """
    Solution 2: Sequential Chain - Multi-step Processing
    
    Complete implementation of content marketing pipeline.
    """
    print("\n" + "="*60)
    print("✅ SOLUTION 2: Sequential Chain - Content Marketing Pipeline")
    print("="*60)
    
    # 1. Research Chain
    research_prompt = PromptTemplate(
        input_variables=["topic", "target_audience"],
        template="""Research the topic: {topic}
Target Audience: {target_audience}

Provide:
1. Key insights and trends
2. Pain points and challenges
3. Relevant statistics and data
4. Expert opinions and quotes
5. Current market landscape

Format as structured research points:"""
    )
    
    research_chain = LLMChain(
        llm=llm,
        prompt=research_prompt,
        output_key="research_points"
    )
    
    # 2. Outline Chain
    outline_prompt = PromptTemplate(
        input_variables=["topic", "research_points"],
        template="""Create a detailed content outline based on this research:

Topic: {topic}
Research: {research_points}

Structure:
1. Introduction (hook, problem statement)
2. Main sections (3-5 key points)
3. Supporting evidence and examples
4. Conclusion with actionable takeaways
5. Call-to-action

Include word count estimates for each section:"""
    )
    
    outline_chain = LLMChain(
        llm=llm,
        prompt=outline_prompt,
        output_key="content_outline"
    )
    
    # 3. Content Writing Chain
    content_prompt = PromptTemplate(
        input_variables=["topic", "content_outline", "target_audience"],
        template="""Write comprehensive content following this outline:

Topic: {topic}
Target Audience: {target_audience}
Outline: {content_outline}

Requirements:
- Engaging and informative writing
- Include relevant examples and data
- Optimize for readability
- Include internal linking opportunities
- 1500-2000 words total

Content:"""
    )
    
    content_chain = LLMChain(
        llm=llm,
        prompt=content_prompt,
        output_key="draft_content"
    )
    
    # 4. SEO Optimization Chain
    seo_prompt = PromptTemplate(
        input_variables=["draft_content", "topic"],
        template="""Optimize this content for SEO:

Topic: {topic}
Content: {draft_content}

Provide:
1. SEO-optimized title (60 characters max)
2. Meta description (160 characters max)
3. Primary and secondary keywords
4. Internal linking suggestions
5. Schema markup recommendations

SEO Optimization:"""
    )
    
    seo_chain = LLMChain(
        llm=llm,
        prompt=seo_prompt,
        output_key="seo_optimization"
    )
    
    # Combine into SequentialChain
    marketing_pipeline = SequentialChain(
        chains=[research_chain, outline_chain, content_chain, seo_chain],
        input_variables=["topic", "target_audience"],
        output_variables=["research_points", "content_outline", "draft_content", "seo_optimization"],
        verbose=True
    )
    
    # Test the pipeline
    test_topics = [
        {"topic": "artificial intelligence in healthcare", "target_audience": "healthcare professionals"},
        {"topic": "sustainable business practices", "target_audience": "small business owners"}
    ]
    
    results = []
    for i, test_case in enumerate(test_topics, 1):
        print(f"\n📊 Test Case {i}: {test_case['topic']}")
        
        try:
            result = marketing_pipeline.invoke(test_case)
            results.append(result)
            
            print(f"   ✅ Research: {len(result['research_points'])} characters")
            print(f"   ✅ Outline: {len(result['content_outline'])} characters")
            print(f"   ✅ Content: {len(result['draft_content'])} characters")
            print(f"   ✅ SEO: {len(result['seo_optimization'])} characters")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return {
        "research_chain": research_chain,
        "outline_chain": outline_chain,
        "content_chain": content_chain,
        "seo_chain": seo_chain,
        "pipeline": marketing_pipeline,
        "test_results": results
    }


def solution_3_router_chain(llm):
    """
    Solution 3: Router Chain - Conditional Logic
    
    Complete implementation of intelligent content routing system.
    """
    print("\n" + "="*60)
    print("✅ SOLUTION 3: Router Chain - Intelligent Content Routing")
    print("="*60)
    
    # Define specialized prompt templates
    technical_template = """You are a technical content expert specializing in software development, 
engineering, and technical documentation.

Question: {input}

Provide a detailed, technical response that includes:
- Clear explanations of technical concepts
- Code examples where relevant
- Best practices and industry standards
- References to documentation or resources
- Step-by-step instructions when applicable

Technical Response:"""

    creative_template = """You are a creative content specialist with expertise in storytelling, 
marketing, and engaging writing.

Question: {input}

Provide a creative, engaging response that includes:
- Compelling storytelling elements
- Emotional appeal and connection
- Vivid descriptions and imagery
- Inspirational or motivational content
- Creative examples and analogies

Creative Response:"""

    business_template = """You are a business consultant with expertise in strategy, 
analysis, and professional communication.

Question: {input}

Provide a professional business response that includes:
- Strategic insights and analysis
- Data-driven recommendations
- ROI considerations and metrics
- Professional tone and structure
- Actionable business advice

Business Response:"""

    general_template = """You are a helpful assistant with broad knowledge across many topics.

Question: {input}

Provide a clear, informative response that:
- Addresses the question directly
- Provides relevant information
- Uses accessible language
- Includes practical examples
- Offers additional resources if helpful

Response:"""

    # Create prompt infos for router
    prompt_infos = [
        {
            "name": "technical",
            "description": "Good for technical questions, programming, engineering, software development, and technical documentation",
            "prompt_template": technical_template
        },
        {
            "name": "creative",
            "description": "Good for creative writing, storytelling, marketing, inspirational content, and engaging narratives",
            "prompt_template": creative_template
        },
        {
            "name": "business",
            "description": "Good for business strategy, analysis, professional communication, ROI, and corporate topics",
            "prompt_template": business_template
        }
    ]
    
    # Create default chain
    default_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(template=general_template, input_variables=["input"])
    )
    
    # Create router chain
    router_chain = MultiPromptChain.from_prompts(
        llm=llm,
        prompt_infos=prompt_infos,
        default_chain=default_chain,
        verbose=True
    )
    
    # Test the router with different query types
    test_queries = [
        "How do I implement a binary search algorithm in Python?",
        "Write a story about a robot learning to paint",
        "What's the ROI of implementing AI in customer service?",
        "What's the weather like today?"
    ]
    
    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n🧭 Test Query {i}: {query}")
        
        try:
            result = router_chain.invoke({"input": query})
            results.append({
                "query": query,
                "destination": result.get("destination", "default"),
                "response": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]
            })
            
            print(f"   ✅ Routed to: {result.get('destination', 'default')}")
            print(f"   ✅ Response preview: {result['text'][:100]}...")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return {
        "technical_chain": None,  # Handled by router
        "creative_chain": None,   # Handled by router
        "business_chain": None,   # Handled by router
        "router_chain": router_chain,
        "test_results": results
    }


def solution_4_transform_chain(llm):
    """
    Solution 4: Transform Chain - Data Processing
    
    Complete implementation of data processing pipeline.
    """
    print("\n" + "="*60)
    print("✅ SOLUTION 4: Transform Chain - Data Processing Pipeline")
    print("="*60)
    
    # 1. Text Cleaning Transform
    def clean_text_transform(inputs: dict) -> dict:
        """Clean and preprocess text data."""
        text = inputs["raw_text"]
        
        # Basic text cleaning
        cleaned = text.strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Remove extra whitespace
        cleaned = cleaned.replace('\n', ' ')
        
        # Count statistics
        word_count = len(cleaned.split())
        char_count = len(cleaned)
        sentence_count = len(re.split(r'[.!?]+', cleaned))
        
        return {
            "cleaned_text": cleaned,
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": sentence_count
        }
    
    clean_transform = TransformChain(
        input_variables=["raw_text"],
        output_variables=["cleaned_text", "word_count", "char_count", "sentence_count"],
        transform=clean_text_transform
    )
    
    # 2. Information Extraction Transform
    def extract_info_transform(inputs: dict) -> dict:
        """Extract key information from text."""
        text = inputs["cleaned_text"]
        
        # Extract key phrases (simple implementation)
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Simple sentiment analysis (basic implementation)
        positive_words = ["good", "great", "excellent", "amazing", "wonderful"]
        negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"]
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        sentiment = "positive" if positive_count > negative_count else "negative" if negative_count > positive_count else "neutral"
        
        return {
            "top_keywords": top_keywords,
            "sentiment": sentiment,
            "positive_score": positive_count,
            "negative_score": negative_count
        }
    
    extract_transform = TransformChain(
        input_variables=["cleaned_text"],
        output_variables=["top_keywords", "sentiment", "positive_score", "negative_score"],
        transform=extract_info_transform
    )
    
    # 3. Formatting Transform
    def format_output_transform(inputs: dict) -> dict:
        """Format data for different output formats."""
        # Create JSON format
        json_output = {
            "text_stats": {
                "word_count": inputs["word_count"],
                "char_count": inputs["char_count"],
                "sentence_count": inputs["sentence_count"]
            },
            "analysis": {
                "keywords": inputs["top_keywords"],
                "sentiment": inputs["sentiment"],
                "sentiment_scores": {
                    "positive": inputs["positive_score"],
                    "negative": inputs["negative_score"]
                }
            }
        }
        
        # Create markdown format
        markdown_output = f"""# Text Analysis Report

## Statistics
- **Word Count**: {inputs['word_count']}
- **Character Count**: {inputs['char_count']}
- **Sentence Count**: {inputs['sentence_count']}

## Analysis
- **Sentiment**: {inputs['sentiment']}
- **Top Keywords**: {', '.join([kw[0] for kw in inputs['top_keywords']])}
- **Sentiment Scores**: Positive: {inputs['positive_score']}, Negative: {inputs['negative_score']}
"""
        
        return {
            "json_output": json.dumps(json_output, indent=2),
            "markdown_output": markdown_output,
            "summary": f"Processed text with {inputs['word_count']} words, {inputs['sentiment']} sentiment"
        }
    
    format_transform = TransformChain(
        input_variables=["word_count", "char_count", "sentence_count", "top_keywords", "sentiment", "positive_score", "negative_score"],
        output_variables=["json_output", "markdown_output", "summary"],
        transform=format_output_transform
    )
    
    # Test the transforms
    test_texts = [
        "This is a wonderful example of great content. I really love this amazing article!",
        "The product was terrible and disappointing. I had a horrible experience with this awful service.",
        "The weather is okay today. It's neither great nor bad, just average conditions."
    ]
    
    results = []
    for i, text in enumerate(test_texts, 1):
        print(f"\n📊 Test Text {i}: {text[:50]}...")
        
        try:
            # Run through the pipeline
            clean_result = clean_transform.invoke({"raw_text": text})
            extract_result = extract_transform.invoke({"cleaned_text": clean_result["cleaned_text"]})
            
            # Combine results for formatting
            format_input = {**clean_result, **extract_result}
            format_result = format_transform.invoke(format_input)
            
            result = {
                "original_text": text,
                "cleaned_text": clean_result["cleaned_text"],
                "stats": {
                    "words": clean_result["word_count"],
                    "chars": clean_result["char_count"],
                    "sentences": clean_result["sentence_count"]
                },
                "analysis": {
                    "keywords": extract_result["top_keywords"],
                    "sentiment": extract_result["sentiment"]
                },
                "formatted_output": format_result
            }
            results.append(result)
            
            print(f"   ✅ Words: {clean_result['word_count']}, Sentiment: {extract_result['sentiment']}")
            print(f"   ✅ Summary: {format_result['summary']}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return {
        "clean_transform": clean_transform,
        "extract_transform": extract_transform,
        "format_transform": format_transform,
        "pipeline": [clean_transform, extract_transform, format_transform],
        "test_results": results
    }


def solution_5_custom_chain(llm):
    """
    Solution 5: Custom Chain - Business Logic
    
    Complete implementation of customer service automation system.
    """
    print("\n" + "="*60)
    print("✅ SOLUTION 5: Custom Chain - Customer Service Automation")
    print("="*60)
    
    # 1. Inquiry Classification Chain
    classify_prompt = PromptTemplate(
        input_variables=["customer_message"],
        template="""Classify this customer inquiry into one of these categories:
- support: Technical issues, account problems, how-to questions
- sales: Product inquiries, pricing questions, feature requests
- complaint: Negative feedback, refund requests, service issues
- general: General questions, information requests, feedback

Customer Message: {customer_message}

Provide only the category name:"""
    )
    
    classification_chain = LLMChain(
        llm=llm,
        prompt=classify_prompt,
        output_key="category"
    )
    
    # 2. Response Generation Chain
    response_prompt = PromptTemplate(
        input_variables=["customer_message", "category"],
        template="""Generate an appropriate response for this {category} inquiry:

Original Message: {customer_message}

Requirements:
- Professional and helpful tone
- Address the specific concern
- Provide actionable next steps
- Include relevant resources if applicable
- Keep response concise but complete

Response:"""
    )
    
    response_chain = LLMChain(
        llm=llm,
        prompt=response_prompt,
        output_key="response"
    )
    
    # 3. Escalation Chain
    escalation_prompt = PromptTemplate(
        input_variables=["customer_message", "category", "response"],
        template="""Determine if this inquiry needs human escalation:

Customer Message: {customer_message}
Category: {category}
Generated Response: {response}

Consider:
- Complexity of the issue
- Customer sentiment
- Technical difficulty
- Business impact

Should this be escalated to a human agent? (yes/no):"""
    )
    
    escalation_chain = LLMChain(
        llm=llm,
        prompt=escalation_prompt,
        output_key="escalation_decision"
    )
    
    # 4. Custom Service Chain Class
    class CustomerServiceChain(Chain):
        """Custom chain for customer service automation."""
        
        def __init__(self, llm, **kwargs):
            super().__init__(**kwargs)
            self.llm = llm
            self.classification_chain = classification_chain
            self.response_chain = response_chain
            self.escalation_chain = escalation_chain
            self.interaction_history = []
        
        @property
        def input_keys(self):
            return ["customer_message"]
        
        @property
        def output_keys(self):
            return ["category", "response", "escalation_decision", "interaction_id"]
        
        def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Execute the customer service chain."""
            customer_message = inputs["customer_message"]
            interaction_id = f"int_{int(time.time())}"
            
            try:
                # Step 1: Classify the inquiry
                classification_result = self.classification_chain.invoke({
                    "customer_message": customer_message
                })
                category = classification_result["category"].strip().lower()
                
                # Step 2: Generate response
                response_result = self.response_chain.invoke({
                    "customer_message": customer_message,
                    "category": category
                })
                response = response_result["response"]
                
                # Step 3: Check for escalation
                escalation_result = self.escalation_chain.invoke({
                    "customer_message": customer_message,
                    "category": category,
                    "response": response
                })
                escalation_decision = escalation_result["escalation_decision"].strip().lower()
                
                # Store interaction
                interaction = {
                    "id": interaction_id,
                    "timestamp": time.time(),
                    "message": customer_message,
                    "category": category,
                    "response": response,
                    "escalated": escalation_decision == "yes"
                }
                self.interaction_history.append(interaction)
                
                return {
                    "category": category,
                    "response": response,
                    "escalation_decision": escalation_decision,
                    "interaction_id": interaction_id
                }
                
            except Exception as e:
                # Fallback response
                return {
                    "category": "general",
                    "response": "I apologize, but I'm having trouble processing your request. Please contact our human support team for assistance.",
                    "escalation_decision": "yes",
                    "interaction_id": interaction_id
                }
        
        def get_interaction_history(self):
            """Get the history of all interactions."""
            return self.interaction_history
    
    # Test the customer service system
    test_inquiries = [
        "I can't log into my account. It says my password is incorrect.",
        "What features does your premium plan include?",
        "This product is terrible! I want my money back immediately!",
        "How do I update my billing information?",
        "Can you tell me more about your company?"
    ]
    
    service_chain = CustomerServiceChain(llm=llm)
    results = []
    
    for i, inquiry in enumerate(test_inquiries, 1):
        print(f"\n💬 Test Inquiry {i}: {inquiry[:50]}...")
        
        try:
            result = service_chain.invoke({"customer_message": inquiry})
            results.append(result)
            
            print(f"   ✅ Category: {result['category']}")
            print(f"   ✅ Escalated: {result['escalation_decision']}")
            print(f"   ✅ Response: {result['response'][:100]}...")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return {
        "classification_chain": classification_chain,
        "response_chain": response_chain,
        "escalation_chain": escalation_chain,
        "service_chain": service_chain,
        "test_results": results
    }


def solution_6_advanced_chain(llm):
    """
    Solution 6: Advanced Chain - Error Handling & Optimization
    
    Complete implementation of robust chain system with advanced features.
    """
    print("\n" + "="*60)
    print("✅ SOLUTION 6: Advanced Chain - Error Handling & Optimization")
    print("="*60)
    
    # 1. Error Handling Chain with Retry Logic
    class RobustChain(Chain):
        """Chain with built-in error handling and retry logic."""
        
        def __init__(self, llm, max_retries=3, backoff_factor=2, **kwargs):
            super().__init__(**kwargs)
            self.llm = llm
            self.max_retries = max_retries
            self.backoff_factor = backoff_factor
            self.success_count = 0
            self.error_count = 0
        
        @property
        def input_keys(self):
            return ["input"]
        
        @property
        def output_keys(self):
            return ["output", "success", "retries", "execution_time"]
        
        def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Execute with retry logic and error handling."""
            start_time = time.time()
            retries = 0
            
            while retries <= self.max_retries:
                try:
                    # Simple prompt for demonstration
                    prompt = f"Process this input: {inputs['input']}\n\nProvide a helpful response:"
                    response = self.llm.invoke(prompt)
                    
                    execution_time = time.time() - start_time
                    self.success_count += 1
                    
                    return {
                        "output": response,
                        "success": True,
                        "retries": retries,
                        "execution_time": execution_time
                    }
                    
                except Exception as e:
                    retries += 1
                    self.error_count += 1
                    
                    if retries > self.max_retries:
                        execution_time = time.time() - start_time
                        return {
                            "output": f"Error after {retries} retries: {str(e)}",
                            "success": False,
                            "retries": retries,
                            "execution_time": execution_time
                        } 