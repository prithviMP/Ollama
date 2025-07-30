#!/usr/bin/env python3
"""
Lesson 3: Chains & Sequential Processing with LangChain

This lesson covers:
1. LLMChain fundamentals and composition
2. Sequential chains for multi-step processing  
3. Router chains for conditional logic
4. Transform and utility chains
5. Advanced chain patterns and optimization

Author: LangChain Course
"""

import os
import sys
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import json

# Add shared resources to path
sys.path.append('../shared-resources')

# Load environment variables
load_dotenv()

# LangChain imports
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain.chains import TransformChain, LLMMathChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import BaseOutputParser
import requests

# Shared utilities
from utils.llm_setup import setup_llm_providers, get_preferred_llm
from utils.error_handling import safe_llm_call


def setup_lesson():
    """Set up the lesson environment and providers."""
    print("🦜🔗 LangChain Course - Lesson 3: Chains & Sequential Processing")
    print("=" * 70)
    
    providers = setup_llm_providers()
    if not providers:
        print("❌ No LLM providers available. Please check your setup.")
        return None, None
    
    llm = get_preferred_llm(providers, prefer_chat=False)
    chat_llm = get_preferred_llm(providers, prefer_chat=True)
    
    return llm, chat_llm


def demonstrate_basic_llm_chains(llm):
    """
    Demonstrate basic LLMChain usage and composition.
    """
    print("\n" + "="*60)
    print("🔗 BASIC LLM CHAINS DEMONSTRATION")
    print("="*60)
    
    # Example 1: Simple LLMChain
    print("\n1. Simple LLMChain: Article Summarization")
    print("-" * 40)
    
    summary_prompt = PromptTemplate(
        input_variables=["article"],
        template="""Summarize the following article in 3-4 sentences:

Article: {article}

Summary:"""
    )
    
    summary_chain = LLMChain(
        llm=llm,
        prompt=summary_prompt,
        output_key="summary"
    )
    
    test_article = """
    Artificial Intelligence (AI) has emerged as one of the most transformative technologies of the 21st century.
    From healthcare to finance, transportation to entertainment, AI is revolutionizing industries and changing 
    how we live and work. Machine learning algorithms can now diagnose diseases, predict market trends, 
    drive cars autonomously, and even create art. However, with great power comes great responsibility. 
    The rapid advancement of AI raises important questions about ethics, privacy, job displacement, and 
    the need for regulation. As we continue to develop these powerful tools, it's crucial that we do so 
    thoughtfully and responsibly, ensuring that AI benefits all of humanity.
    """
    
    if llm:
        result = summary_chain.invoke({"article": test_article})
        print(f"Original Article Length: {len(test_article)} characters")
        print(f"Summary: {result['summary']}")
    else:
        print("LLM not available for demonstration")
    
    # Example 2: LLMChain with Custom Output Processing
    print("\n\n2. LLMChain with Custom Processing: Keyword Extraction")
    print("-" * 50)
    
    keyword_prompt = PromptTemplate(
        input_variables=["text"],
        template="""Extract the 5 most important keywords from the following text.
Return them as a comma-separated list.

Text: {text}

Keywords:"""
    )
    
    keyword_chain = LLMChain(
        llm=llm,
        prompt=keyword_prompt,
        output_key="keywords"
    )
    
    if llm:
        keyword_result = keyword_chain.invoke({"text": test_article})
        keywords = keyword_result['keywords'].strip().split(',')
        print(f"Extracted Keywords: {[kw.strip() for kw in keywords]}")
    else:
        print("LLM not available for demonstration")


def demonstrate_sequential_chains(llm):
    """
    Demonstrate sequential chain patterns for multi-step processing.
    """
    print("\n" + "="*60)
    print("⛓️  SEQUENTIAL CHAINS DEMONSTRATION")
    print("="*60)
    
    # Example 1: SimpleSequentialChain
    print("\n1. SimpleSequentialChain: Story Generation Pipeline")
    print("-" * 50)
    
    # First chain: Generate story outline - FIXED to use single input
    outline_prompt = PromptTemplate(
        input_variables=["story_request"],  # Changed from ["genre", "setting"]
        template="""Create a brief story outline based on this request: {story_request}
Include the main character, conflict, and basic plot structure in 2-3 sentences."""
    )
    outline_chain = LLMChain(llm=llm, prompt=outline_prompt)
    
    # Second chain: Expand outline into story
    story_prompt = PromptTemplate(
        input_variables=["outline"],
        template="""Based on this outline, write a compelling short story (4-5 paragraphs):

Outline: {outline}

Story:"""
    )
    story_chain = LLMChain(llm=llm, prompt=story_prompt)
    
    # Combine into simple sequential chain
    story_pipeline = SimpleSequentialChain(
        chains=[outline_chain, story_chain],
        verbose=True
    )
    
    if llm:
        story_result = story_pipeline.invoke({"input": "science fiction story set on Mars colony"})
        print("Generated Story:")
        print(story_result['output'][:300] + "..." if len(story_result['output']) > 300 else story_result['output'])
    else:
        print("LLM not available for story generation")
    
    # Example 2: SequentialChain with Multiple Inputs/Outputs
    print("\n\n2. SequentialChain: Content Marketing Pipeline")
    print("-" * 50)
    
    # Chain 1: Research phase
    research_prompt = PromptTemplate(
        input_variables=["topic", "audience"],
        template="""Research key points about {topic} for {audience}.
Provide 3-4 main points with brief explanations."""
    )
    research_chain = LLMChain(
        llm=llm,
        prompt=research_prompt,
        output_key="research_points"
    )
    
    # Chain 2: Content creation
    content_prompt = PromptTemplate(
        input_variables=["topic", "research_points"],
        template="""Create engaging content about {topic} based on these research points:

{research_points}

Write a compelling introduction and main body (2-3 paragraphs)."""
    )
    content_chain = LLMChain(
        llm=llm,
        prompt=content_prompt,
        output_key="content"
    )
    
    # Chain 3: SEO optimization
    seo_prompt = PromptTemplate(
        input_variables=["content", "topic"],
        template="""Suggest 5 SEO-optimized headlines for this content about {topic}:

Content: {content}

Headlines:"""
    )
    seo_chain = LLMChain(
        llm=llm,
        prompt=seo_prompt,
        output_key="headlines"
    )
    
    # Combine into sequential chain
    marketing_pipeline = SequentialChain(
        chains=[research_chain, content_chain, seo_chain],
        input_variables=["topic", "audience"],
        output_variables=["research_points", "content", "headlines"],
        verbose=True
    )
    
    if llm:
        marketing_result = marketing_pipeline.invoke({
            "topic": "sustainable energy solutions",
            "audience": "small business owners"
        })
        
        print("Marketing Pipeline Results:")
        print(f"Research Points: {marketing_result['research_points'][:200]}...")
        print(f"Content: {marketing_result['content'][:200]}...")
        print(f"Headlines: {marketing_result['headlines']}")
    else:
        print("LLM not available for marketing pipeline")


def demonstrate_router_chains(llm):
    """
    Demonstrate router chains for conditional logic and intelligent routing.
    """
    print("\n" + "="*60)
    print("🧭 ROUTER CHAINS DEMONSTRATION")
    print("="*60)
    
    # Define specialized prompt templates
    math_template = """You are a mathematics expert. Solve this math problem step by step:

Problem: {input}

Solution:"""

    science_template = """You are a science expert. Explain this scientific concept clearly:

Question: {input}

Explanation:"""

    history_template = """You are a history expert. Provide historical context and information:

Question: {input}

Historical Response:"""

    general_template = """You are a helpful assistant. Answer this general question:

Question: {input}

Answer:"""
    
    # Create prompt infos for router
    prompt_infos = [
        {
            "name": "mathematics",
            "description": "Good for math problems, calculations, equations, and mathematical concepts",
            "prompt_template": math_template
        },
        {
            "name": "science", 
            "description": "Good for scientific questions, physics, chemistry, biology, and natural phenomena",
            "prompt_template": science_template
        },
        {
            "name": "history",
            "description": "Good for historical events, dates, historical figures, and past civilizations",
            "prompt_template": history_template
        }
    ]
    
    # Create default chain
    default_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(template=general_template, input_variables=["input"])
    )
    
    # Create router chain
    if llm:
        router_chain = MultiPromptChain.from_prompts(
            llm=llm,
            prompt_infos=prompt_infos,
            default_chain=default_chain,
            verbose=True
        )
        
        # Test the router with different types of questions
        test_questions = [
            "What is the square root of 144?",
            "How does photosynthesis work?",
            "When did World War II end?",
            "What's your favorite color?"
        ]
        
        print("Testing Router Chain with Various Questions:")
        print("-" * 50)
        
        for question in test_questions:
            print(f"\nQuestion: {question}")
            try:
                result = router_chain.invoke({"input": question})
                print(f"Routed to: {result.get('destination', 'default')}")
                print(f"Response: {result['text'][:150]}...")
            except Exception as e:
                print(f"Error processing question: {e}")
    else:
        print("LLM not available for router demonstration")


def demonstrate_transform_chains(llm):
    """
    Demonstrate transform chains for data preprocessing and utility operations.
    """
    print("\n" + "="*60)
    print("🔄 TRANSFORM CHAINS DEMONSTRATION")
    print("="*60)
    
    # Example 1: Data preprocessing with TransformChain
    print("\n1. TransformChain: Data Cleaning Pipeline")
    print("-" * 40)
    
    def preprocess_text(inputs: dict) -> dict:
        """Clean and preprocess text data."""
        text = inputs["raw_text"]
        
        # Basic text cleaning
        cleaned = text.strip()
        cleaned = ' '.join(cleaned.split())  # Remove extra whitespace
        cleaned = cleaned.replace('\n', ' ')
        
        # Count words and characters
        word_count = len(cleaned.split())
        char_count = len(cleaned)
        
        return {
            "cleaned_text": cleaned,
            "word_count": word_count,
            "char_count": char_count
        }
    
    # Create transform chain
    preprocess_chain = TransformChain(
        input_variables=["raw_text"],
        output_variables=["cleaned_text", "word_count", "char_count"],
        transform=preprocess_text
    )
    
    # Test data preprocessing
    messy_text = """   This is some    messy text with  
    irregular   spacing and
    line breaks.     """
    
    preprocess_result = preprocess_chain.invoke({"raw_text": messy_text})
    print(f"Original: '{messy_text}'")
    print(f"Cleaned: '{preprocess_result['cleaned_text']}'")
    print(f"Word Count: {preprocess_result['word_count']}")
    print(f"Character Count: {preprocess_result['char_count']}")
    
    # Example 2: Math Chain for calculations
    print("\n\n2. LLMMathChain: Mathematical Problem Solving")
    print("-" * 50)
    
    if llm:
        math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
        
        math_problems = [
            "What is 15% of 240?",
            "If I have 50 apples and give away 30% of them, how many do I have left?",
            "Calculate the area of a circle with radius 7"
        ]
        
        for problem in math_problems:
            print(f"\nProblem: {problem}")
            try:
                result = math_chain.invoke({"question": problem})
                print(f"Answer: {result['answer']}")
            except Exception as e:
                print(f"Error solving problem: {e}")
    else:
        print("LLM not available for math chain demonstration")


def demonstrate_custom_chains(llm):
    """
    Demonstrate custom chain creation for specific business logic.
    """
    print("\n" + "="*60)
    print("🛠️  CUSTOM CHAINS DEMONSTRATION")
    print("="*60)
    
    # Example 1: Email Classification and Response Chain
    print("\n1. Custom Email Processing Chain")
    print("-" * 40)
    
    # Step 1: Classify email
    classify_prompt = PromptTemplate(
        input_variables=["email_content"],
        template="""Classify this email into one of these categories:
- support: Customer support questions
- sales: Sales inquiries
- complaint: Customer complaints  
- spam: Spam or promotional emails

Email: {email_content}

Category:"""
    )
    
    classify_chain = LLMChain(
        llm=llm,
        prompt=classify_prompt,
        output_key="category"
    )
    
    # Step 2: Generate appropriate response based on category
    response_prompt = PromptTemplate(
        input_variables=["email_content", "category"],
        template="""Generate an appropriate response for this {category} email:

Original Email: {email_content}

Response:"""
    )
    
    response_chain = LLMChain(
        llm=llm,
        prompt=response_prompt,
        output_key="response"
    )
    
    # Combine into email processing pipeline
    email_pipeline = SequentialChain(
        chains=[classify_chain, response_chain],
        input_variables=["email_content"],
        output_variables=["category", "response"],
        verbose=True
    )
    
    # Test emails
    test_emails = [
        "Hi, I'm having trouble logging into my account. Can you help?",
        "I'm interested in your premium plan. What features does it include?",
        "This product is terrible! I want a full refund immediately!"
    ]
    
    if llm:
        for email in test_emails:
            print(f"\nEmail: {email}")
            result = email_pipeline.invoke({"email_content": email})
            print(f"Category: {result['category'].strip()}")
            print(f"Response: {result['response'][:200]}...")
    else:
        print("LLM not available for email processing demonstration")


def demonstrate_advanced_chain_patterns(llm):
    """
    Demonstrate advanced chain patterns and optimization techniques.
    """
    print("\n" + "="*60)
    print("🚀 ADVANCED CHAIN PATTERNS DEMONSTRATION")
    print("="*60)
    
    # Example 1: Chain with Error Handling and Fallbacks
    print("\n1. Robust Chain with Error Handling")
    print("-" * 40)
    
    def create_robust_chain(llm):
        """Create a chain with built-in error handling."""
        
        # Primary chain
        primary_prompt = PromptTemplate(
            input_variables=["question"],
            template="Provide a detailed answer to: {question}"
        )
        primary_chain = LLMChain(llm=llm, prompt=primary_prompt)
        
        # Fallback chain (simpler)
        fallback_prompt = PromptTemplate(
            input_variables=["question"],
            template="Give a brief answer to: {question}"
        )
        fallback_chain = LLMChain(llm=llm, prompt=fallback_prompt)
        
        def robust_invoke(inputs):
            """Invoke with fallback handling."""
            try:
                return primary_chain.invoke(inputs)
            except Exception as e:
                print(f"Primary chain failed: {e}. Using fallback...")
                try:
                    return fallback_chain.invoke(inputs)
                except Exception as e2:
                    print(f"Fallback also failed: {e2}")
                    return {"text": "Unable to process request at this time."}
        
        return robust_invoke
    
    if llm:
        robust_chain = create_robust_chain(llm)
        test_question = "What are the benefits of renewable energy?"
        
        result = robust_chain({"question": test_question})
        print(f"Question: {test_question}")
        print(f"Answer: {result.get('text', 'No response')[:200]}...")
    
    # Example 2: Performance Monitoring Chain
    print("\n\n2. Chain Performance Monitoring")
    print("-" * 40)
    
    import time
    
    def monitored_chain_invoke(chain, inputs):
        """Invoke chain with performance monitoring."""
        start_time = time.time()
        
        try:
            result = chain.invoke(inputs)
            end_time = time.time()
            
            execution_time = end_time - start_time
            print(f"⏱️  Chain executed in {execution_time:.2f} seconds")
            print(f"📊 Input tokens: ~{len(str(inputs)) // 4}")
            print(f"📊 Output tokens: ~{len(str(result)) // 4}")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"❌ Chain failed after {execution_time:.2f} seconds: {e}")
            return None
    
    if llm:
        simple_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=["topic"],
                template="Explain {topic} in simple terms."
            )
        )
        
        print("Testing chain performance monitoring:")
        result = monitored_chain_invoke(simple_chain, {"topic": "quantum computing"})
        if result:
            print(f"Result: {result['text'][:100]}...")


def interactive_chain_builder(llm):
    """
    Interactive chain building demonstration.
    """
    print("\n" + "="*60)
    print("🎮 INTERACTIVE CHAIN BUILDER")
    print("="*60)
    
    print("Welcome to the Interactive Chain Builder!")
    print("Let's build a custom chain together.")
    
    chain_types = {
        "1": "Content Generation Chain",
        "2": "Data Analysis Chain", 
        "3": "Question Answering Chain",
        "4": "Creative Writing Chain"
    }
    
    print("\nAvailable chain types:")
    for key, value in chain_types.items():
        print(f"  {key}. {value}")
    
    choice = input("\nSelect a chain type (1-4) or 'quit': ").strip()
    
    if choice.lower() in ['quit', 'q']:
        print("Thanks for using the Chain Builder!")
        return
    
    if choice not in chain_types:
        print("Invalid choice.")
        return
    
    selected_type = chain_types[choice]
    print(f"\nBuilding: {selected_type}")
    
    if choice == "1" and llm:
        # Content Generation Chain
        topic = input("Enter a topic for content generation: ").strip()
        audience = input("Enter target audience: ").strip()
        
        if topic and audience:
            # Create a quick content generation chain
            content_prompt = PromptTemplate(
                input_variables=["topic", "audience"],
                template="""Create engaging content about {topic} for {audience}.
Include an attention-grabbing headline and 2-3 paragraphs of content."""
            )
            
            content_chain = LLMChain(llm=llm, prompt=content_prompt)
            
            print(f"\nGenerating content about '{topic}' for '{audience}'...")
            result = content_chain.invoke({"topic": topic, "audience": audience})
            print(f"\nGenerated Content:\n{result['text']}")
    
    elif choice == "3" and llm:
        # Question Answering Chain
        context = input("Enter some context/background information: ").strip()
        question = input("Enter your question: ").strip()
        
        if context and question:
            qa_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""Based on this context: {context}

Answer this question: {question}

Provide a clear, accurate answer based on the given context."""
            )
            
            qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
            
            print(f"\nAnswering question based on provided context...")
            result = qa_chain.invoke({"context": context, "question": question})
            print(f"\nAnswer:\n{result['text']}")
    
    else:
        print("This chain type is not implemented in the interactive demo.")
        print("Try building it yourself using the patterns from this lesson!")


def main():
    """
    Main function to run all chain demonstrations.
    """
    llm, chat_llm = setup_lesson()
    
    if not llm and not chat_llm:
        print("❌ Cannot proceed without LLM providers. Please check your setup.")
        return
    
    # Use the available model for demonstrations
    demo_llm = llm if llm else chat_llm
    
    print(f"\n🔧 Using LLM: {type(demo_llm).__name__}")
    
    try:
        # Run all demonstrations
        demonstrate_basic_llm_chains(demo_llm)
        demonstrate_sequential_chains(demo_llm)
        demonstrate_router_chains(demo_llm)
        demonstrate_transform_chains(demo_llm)
        demonstrate_custom_chains(demo_llm)
        demonstrate_advanced_chain_patterns(demo_llm)
        
        # Interactive builder
        print("\n🎉 Core demonstrations completed!")
        
        run_builder = input("\nWould you like to try the Interactive Chain Builder? (y/n): ").strip().lower()
        if run_builder in ['y', 'yes']:
            interactive_chain_builder(demo_llm)
        
        print("\n✨ Lesson 3 completed! You've mastered LangChain chains and sequential processing.")
        print("\n📚 Key Skills Acquired:")
        print("   • LLMChain composition and execution")
        print("   • Sequential chains for multi-step workflows")
        print("   • Router chains for intelligent decision making")
        print("   • Transform chains for data preprocessing")
        print("   • Custom chain development")
        print("   • Advanced chain patterns and optimization")
        
        print("\n🔗 Next: Lesson 4 - Memory & Conversation Management")
        
    except KeyboardInterrupt:
        print("\n\n👋 Lesson interrupted. You can resume anytime!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    main() 