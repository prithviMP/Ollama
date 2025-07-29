#!/usr/bin/env python3
"""
Lesson 2: Prompt Engineering - Practice Exercises

Complete these exercises to master advanced prompt engineering techniques.
Each exercise focuses on a specific aspect of prompt engineering covered in this lesson.

Instructions:
1. Implement each exercise function
2. Run this file to test your implementations
3. Check solutions.py for reference implementations
"""

import os
import sys
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Add shared resources to path
sys.path.append('../shared-resources')

# Load environment variables
load_dotenv()

# LangChain imports
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# Shared utilities
from utils.llm_setup import setup_llm_providers, get_preferred_llm
from utils.error_handling import safe_llm_call

# Set up LLM providers
providers = setup_llm_providers()
llm = get_preferred_llm(providers, prefer_chat=False) if providers else None
chat_llm = get_preferred_llm(providers, prefer_chat=True) if providers else None


def exercise_1_code_documentation_generator():
    """
    Exercise 1: Custom Few-Shot Template for Code Documentation
    
    Task: Create a few-shot prompt template that generates documentation for Python functions.
    The template should:
    1. Use 3-4 examples of function->documentation pairs
    2. Generate comprehensive docstrings with parameters, returns, and examples
    3. Handle different types of functions (simple, complex, with multiple parameters)
    
    Hint: Use FewShotPromptTemplate with examples that show good documentation patterns
    """
    
    def create_code_documentation_template():
        """
        Create a few-shot template for generating function documentation.
        
        Returns:
            FewShotPromptTemplate: The configured template
        """
        # TODO: Create examples of function->documentation pairs
        # Include examples like:
        # - Simple function with one parameter
        # - Function with multiple parameters and return value
        # - Function with complex logic
        
        examples = [
            # Add your examples here
        ]
        
        # TODO: Create the example template format
        example_template = PromptTemplate(
            input_variables=["function", "documentation"],
            template="Function:\n{function}\n\nDocumentation:\n{documentation}"
        )
        
        # TODO: Create the few-shot template
        few_shot_template = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_template,
            prefix="Generate comprehensive documentation for Python functions:",
            suffix="Function:\n{new_function}\n\nDocumentation:",
            input_variables=["new_function"]
        )
        
        return few_shot_template
    
    def test_documentation_generator(template):
        """Test the documentation generator with sample functions."""
        test_functions = [
            """def calculate_bmi(weight, height):
    return weight / (height ** 2)""",
            
            """def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)""",
            
            """def send_email(to, subject, body, attachments=None):
    if attachments is None:
        attachments = []
    # Implementation here
    return True"""
        ]
        
        for i, func in enumerate(test_functions, 1):
            print(f"\nTest {i}:")
            print(f"Function:\n{func}")
            
            if template and llm:
                formatted_prompt = template.format(new_function=func)
                response = safe_llm_call(llm, formatted_prompt)
                if response:
                    print(f"Generated Documentation:\n{response.strip()}")
            else:
                print("Template or LLM not available")
    
    print("=" * 60)
    print("EXERCISE 1: Code Documentation Generator")
    print("=" * 60)
    
    template = create_code_documentation_template()
    test_documentation_generator(template)


def exercise_2_mathematical_chain_of_thought():
    """
    Exercise 2: Mathematical Chain-of-Thought System
    
    Task: Build a comprehensive CoT system for solving multi-step math problems.
    The system should:
    1. Break down problems into logical steps
    2. Show intermediate calculations
    3. Verify the final answer
    4. Handle different types of math problems (algebra, geometry, word problems)
    
    Hint: Create templates for different problem types with structured reasoning
    """
    
    def create_math_cot_templates():
        """
        Create chain-of-thought templates for different math problem types.
        
        Returns:
            Dict[str, PromptTemplate]: Templates for different math types
        """
        templates = {}
        
        # TODO: Create templates for different math problem types
        
        # Word Problem Template
        word_problem_template = """
        Solve this word problem step by step:
        
        Problem: {problem}
        
        Step 1: Read and understand what the problem is asking
        Step 2: Identify the given information and what we need to find
        Step 3: Determine the mathematical operations needed
        Step 4: Set up the equation or calculation
        Step 5: Solve step by step
        Step 6: Check if the answer makes sense in context
        
        Solution:
        """
        
        # TODO: Add more templates for other problem types
        # - Algebra problems
        # - Geometry problems  
        # - Percentage/ratio problems
        
        templates["word_problem"] = PromptTemplate(
            input_variables=["problem"],
            template=word_problem_template
        )
        
        return templates
    
    def test_math_cot_system(templates):
        """Test the math CoT system with various problems."""
        test_problems = [
            ("word_problem", "A store is having a 25% off sale. If a jacket originally costs $80, and you have a $10 coupon, how much will you pay?"),
            ("word_problem", "A recipe calls for 2 cups of flour for 12 cookies. How many cups of flour are needed for 30 cookies?"),
            ("word_problem", "Tom drives 60 mph for 2 hours, then 40 mph for 1.5 hours. What is his average speed for the entire trip?")
        ]
        
        for problem_type, problem in test_problems:
            print(f"\nProblem Type: {problem_type}")
            print(f"Problem: {problem}")
            
            if problem_type in templates and llm:
                template = templates[problem_type]
                formatted_prompt = template.format(problem=problem)
                response = safe_llm_call(llm, formatted_prompt)
                if response:
                    print(f"Solution:\n{response.strip()}")
            else:
                print("Template not implemented or LLM not available")
            print("-" * 50)
    
    print("\n" + "=" * 60)
    print("EXERCISE 2: Mathematical Chain-of-Thought System")
    print("=" * 60)
    
    templates = create_math_cot_templates()
    test_math_cot_system(templates)


def exercise_3_structured_data_extraction():
    """
    Exercise 3: Structured Data Extraction with Pydantic
    
    Task: Create a system to extract structured information from unstructured text.
    The system should:
    1. Define Pydantic models for different data types
    2. Create prompts that enforce structured output
    3. Handle parsing errors gracefully
    4. Extract multiple entities from complex text
    
    Hint: Use PydanticOutputParser with detailed field descriptions
    """
    
    # TODO: Define Pydantic models for structured data
    class ProductInfo(BaseModel):
        """Model for extracting product information from text."""
        # TODO: Add fields for product extraction
        # name, price, category, rating, availability, etc.
        pass
    
    class EventInfo(BaseModel):
        """Model for extracting event information from text."""
        # TODO: Add fields for event extraction
        # name, date, time, location, organizer, etc.
        pass
    
    def create_extraction_templates():
        """
        Create templates for structured data extraction.
        
        Returns:
            Dict[str, tuple]: Templates and parsers for different data types
        """
        templates = {}
        
        # TODO: Create templates and parsers for each model
        # Example structure:
        # parser = PydanticOutputParser(pydantic_object=ProductInfo)
        # template = PromptTemplate(...)
        # templates["product"] = (template, parser)
        
        return templates
    
    def test_data_extraction(templates):
        """Test structured data extraction with sample texts."""
        test_data = [
            ("product", "The new iPhone 15 Pro is available for $999. It has a 4.5-star rating and features a titanium design. Currently in stock at Apple stores."),
            ("product", "Samsung Galaxy S24 Ultra - $1,199.99. Premium smartphone with S Pen. Rating: 4.7/5 stars. Limited availability."),
            ("event", "Join us for the Annual Tech Conference on March 15, 2024, at 9:00 AM. Located at Convention Center, organized by TechCorp.")
        ]
        
        for data_type, text in test_data:
            print(f"\nData Type: {data_type}")
            print(f"Text: {text}")
            
            if data_type in templates and chat_llm:
                template, parser = templates[data_type]
                formatted_prompt = template.format(text=text)
                response = safe_llm_call(chat_llm, formatted_prompt)
                
                if response:
                    try:
                        parsed_data = parser.parse(response)
                        print(f"Extracted Data: {parsed_data}")
                    except Exception as e:
                        print(f"Parsing failed: {e}")
                        print(f"Raw response: {response}")
            else:
                print("Template not implemented or chat LLM not available")
            print("-" * 50)
    
    print("\n" + "=" * 60)
    print("EXERCISE 3: Structured Data Extraction")
    print("=" * 60)
    
    templates = create_extraction_templates()
    test_data_extraction(templates)


def exercise_4_expert_role_prompting():
    """
    Exercise 4: Advanced Role-Based Expert Prompting
    
    Task: Create expert-level prompts for different professional domains.
    The system should:
    1. Define detailed expert personas with specific expertise
    2. Create domain-specific prompting patterns
    3. Generate responses appropriate to the expert's field
    4. Handle cross-domain consultation scenarios
    
    Hint: Use ChatPromptTemplate with detailed system messages for expert personas
    """
    
    def create_expert_personas():
        """
        Create detailed expert personas for different domains.
        
        Returns:
            Dict[str, dict]: Expert personas with their specializations
        """
        experts = {}
        
        # TODO: Define expert personas with detailed backgrounds
        # Include: background, expertise, communication style, typical approaches
        
        experts["legal_advisor"] = {
            "persona": "You are a senior legal advisor with 20 years of experience in corporate law and compliance.",
            "expertise": ["contract law", "regulatory compliance", "risk assessment", "legal strategy"],
            "style": "precise, thorough, risk-aware"
        }
        
        # TODO: Add more expert personas
        # - Medical professional
        # - Financial advisor
        # - Environmental scientist
        # - Educational psychologist
        
        return experts
    
    def create_expert_consultation_system(experts):
        """
        Create a consultation system using expert personas.
        
        Args:
            experts (Dict): Expert persona definitions
            
        Returns:
            Dict[str, ChatPromptTemplate]: Templates for expert consultations
        """
        templates = {}
        
        for expert_type, expert_data in experts.items():
            # TODO: Create ChatPromptTemplate for each expert
            # Include system message with persona and expertise
            # Include human message template for questions
            pass
        
        return templates
    
    def test_expert_consultation(templates):
        """Test expert consultation with domain-specific questions."""
        test_questions = [
            ("legal_advisor", "What are the key legal considerations when launching a SaaS product in multiple countries?"),
            ("legal_advisor", "How should we structure our terms of service to minimize liability?")
        ]
        
        for expert_type, question in test_questions:
            print(f"\nConsulting: {expert_type}")
            print(f"Question: {question}")
            
            if expert_type in templates and chat_llm:
                template = templates[expert_type]
                messages = template.format_messages(question=question)
                response = safe_llm_call(chat_llm, messages)
                if response:
                    print(f"Expert Response:\n{response.strip()}")
            else:
                print("Expert template not implemented or chat LLM not available")
            print("-" * 50)
    
    print("\n" + "=" * 60)
    print("EXERCISE 4: Expert Role-Based Prompting")
    print("=" * 60)
    
    experts = create_expert_personas()
    templates = create_expert_consultation_system(experts)
    test_expert_consultation(templates)


def exercise_5_prompt_optimization_system():
    """
    Exercise 5: Prompt Optimization and A/B Testing System
    
    Task: Build a system to test and optimize prompt performance.
    The system should:
    1. Create multiple variations of the same prompt
    2. Test prompts with different inputs
    3. Compare response quality and consistency
    4. Provide optimization recommendations
    
    Hint: Create a framework that can systematically test prompt variations
    """
    
    def create_prompt_variations(base_task):
        """
        Create multiple variations of a prompt for A/B testing.
        
        Args:
            base_task (str): The base task to create variations for
            
        Returns:
            Dict[str, str]: Different prompt variations
        """
        variations = {}
        
        # TODO: Create systematic prompt variations
        # Include different approaches:
        # - Direct vs contextual
        # - Structured vs free-form
        # - Role-based vs neutral
        # - With examples vs without examples
        
        variations["direct"] = f"{base_task}"
        
        # TODO: Add more sophisticated variations
        
        return variations
    
    def evaluate_prompt_performance(variations, test_inputs):
        """
        Evaluate the performance of different prompt variations.
        
        Args:
            variations (Dict[str, str]): Prompt variations to test
            test_inputs (List[str]): Inputs to test prompts with
            
        Returns:
            Dict[str, dict]: Performance metrics for each variation
        """
        results = {}
        
        for variation_name, prompt_template in variations.items():
            results[variation_name] = {
                "responses": [],
                "avg_length": 0,
                "consistency": 0
            }
            
            # TODO: Test each variation with all inputs
            # Collect responses and calculate metrics
            # Consider: response length, consistency, relevance
            
            for test_input in test_inputs:
                # TODO: Format prompt and get response
                # Store response and calculate metrics
                pass
        
        return results
    
    def test_optimization_system():
        """Test the prompt optimization system."""
        base_task = "Explain a complex technical concept in simple terms"
        
        test_inputs = [
            "machine learning algorithms",
            "blockchain technology",
            "quantum computing",
            "neural networks"
        ]
        
        print("Base Task:", base_task)
        print("Test Inputs:", test_inputs)
        
        variations = create_prompt_variations(base_task)
        print(f"\nCreated {len(variations)} prompt variations")
        
        # TODO: Run evaluation and display results
        # Show which variations perform best for different metrics
        
        if variations and llm:
            print("\nPrompt Variations:")
            for name, prompt in variations.items():
                print(f"{name}: {prompt[:100]}...")
        else:
            print("LLM not available for testing")
    
    print("\n" + "=" * 60)
    print("EXERCISE 5: Prompt Optimization System")
    print("=" * 60)
    
    test_optimization_system()


def exercise_6_advanced_output_parsing():
    """
    Exercise 6: Advanced Output Parsing with Error Handling
    
    Task: Create a robust output parsing system that handles various formats and errors.
    The system should:
    1. Parse multiple output formats (JSON, lists, structured text)
    2. Handle malformed outputs gracefully
    3. Implement retry logic with corrective prompts
    4. Provide fallback parsing strategies
    
    Hint: Use OutputFixingParser and custom validation logic
    """
    
    class ComplexAnalysis(BaseModel):
        """Model for complex analysis outputs."""
        # TODO: Define fields for complex analysis
        # summary, key_points, recommendations, confidence_score, etc.
        pass
    
    def create_robust_parsing_system():
        """
        Create a robust parsing system with error handling.
        
        Returns:
            tuple: Parser and error handling functions
        """
        # TODO: Create parser with error handling
        # Use OutputFixingParser for automatic error correction
        # Implement custom validation and retry logic
        
        return None, None
    
    def test_parsing_robustness():
        """Test parsing system with various inputs including malformed ones."""
        test_cases = [
            "Well-formed input that should parse correctly",
            "Partially malformed input with missing fields",
            "Completely malformed input that needs fixing"
        ]
        
        # TODO: Test parsing system with various inputs
        # Show how error handling and correction works
        
        print("Testing parsing robustness...")
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}: {test_case}")
            # TODO: Implement testing logic
    
    print("\n" + "=" * 60)
    print("EXERCISE 6: Advanced Output Parsing")
    print("=" * 60)
    
    test_parsing_robustness()


def bonus_exercise_prompt_engineering_framework():
    """
    Bonus Exercise: Complete Prompt Engineering Framework
    
    Task: Combine all techniques into a comprehensive prompt engineering framework.
    The framework should:
    1. Automatically select appropriate prompting techniques
    2. Optimize prompts based on task type
    3. Handle multiple input/output formats
    4. Provide performance analytics
    5. Support easy experimentation and iteration
    
    This is an advanced exercise that combines all previous concepts.
    """
    
    class PromptEngineeringFramework:
        """A comprehensive framework for prompt engineering."""
        
        def __init__(self, llm, chat_llm):
            self.llm = llm
            self.chat_llm = chat_llm
            # TODO: Initialize framework components
            pass
        
        def auto_select_technique(self, task_type, complexity):
            """Automatically select the best prompting technique."""
            # TODO: Implement technique selection logic
            pass
        
        def optimize_prompt(self, base_prompt, optimization_goals):
            """Optimize a prompt for specific goals."""
            # TODO: Implement prompt optimization
            pass
        
        def analyze_performance(self, prompt, test_cases):
            """Analyze prompt performance across test cases."""
            # TODO: Implement performance analysis
            pass
    
    def test_framework():
        """Test the complete prompt engineering framework."""
        if llm or chat_llm:
            framework = PromptEngineeringFramework(llm, chat_llm)
            print("Framework initialized successfully!")
            
            # TODO: Demonstrate framework capabilities
            # Show automatic technique selection
            # Show prompt optimization
            # Show performance analysis
        else:
            print("LLM not available for framework testing")
    
    print("\n" + "=" * 60)
    print("BONUS EXERCISE: Complete Prompt Engineering Framework")
    print("=" * 60)
    
    test_framework()


if __name__ == "__main__":
    """
    Run all exercises when the script is executed directly.
    """
    print("🦜🔗 LangChain Course - Lesson 2 Exercises")
    print("=" * 60)
    
    if not (llm or chat_llm):
        print("⚠️  No LLM providers available. Please configure your API keys.")
        print("Set OPENAI_API_KEY in your .env file or use another provider.")
        print("Some exercises will show structure but won't generate responses.")
    
    print("\n🏋️ Starting prompt engineering exercises...")
    
    # Run all exercises
    exercises = [
        exercise_1_code_documentation_generator,
        exercise_2_mathematical_chain_of_thought,
        exercise_3_structured_data_extraction,
        exercise_4_expert_role_prompting,
        exercise_5_prompt_optimization_system,
        exercise_6_advanced_output_parsing,
        bonus_exercise_prompt_engineering_framework
    ]
    
    for i, exercise in enumerate(exercises, 1):
        try:
            exercise()
        except Exception as e:
            print(f"Exercise {i} error: {e}")
    
    print("\n🎯 Exercises completed!")
    print("Check solutions.py for reference implementations.")
    print("Practice these techniques to master prompt engineering!") 