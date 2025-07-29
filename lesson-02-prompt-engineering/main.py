#!/usr/bin/env python3
"""
Lesson 2: Prompt Engineering with LangChain

This lesson covers:
1. Few-shot prompting with examples
2. Chain-of-thought reasoning
3. Structured output parsing
4. Advanced prompt engineering patterns
5. Production prompt optimization

Author: LangChain Course
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
from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

# Shared utilities
from utils.llm_setup import setup_llm_providers, get_preferred_llm
from utils.error_handling import safe_llm_call
from utils.prompt_helpers import create_few_shot_template, create_chain_of_thought_template


# Pydantic models for structured outputs
class PersonInfo(BaseModel):
    """Model for extracting person information."""
    name: str = Field(description="person's full name")
    age: int = Field(description="person's age in years", ge=0, le=150)
    occupation: str = Field(description="person's job title or profession")
    location: Optional[str] = Field(description="person's location or city", default=None)


class MathProblemSolution(BaseModel):
    """Model for structured math problem solutions."""
    problem_type: str = Field(description="type of math problem (algebra, geometry, etc.)")
    steps: List[str] = Field(description="list of solution steps")
    final_answer: str = Field(description="the final numerical answer")
    explanation: str = Field(description="brief explanation of the solution approach")


class CodeDocumentation(BaseModel):
    """Model for code documentation."""
    function_name: str = Field(description="name of the function")
    purpose: str = Field(description="what the function does")
    parameters: List[Dict[str, str]] = Field(description="list of parameters with types and descriptions")
    returns: str = Field(description="description of return value")
    example_usage: str = Field(description="example of how to use the function")


def setup_lesson():
    """Set up the lesson environment and providers."""
    print("🦜🔗 LangChain Course - Lesson 2: Prompt Engineering")
    print("=" * 60)
    
    providers = setup_llm_providers()
    if not providers:
        print("❌ No LLM providers available. Please check your setup.")
        return None, None
    
    llm = get_preferred_llm(providers, prefer_chat=False)
    chat_llm = get_preferred_llm(providers, prefer_chat=True)
    
    return llm, chat_llm


def demonstrate_few_shot_prompting(llm):
    """
    Demonstrate few-shot prompting techniques.
    """
    print("\n" + "="*50)
    print("🎯 FEW-SHOT PROMPTING DEMONSTRATION")
    print("="*50)
    
    # Example 1: Simple Few-Shot with Static Examples
    print("\n1. Simple Few-Shot: Emotion to Emoji Conversion")
    print("-" * 40)
    
    examples = [
        {"emotion": "happy", "emoji": "😊"},
        {"emotion": "sad", "emoji": "😢"},
        {"emotion": "angry", "emoji": "😠"},
        {"emotion": "surprised", "emoji": "😮"},
        {"emotion": "confused", "emoji": "😕"}
    ]
    
    example_template = PromptTemplate(
        input_variables=["emotion", "emoji"],
        template="Emotion: {emotion} → Emoji: {emoji}"
    )
    
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_template,
        prefix="Convert emotions to appropriate emojis:",
        suffix="Emotion: {input_emotion} → Emoji:",
        input_variables=["input_emotion"]
    )
    
    test_emotions = ["excited", "tired", "worried"]
    
    for emotion in test_emotions:
        formatted_prompt = few_shot_prompt.format(input_emotion=emotion)
        print(f"\nTesting emotion: {emotion}")
        print(f"Prompt preview: ...{formatted_prompt[-100:]}")
        
        response = safe_llm_call(llm, formatted_prompt)
        if response:
            print(f"Result: {emotion} → {response.strip()}")
    
    # Example 2: Code Comment Generation
    print("\n\n2. Advanced Few-Shot: Code Comment Generation")
    print("-" * 40)
    
    code_examples = [
        {
            "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
            "comment": "Calculates the factorial of a number using recursion"
        },
        {
            "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "comment": "Generates the nth Fibonacci number using recursive approach"
        },
        {
            "code": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
            "comment": "Performs binary search on a sorted array to find target element"
        }
    ]
    
    code_comment_template = PromptTemplate(
        input_variables=["code", "comment"],
        template="Code:\n{code}\n\nComment: {comment}"
    )
    
    code_few_shot = FewShotPromptTemplate(
        examples=code_examples,
        example_prompt=code_comment_template,
        prefix="Generate concise, descriptive comments for the following code functions:",
        suffix="Code:\n{new_code}\n\nComment:",
        input_variables=["new_code"]
    )
    
    test_code = """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)"""
    
    formatted_prompt = code_few_shot.format(new_code=test_code)
    print(f"Generating comment for:\n{test_code}")
    
    response = safe_llm_call(llm, formatted_prompt)
    if response:
        print(f"\nGenerated comment: {response.strip()}")


def demonstrate_chain_of_thought(llm):
    """
    Demonstrate chain-of-thought reasoning patterns.
    """
    print("\n" + "="*50)
    print("🧠 CHAIN-OF-THOUGHT REASONING DEMONSTRATION")
    print("="*50)
    
    # Example 1: Mathematical Problem Solving
    print("\n1. Mathematical Chain-of-Thought")
    print("-" * 40)
    
    math_cot_template = """
    Solve this math problem step by step:

    Problem: {problem}

    Let me work through this systematically:

    Step 1: Understand what we're given and what we need to find
    Step 2: Identify the appropriate mathematical approach
    Step 3: Set up the equation or formula
    Step 4: Perform the calculations step by step
    Step 5: Verify the answer makes sense

    Solution:
    """
    
    math_prompt = PromptTemplate(
        input_variables=["problem"],
        template=math_cot_template
    )
    
    math_problems = [
        "A train travels 120 miles in 2 hours. If it maintains the same speed, how long will it take to travel 300 miles?",
        "Sarah has $50. She buys 3 books for $12 each and 2 pens for $3 each. How much money does she have left?",
        "A rectangle has a length of 15 cm and a width of 8 cm. What is its area and perimeter?"
    ]
    
    for problem in math_problems:
        print(f"\nProblem: {problem}")
        formatted_prompt = math_prompt.format(problem=problem)
        
        response = safe_llm_call(llm, formatted_prompt)
        if response:
            print(f"Solution:\n{response.strip()}\n")
            print("-" * 60)
    
    # Example 2: Logical Reasoning
    print("\n2. Logical Chain-of-Thought")
    print("-" * 40)
    
    logic_cot_template = """
    Solve this logical reasoning problem step by step:

    Problem: {problem}

    Let me reason through this step by step:

    Step 1: Identify the key information and constraints
    Step 2: Determine what logical rules apply
    Step 3: Work through the implications systematically
    Step 4: Eliminate impossible options
    Step 5: Arrive at the logical conclusion

    Reasoning:
    """
    
    logic_prompt = PromptTemplate(
        input_variables=["problem"],
        template=logic_cot_template
    )
    
    logic_problem = """
    There are three boxes: Red, Blue, and Green.
    - One box contains only apples
    - One box contains only oranges  
    - One box contains both apples and oranges
    
    Each box is labeled, but all labels are wrong.
    You can pick one fruit from one box to determine the contents of all boxes.
    
    If you pick an apple from the box labeled "Oranges", what are the contents of each box?
    """
    
    print(f"Problem: {logic_problem}")
    formatted_prompt = logic_prompt.format(problem=logic_problem)
    
    response = safe_llm_call(llm, formatted_prompt)
    if response:
        print(f"Solution:\n{response.strip()}")


def demonstrate_structured_output_parsing(chat_llm):
    """
    Demonstrate structured output parsing with Pydantic.
    """
    print("\n" + "="*50)
    print("📊 STRUCTURED OUTPUT PARSING DEMONSTRATION")
    print("="*50)
    
    # Example 1: Person Information Extraction
    print("\n1. Person Information Extraction")
    print("-" * 40)
    
    person_parser = PydanticOutputParser(pydantic_object=PersonInfo)
    
    person_template = """
    Extract information about the person mentioned in the following text.
    
    Text: {text}
    
    {format_instructions}
    """
    
    person_prompt = PromptTemplate(
        template=person_template,
        input_variables=["text"],
        partial_variables={"format_instructions": person_parser.get_format_instructions()}
    )
    
    test_texts = [
        "Dr. Sarah Johnson is a 35-year-old neurosurgeon working at Johns Hopkins Hospital in Baltimore.",
        "Meet Alex Chen, 28, a software engineer at Google who loves hiking and lives in San Francisco.",
        "Professor Maria Rodriguez, aged 52, teaches literature at Harvard University and has written several books."
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        formatted_prompt = person_prompt.format(text=text)
        
        response = safe_llm_call(chat_llm, formatted_prompt)
        if response:
            try:
                parsed_info = person_parser.parse(response)
                print(f"Extracted Info:")
                print(f"  Name: {parsed_info.name}")
                print(f"  Age: {parsed_info.age}")
                print(f"  Occupation: {parsed_info.occupation}")
                print(f"  Location: {parsed_info.location}")
            except Exception as e:
                print(f"Parsing failed: {e}")
                print(f"Raw response: {response}")
    
    # Example 2: Math Problem Solution Structure
    print("\n\n2. Structured Math Problem Solutions")
    print("-" * 40)
    
    math_parser = PydanticOutputParser(pydantic_object=MathProblemSolution)
    
    math_structure_template = """
    Solve the following math problem and provide a structured response:
    
    Problem: {problem}
    
    {format_instructions}
    """
    
    math_structure_prompt = PromptTemplate(
        template=math_structure_template,
        input_variables=["problem"],
        partial_variables={"format_instructions": math_parser.get_format_instructions()}
    )
    
    math_problem = "Find the area of a circle with radius 7 cm. Use π = 3.14159."
    
    print(f"Problem: {math_problem}")
    formatted_prompt = math_structure_prompt.format(problem=math_problem)
    
    response = safe_llm_call(chat_llm, formatted_prompt)
    if response:
        try:
            parsed_solution = math_parser.parse(response)
            print(f"\nStructured Solution:")
            print(f"  Problem Type: {parsed_solution.problem_type}")
            print(f"  Steps:")
            for i, step in enumerate(parsed_solution.steps, 1):
                print(f"    {i}. {step}")
            print(f"  Final Answer: {parsed_solution.final_answer}")
            print(f"  Explanation: {parsed_solution.explanation}")
        except Exception as e:
            print(f"Parsing failed: {e}")
            print(f"Raw response: {response}")


def demonstrate_advanced_prompting_patterns(chat_llm):
    """
    Demonstrate advanced prompting patterns and techniques.
    """
    print("\n" + "="*50)
    print("🚀 ADVANCED PROMPTING PATTERNS DEMONSTRATION")
    print("="*50)
    
    # Example 1: Role-Based Prompting
    print("\n1. Role-Based Expert Prompting")
    print("-" * 40)
    
    expert_roles = {
        "Software Architect": "You are a senior software architect with 15 years of experience in designing scalable systems.",
        "Data Scientist": "You are a data scientist with expertise in machine learning and statistical analysis.",
        "UX Designer": "You are a UX designer with deep knowledge of user psychology and interface design.",
        "Security Expert": "You are a cybersecurity expert specializing in application security and threat assessment."
    }
    
    problem = "How would you approach building a real-time chat application that needs to handle 1 million concurrent users?"
    
    for role, persona in expert_roles.items():
        print(f"\n{role} Perspective:")
        print("-" * 30)
        
        role_template = ChatPromptTemplate.from_messages([
            ("system", f"{persona} Provide expert advice from your specialized perspective."),
            ("human", f"Problem: {problem}\n\nPlease provide your expert recommendation in 2-3 key points.")
        ])
        
        messages = role_template.format_messages()
        response = safe_llm_call(chat_llm, messages)
        if response:
            print(f"{response.strip()}")
    
    # Example 2: Multi-Step Instruction Hierarchy
    print("\n\n2. Hierarchical Instruction Following")
    print("-" * 40)
    
    hierarchy_template = """
    You are an AI assistant that follows instructions in hierarchical order.
    
    PRIMARY OBJECTIVE: {primary_objective}
    
    SECONDARY REQUIREMENTS:
    {secondary_requirements}
    
    CONSTRAINTS:
    {constraints}
    
    TASK: {task}
    
    Please complete the task while respecting the hierarchy: Primary objective first, then secondary requirements, then constraints.
    """
    
    hierarchy_prompt = PromptTemplate(
        input_variables=["primary_objective", "secondary_requirements", "constraints", "task"],
        template=hierarchy_template
    )
    
    formatted_prompt = hierarchy_prompt.format(
        primary_objective="Provide accurate and helpful information",
        secondary_requirements="• Be concise and clear\n• Include practical examples\n• Suggest best practices",
        constraints="• Keep response under 200 words\n• Use bullet points for structure\n• Include at least one code example",
        task="Explain the difference between synchronous and asynchronous programming in Python"
    )
    
    print("Hierarchical Instruction Example:")
    response = safe_llm_call(chat_llm, formatted_prompt)
    if response:
        print(f"\nResponse:\n{response.strip()}")


def demonstrate_prompt_optimization(llm, chat_llm):
    """
    Demonstrate prompt optimization techniques.
    """
    print("\n" + "="*50)
    print("⚡ PROMPT OPTIMIZATION DEMONSTRATION")
    print("="*50)
    
    # Example: A/B Testing Different Prompt Versions
    print("\n1. A/B Testing Prompt Variations")
    print("-" * 40)
    
    base_task = "Explain machine learning to a beginner"
    
    prompt_variations = {
        "Version A - Direct": "Explain machine learning to a beginner.",
        
        "Version B - Context": "You are teaching a complete beginner who has no technical background. Explain machine learning in simple terms.",
        
        "Version C - Structured": """Explain machine learning to a beginner using this structure:
1. What it is in simple terms
2. A real-world analogy
3. One practical example
4. Why it matters

Keep it conversational and avoid technical jargon.""",
        
        "Version D - Interactive": """Imagine you're having a friendly conversation with someone who just asked "What is machine learning?" 
Respond as if you're sitting across from them at a coffee shop, using everyday language and relatable examples."""
    }
    
    print(f"Task: {base_task}\n")
    
    for version, prompt in prompt_variations.items():
        print(f"{version}:")
        print(f"Prompt: {prompt}")
        
        if "A" in version or "B" in version:
            response = safe_llm_call(llm, prompt)
        else:
            response = safe_llm_call(chat_llm, prompt)
        
        if response:
            # Show first 150 characters as preview
            preview = response.strip()[:150] + "..." if len(response) > 150 else response.strip()
            print(f"Response: {preview}")
        print("-" * 50)
    
    # Example 2: Performance Metrics
    print("\n2. Prompt Performance Considerations")
    print("-" * 40)
    
    performance_tips = [
        "• **Token Efficiency**: Shorter prompts = lower costs and faster responses",
        "• **Clarity**: Specific instructions reduce ambiguity and improve results",
        "• **Examples**: Few-shot examples dramatically improve performance",
        "• **Structure**: Well-organized prompts are easier for models to follow",
        "• **Testing**: Always test prompts with multiple inputs and edge cases",
        "• **Iterative Improvement**: Refine prompts based on real usage patterns"
    ]
    
    print("Key Performance Optimization Tips:")
    for tip in performance_tips:
        print(tip)


def interactive_prompt_laboratory(llm, chat_llm):
    """
    Interactive prompt engineering laboratory.
    """
    print("\n" + "="*50)
    print("🧪 INTERACTIVE PROMPT LABORATORY")
    print("="*50)
    
    print("Welcome to the Prompt Engineering Lab!")
    print("Try different techniques and see how they affect the results.")
    print("Type 'help' for available techniques or 'quit' to exit.")
    
    techniques = {
        "1": "Few-shot prompting",
        "2": "Chain-of-thought reasoning",
        "3": "Role-based prompting",
        "4": "Structured output",
        "5": "Custom prompt"
    }
    
    while True:
        print("\nAvailable techniques:")
        for key, value in techniques.items():
            print(f"  {key}. {value}")
        
        choice = input("\nSelect a technique (1-5) or 'quit': ").strip()
        
        if choice.lower() in ['quit', 'q', 'exit']:
            print("Thanks for using the Prompt Laboratory!")
            break
        elif choice == "help":
            print("\nTechnique descriptions:")
            print("1. Few-shot: Learn from examples")
            print("2. Chain-of-thought: Step-by-step reasoning")
            print("3. Role-based: Expert perspectives")
            print("4. Structured: Formatted outputs")
            print("5. Custom: Your own prompt")
            continue
        elif choice not in techniques:
            print("Invalid choice. Please select 1-5 or 'quit'.")
            continue
        
        user_input = input("Enter your query/problem: ").strip()
        if not user_input:
            print("Empty input, skipping...")
            continue
        
        if choice == "1":
            # Few-shot example
            prompt = f"""Here are some examples of how to answer questions clearly:

Q: What is Python?
A: Python is a beginner-friendly programming language known for its simple syntax and versatility.

Q: What is machine learning?
A: Machine learning is a type of AI that learns patterns from data to make predictions or decisions.

Q: {user_input}
A:"""
            
        elif choice == "2":
            # Chain-of-thought
            prompt = f"""Let me think about this step by step:

Question: {user_input}

Step 1: Understanding the question
Step 2: Identifying key concepts
Step 3: Analyzing the components
Step 4: Forming a comprehensive answer

Answer:"""
            
        elif choice == "3":
            # Role-based
            prompt = f"""You are an expert teacher who excels at explaining complex topics in simple terms. 
Your student has asked: {user_input}

Provide a clear, helpful explanation:"""
            
        elif choice == "4":
            # Structured output
            prompt = f"""Please answer the following question using this structure:

Question: {user_input}

Answer:
- Main Point: [key concept]
- Explanation: [detailed explanation]
- Example: [practical example]
- Why it matters: [significance]

Response:"""
            
        elif choice == "5":
            # Custom prompt
            custom_prompt = input("Enter your custom prompt template (use {input} for the query): ").strip()
            if "{input}" in custom_prompt:
                prompt = custom_prompt.format(input=user_input)
            else:
                prompt = f"{custom_prompt}\n\nQuery: {user_input}"
        
        print(f"\nUsing technique: {techniques[choice]}")
        print("Generating response...")
        
        # Use chat model for most techniques
        if choice in ["3", "4"]:
            response = safe_llm_call(chat_llm, prompt)
        else:
            response = safe_llm_call(llm, prompt)
        
        if response:
            print(f"\nResponse:\n{response.strip()}")
        else:
            print("Failed to generate response.")
        
        print("\n" + "="*50)


def main():
    """
    Main function to run all prompt engineering demonstrations.
    """
    llm, chat_llm = setup_lesson()
    
    if not llm and not chat_llm:
        print("❌ Cannot proceed without LLM providers. Please check your setup.")
        return
    
    # Use the available model for demonstrations
    demo_llm = llm if llm else chat_llm
    demo_chat_llm = chat_llm if chat_llm else llm
    
    print(f"\n🔧 Using LLM: {type(demo_llm).__name__}")
    print(f"🔧 Using Chat LLM: {type(demo_chat_llm).__name__}")
    
    try:
        # Run all demonstrations
        demonstrate_few_shot_prompting(demo_llm)
        demonstrate_chain_of_thought(demo_llm)
        demonstrate_structured_output_parsing(demo_chat_llm)
        demonstrate_advanced_prompting_patterns(demo_chat_llm)
        demonstrate_prompt_optimization(demo_llm, demo_chat_llm)
        
        # Interactive lab
        print("\n🎉 Core demonstrations completed!")
        
        run_lab = input("\nWould you like to try the Interactive Prompt Laboratory? (y/n): ").strip().lower()
        if run_lab in ['y', 'yes']:
            interactive_prompt_laboratory(demo_llm, demo_chat_llm)
        
        print("\n✨ Lesson 2 completed! You've mastered advanced prompt engineering techniques.")
        print("\n📚 Key Skills Acquired:")
        print("   • Few-shot prompting with examples")
        print("   • Chain-of-thought reasoning")
        print("   • Structured output parsing")
        print("   • Role-based expert prompting")
        print("   • Prompt optimization strategies")
        
        print("\n🔗 Next: Lesson 3 - Chains & Sequential Processing")
        
    except KeyboardInterrupt:
        print("\n\n👋 Lesson interrupted. You can resume anytime!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    main() 