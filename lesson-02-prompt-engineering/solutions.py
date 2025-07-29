#!/usr/bin/env python3
"""
Lesson 2: Prompt Engineering - Exercise Solutions

These are reference implementations for the exercises in exercises.py.
Study these solutions to understand advanced prompt engineering patterns.
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
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

# Shared utilities
from utils.llm_setup import setup_llm_providers, get_preferred_llm
from utils.error_handling import safe_llm_call

# Set up LLM providers
providers = setup_llm_providers()
llm = get_preferred_llm(providers, prefer_chat=False) if providers else None
chat_llm = get_preferred_llm(providers, prefer_chat=True) if providers else None


def solution_1_code_documentation_generator():
    """
    Solution 1: Custom Few-Shot Template for Code Documentation
    """
    
    def create_code_documentation_template():
        """Create a few-shot template for generating function documentation."""
        
        examples = [
            {
                "function": "def add_numbers(a, b):\n    return a + b",
                "documentation": '''"""
Add two numbers together.

Args:
    a (int|float): The first number
    b (int|float): The second number

Returns:
    int|float: The sum of a and b

Example:
    >>> add_numbers(5, 3)
    8
    >>> add_numbers(2.5, 1.5)
    4.0
"""'''
            },
            {
                "function": "def find_max(numbers):\n    if not numbers:\n        return None\n    return max(numbers)",
                "documentation": '''"""
Find the maximum value in a list of numbers.

Args:
    numbers (List[int|float]): List of numbers to search

Returns:
    int|float|None: The maximum value, or None if list is empty

Raises:
    TypeError: If input is not a list or contains non-numeric values

Example:
    >>> find_max([1, 5, 3, 9, 2])
    9
    >>> find_max([])
    None
"""'''
            },
            {
                "function": "def process_user_data(user_dict, required_fields=None):\n    if required_fields is None:\n        required_fields = ['name', 'email']\n    \n    for field in required_fields:\n        if field not in user_dict:\n            raise ValueError(f'Missing required field: {field}')\n    \n    return {k: v.strip() if isinstance(v, str) else v for k, v in user_dict.items()}",
                "documentation": '''"""
Process and validate user data dictionary.

Args:
    user_dict (Dict[str, Any]): Dictionary containing user information
    required_fields (List[str], optional): List of required field names.
        Defaults to ['name', 'email'].

Returns:
    Dict[str, Any]: Processed user data with stripped string values

Raises:
    ValueError: If any required field is missing from user_dict

Example:
    >>> user = {'name': '  John Doe  ', 'email': 'john@example.com', 'age': 30}
    >>> process_user_data(user)
    {'name': 'John Doe', 'email': 'john@example.com', 'age': 30}
"""'''
            }
        ]
        
        example_template = PromptTemplate(
            input_variables=["function", "documentation"],
            template="Function:\n{function}\n\nDocumentation:\n{documentation}"
        )
        
        few_shot_template = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_template,
            prefix="Generate comprehensive Python docstrings following Google style:",
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
    # Email sending implementation
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
    print("SOLUTION 1: Code Documentation Generator")
    print("=" * 60)
    
    template = create_code_documentation_template()
    test_documentation_generator(template)


def solution_2_mathematical_chain_of_thought():
    """
    Solution 2: Mathematical Chain-of-Thought System
    """
    
    def create_math_cot_templates():
        """Create chain-of-thought templates for different math problem types."""
        templates = {}
        
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
        
        # Algebra Template
        algebra_template = """
Solve this algebra problem step by step:

Problem: {problem}

Step 1: Identify the type of algebraic equation
Step 2: Isolate the variable on one side
Step 3: Perform inverse operations systematically
Step 4: Simplify at each step
Step 5: Check the solution by substitution
Step 6: State the final answer

Solution:
        """
        
        # Geometry Template
        geometry_template = """
Solve this geometry problem step by step:

Problem: {problem}

Step 1: Identify the geometric shapes and given measurements
Step 2: Recall relevant formulas (area, perimeter, volume, etc.)
Step 3: Draw a diagram if helpful
Step 4: Substitute known values into formulas
Step 5: Calculate step by step
Step 6: Include proper units in the final answer

Solution:
        """
        
        # Percentage/Ratio Template
        percentage_template = """
Solve this percentage/ratio problem step by step:

Problem: {problem}

Step 1: Identify what percentage or ratio we're working with
Step 2: Determine the base amount and the percentage/ratio
Step 3: Convert percentages to decimals or fractions if needed
Step 4: Set up the calculation (multiplication, division, or proportion)
Step 5: Calculate the result
Step 6: Express the answer in appropriate format

Solution:
        """
        
        templates["word_problem"] = PromptTemplate(
            input_variables=["problem"],
            template=word_problem_template
        )
        
        templates["algebra"] = PromptTemplate(
            input_variables=["problem"],
            template=algebra_template
        )
        
        templates["geometry"] = PromptTemplate(
            input_variables=["problem"],
            template=geometry_template
        )
        
        templates["percentage"] = PromptTemplate(
            input_variables=["problem"],
            template=percentage_template
        )
        
        return templates
    
    def test_math_cot_system(templates):
        """Test the math CoT system with various problems."""
        test_problems = [
            ("word_problem", "A store is having a 25% off sale. If a jacket originally costs $80, and you have a $10 coupon, how much will you pay?"),
            ("algebra", "Solve for x: 3x + 7 = 2x - 5"),
            ("geometry", "Find the area of a circle with radius 6 cm. Use π = 3.14159."),
            ("percentage", "If 15% of a number is 45, what is the original number?"),
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
    print("SOLUTION 2: Mathematical Chain-of-Thought System")
    print("=" * 60)
    
    templates = create_math_cot_templates()
    test_math_cot_system(templates)


def solution_3_structured_data_extraction():
    """
    Solution 3: Structured Data Extraction with Pydantic
    """
    
    class ProductInfo(BaseModel):
        """Model for extracting product information from text."""
        name: str = Field(description="product name or title")
        price: Optional[float] = Field(description="product price in dollars", default=None)
        category: Optional[str] = Field(description="product category or type", default=None)
        rating: Optional[float] = Field(description="product rating (1-5 scale)", ge=1.0, le=5.0, default=None)
        availability: Optional[str] = Field(description="availability status (in stock, out of stock, limited)", default=None)
        brand: Optional[str] = Field(description="product brand or manufacturer", default=None)
    
    class EventInfo(BaseModel):
        """Model for extracting event information from text."""
        name: str = Field(description="event name or title")
        date: Optional[str] = Field(description="event date (YYYY-MM-DD format preferred)", default=None)
        time: Optional[str] = Field(description="event time (HH:MM format preferred)", default=None)
        location: Optional[str] = Field(description="event location or venue", default=None)
        organizer: Optional[str] = Field(description="event organizer or host", default=None)
        description: Optional[str] = Field(description="brief event description", default=None)
    
    def create_extraction_templates():
        """Create templates for structured data extraction."""
        templates = {}
        
        # Product extraction
        product_parser = PydanticOutputParser(pydantic_object=ProductInfo)
        product_template = PromptTemplate(
            template="""Extract product information from the following text and format it as JSON.

Text: {text}

{format_instructions}""",
            input_variables=["text"],
            partial_variables={"format_instructions": product_parser.get_format_instructions()}
        )
        templates["product"] = (product_template, product_parser)
        
        # Event extraction
        event_parser = PydanticOutputParser(pydantic_object=EventInfo)
        event_template = PromptTemplate(
            template="""Extract event information from the following text and format it as JSON.

Text: {text}

{format_instructions}""",
            input_variables=["text"],
            partial_variables={"format_instructions": event_parser.get_format_instructions()}
        )
        templates["event"] = (event_template, event_parser)
        
        return templates
    
    def test_data_extraction(templates):
        """Test structured data extraction with sample texts."""
        test_data = [
            ("product", "The new iPhone 15 Pro is available for $999. It has a 4.5-star rating and features a titanium design. Currently in stock at Apple stores."),
            ("product", "Samsung Galaxy S24 Ultra - $1,199.99. Premium smartphone with S Pen. Rating: 4.7/5 stars. Limited availability."),
            ("product", "MacBook Air M2 by Apple, priced at $1,199. Laptop category. 4.8 stars. Available for immediate shipping."),
            ("event", "Join us for the Annual Tech Conference on March 15, 2024, at 9:00 AM. Located at Convention Center, organized by TechCorp."),
            ("event", "Python Meetup: Building Web Apps with FastAPI. February 20, 2024, 6:30 PM. Downtown Library, hosted by Python Users Group.")
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
                        print(f"Extracted Data:")
                        for field, value in parsed_data.dict().items():
                            if value is not None:
                                print(f"  {field}: {value}")
                    except Exception as e:
                        print(f"Parsing failed: {e}")
                        print(f"Raw response: {response}")
            else:
                print("Template not implemented or chat LLM not available")
            print("-" * 50)
    
    print("\n" + "=" * 60)
    print("SOLUTION 3: Structured Data Extraction")
    print("=" * 60)
    
    templates = create_extraction_templates()
    test_data_extraction(templates)


def solution_4_expert_role_prompting():
    """
    Solution 4: Advanced Role-Based Expert Prompting
    """
    
    def create_expert_personas():
        """Create detailed expert personas for different domains."""
        experts = {
            "legal_advisor": {
                "persona": "You are a senior legal advisor with 20 years of experience in corporate law and compliance. You specialize in technology law, intellectual property, and international business regulations.",
                "expertise": ["contract law", "regulatory compliance", "risk assessment", "legal strategy", "IP protection"],
                "style": "precise, thorough, risk-aware, practical"
            },
            "medical_professional": {
                "persona": "You are a board-certified physician with expertise in internal medicine and 15 years of clinical experience. You stay current with medical research and evidence-based practices.",
                "expertise": ["diagnosis", "treatment planning", "preventive care", "medical research", "patient communication"],
                "style": "evidence-based, compassionate, clear, thorough"
            },
            "financial_advisor": {
                "persona": "You are a certified financial planner (CFP) with 18 years of experience helping individuals and businesses with investment strategy and financial planning.",
                "expertise": ["investment planning", "risk management", "retirement planning", "tax strategy", "market analysis"],
                "style": "analytical, conservative, goal-oriented, educational"
            },
            "environmental_scientist": {
                "persona": "You are an environmental scientist with a PhD in Environmental Engineering and 12 years of research experience in sustainability and climate change.",
                "expertise": ["environmental impact", "sustainability", "climate science", "policy analysis", "green technology"],
                "style": "data-driven, holistic, forward-thinking, collaborative"
            }
        }
        
        return experts
    
    def create_expert_consultation_system(experts):
        """Create a consultation system using expert personas."""
        templates = {}
        
        for expert_type, expert_data in experts.items():
            template = ChatPromptTemplate.from_messages([
                ("system", f"""{expert_data['persona']}

Your areas of expertise include: {', '.join(expert_data['expertise'])}

Communication style: {expert_data['style']}

When providing advice:
1. Draw from your extensive experience and expertise
2. Consider both immediate and long-term implications
3. Provide actionable recommendations
4. Highlight any risks or important considerations
5. Use your professional judgment and industry best practices"""),
                ("human", "Question: {question}\n\nPlease provide your expert advice and recommendations.")
            ])
            
            templates[expert_type] = template
        
        return templates
    
    def test_expert_consultation(templates):
        """Test expert consultation with domain-specific questions."""
        test_questions = [
            ("legal_advisor", "What are the key legal considerations when launching a SaaS product in multiple countries?"),
            ("medical_professional", "What are the most important preventive health measures for someone in their 40s?"),
            ("financial_advisor", "How should a 30-year-old with $50K income start planning for retirement?"),
            ("environmental_scientist", "What are the most effective ways a small business can reduce its carbon footprint?"),
            ("legal_advisor", "How should we structure our terms of service to minimize liability while being fair to users?")
        ]
        
        for expert_type, question in test_questions:
            print(f"\nConsulting: {expert_type.replace('_', ' ').title()}")
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
    print("SOLUTION 4: Expert Role-Based Prompting")
    print("=" * 60)
    
    experts = create_expert_personas()
    templates = create_expert_consultation_system(experts)
    test_expert_consultation(templates)


def solution_5_prompt_optimization_system():
    """
    Solution 5: Prompt Optimization and A/B Testing System
    """
    
    def create_prompt_variations(base_task):
        """Create multiple variations of a prompt for A/B testing."""
        variations = {
            "direct": f"{base_task}",
            
            "contextual": f"You are an expert educator who specializes in making complex topics accessible. {base_task}",
            
            "structured": f"""{base_task}

Please use this structure:
1. Simple definition
2. Key concepts
3. Real-world example
4. Why it matters

Keep your explanation clear and engaging.""",
            
            "analogical": f"""{base_task}

Use analogies and metaphors to make the concept relatable. Think of everyday situations that mirror the technical concepts.""",
            
            "interactive": f"""Imagine you're having a conversation with a curious friend who asked about this topic. {base_task}

Be conversational, use examples they can relate to, and anticipate follow-up questions.""",
            
            "step_by_step": f"""{base_task}

Break down your explanation into clear steps:
- Start with the basics
- Build complexity gradually  
- Use specific examples
- Connect to practical applications"""
        }
        
        return variations
    
    def evaluate_prompt_performance(variations, test_inputs):
        """Evaluate the performance of different prompt variations."""
        results = {}
        
        for variation_name, prompt_template in variations.items():
            results[variation_name] = {
                "responses": [],
                "avg_length": 0,
                "response_count": 0
            }
            
            for test_input in test_inputs:
                full_prompt = f"{prompt_template}: {test_input}"
                
                if llm:
                    response = safe_llm_call(llm, full_prompt)
                    if response:
                        results[variation_name]["responses"].append(response)
                        results[variation_name]["response_count"] += 1
            
            # Calculate average length
            if results[variation_name]["responses"]:
                total_length = sum(len(resp) for resp in results[variation_name]["responses"])
                results[variation_name]["avg_length"] = total_length / len(results[variation_name]["responses"])
        
        return results
    
    def display_optimization_results(results):
        """Display optimization results in a readable format."""
        print("\nPrompt Performance Analysis:")
        print("=" * 60)
        
        for variation, metrics in results.items():
            print(f"\n{variation.upper()} Variation:")
            print(f"  Successful responses: {metrics['response_count']}")
            print(f"  Average length: {metrics['avg_length']:.0f} characters")
            
            if metrics['responses']:
                print(f"  Sample response: {metrics['responses'][0][:150]}...")
        
        # Recommendations
        print(f"\n{'RECOMMENDATIONS':=^60}")
        if results:
            best_response_rate = max(r['response_count'] for r in results.values())
            best_variations = [name for name, r in results.items() if r['response_count'] == best_response_rate]
            
            print(f"• Best response rate: {', '.join(best_variations)}")
            
            avg_lengths = {name: r['avg_length'] for name, r in results.items() if r['avg_length'] > 0}
            if avg_lengths:
                most_detailed = max(avg_lengths, key=avg_lengths.get)
                most_concise = min(avg_lengths, key=avg_lengths.get)
                print(f"• Most detailed responses: {most_detailed}")
                print(f"• Most concise responses: {most_concise}")
    
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
        
        if llm:
            results = evaluate_prompt_performance(variations, test_inputs)
            display_optimization_results(results)
        else:
            print("\nPrompt Variations Created:")
            for name, prompt in variations.items():
                print(f"{name}: {prompt[:100]}...")
            print("\nLLM not available for performance testing")
    
    print("\n" + "=" * 60)
    print("SOLUTION 5: Prompt Optimization System")
    print("=" * 60)
    
    test_optimization_system()


def solution_6_advanced_output_parsing():
    """
    Solution 6: Advanced Output Parsing with Error Handling
    """
    
    class ComplexAnalysis(BaseModel):
        """Model for complex analysis outputs."""
        summary: str = Field(description="brief summary of the analysis")
        key_points: List[str] = Field(description="list of main points or findings")
        recommendations: List[str] = Field(description="actionable recommendations")
        confidence_score: float = Field(description="confidence level (0.0 to 1.0)", ge=0.0, le=1.0)
        methodology: Optional[str] = Field(description="analysis methodology used", default=None)
    
    def create_robust_parsing_system():
        """Create a robust parsing system with error handling."""
        parser = PydanticOutputParser(pydantic_object=ComplexAnalysis)
        
        # Create an output fixing parser that can handle malformed outputs
        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
        
        def parse_with_retry(response_text, max_retries=3):
            """Parse with retry logic and progressive fixing."""
            for attempt in range(max_retries):
                try:
                    if attempt == 0:
                        # First try: normal parsing
                        return parser.parse(response_text)
                    else:
                        # Subsequent tries: use fixing parser
                        return fixing_parser.parse(response_text)
                        
                except Exception as e:
                    print(f"Parse attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        # Last resort: return a default structure
                        return ComplexAnalysis(
                            summary="Parsing failed - raw response available",
                            key_points=[f"Raw response: {response_text[:200]}..."],
                            recommendations=["Review and correct the response format"],
                            confidence_score=0.0,
                            methodology="Error handling fallback"
                        )
            
            return None
        
        return parser, parse_with_retry
    
    def test_parsing_robustness():
        """Test parsing system with various inputs including malformed ones."""
        parser, parse_with_retry = create_robust_parsing_system()
        
        # Test cases with different levels of formatting issues
        test_cases = [
            {
                "description": "Well-formed JSON",
                "response": '''{
    "summary": "Analysis of market trends shows positive growth",
    "key_points": ["Market up 15%", "Tech sector leading", "Consumer confidence high"],
    "recommendations": ["Invest in tech stocks", "Monitor quarterly reports"],
    "confidence_score": 0.85,
    "methodology": "Statistical analysis of Q4 data"
}'''
            },
            {
                "description": "Missing field (methodology)",
                "response": '''{
    "summary": "Product analysis reveals strong performance",
    "key_points": ["Sales increased 20%", "Customer satisfaction high"],
    "recommendations": ["Expand marketing", "Increase production"],
    "confidence_score": 0.75
}'''
            },
            {
                "description": "Malformed JSON (missing quotes)",
                "response": '''{
    summary: Analysis shows mixed results,
    key_points: [Market volatility, Uncertain outlook],
    recommendations: [Wait and see, Diversify portfolio],
    confidence_score: 0.60
}'''
            },
            {
                "description": "Natural language response",
                "response": """The analysis summary shows that we have strong market position. 
The key points are: revenue growth of 25%, customer retention at 90%, and market share increased.
My recommendations are: expand to new markets, improve customer service, invest in R&D.
I'm 80% confident in this analysis using market research methodology."""
            }
        ]
        
        print("Testing parsing robustness with various input formats:")
        print("=" * 60)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}: {test_case['description']}")
            print(f"Input: {test_case['response'][:100]}...")
            
            if parser:
                try:
                    parsed_result = parse_with_retry(test_case['response'])
                    if parsed_result:
                        print(f"✅ Successfully parsed:")
                        print(f"   Summary: {parsed_result.summary}")
                        print(f"   Key Points: {len(parsed_result.key_points)} items")
                        print(f"   Recommendations: {len(parsed_result.recommendations)} items")
                        print(f"   Confidence: {parsed_result.confidence_score}")
                    else:
                        print("❌ Parsing failed completely")
                except Exception as e:
                    print(f"❌ Unexpected error: {e}")
            else:
                print("Parser not available")
            
            print("-" * 40)
    
    print("\n" + "=" * 60)
    print("SOLUTION 6: Advanced Output Parsing")
    print("=" * 60)
    
    test_parsing_robustness()


def solution_bonus_prompt_engineering_framework():
    """
    Bonus Solution: Complete Prompt Engineering Framework
    """
    
    class PromptEngineeringFramework:
        """A comprehensive framework for prompt engineering."""
        
        def __init__(self, llm, chat_llm):
            self.llm = llm
            self.chat_llm = chat_llm
            self.technique_map = {
                "simple": self._create_simple_prompt,
                "few_shot": self._create_few_shot_prompt,
                "chain_of_thought": self._create_cot_prompt,
                "role_based": self._create_role_prompt,
                "structured": self._create_structured_prompt
            }
        
        def auto_select_technique(self, task_type, complexity):
            """Automatically select the best prompting technique."""
            if complexity == "low":
                return "simple"
            elif task_type == "math" or task_type == "logic":
                return "chain_of_thought"
            elif task_type == "examples" or complexity == "medium":
                return "few_shot"
            elif task_type == "expert" or task_type == "domain_specific":
                return "role_based"
            elif task_type == "data_extraction":
                return "structured"
            else:
                return "few_shot"  # Default fallback
        
        def _create_simple_prompt(self, task, **kwargs):
            """Create a simple, direct prompt."""
            return f"{task}"
        
        def _create_few_shot_prompt(self, task, examples=None, **kwargs):
            """Create a few-shot prompt with examples."""
            if not examples:
                examples = [
                    "Example 1: Simple case with clear outcome",
                    "Example 2: Moderate complexity with detailed solution",
                    "Example 3: Complex case with step-by-step approach"
                ]
            
            prompt = f"Here are some examples:\n\n"
            for i, example in enumerate(examples, 1):
                prompt += f"{i}. {example}\n"
            prompt += f"\nNow apply this to: {task}"
            return prompt
        
        def _create_cot_prompt(self, task, **kwargs):
            """Create a chain-of-thought prompt."""
            return f"""Let me solve this step by step:

Task: {task}

Step 1: Understand the problem
Step 2: Identify key components
Step 3: Work through systematically
Step 4: Verify the solution

Solution:"""
        
        def _create_role_prompt(self, task, role="expert", **kwargs):
            """Create a role-based prompt."""
            role_descriptions = {
                "expert": "You are a highly experienced expert in this field",
                "teacher": "You are a skilled teacher who excels at clear explanations",
                "consultant": "You are a professional consultant providing strategic advice",
                "analyst": "You are a thorough analyst with attention to detail"
            }
            
            description = role_descriptions.get(role, role_descriptions["expert"])
            return f"{description}. {task}"
        
        def _create_structured_prompt(self, task, structure=None, **kwargs):
            """Create a structured prompt with specific output format."""
            if not structure:
                structure = [
                    "Main Point: [key insight]",
                    "Details: [supporting information]",
                    "Example: [concrete example]",
                    "Conclusion: [final thoughts]"
                ]
            
            prompt = f"{task}\n\nPlease format your response as follows:\n"
            for item in structure:
                prompt += f"- {item}\n"
            return prompt
        
        def optimize_prompt(self, base_prompt, optimization_goals):
            """Optimize a prompt for specific goals."""
            optimizations = []
            
            if "clarity" in optimization_goals:
                optimizations.append("Be clear and specific in your instructions.")
            if "brevity" in optimization_goals:
                optimizations.append("Keep responses concise and to the point.")
            if "examples" in optimization_goals:
                optimizations.append("Include concrete examples.")
            if "structure" in optimization_goals:
                optimizations.append("Organize your response with clear headings.")
            
            if optimizations:
                optimization_text = " ".join(optimizations)
                return f"{base_prompt}\n\nAdditional instructions: {optimization_text}"
            
            return base_prompt
        
        def analyze_performance(self, prompt, test_cases):
            """Analyze prompt performance across test cases."""
            results = {
                "total_cases": len(test_cases),
                "successful_responses": 0,
                "average_length": 0,
                "response_quality": "unknown"
            }
            
            responses = []
            for test_case in test_cases:
                full_prompt = f"{prompt}\n\nTest case: {test_case}"
                response = safe_llm_call(self.llm, full_prompt) if self.llm else None
                
                if response:
                    responses.append(response)
                    results["successful_responses"] += 1
            
            if responses:
                total_length = sum(len(r) for r in responses)
                results["average_length"] = total_length / len(responses)
                
                # Simple quality heuristic
                if results["successful_responses"] == len(test_cases):
                    results["response_quality"] = "excellent"
                elif results["successful_responses"] > len(test_cases) * 0.7:
                    results["response_quality"] = "good"
                else:
                    results["response_quality"] = "needs_improvement"
            
            return results
        
        def create_prompt(self, task, technique=None, **kwargs):
            """Create a prompt using the specified technique."""
            if technique is None:
                technique = self.auto_select_technique(
                    kwargs.get("task_type", "general"),
                    kwargs.get("complexity", "medium")
                )
            
            if technique in self.technique_map:
                return self.technique_map[technique](task, **kwargs)
            else:
                return self._create_simple_prompt(task)
    
    def test_framework():
        """Test the complete prompt engineering framework."""
        if llm or chat_llm:
            framework = PromptEngineeringFramework(llm, chat_llm)
            print("Framework initialized successfully!")
            
            # Test automatic technique selection
            print("\n1. Automatic Technique Selection:")
            test_scenarios = [
                ("Solve this math problem: 2x + 5 = 15", {"task_type": "math", "complexity": "medium"}),
                ("Explain quantum computing", {"task_type": "general", "complexity": "high"}),
                ("Write a business email", {"task_type": "general", "complexity": "low"})
            ]
            
            for task, params in test_scenarios:
                technique = framework.auto_select_technique(params["task_type"], params["complexity"])
                print(f"   Task: {task[:40]}...")
                print(f"   Selected technique: {technique}")
            
            # Test prompt creation
            print("\n2. Prompt Creation with Different Techniques:")
            sample_task = "Explain the benefits of renewable energy"
            
            for technique in ["simple", "few_shot", "chain_of_thought", "role_based"]:
                prompt = framework.create_prompt(sample_task, technique=technique, role="expert")
                print(f"\n   {technique.upper()}:")
                print(f"   {prompt[:100]}...")
            
            # Test optimization
            print("\n3. Prompt Optimization:")
            base_prompt = "Explain machine learning"
            goals = ["clarity", "examples", "structure"]
            optimized = framework.optimize_prompt(base_prompt, goals)
            print(f"   Original: {base_prompt}")
            print(f"   Optimized: {optimized}")
            
            # Test performance analysis
            print("\n4. Performance Analysis:")
            test_cases = ["neural networks", "deep learning", "AI ethics"]
            simple_prompt = framework.create_prompt("Explain this concept", technique="simple")
            
            if llm:
                results = framework.analyze_performance(simple_prompt, test_cases[:1])  # Test with one case
                print(f"   Successful responses: {results['successful_responses']}/{results['total_cases']}")
                print(f"   Quality assessment: {results['response_quality']}")
            else:
                print("   LLM not available for performance testing")
            
        else:
            print("LLM not available for framework testing")
    
    print("\n" + "=" * 60)
    print("BONUS SOLUTION: Complete Prompt Engineering Framework")
    print("=" * 60)
    
    test_framework()


if __name__ == "__main__":
    """
    Run all solutions when the script is executed directly.
    """
    print("🦜🔗 LangChain Course - Lesson 2 Solutions")
    print("=" * 60)
    
    if not (llm or chat_llm):
        print("⚠️  No LLM providers available. Solutions will show structure and logic.")
        print("Configure OPENAI_API_KEY in your .env file to see full functionality.\n")
    
    # Run all solutions
    solutions = [
        solution_1_code_documentation_generator,
        solution_2_mathematical_chain_of_thought,
        solution_3_structured_data_extraction,
        solution_4_expert_role_prompting,
        solution_5_prompt_optimization_system,
        solution_6_advanced_output_parsing,
        solution_bonus_prompt_engineering_framework
    ]
    
    for i, solution in enumerate(solutions, 1):
        print(f"\n{'='*20} SOLUTION {i} {'='*20}")
        try:
            solution()
        except Exception as e:
            print(f"Solution {i} error: {e}")
    
    print("\n🎉 All solutions demonstrated!")
    print("These implementations show advanced prompt engineering patterns.")
    print("Practice with these techniques to master prompt engineering!") 