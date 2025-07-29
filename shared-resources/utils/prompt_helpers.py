"""
Prompt Helper Utilities

Common functions for creating and managing prompts across lessons.
"""

from typing import List, Dict, Any, Optional
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage


def create_prompt_template(template: str, variables: List[str]) -> PromptTemplate:
    """
    Create a PromptTemplate with validation.
    
    Args:
        template (str): The prompt template string
        variables (List[str]): List of variable names used in the template
        
    Returns:
        PromptTemplate: The created prompt template
    """
    return PromptTemplate(
        input_variables=variables,
        template=template
    )


def create_chat_prompt_template(messages: List[tuple]) -> ChatPromptTemplate:
    """
    Create a ChatPromptTemplate from a list of (role, content) tuples.
    
    Args:
        messages (List[tuple]): List of (role, content) tuples
        
    Returns:
        ChatPromptTemplate: The created chat prompt template
    """
    return ChatPromptTemplate.from_messages(messages)


def format_response(response: Any, max_length: Optional[int] = None) -> str:
    """
    Format LLM response for display.
    
    Args:
        response (Any): The LLM response
        max_length (Optional[int]): Maximum length to display
        
    Returns:
        str: Formatted response string
    """
    if hasattr(response, 'content'):
        content = response.content
    else:
        content = str(response)
    
    if max_length and len(content) > max_length:
        content = content[:max_length] + "..."
    
    return content.strip()


def create_conversation_template(system_prompt: str, conversation_turns: List[str]) -> ChatPromptTemplate:
    """
    Create a conversation template with system prompt and multiple turns.
    
    Args:
        system_prompt (str): The system message
        conversation_turns (List[str]): List of human messages
        
    Returns:
        ChatPromptTemplate: The conversation template
    """
    messages = [("system", system_prompt)]
    
    for i, turn in enumerate(conversation_turns):
        if i % 2 == 0:  # Even indices are human messages
            messages.append(("human", turn))
        else:  # Odd indices are AI messages
            messages.append(("ai", turn))
    
    return ChatPromptTemplate.from_messages(messages)


def validate_template_variables(template: str, provided_variables: Dict[str, Any]) -> bool:
    """
    Validate that all template variables are provided.
    
    Args:
        template (str): The template string
        provided_variables (Dict[str, Any]): Variables to check
        
    Returns:
        bool: True if all variables are provided
    """
    import re
    
    # Find all variables in template using regex
    variables_in_template = set(re.findall(r'\{(\w+)\}', template))
    provided_vars = set(provided_variables.keys())
    
    missing_vars = variables_in_template - provided_vars
    
    if missing_vars:
        print(f"Missing variables: {missing_vars}")
        return False
    
    return True


def create_few_shot_template(examples: List[Dict[str, str]], 
                           input_key: str, 
                           output_key: str,
                           task_description: str = "") -> PromptTemplate:
    """
    Create a few-shot prompting template.
    
    Args:
        examples (List[Dict[str, str]]): List of example dictionaries
        input_key (str): Key for input in examples
        output_key (str): Key for output in examples
        task_description (str): Optional task description
        
    Returns:
        PromptTemplate: Few-shot prompt template
    """
    template_parts = []
    
    if task_description:
        template_parts.append(task_description)
        template_parts.append("")
    
    template_parts.append("Here are some examples:")
    template_parts.append("")
    
    for i, example in enumerate(examples, 1):
        template_parts.append(f"Example {i}:")
        template_parts.append(f"Input: {example[input_key]}")
        template_parts.append(f"Output: {example[output_key]}")
        template_parts.append("")
    
    template_parts.append("Now, please provide the output for this input:")
    template_parts.append("Input: {input}")
    template_parts.append("Output:")
    
    template = "\n".join(template_parts)
    
    return PromptTemplate(
        input_variables=["input"],
        template=template
    )


def create_chain_of_thought_template(problem_type: str = "general") -> PromptTemplate:
    """
    Create a chain-of-thought prompting template.
    
    Args:
        problem_type (str): Type of problem (general, math, logic, etc.)
        
    Returns:
        PromptTemplate: Chain-of-thought prompt template
    """
    if problem_type == "math":
        template = """
        Solve this math problem step by step:
        
        Problem: {problem}
        
        Let me think through this step by step:
        1. First, I'll identify what we know and what we need to find.
        2. Then, I'll determine the appropriate method or formula to use.
        3. Next, I'll work through the calculations step by step.
        4. Finally, I'll verify my answer makes sense.
        
        Step-by-step solution:
        """
    elif problem_type == "logic":
        template = """
        Solve this logical reasoning problem step by step:
        
        Problem: {problem}
        
        Let me reason through this step by step:
        1. First, I'll identify the key information and constraints.
        2. Then, I'll consider what logical rules or principles apply.
        3. Next, I'll work through the logical deductions.
        4. Finally, I'll arrive at the conclusion.
        
        Step-by-step reasoning:
        """
    else:
        template = """
        Let's think about this problem step by step:
        
        Problem: {problem}
        
        I'll break this down into steps:
        1. First, let me understand what the problem is asking.
        2. Then, I'll identify the key factors to consider.
        3. Next, I'll work through each aspect systematically.
        4. Finally, I'll synthesize my findings into a conclusion.
        
        Step-by-step analysis:
        """
    
    return PromptTemplate(
        input_variables=["problem"],
        template=template
    )


if __name__ == "__main__":
    """Test the prompt helper utilities."""
    print("🦜🔗 LangChain Course - Prompt Helpers Test")
    print("=" * 50)
    
    # Test basic template creation
    template = create_prompt_template(
        "Tell me about {topic} in {style} style.",
        ["topic", "style"]
    )
    
    formatted = template.format(topic="Python", style="beginner-friendly")
    print(f"Basic template: {formatted}")
    
    # Test few-shot template
    examples = [
        {"input": "sky", "output": "blue"},
        {"input": "grass", "output": "green"},
        {"input": "sun", "output": "yellow"}
    ]
    
    few_shot = create_few_shot_template(examples, "input", "output", "Identify the color:")
    print(f"\nFew-shot template created with {len(examples)} examples")
    
    # Test chain-of-thought
    cot = create_chain_of_thought_template("math")
    print(f"\nChain-of-thought template for math problems created")
    
    print("\n✅ All prompt helper tests completed!") 