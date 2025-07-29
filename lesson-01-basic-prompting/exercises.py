#!/usr/bin/env python3
"""
Lesson 1: Basic Prompting - Practice Exercises

Complete these exercises to reinforce your understanding of basic LangChain concepts.
Each exercise builds upon the previous one and includes hints to guide you.

Instructions:
1. Implement each exercise function
2. Run this file to test your implementations
3. Check solutions.py for reference implementations
"""

import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv()

# Import your preferred LLM provider
try:
    from langchain_openai import OpenAI, ChatOpenAI
    llm = OpenAI(temperature=0.7) if os.getenv("OPENAI_API_KEY") else None
    chat_llm = ChatOpenAI(temperature=0.7) if os.getenv("OPENAI_API_KEY") else None
except ImportError:
    llm = None
    chat_llm = None
    print("Please install langchain-openai or configure another provider")


def exercise_1_basic_llm_setup():
    """
    Exercise 1: Basic LLM Setup and Text Generation
    
    Task: Create a function that:
    1. Takes a topic as input
    2. Generates a brief explanation (2-3 sentences) about that topic
    3. Returns the generated text
    
    Hint: Use the llm.invoke() method with a simple prompt
    """
    
    def explain_topic(topic: str) -> str:
        """
        Generate a brief explanation about the given topic.
        
        Args:
            topic (str): The topic to explain
            
        Returns:
            str: Generated explanation
        """
        # TODO: Implement this function
        # Use the global 'llm' variable to generate an explanation
        pass
    
    # Test your implementation
    if llm:
        result = explain_topic("machine learning")
        print(f"Exercise 1 Result: {result}")
    else:
        print("Exercise 1: LLM not available")


def exercise_2_prompt_template():
    """
    Exercise 2: Create a Personal Assistant Prompt Template
    
    Task: Create a prompt template for a personal assistant that:
    1. Takes a task description and priority level (high/medium/low)
    2. Generates a formatted response with time estimates and suggestions
    3. Uses PromptTemplate with variables
    
    Hint: Use PromptTemplate with input_variables and template string
    """
    
    def create_task_assistant(task: str, priority: str) -> str:
        """
        Create a task management response using a prompt template.
        
        Args:
            task (str): Description of the task
            priority (str): Priority level (high/medium/low)
            
        Returns:
            str: Formatted assistant response
        """
        # TODO: Create a PromptTemplate
        # Include variables for task, priority
        # Make the template provide time estimates and suggestions
        pass
    
    # Test your implementation
    if llm:
        result = create_task_assistant("Prepare presentation for client meeting", "high")
        print(f"Exercise 2 Result: {result}")
    else:
        print("Exercise 2: LLM not available")


def exercise_3_chat_conversation():
    """
    Exercise 3: Multi-turn Conversation System
    
    Task: Create a simple chat system that:
    1. Maintains conversation context across multiple exchanges
    2. Uses SystemMessage to set the AI's personality
    3. Handles a conversation flow with at least 3 exchanges
    
    Hint: Use a list to store messages and append new ones for each turn
    """
    
    def simulate_conversation():
        """
        Simulate a multi-turn conversation with context.
        
        Returns:
            list: List of all messages in the conversation
        """
        # TODO: Create a conversation with:
        # 1. SystemMessage setting the AI as a helpful coding mentor
        # 2. Human asking about learning Python
        # 3. AI response
        # 4. Follow-up human question about specific Python topics
        # 5. Final AI response
        
        messages = []
        
        # Add your messages here
        
        return messages
    
    # Test your implementation
    if chat_llm:
        conversation = simulate_conversation()
        print("Exercise 3 - Conversation:")
        for i, message in enumerate(conversation):
            print(f"  {i+1}. {type(message).__name__}: {message.content}")
    else:
        print("Exercise 3: Chat LLM not available")


def exercise_4_creative_writing():
    """
    Exercise 4: Creative Writing Generator
    
    Task: Create a story generator that:
    1. Takes character name, setting, and genre as inputs
    2. Uses ChatPromptTemplate for structured prompts
    3. Generates a short story opening (2-3 paragraphs)
    
    Hint: Use ChatPromptTemplate.from_messages() with system and human messages
    """
    
    def generate_story_opening(character: str, setting: str, genre: str) -> str:
        """
        Generate a creative story opening using chat prompt templates.
        
        Args:
            character (str): Main character name
            setting (str): Story setting/location
            genre (str): Story genre (mystery, sci-fi, fantasy, etc.)
            
        Returns:
            str: Generated story opening
        """
        # TODO: Create a ChatPromptTemplate with:
        # 1. System message defining the AI as a creative writer
        # 2. Human message with variables for character, setting, genre
        pass
    
    # Test your implementation
    if chat_llm:
        story = generate_story_opening("Alex", "space station", "sci-fi")
        print(f"Exercise 4 Result: {story}")
    else:
        print("Exercise 4: Chat LLM not available")


def exercise_5_advanced_template():
    """
    Exercise 5: Advanced Prompt Template with Multiple Variables
    
    Task: Create a comprehensive prompt template for a learning assistant that:
    1. Takes subject, difficulty level, learning style, and time available
    2. Generates a personalized study plan
    3. Uses conditional formatting based on difficulty level
    
    Hint: Use string formatting and conditional logic in your template
    """
    
    def create_study_plan(subject: str, difficulty: str, learning_style: str, time_minutes: int) -> str:
        """
        Create a personalized study plan using advanced prompt templates.
        
        Args:
            subject (str): Subject to study
            difficulty (str): Difficulty level (beginner/intermediate/advanced)
            learning_style (str): Preferred learning style (visual/auditory/kinesthetic)
            time_minutes (int): Available time in minutes
            
        Returns:
            str: Personalized study plan
        """
        # TODO: Create an advanced template that:
        # 1. Adapts content based on difficulty level
        # 2. Suggests learning methods based on learning style
        # 3. Structures plan based on available time
        pass
    
    # Test your implementation
    if llm:
        plan = create_study_plan("Python programming", "beginner", "visual", 60)
        print(f"Exercise 5 Result: {plan}")
    else:
        print("Exercise 5: LLM not available")


def exercise_6_error_handling():
    """
    Exercise 6: Error Handling and Fallback
    
    Task: Create a robust function that:
    1. Tries to use the primary LLM
    2. Falls back to a chat model if available
    3. Provides a default response if no models are available
    4. Handles API errors gracefully
    
    Hint: Use try-except blocks and check model availability
    """
    
    def robust_text_generation(prompt: str) -> str:
        """
        Generate text with error handling and fallback mechanisms.
        
        Args:
            prompt (str): Input prompt
            
        Returns:
            str: Generated text or fallback message
        """
        # TODO: Implement robust generation with:
        # 1. Try primary LLM first
        # 2. Fall back to chat model
        # 3. Provide helpful error messages
        # 4. Handle exceptions gracefully
        pass
    
    # Test your implementation
    result = robust_text_generation("Explain the importance of error handling in software development")
    print(f"Exercise 6 Result: {result}")


def bonus_exercise_interactive_cli():
    """
    Bonus Exercise: Interactive Command Line Interface
    
    Task: Create an interactive CLI that:
    1. Allows users to choose between different prompt styles
    2. Lets users input custom prompts
    3. Displays results with proper formatting
    4. Includes a help system
    
    This is a more advanced exercise combining multiple concepts.
    """
    
    def interactive_prompt_tool():
        """
        Create an interactive prompt tool with multiple options.
        """
        # TODO: Implement an interactive CLI with:
        # 1. Menu system for different prompt types
        # 2. User input handling
        # 3. Results formatting
        # 4. Help and exit options
        pass
    
    print("Bonus Exercise: Run interactive_prompt_tool() to test your CLI")


if __name__ == "__main__":
    """
    Run all exercises when the script is executed directly.
    """
    print("🦜🔗 LangChain Course - Lesson 1 Exercises")
    print("=" * 50)
    
    if not (llm or chat_llm):
        print("⚠️  No LLM providers available. Please configure your API keys.")
        print("Set OPENAI_API_KEY in your .env file or use another provider.")
        exit(1)
    
    print("\n🏋️ Starting exercises...")
    
    # Run all exercises
    exercises = [
        exercise_1_basic_llm_setup,
        exercise_2_prompt_template,
        exercise_3_chat_conversation,
        exercise_4_creative_writing,
        exercise_5_advanced_template,
        exercise_6_error_handling
    ]
    
    for i, exercise in enumerate(exercises, 1):
        print(f"\n--- Exercise {i} ---")
        try:
            exercise()
        except Exception as e:
            print(f"Exercise {i} error: {e}")
    
    print("\n🎯 Exercises completed!")
    print("Check solutions.py for reference implementations.")
    print("Try the bonus exercise for additional practice!") 