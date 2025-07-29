#!/usr/bin/env python3
"""
Lesson 1: Basic Prompting - Exercise Solutions

These are reference implementations for the exercises in exercises.py.
Study these solutions to understand the concepts and compare with your implementations.
"""

import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv()

# Import LLM providers with error handling
try:
    from langchain_openai import OpenAI, ChatOpenAI
    llm = OpenAI(temperature=0.7) if os.getenv("OPENAI_API_KEY") else None
    chat_llm = ChatOpenAI(temperature=0.7) if os.getenv("OPENAI_API_KEY") else None
except ImportError:
    llm = None
    chat_llm = None


def solution_1_basic_llm_setup():
    """
    Solution 1: Basic LLM Setup and Text Generation
    """
    
    def explain_topic(topic: str) -> str:
        """Generate a brief explanation about the given topic."""
        if not llm:
            return "Error: No LLM available"
        
        prompt = f"Explain {topic} in 2-3 sentences. Be clear and concise."
        
        try:
            result = llm.invoke(prompt)
            return result.strip()
        except Exception as e:
            return f"Error generating explanation: {e}"
    
    # Test the solution
    if llm:
        result = explain_topic("machine learning")
        print(f"Solution 1 Result:\n{result}\n")
    else:
        print("Solution 1: LLM not available\n")


def solution_2_prompt_template():
    """
    Solution 2: Personal Assistant Prompt Template
    """
    
    def create_task_assistant(task: str, priority: str) -> str:
        """Create a task management response using a prompt template."""
        if not llm:
            return "Error: No LLM available"
        
        template = """
        You are a personal productivity assistant. Help with the following task:

        Task: {task}
        Priority: {priority}

        Please provide:
        1. Estimated time to complete
        2. Key steps to accomplish this task
        3. Any helpful tips or suggestions
        4. Potential obstacles and how to overcome them

        Format your response in a clear, actionable manner.
        """
        
        prompt = PromptTemplate(
            input_variables=["task", "priority"],
            template=template
        )
        
        formatted_prompt = prompt.format(task=task, priority=priority)
        
        try:
            result = llm.invoke(formatted_prompt)
            return result.strip()
        except Exception as e:
            return f"Error: {e}"
    
    # Test the solution
    if llm:
        result = create_task_assistant("Prepare presentation for client meeting", "high")
        print(f"Solution 2 Result:\n{result}\n")
    else:
        print("Solution 2: LLM not available\n")


def solution_3_chat_conversation():
    """
    Solution 3: Multi-turn Conversation System
    """
    
    def simulate_conversation():
        """Simulate a multi-turn conversation with context."""
        if not chat_llm:
            return [SystemMessage(content="Chat LLM not available")]
        
        messages = [
            SystemMessage(content="You are a helpful coding mentor. Be encouraging, clear, and provide practical advice. Keep responses concise but informative."),
            HumanMessage(content="I'm new to programming and want to learn Python. Where should I start?")
        ]
        
        try:
            # Get first AI response
            response1 = chat_llm.invoke(messages)
            messages.append(AIMessage(content=response1.content))
            
            # Add follow-up question
            messages.append(HumanMessage(content="That's helpful! Can you recommend some specific Python topics I should focus on first, and any good resources?"))
            
            # Get final AI response
            response2 = chat_llm.invoke(messages)
            messages.append(AIMessage(content=response2.content))
            
        except Exception as e:
            messages.append(AIMessage(content=f"Error in conversation: {e}"))
        
        return messages
    
    # Test the solution
    if chat_llm:
        conversation = simulate_conversation()
        print("Solution 3 - Conversation:")
        for i, message in enumerate(conversation):
            print(f"  {i+1}. {type(message).__name__}: {message.content[:100]}...")
        print()
    else:
        print("Solution 3: Chat LLM not available\n")


def solution_4_creative_writing():
    """
    Solution 4: Creative Writing Generator
    """
    
    def generate_story_opening(character: str, setting: str, genre: str) -> str:
        """Generate a creative story opening using chat prompt templates."""
        if not chat_llm:
            return "Error: Chat LLM not available"
        
        chat_template = ChatPromptTemplate.from_messages([
            ("system", "You are a creative writer specializing in {genre} stories. Write engaging, vivid openings that draw readers in immediately. Use descriptive language and create atmosphere."),
            ("human", "Write a compelling story opening (2-3 paragraphs) featuring a character named {character} in the setting of {setting}. Make it a {genre} story with an intriguing hook.")
        ])
        
        try:
            formatted_messages = chat_template.format_messages(
                character=character,
                setting=setting,
                genre=genre
            )
            
            response = chat_llm.invoke(formatted_messages)
            return response.content
        except Exception as e:
            return f"Error generating story: {e}"
    
    # Test the solution
    if chat_llm:
        story = generate_story_opening("Alex", "space station", "sci-fi")
        print(f"Solution 4 Result:\n{story}\n")
    else:
        print("Solution 4: Chat LLM not available\n")


def solution_5_advanced_template():
    """
    Solution 5: Advanced Prompt Template with Multiple Variables
    """
    
    def create_study_plan(subject: str, difficulty: str, learning_style: str, time_minutes: int) -> str:
        """Create a personalized study plan using advanced prompt templates."""
        if not llm:
            return "Error: No LLM available"
        
        # Determine study approach based on difficulty
        difficulty_approaches = {
            "beginner": "Focus on fundamentals and basic concepts. Start with simple examples.",
            "intermediate": "Build on existing knowledge. Include practical applications.",
            "advanced": "Dive deep into complex topics. Emphasize advanced techniques and edge cases."
        }
        
        # Learning style suggestions
        style_methods = {
            "visual": "diagrams, charts, code examples, mind maps, and visual aids",
            "auditory": "explanations, discussions, verbal repetition, and audio resources",
            "kinesthetic": "hands-on coding, interactive exercises, and practical projects"
        }
        
        approach = difficulty_approaches.get(difficulty, "Adapt content to your current level")
        methods = style_methods.get(learning_style, "varied learning methods")
        
        template = """
        Create a personalized {time_minutes}-minute study plan for learning {subject}.
        
        Student Profile:
        - Difficulty Level: {difficulty}
        - Learning Style: {learning_style}
        - Available Time: {time_minutes} minutes
        
        Study Approach: {approach}
        Recommended Methods: {methods}
        
        Please provide:
        1. Time breakdown for different activities
        2. Specific topics to cover in order
        3. Learning activities suited to the learning style
        4. Quick assessment or practice suggestions
        5. Next steps for continued learning
        
        Make the plan actionable and realistic for the time available.
        """
        
        prompt = PromptTemplate(
            input_variables=["subject", "difficulty", "learning_style", "time_minutes", "approach", "methods"],
            template=template
        )
        
        try:
            formatted_prompt = prompt.format(
                subject=subject,
                difficulty=difficulty,
                learning_style=learning_style,
                time_minutes=time_minutes,
                approach=approach,
                methods=methods
            )
            
            result = llm.invoke(formatted_prompt)
            return result.strip()
        except Exception as e:
            return f"Error creating study plan: {e}"
    
    # Test the solution
    if llm:
        plan = create_study_plan("Python programming", "beginner", "visual", 60)
        print(f"Solution 5 Result:\n{plan}\n")
    else:
        print("Solution 5: LLM not available\n")


def solution_6_error_handling():
    """
    Solution 6: Error Handling and Fallback
    """
    
    def robust_text_generation(prompt: str) -> str:
        """Generate text with error handling and fallback mechanisms."""
        
        # Try primary LLM first
        if llm:
            try:
                result = llm.invoke(prompt)
                return f"✅ Primary LLM: {result.strip()}"
            except Exception as e:
                print(f"⚠️ Primary LLM failed: {e}")
        
        # Fall back to chat model
        if chat_llm:
            try:
                messages = [HumanMessage(content=prompt)]
                response = chat_llm.invoke(messages)
                return f"✅ Fallback Chat LLM: {response.content.strip()}"
            except Exception as e:
                print(f"⚠️ Chat LLM fallback failed: {e}")
        
        # Final fallback - provide helpful default response
        return """
        ❌ No LLM providers available. 
        
        Error handling in software development is crucial because:
        1. It prevents applications from crashing unexpectedly
        2. It provides meaningful feedback to users when things go wrong
        3. It allows developers to anticipate and handle edge cases
        4. It improves user experience and application reliability
        
        To implement proper error handling:
        - Use try-except blocks for risky operations
        - Provide fallback mechanisms when possible
        - Log errors for debugging purposes
        - Give users clear, actionable error messages
        """
    
    # Test the solution
    result = robust_text_generation("Explain the importance of error handling in software development")
    print(f"Solution 6 Result:\n{result}\n")


def solution_bonus_interactive_cli():
    """
    Bonus Solution: Interactive Command Line Interface
    """
    
    def interactive_prompt_tool():
        """Create an interactive prompt tool with multiple options."""
        
        def show_menu():
            print("\n" + "="*50)
            print("🛠️  INTERACTIVE PROMPT TOOL")
            print("="*50)
            print("1. Simple Question Answering")
            print("2. Creative Writing")
            print("3. Code Explanation")
            print("4. Study Plan Generator")
            print("5. Custom Prompt")
            print("6. Help")
            print("7. Exit")
            print("-" * 50)
        
        def simple_qa():
            question = input("Ask any question: ").strip()
            if not question:
                print("No question provided.")
                return
            
            if llm:
                try:
                    response = llm.invoke(f"Answer this question clearly and concisely: {question}")
                    print(f"\n📝 Answer: {response}")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("LLM not available for Q&A")
        
        def creative_writing():
            genre = input("Enter genre (sci-fi, fantasy, mystery, etc.): ").strip()
            character = input("Enter main character name: ").strip()
            setting = input("Enter setting: ").strip()
            
            if not all([genre, character, setting]):
                print("All fields are required for creative writing.")
                return
            
            if chat_llm:
                try:
                    template = ChatPromptTemplate.from_messages([
                        ("system", f"You are a creative writer specializing in {genre} stories."),
                        ("human", f"Write a story opening featuring {character} in {setting}.")
                    ])
                    messages = template.format_messages()
                    response = chat_llm.invoke(messages)
                    print(f"\n📖 Story Opening:\n{response.content}")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Chat LLM not available for creative writing")
        
        def code_explanation():
            code = input("Enter code to explain: ").strip()
            if not code:
                print("No code provided.")
                return
            
            prompt = f"Explain this code step by step:\n\n{code}"
            
            if llm:
                try:
                    response = llm.invoke(prompt)
                    print(f"\n💻 Code Explanation:\n{response}")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("LLM not available for code explanation")
        
        def custom_prompt():
            prompt = input("Enter your custom prompt: ").strip()
            if not prompt:
                print("No prompt provided.")
                return
            
            if llm:
                try:
                    response = llm.invoke(prompt)
                    print(f"\n🎯 Response:\n{response}")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("LLM not available for custom prompts")
        
        def show_help():
            print("""
            📚 HELP - How to use this tool:
            
            1. Simple Q&A: Ask any question and get a direct answer
            2. Creative Writing: Generate story openings with custom parameters
            3. Code Explanation: Get detailed explanations of code snippets
            4. Study Plan: Create personalized learning plans
            5. Custom Prompt: Enter any prompt for free-form generation
            6. Help: Show this help message
            7. Exit: Quit the application
            
            💡 Tips:
            - Be specific in your prompts for better results
            - Try different options to explore LangChain capabilities
            - Check your .env file if you get "LLM not available" errors
            """)
        
        # Main loop
        while True:
            show_menu()
            choice = input("Enter your choice (1-7): ").strip()
            
            if choice == "1":
                simple_qa()
            elif choice == "2":
                creative_writing()
            elif choice == "3":
                code_explanation()
            elif choice == "4":
                print("Study plan generator would use solution_5_advanced_template()")
            elif choice == "5":
                custom_prompt()
            elif choice == "6":
                show_help()
            elif choice == "7":
                print("👋 Goodbye! Thanks for using the interactive prompt tool.")
                break
            else:
                print("Invalid choice. Please enter 1-7.")
            
            input("\nPress Enter to continue...")
    
    print("Bonus Solution: Interactive CLI implemented!")
    print("Call interactive_prompt_tool() to start the interactive interface.")
    
    # Uncomment to run interactively:
    # interactive_prompt_tool()


if __name__ == "__main__":
    """
    Run all solutions when the script is executed directly.
    """
    print("🦜🔗 LangChain Course - Lesson 1 Solutions")
    print("=" * 60)
    
    if not (llm or chat_llm):
        print("⚠️  No LLM providers available. Solutions will show error handling.")
        print("Configure OPENAI_API_KEY in your .env file to see full functionality.\n")
    
    # Run all solutions
    solutions = [
        solution_1_basic_llm_setup,
        solution_2_prompt_template,
        solution_3_chat_conversation,
        solution_4_creative_writing,
        solution_5_advanced_template,
        solution_6_error_handling,
        solution_bonus_interactive_cli
    ]
    
    for i, solution in enumerate(solutions, 1):
        print(f"\n{'='*20} SOLUTION {i} {'='*20}")
        try:
            solution()
        except Exception as e:
            print(f"Solution {i} error: {e}")
    
    print("\n🎉 All solutions demonstrated!")
    print("Compare these with your implementations in exercises.py") 