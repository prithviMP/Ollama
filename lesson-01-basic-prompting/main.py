#!/usr/bin/env python3
"""
Lesson 1: Basic Prompting with LangChain

This lesson covers:
1. Setting up LangChain with different LLM providers
2. Basic prompt templates and variables
3. Chat models for conversational AI
4. Error handling and best practices

Author: LangChain Course
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain imports
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_openai import OpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

# Optional imports with error handling
try:
    from langchain_openai import OpenAI, ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI not available. Install with: pip install langchain-openai")

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠️  Anthropic not available. Install with: pip install langchain-anthropic")

try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️  Ollama not available. Install with: pip install langchain-ollama")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("⚠️  Google Gemini not available. Install with: pip install langchain-google-genai")

try:
    from langchain_deepseek import ChatDeepSeek
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    print("⚠️  DeepSeek not available. Install with: pip install langchain-deepseek")

# OpenRouter can use OpenAI client with custom base URL
try:
    from langchain_openai import ChatOpenAI
    OPENROUTER_AVAILABLE = OPENAI_AVAILABLE  # Reuse OpenAI availability
except ImportError:
    OPENROUTER_AVAILABLE = False
    print("⚠️  OpenRouter not available. Install with: pip install langchain-openai")


def setup_llm_providers():
    """
    Set up different LLM providers based on available API keys.
    Returns a dictionary of available providers.
    """
    providers = {}
    
    # OpenAI Setup
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            providers["openai"] = OpenAI(
                temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.7)),
                max_tokens=int(os.getenv("MAX_TOKENS", 1000))
            )
            providers["chat_openai"] = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.7))
            )
            print("✅ OpenAI configured successfully")
        except Exception as e:
            print(f"❌ OpenAI configuration failed: {e}")
    
    # Anthropic Setup
    if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
        try:
            providers["anthropic"] = ChatAnthropic(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.7))
            )
            print("✅ Anthropic configured successfully")
        except Exception as e:
            print(f"❌ Anthropic configuration failed: {e}")
    
    # Ollama Setup (local model)
    if OLLAMA_AVAILABLE:
        try:
            providers["ollama"] = ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "llama2"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
            print("✅ Ollama configured successfully")
        except Exception as e:
            print(f"❌ Ollama configuration failed: {e}")
    
    # Google Gemini Setup
    if GOOGLE_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
        try:
            providers["google"] = ChatGoogleGenerativeAI(
                model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
                temperature=float(os.getenv("GOOGLE_TEMPERATURE", 0.7)),
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            print("✅ Google Gemini configured successfully")
        except Exception as e:
            print(f"❌ Google Gemini configuration failed: {e}")
    
    # DeepSeek Setup
    if DEEPSEEK_AVAILABLE and os.getenv("DEEPSEEK_API_KEY"):
        try:
            providers["deepseek"] = ChatDeepSeek(
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", 0.7)),
                api_key=os.getenv("DEEPSEEK_API_KEY")
            )
            print("✅ DeepSeek configured successfully")
        except Exception as e:
            print(f"❌ DeepSeek configuration failed: {e}")
    
    # OpenRouter Setup (supports multiple models including DeepSeek)
    if OPENROUTER_AVAILABLE and os.getenv("OPENROUTER_API_KEY"):
        try:
            providers["openrouter"] = ChatOpenAI(
                model=os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat"),
                temperature=float(os.getenv("OPENROUTER_TEMPERATURE", 0.7)),
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1"
            )
            print("✅ OpenRouter configured successfully")
        except Exception as e:
            print(f"❌ OpenRouter configuration failed: {e}")
    
    if not providers:
        print("❌ No LLM providers available. Please check your API keys and installations.")
        return None
    
    return providers


def basic_llm_example(llm):
    """
    Demonstrate basic LLM usage with simple text generation.
    """
    print("\n" + "="*50)
    print("📝 BASIC LLM EXAMPLE")
    print("="*50)
    
    try:
        # Simple prompt
        prompt = "What is the average runrate of virat kohli in odi cricket?"
        print(f"Prompt: {prompt}")
        
        result = llm.invoke(prompt)
        print(f"Response: {result}")
        
    except Exception as e:
        print(f"Error in basic LLM example: {e}")


def prompt_template_example(llm):
    """
    Demonstrate prompt templates with variables.
    """
    print("\n" + "="*50)
    print("🎯 PROMPT TEMPLATE EXAMPLE")
    print("="*50)
    
    try:
        # Create a prompt template
        template = """
        You are a {role} with expertise in {domain}.
        Please {task} about {topic} in a {style} manner.
        Keep your response under {word_limit} words.
        """
        
        prompt = PromptTemplate(
            input_variables=["role", "domain", "task", "topic", "style", "word_limit"],
            template=template
        )
        
        # Format the prompt with specific values
        formatted_prompt = prompt.format(
            role="Commentator",
            domain="cricket",
            task="comment on the performance of virat kohli in odi cricket",
            topic="virat kohli",
            style="Sasuke Uchiha",
            word_limit="100"
        )
        
        print(f"Template: {template.strip()}")
        print(f"\nFormatted Prompt: {formatted_prompt}")
        
        result = llm.invoke(formatted_prompt)
        print(f"\nResponse: {result}")
        
    except Exception as e:
        print(f"Error in prompt template example: {e}")


def chat_model_example(chat_llm):
    """
    Demonstrate chat models with conversation context.
    """
    print("\n" + "="*50)
    print("💬 CHAT MODEL EXAMPLE")
    print("="*50)
    
    try:
        # Create a conversation with different message types
        messages = [
            SystemMessage(content="You are a helpful coding assistant. Be concise but informative."),
            HumanMessage(content="What's the difference between lists and tuples in Python?"),
        ]
        
        print("Messages:")
        for i, msg in enumerate(messages, 1):
            print(f"  {i}. {type(msg).__name__}: {msg.content}")
        
        # Get response
        response = chat_llm.invoke(messages)
        print(f"\nAI Response: {response.content}")
        
        # Continue the conversation
        messages.append(AIMessage(content=response.content))
        messages.append(HumanMessage(content="Can you give me a practical example?"))
        
        print(f"\nFollow-up: {messages[-1].content}")
        
        final_response = chat_llm.invoke(messages)
        print(f"AI Response: {final_response.content}")
        
    except Exception as e:
        print(f"Error in chat model example: {e}")


def chat_prompt_template_example(chat_llm):
    """
    Demonstrate chat prompt templates for structured conversations.
    """
    print("\n" + "="*50)
    print("💬 CHAT PROMPT TEMPLATE EXAMPLE")
    print("="*50)
    
    try:
        # Create a chat prompt template
        chat_template = ChatPromptTemplate.from_messages([
            ("system", "You are a {profession} with {years} years of experience."),
            ("human", "I need advice about {topic}. Specifically: {question}"),
        ])
        
        # Format the template
        formatted_messages = chat_template.format_messages(
            profession="data scientist",
            years="10",
            topic="machine learning",
            question="How do I prevent overfitting in my neural network?"
        )
        
        print("Chat Template Messages:")
        for i, msg in enumerate(formatted_messages, 1):
            print(f"  {i}. {type(msg).__name__}: {msg.content}")
        
        response = chat_llm.invoke(formatted_messages)
        print(f"\nAI Response: {response.content}")
        
    except Exception as e:
        print(f"Error in chat prompt template example: {e}")


def interactive_demo(providers):
    """
    Interactive demonstration allowing users to try different providers.
    """
    print("\n" + "="*50)
    print("🎮 INTERACTIVE DEMO")
    print("="*50)
    
    print("Available providers:")
    for i, provider in enumerate(providers.keys(), 1):
        print(f"  {i}. {provider}")
    
    try:
        choice = input("\nEnter provider number (or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            return
        
        provider_names = list(providers.keys())
        selected_provider = provider_names[int(choice) - 1]
        llm = providers[selected_provider]
        
        user_prompt = input("Enter your prompt: ").strip()
        
        if not user_prompt:
            print("Empty prompt, skipping...")
            return
        
        print(f"\nUsing {selected_provider}...")
        
        if "chat" in selected_provider or "anthropic" in selected_provider or "google" in selected_provider or "deepseek" in selected_provider or "openrouter" in selected_provider:
            # Use chat interface for chat models
            messages = [HumanMessage(content=user_prompt)]
            response = llm.invoke(messages)
            print(f"Response: {response.content}")
        else:
            # Use regular interface for text models
            response = llm.invoke(user_prompt)
            print(f"Response: {response}")
            
    except (ValueError, IndexError):
        print("Invalid selection.")
    except Exception as e:
        print(f"Error in interactive demo: {e}")


def main():
    """
    Main function to run all examples.
    """
    print("🦜🔗 LangChain Course - Lesson 1: Basic Prompting")
    print("=" * 60)
    
    # Set up providers
    providers = setup_llm_providers()
    
    if not providers:
        print("❌ Cannot proceed without LLM providers. Please check your setup.")
        return
    
    # Get the first available provider for examples
    text_llm = None
    chat_llm = None
    
    # Find text and chat models
    for name, llm in providers.items():
        if "chat" in name or "anthropic" in name or "google" in name or "deepseek" in name or "openrouter" in name:
            if chat_llm is None:
                chat_llm = llm
        else:
            if text_llm is None:
                text_llm = llm
    
    # If no separate text model, use chat model for both
    if text_llm is None and chat_llm is not None:
        text_llm = chat_llm
    
    # Run examples
    if text_llm:
       # basic_llm_example(text_llm)
       prompt_template_example(text_llm)
    
    # if chat_llm:
    #     chat_model_example(chat_llm)
    #     chat_prompt_template_example(chat_llm)
    
    # # Interactive demo
    interactive_demo(providers)
    
    # print("\n🎉 Lesson 1 completed! Check out the exercises to practice more.")


if __name__ == "__main__":
    main() 