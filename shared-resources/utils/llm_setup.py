"""
LLM Setup Utilities

Common functions for setting up and managing LLM providers across lessons.
"""

import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def setup_llm_providers() -> Dict[str, Any]:
    """
    Set up all available LLM providers based on environment configuration.
    
    Returns:
        Dict[str, Any]: Dictionary of available LLM providers
    """
    providers = {}
    
    # OpenAI Setup
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAI, ChatOpenAI
            
            providers["openai"] = OpenAI(
                temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.7)),
                max_tokens=int(os.getenv("MAX_TOKENS", 1000))
            )
            providers["chat_openai"] = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.7))
            )
            print("✅ OpenAI configured successfully")
        except ImportError:
            print("⚠️  OpenAI package not installed. Run: pip install langchain-openai")
        except Exception as e:
            print(f"❌ OpenAI configuration failed: {e}")
    
    # Anthropic Setup
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
            
            providers["anthropic"] = ChatAnthropic(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
                temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", 0.7))
            )
            print("✅ Anthropic configured successfully")
        except ImportError:
            print("⚠️  Anthropic package not installed. Run: pip install langchain-anthropic")
        except Exception as e:
            print(f"❌ Anthropic configuration failed: {e}")
    
    # Ollama Setup (local models)
    try:
        from langchain_ollama import ChatOllama
        
        providers["ollama"] = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", 0.7))
        )
        print("✅ Ollama configured successfully")
    except ImportError:
        print("⚠️  Ollama package not installed. Run: pip install langchain-ollama")
    except Exception as e:
        print(f"❌ Ollama configuration failed: {e}")
    
    if not providers:
        print("❌ No LLM providers available. Please check your API keys and installations.")
        print("Available options:")
        print("  - Set OPENAI_API_KEY for OpenAI models")
        print("  - Set ANTHROPIC_API_KEY for Claude models") 
        print("  - Install and run Ollama for local models")
    
    return providers


def get_available_providers() -> List[str]:
    """
    Get a list of available LLM provider names.
    
    Returns:
        List[str]: List of available provider names
    """
    providers = setup_llm_providers()
    return list(providers.keys())


def get_preferred_llm(providers: Dict[str, Any], prefer_chat: bool = False) -> Optional[Any]:
    """
    Get the preferred LLM from available providers.
    
    Args:
        providers (Dict[str, Any]): Available providers
        prefer_chat (bool): Whether to prefer chat models
        
    Returns:
        Optional[Any]: The preferred LLM instance or None
    """
    if not providers:
        return None
    
    # Priority order for different types
    if prefer_chat:
        priority = ["chat_openai", "anthropic", "ollama", "openai"]
    else:
        priority = ["openai", "chat_openai", "anthropic", "ollama"]
    
    for provider_name in priority:
        if provider_name in providers:
            return providers[provider_name]
    
    # Return first available if none match priority
    return list(providers.values())[0]


def validate_provider_config() -> Dict[str, bool]:
    """
    Validate the configuration for each provider.
    
    Returns:
        Dict[str, bool]: Validation status for each provider
    """
    validation = {}
    
    # Check OpenAI
    validation["openai"] = bool(os.getenv("OPENAI_API_KEY"))
    
    # Check Anthropic
    validation["anthropic"] = bool(os.getenv("ANTHROPIC_API_KEY"))
    
    # Check Ollama (assume available if no error in import)
    try:
        from langchain_ollama import ChatOllama
        validation["ollama"] = True
    except ImportError:
        validation["ollama"] = False
    
    return validation


def print_provider_status():
    """
    Print the status of all LLM providers.
    """
    print("\n🔍 LLM Provider Status:")
    print("-" * 40)
    
    validation = validate_provider_config()
    providers = setup_llm_providers()
    
    for provider, is_configured in validation.items():
        status = "✅ Available" if provider in providers else "❌ Not available"
        config_status = "🔧 Configured" if is_configured else "⚙️  Not configured"
        
        print(f"{provider.capitalize():12} | {status:13} | {config_status}")
    
    print("-" * 40)
    print(f"Total available: {len(providers)}")


if __name__ == "__main__":
    """Test the LLM setup utilities."""
    print("🦜🔗 LangChain Course - LLM Setup Test")
    print("=" * 50)
    
    print_provider_status()
    
    providers = setup_llm_providers()
    print(f"\nAvailable providers: {list(providers.keys())}")
    
    if providers:
        preferred = get_preferred_llm(providers)
        print(f"Preferred LLM: {type(preferred).__name__}")
        
        chat_preferred = get_preferred_llm(providers, prefer_chat=True)
        print(f"Preferred Chat LLM: {type(chat_preferred).__name__}")
    else:
        print("No providers available for testing.") 