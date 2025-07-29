"""
Error Handling Utilities

Common functions for handling errors and implementing fallback mechanisms
in LangChain applications.
"""

import time
from typing import Any, Optional, Callable, List
from functools import wraps


def safe_llm_call(llm: Any, prompt: Any, max_retries: int = 3, delay: float = 1.0) -> Optional[str]:
    """
    Safely call an LLM with retry logic and error handling.
    
    Args:
        llm (Any): The LLM instance to call
        prompt (Any): The prompt to send
        max_retries (int): Maximum number of retry attempts
        delay (float): Delay between retries in seconds
        
    Returns:
        Optional[str]: The LLM response or None if all attempts failed
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            
            # Handle different response types
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            last_error = e
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
    
    print(f"All {max_retries} attempts failed. Last error: {last_error}")
    return None


def handle_api_errors(func: Callable) -> Callable:
    """
    Decorator to handle common API errors gracefully.
    
    Args:
        func (Callable): Function to wrap with error handling
        
    Returns:
        Callable: Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ImportError as e:
            print(f"❌ Import Error: {e}")
            print("Please install the required package or check your installation.")
            return None
        except KeyError as e:
            print(f"❌ Configuration Error: Missing environment variable {e}")
            print("Please check your .env file configuration.")
            return None
        except ConnectionError as e:
            print(f"❌ Connection Error: {e}")
            print("Please check your internet connection and API endpoints.")
            return None
        except Exception as e:
            error_type = type(e).__name__
            print(f"❌ {error_type}: {e}")
            return None
    
    return wrapper


def fallback_chain(*llms: Any) -> Callable:
    """
    Create a fallback chain that tries multiple LLMs in order.
    
    Args:
        *llms: Variable number of LLM instances to try
        
    Returns:
        Callable: Function that tries LLMs in fallback order
    """
    def generate_with_fallback(prompt: Any) -> Optional[str]:
        """
        Generate response using fallback chain.
        
        Args:
            prompt (Any): The prompt to send
            
        Returns:
            Optional[str]: Generated response or None if all failed
        """
        for i, llm in enumerate(llms):
            if llm is None:
                continue
                
            try:
                print(f"Trying LLM {i + 1}/{len(llms)}: {type(llm).__name__}")
                response = safe_llm_call(llm, prompt, max_retries=2)
                
                if response:
                    print(f"✅ Success with {type(llm).__name__}")
                    return response
                    
            except Exception as e:
                print(f"⚠️ {type(llm).__name__} failed: {e}")
                continue
        
        print("❌ All LLMs in fallback chain failed")
        return None
    
    return generate_with_fallback


def validate_response(response: str, 
                     min_length: int = 10, 
                     max_length: int = 5000,
                     required_keywords: Optional[List[str]] = None) -> bool:
    """
    Validate an LLM response against various criteria.
    
    Args:
        response (str): The response to validate
        min_length (int): Minimum acceptable length
        max_length (int): Maximum acceptable length
        required_keywords (Optional[List[str]]): Keywords that must be present
        
    Returns:
        bool: True if response passes validation
    """
    if not response or not isinstance(response, str):
        return False
    
    # Length validation
    if len(response) < min_length or len(response) > max_length:
        return False
    
    # Keyword validation
    if required_keywords:
        response_lower = response.lower()
        for keyword in required_keywords:
            if keyword.lower() not in response_lower:
                return False
    
    return True


def create_error_fallback(default_message: str = "Sorry, I'm unable to process that request right now."):
    """
    Create a fallback function that returns a default message.
    
    Args:
        default_message (str): Default message to return
        
    Returns:
        Callable: Function that returns the default message
    """
    def fallback_response(*args, **kwargs) -> str:
        return default_message
    
    return fallback_response


class RateLimitHandler:
    """
    Handle rate limiting with exponential backoff.
    """
    
    def __init__(self, initial_delay: float = 1.0, max_delay: float = 60.0):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.current_delay = initial_delay
    
    def wait(self):
        """Wait for the current delay period and increase delay."""
        print(f"Rate limit hit. Waiting {self.current_delay:.1f} seconds...")
        time.sleep(self.current_delay)
        self.current_delay = min(self.current_delay * 2, self.max_delay)
    
    def reset(self):
        """Reset delay to initial value."""
        self.current_delay = self.initial_delay


def robust_llm_generate(llm_list: List[Any], 
                       prompt: Any,
                       validation_func: Optional[Callable] = None,
                       max_attempts: int = 5) -> Optional[str]:
    """
    Generate text with multiple LLMs and validation.
    
    Args:
        llm_list (List[Any]): List of LLM instances to try
        prompt (Any): The prompt to send
        validation_func (Optional[Callable]): Function to validate responses
        max_attempts (int): Maximum total attempts across all LLMs
        
    Returns:
        Optional[str]: Valid response or None
    """
    attempts = 0
    rate_handler = RateLimitHandler()
    
    while attempts < max_attempts:
        for llm in llm_list:
            if llm is None:
                continue
            
            attempts += 1
            if attempts > max_attempts:
                break
            
            try:
                response = safe_llm_call(llm, prompt, max_retries=1)
                
                if response:
                    # Validate response if validation function provided
                    if validation_func and not validation_func(response):
                        print(f"Response failed validation, trying next LLM...")
                        continue
                    
                    rate_handler.reset()
                    return response
            
            except Exception as e:
                if "rate limit" in str(e).lower():
                    rate_handler.wait()
                else:
                    print(f"Error with {type(llm).__name__}: {e}")
    
    return None


if __name__ == "__main__":
    """Test the error handling utilities."""
    print("🦜🔗 LangChain Course - Error Handling Test")
    print("=" * 50)
    
    # Test validation
    test_responses = [
        "This is a valid response.",
        "",  # Too short
        "x" * 6000,  # Too long
        "This response contains Python keyword"
    ]
    
    print("Testing response validation:")
    for i, response in enumerate(test_responses):
        valid = validate_response(
            response, 
            min_length=5, 
            max_length=100,
            required_keywords=["valid", "response"]
        )
        print(f"  Response {i+1}: {'✅ Valid' if valid else '❌ Invalid'}")
    
    # Test fallback function
    fallback = create_error_fallback("Service temporarily unavailable.")
    print(f"\nFallback message: {fallback()}")
    
    # Test rate limit handler
    print("\nTesting rate limit handler:")
    rate_handler = RateLimitHandler(0.1, 1.0)  # Fast for testing
    print(f"Initial delay: {rate_handler.current_delay}")
    rate_handler.wait()
    print(f"After first wait: {rate_handler.current_delay}")
    
    print("\n✅ Error handling tests completed!") 