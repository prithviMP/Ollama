# Lesson 1: Basic Prompting with LangChain

## 🎯 Learning Objectives

By the end of this lesson, you will be able to:
- Set up LangChain with different LLM providers (OpenAI, Anthropic, Ollama)
- Create basic prompt templates and variables
- Use chat models for conversational AI
- Handle different response formats
- Implement basic error handling and best practices

## 📚 Concepts Covered

### 1. LangChain Installation and Setup
- Installing LangChain and provider packages
- Environment configuration
- API key management

### 2. Large Language Models (LLMs)
- Understanding different LLM providers
- Configuring OpenAI GPT models
- Setting up Anthropic Claude
- Using local models with Ollama

### 3. Basic Prompting
- Simple text generation
- Prompt templates with variables
- PromptTemplate class usage
- String formatting in prompts

### 4. Chat Models
- ChatOpenAI vs OpenAI
- Message types (System, Human, AI)
- Conversation structure
- Response handling

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Poetry installed
- At least one API key (OpenAI, Anthropic) or Ollama running locally

### Setup
1. Navigate to this lesson directory:
```bash
cd lesson-01-basic-prompting
```

2. Install dependencies:
```bash
poetry install
poetry shell
```

3. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

4. Run the main lesson:
```bash
python main.py
```

## 📝 Code Examples

### Basic LLM Usage
```python
from langchain_openai import OpenAI

llm = OpenAI(temperature=0.7)
result = llm.invoke("What is artificial intelligence?")
print(result)
```

### Prompt Templates
```python
from langchain.prompts import PromptTemplate

template = "Tell me a {adjective} joke about {topic}."
prompt = PromptTemplate(
    input_variables=["adjective", "topic"],
    template=template
)

formatted_prompt = prompt.format(adjective="funny", topic="programming")
```

### Chat Models
```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(temperature=0.7)
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hello! How are you?")
]
response = chat.invoke(messages)
```

## 🏋️ Exercises

### Exercise 1: Basic LLM Setup
Set up three different LLM providers and test basic text generation.

### Exercise 2: Personal Assistant
Create a prompt template for a personal assistant that takes a task and priority level.

### Exercise 3: Multi-turn Conversation
Build a simple chat system that maintains conversation context.

### Exercise 4: Creative Writing
Use prompt templates to generate stories with user-defined characters and settings.

## 📖 Additional Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic Claude Documentation](https://docs.anthropic.com/)
- [Ollama Documentation](https://ollama.ai/docs)

## 🔗 Next Lesson

[Lesson 2: Prompt Engineering](../lesson-02-prompt-engineering/) - Learn advanced prompting techniques including few-shot prompting, chain-of-thought, and structured outputs.

## 💡 Key Takeaways

1. **LangChain Abstractions**: LangChain provides unified interfaces for different LLM providers
2. **Prompt Templates**: Use templates for reusable and dynamic prompts
3. **Chat vs Text Models**: Chat models are better for conversational AI
4. **Environment Management**: Always use environment variables for API keys
5. **Error Handling**: Implement proper error handling for production applications

## ⚠️ Common Pitfalls

- Forgetting to set environment variables
- Not handling API rate limits
- Using incorrect message types with chat models
- Hardcoding prompts instead of using templates
- Not considering token limits and costs

---

**Duration:** ~1 hour  
**Difficulty:** Beginner  
**Prerequisites:** Basic Python knowledge 