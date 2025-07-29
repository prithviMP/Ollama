# Lesson 3: Chains & Sequential Processing with LangChain

## 🎯 Learning Objectives

By the end of this lesson, you will be able to:
- Master LLMChain fundamentals and composition patterns
- Build complex sequential chains for multi-step processing
- Implement router chains for conditional logic and branching
- Create custom chains for specific use cases
- Design production-ready chain architectures

## 📚 Concepts Covered

### 1. LLMChain Fundamentals
- Basic LLMChain construction and execution
- Chain composition and chaining patterns
- Input/output handling and data flow
- Chain debugging and monitoring

### 2. Sequential Chains
- SimpleSequentialChain for linear processing
- SequentialChain for complex multi-input/output flows
- Chain memory and state management
- Error handling in chain sequences

### 3. Router Chains
- MultiPromptChain for conditional routing
- LLMRouterChain for intelligent decision making
- Custom routing logic implementation
- Destination chains and fallback handling

### 4. Transform and Utility Chains
- TransformChain for data preprocessing
- LLMMathChain for mathematical operations
- APIChain for external service integration
- Custom utility chain development

### 5. Advanced Chain Patterns
- Parallel chain execution
- Chain composition strategies
- Performance optimization techniques
- Production deployment patterns

## 🚀 Getting Started

### Prerequisites
- Completed Lesson 1 (Basic Prompting) and Lesson 2 (Prompt Engineering)
- Understanding of prompt templates and LLM interactions
- Python 3.11+ and Poetry installed

### Setup
1. Navigate to this lesson directory:
```bash
cd lesson-03-chains
```

2. Install dependencies:
```bash
poetry install
poetry shell
```

3. Set up environment variables:
```bash
cp env.example .env
# Edit .env file with your API keys
```

4. Run the main lesson:
```bash
python main.py
```

## 📝 Code Examples

### Basic LLMChain
```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a brief summary about {topic}"
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("artificial intelligence")
```

### Sequential Chain
```python
from langchain.chains import SequentialChain

# First chain: Generate content
content_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["topic"],
        template="Write content about {topic}"
    ),
    output_key="content"
)

# Second chain: Summarize content
summary_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["content"],
        template="Summarize this content: {content}"
    ),
    output_key="summary"
)

# Combine chains
overall_chain = SequentialChain(
    chains=[content_chain, summary_chain],
    input_variables=["topic"],
    output_variables=["content", "summary"]
)
```

### Router Chain
```python
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain

# Define destination chains
math_chain = LLMChain(llm=llm, prompt=math_prompt)
science_chain = LLMChain(llm=llm, prompt=science_prompt)

# Create router
destinations = [
    {"name": "math", "description": "Good for math problems", "prompt_template": math_prompt},
    {"name": "science", "description": "Good for science questions", "prompt_template": science_prompt}
]

router_chain = MultiPromptChain.from_prompts(
    llm=llm,
    prompt_infos=destinations,
    default_chain=default_chain
)
```

## 🏋️ Exercises

### Exercise 1: Content Creation Pipeline
Build a sequential chain that generates, reviews, and optimizes content.

### Exercise 2: Research Assistant Chain
Create a chain that researches topics, synthesizes information, and formats reports.

### Exercise 3: Multi-Domain Question Router
Implement a router that directs questions to specialized domain experts.

### Exercise 4: Data Processing Pipeline
Build chains that transform, validate, and enrich data.

### Exercise 5: Custom Business Logic Chain
Design a chain for specific business workflow automation.

## 📖 Additional Resources

- [LangChain Chains Documentation](https://python.langchain.com/docs/modules/chains/)
- [Sequential Chains Guide](https://python.langchain.com/docs/modules/chains/foundational/sequential_chains)
- [Router Chains Tutorial](https://python.langchain.com/docs/modules/chains/foundational/router)
- [Custom Chains Development](https://python.langchain.com/docs/modules/chains/how_to/custom_chain)

## 🔗 Next Lesson

[Lesson 4: Memory & Conversation Management](../lesson-04-memory/) - Learn to build stateful applications with persistent memory and conversation context.

## 💡 Key Takeaways

1. **Chain Composition**: Chains enable complex multi-step LLM workflows
2. **Sequential Processing**: Break complex tasks into manageable sequential steps
3. **Conditional Logic**: Router chains provide intelligent decision-making capabilities
4. **Modularity**: Well-designed chains are reusable and maintainable
5. **Error Handling**: Robust chains handle failures gracefully with fallbacks

## ⚠️ Common Pitfalls

- Not handling chain failures and edge cases
- Over-complicated chain architectures
- Inefficient sequential processing
- Poor input/output variable management
- Ignoring chain performance implications

---

**Duration:** ~1 hour  
**Difficulty:** Intermediate  
**Prerequisites:** Lessons 1-2 completed 