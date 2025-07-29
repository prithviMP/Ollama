# Lesson 2: Prompt Engineering with LangChain

## 🎯 Learning Objectives

By the end of this lesson, you will be able to:
- Master few-shot prompting techniques with examples
- Implement chain-of-thought reasoning for complex problems
- Create structured outputs with parsing and validation
- Use advanced prompt engineering patterns
- Build robust prompt templates for production use

## 📚 Concepts Covered

### 1. Few-Shot Prompting
- Example-driven learning patterns
- Dynamic example selection
- Example formatting and structure
- Context length optimization

### 2. Chain-of-Thought (CoT) Reasoning
- Step-by-step problem solving
- Reasoning pattern templates
- Mathematical and logical reasoning
- CoT with few-shot examples

### 3. Structured Output Parsing
- Pydantic output parsers
- JSON schema validation
- Custom output formats
- Error handling for malformed outputs

### 4. Advanced Prompt Techniques
- Role-based prompting
- Instruction hierarchies
- Prompt compression techniques
- Multi-modal prompt patterns

### 5. Production Prompt Patterns
- Template versioning
- A/B testing prompts
- Performance optimization
- Monitoring and analytics

## 🚀 Getting Started

### Prerequisites
- Completed Lesson 1 (Basic Prompting)
- Python 3.11+
- Poetry installed
- LLM provider API keys configured

### Setup
1. Navigate to this lesson directory:
```bash
cd lesson-02-prompt-engineering
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

### Few-Shot Prompting
```python
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts import PromptTemplate

examples = [
    {"input": "happy", "output": "😊"},
    {"input": "sad", "output": "😢"},
    {"input": "angry", "output": "😠"}
]

example_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Convert emotions to emojis:",
    suffix="Input: {emotion}\nOutput:",
    input_variables=["emotion"]
)
```

### Chain-of-Thought Reasoning
```python
cot_template = """
Solve this step by step:

Problem: {problem}

Let me think through this step by step:
1. First, I'll identify what we know
2. Then, I'll determine what we need to find
3. Next, I'll work through the solution
4. Finally, I'll verify the answer

Step-by-step solution:
"""

prompt = PromptTemplate(
    input_variables=["problem"],
    template=cot_template
)
```

### Structured Output Parsing
```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class PersonInfo(BaseModel):
    name: str = Field(description="person's full name")
    age: int = Field(description="person's age in years")
    occupation: str = Field(description="person's job title")

parser = PydanticOutputParser(pydantic_object=PersonInfo)

prompt = PromptTemplate(
    template="Extract person information from this text: {text}\n{format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
```

## 🏋️ Exercises

### Exercise 1: Custom Few-Shot Template
Create a few-shot prompt for code documentation generation.

### Exercise 2: Mathematical Chain-of-Thought
Build a CoT system for solving multi-step math problems.

### Exercise 3: Structured Data Extraction
Extract structured information from unstructured text using Pydantic.

### Exercise 4: Advanced Role-Based Prompting
Create expert-level prompts for different domains (legal, medical, technical).

### Exercise 5: Prompt Optimization System
Build a system to test and optimize prompt performance.

## 📖 Additional Resources

- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [LangChain Output Parsers](https://python.langchain.com/docs/modules/model_io/output_parsers/)
- [Few-Shot Learning](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/few_shot_examples)

## 🔗 Next Lesson

[Lesson 3: Chains & Sequential Processing](../lesson-03-chains/) - Learn to combine multiple LLM calls into powerful sequential workflows.

## 💡 Key Takeaways

1. **Few-Shot Learning**: Examples dramatically improve LLM performance
2. **Chain-of-Thought**: Step-by-step reasoning produces better results
3. **Structured Outputs**: Parsing ensures reliable, usable responses
4. **Prompt Engineering**: Small changes can yield massive improvements
5. **Production Patterns**: Systematic approaches scale better than ad-hoc prompts

## ⚠️ Common Pitfalls

- Using too many examples (context window limits)
- Inconsistent example formatting
- Ignoring output validation
- Not testing prompts across different inputs
- Hardcoding prompts instead of using templates

---

**Duration:** ~1 hour  
**Difficulty:** Intermediate  
**Prerequisites:** Lesson 1 completed 