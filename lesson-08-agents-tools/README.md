# Lesson 8: Agents & Tools with LangChain

## 🎯 Learning Objectives

By the end of this lesson, you will be able to:
- Understand agent architectures and reasoning frameworks
- Build autonomous agents with tool-calling capabilities
- Create custom tools for domain-specific tasks
- Implement multi-agent coordination and collaboration
- Deploy production agents with monitoring and safety controls

## 📚 Concepts Covered

### 1. Agent Fundamentals
- Agent types: ReAct, Plan-and-Execute, Conversational
- Reasoning patterns and decision-making processes
- Tool selection and execution strategies
- Agent memory and state management

### 2. Built-in Tools Ecosystem
- Search tools (Google, DuckDuckGo, Wikipedia)
- Calculator and math tools
- Code execution and Python REPL
- API calling and web interaction tools
- File system and data manipulation tools

### 3. Custom Tool Development
- Tool interface design and implementation
- Input validation and error handling
- Tool composition and chaining
- Performance optimization for tool execution

### 4. Agent Orchestration
- Single-agent workflows and planning
- Multi-agent systems and collaboration
- Agent communication protocols
- Task delegation and coordination

### 5. Production Agent Systems
- Safety mechanisms and output validation
- Agent monitoring and logging
- Cost control and resource management
- Human-in-the-loop patterns

## 🚀 Getting Started

### Prerequisites
- Completed Lessons 1-7 (through RAG Systems)
- Understanding of API integrations and tool interfaces
- Experience with planning and reasoning systems

### Setup
```bash
cd lesson-08-agents-tools
poetry install && poetry shell
cp env.example .env
python main.py
```

## 📝 Code Examples

### Basic Agent with Tools
```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import DuckDuckGoSearchRun, Calculator
from langchain.hub import pull

# Initialize tools
search = DuckDuckGoSearchRun()
calculator = Calculator()
tools = [search, calculator]

# Create agent
prompt = pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)

# Execute with tools
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

result = agent_executor.invoke({
    "input": "What's the population of Tokyo and calculate 10% of that number?"
})
```

### Custom Tool Implementation
```python
from langchain.tools import BaseTool
from typing import Optional
from pydantic import Field

class WeatherTool(BaseTool):
    name = "weather_lookup"
    description = "Get current weather for a specific city"
    api_key: str = Field()
    
    def _run(self, city: str) -> str:
        """Get weather for the specified city."""
        # Implementation with weather API
        response = self._fetch_weather(city)
        return f"Weather in {city}: {response['description']}, {response['temp']}°F"
    
    async def _arun(self, city: str) -> str:
        """Async version of the tool."""
        return self._run(city)

# Custom business tool
class CRMTool(BaseTool):
    name = "crm_lookup"
    description = "Look up customer information in CRM system"
    
    def _run(self, customer_id: str) -> str:
        # Integration with CRM API
        customer_data = self._query_crm(customer_id)
        return f"Customer {customer_id}: {customer_data['name']}, Status: {customer_data['status']}"
```

### Multi-Agent System
```python
from langchain.agents import AgentExecutor
from langchain.schema import BaseMessage

class MultiAgentOrchestrator:
    def __init__(self):
        self.research_agent = self._create_research_agent()
        self.writing_agent = self._create_writing_agent()
        self.review_agent = self._create_review_agent()
    
    def process_request(self, request: str):
        # Step 1: Research
        research_results = self.research_agent.invoke({
            "input": f"Research this topic: {request}"
        })
        
        # Step 2: Write content
        content = self.writing_agent.invoke({
            "input": f"Write content based on: {research_results['output']}"
        })
        
        # Step 3: Review and improve
        final_output = self.review_agent.invoke({
            "input": f"Review and improve: {content['output']}"
        })
        
        return final_output
```

### Production Agent with Safety
```python
class SafeProductionAgent:
    def __init__(self, config):
        self.agent = self._create_agent(config)
        self.safety_checker = SafetyValidator()
        self.cost_monitor = CostMonitor(config.max_cost_per_session)
        self.logger = AgentLogger()
    
    def execute(self, user_input: str, user_id: str):
        # Pre-execution safety check
        if not self.safety_checker.validate_input(user_input):
            return {"error": "Input violates safety policies"}
        
        # Cost monitoring
        if self.cost_monitor.would_exceed_limit(user_input):
            return {"error": "Cost limit would be exceeded"}
        
        # Execute with monitoring
        try:
            result = self.agent.invoke({"input": user_input})
            
            # Post-execution validation
            validated_output = self.safety_checker.validate_output(result)
            
            # Log interaction
            self.logger.log_interaction(user_id, user_input, validated_output)
            
            return validated_output
            
        except Exception as e:
            self.logger.log_error(user_id, user_input, str(e))
            return {"error": "Agent execution failed"}
```

## 🏋️ Exercises

### Exercise 1: Research Assistant Agent
Build an agent that can research topics using multiple tools and compile comprehensive reports.

### Exercise 2: Customer Service Agent
Create an agent with access to CRM, knowledge base, and escalation tools.

### Exercise 3: Code Analysis Agent
Implement an agent that can analyze codebases, run tests, and suggest improvements.

### Exercise 4: Multi-Agent Collaboration
Design a system where specialized agents collaborate on complex tasks.

### Exercise 5: Production Agent Platform
Build a complete agent platform with monitoring, safety controls, and management interfaces.

## 💡 Key Takeaways

1. **Tool Design**: Well-designed tools enable agents to perform complex, real-world tasks
2. **Safety First**: Production agents require robust safety mechanisms and output validation
3. **Cost Management**: Monitor and control agent resource usage in production environments
4. **Specialization**: Specialized agents often outperform general-purpose ones for specific domains
5. **Human Oversight**: Maintain human-in-the-loop capabilities for critical decision points

## 🔗 Next Lesson

[Lesson 9: Production & Advanced Patterns](../lesson-09-production/) - Learn deployment strategies, monitoring, optimization, and advanced LangChain patterns for production systems.

---

**Duration:** ~1.5 hours  
**Difficulty:** Advanced  
**Prerequisites:** Lessons 1-7 completed 