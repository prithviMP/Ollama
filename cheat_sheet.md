# LangChain Course Cheat Sheet 📚

A comprehensive reference guide for all functions, classes, and concepts used across the LangChain course lessons.

## Table of Contents
- [🤖 LLM Providers & Setup](#-llm-providers--setup)
- [💬 Basic LLM Operations](#-basic-llm-operations)
- [📝 Prompt Templates](#-prompt-templates)
- [🧠 Memory Management](#-memory-management)
- [📄 Document Processing](#-document-processing)
- [🔗 Chains](#-chains)
- [🛠 Utility Functions](#-utility-functions)
- [⚙️ Environment Variables](#️-environment-variables)

---

## 🤖 LLM Providers & Setup

### `setup_llm_providers()`
**Purpose**: Initialize and configure all available LLM providers based on environment variables.

**Function Signature**:
```python
def setup_llm_providers() -> Dict[str, Any]
```

**Returns**: Dictionary containing initialized LLM instances and configuration.

**Example**:
```python
from shared_resources.utils.llm_setup import setup_llm_providers

providers = setup_llm_providers()
# Returns: {
#     'openai': ChatOpenAI(...),
#     'anthropic': ChatAnthropic(...),
#     'google': ChatGoogleGenerativeAI(...),
#     'deepseek': ChatDeepSeek(...),
#     'ollama': ChatOllama(...),
#     'config': {...}
# }
```

### `get_preferred_llm(llm_type)`
**Purpose**: Get the best available LLM based on type and available API keys.

**Function Signature**:
```python
def get_preferred_llm(llm_type: str = "chat") -> Any
```

**Parameters**:
- `llm_type` (str): Either "chat" or "completion"

**Example**:
```python
chat_model = get_preferred_llm("chat")
response = chat_model.invoke("Hello, world!")
```

### `validate_llm_provider(provider_name)`
**Purpose**: Check if a specific LLM provider is properly configured.

**Function Signature**:
```python
def validate_llm_provider(provider_name: str) -> bool
```

**Parameters**:
- `provider_name` (str): Name of the provider ("openai", "anthropic", etc.)

**Example**:
```python
if validate_llm_provider("openai"):
    print("OpenAI is configured and ready!")
```

---

## 💬 Basic LLM Operations

### Chat Models
**Purpose**: Direct interaction with chat-based LLMs.

**Common Chat Models**:
```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama

# Initialize
chat_model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=1500
)

# Basic usage
response = chat_model.invoke("Explain quantum computing in simple terms")
print(response.content)
```

### LLM Models (Completion)
**Purpose**: Text completion with traditional LLM models.

**Example**:
```python
from langchain_openai import OpenAI

llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0.7,
    max_tokens=1500
)

response = llm.invoke("The future of AI is")
print(response)
```

---

## 📝 Prompt Templates

### `PromptTemplate`
**Purpose**: Create reusable prompt templates with variables.

**Function Signature**:
```python
PromptTemplate(
    input_variables: List[str],
    template: str
)
```

**Parameters**:
- `input_variables` (List[str]): List of variable names used in the template
- `template` (str): The prompt template with {variable} placeholders

**Example**:
```python
from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["topic", "audience"],
    template="Explain {topic} to a {audience} in simple terms."
)

prompt = template.format(topic="machine learning", audience="child")
# Output: "Explain machine learning to a child in simple terms."
```

### `ChatPromptTemplate`
**Purpose**: Create structured chat conversations with system and human messages.

**Function Signature**:
```python
ChatPromptTemplate.from_messages([
    ("system", "system_message"),
    ("human", "human_message")
])
```

**Example**:
```python
from langchain.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant specialized in {domain}."),
    ("human", "Please help me with: {question}")
])

messages = chat_template.format_messages(
    domain="programming",
    question="How do I optimize Python code?"
)
```

### Message Types
**Purpose**: Structure individual messages in chat conversations.

**Classes**:
```python
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# System message (sets AI behavior)
system_msg = SystemMessage(content="You are a helpful assistant.")

# Human message (user input)
human_msg = HumanMessage(content="What is the capital of France?")

# AI message (assistant response)
ai_msg = AIMessage(content="The capital of France is Paris.")
```

---

## 🧠 Memory Management

### `ConversationBufferMemory`
**Purpose**: Store complete conversation history in memory.

**Function Signature**:
```python
ConversationBufferMemory(
    memory_key: str = "history",
    return_messages: bool = False
)
```

**Parameters**:
- `memory_key` (str): Key to store memory in chain context
- `return_messages` (bool): Return as message objects or string

**Example**:
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Add to memory
memory.save_context(
    {"input": "Hi there!"},
    {"output": "Hello! How can I help you?"}
)

# Retrieve memory
history = memory.load_memory_variables({})
```

### `ConversationSummaryMemory`
**Purpose**: Summarize long conversations to fit within context limits.

**Function Signature**:
```python
ConversationSummaryMemory(
    llm: BaseLanguageModel,
    memory_key: str = "history",
    return_messages: bool = False
)
```

**Parameters**:
- `llm` (BaseLanguageModel): LLM to use for summarization
- `memory_key` (str): Key to store memory in chain context
- `return_messages` (bool): Return as message objects or string

**Example**:
```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(
    llm=chat_model,
    memory_key="chat_history"
)
```

### `ConversationBufferWindowMemory`
**Purpose**: Keep only the last N messages in memory.

**Function Signature**:
```python
ConversationBufferWindowMemory(
    k: int,
    memory_key: str = "history",
    return_messages: bool = False
)
```

**Parameters**:
- `k` (int): Number of recent messages to keep
- `memory_key` (str): Key to store memory in chain context
- `return_messages` (bool): Return as message objects or string

**Example**:
```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    k=5,  # Keep last 5 messages
    memory_key="chat_history",
    return_messages=True
)
```

### `VectorStoreRetrieverMemory`
**Purpose**: Use vector similarity for semantic memory retrieval.

**Function Signature**:
```python
VectorStoreRetrieverMemory(
    retriever: VectorStoreRetriever,
    memory_key: str = "history",
    input_key: str = None,
    return_docs: bool = False
)
```

**Example**:
```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Create vector store
vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

memory = VectorStoreRetrieverMemory(
    retriever=retriever,
    memory_key="chat_history"
)
```

---

## 📄 Document Processing

### Document Loaders

#### `PyPDFLoader`
**Purpose**: Load and parse PDF documents.

**Function Signature**:
```python
PyPDFLoader(file_path: str)
```

**Example**:
```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("path/to/document.pdf")
documents = loader.load()

# Each document has .page_content and .metadata
for doc in documents:
    print(f"Page {doc.metadata['page']}: {doc.page_content[:100]}...")
```

#### `TextLoader`
**Purpose**: Load plain text files.

**Function Signature**:
```python
TextLoader(file_path: str, encoding: str = "utf-8")
```

**Example**:
```python
from langchain.document_loaders import TextLoader

loader = TextLoader("path/to/document.txt")
documents = loader.load()
```

#### `WebBaseLoader`
**Purpose**: Load content from web pages.

**Function Signature**:
```python
WebBaseLoader(web_paths: List[str])
```

**Example**:
```python
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader(["https://example.com/article"])
documents = loader.load()
```

#### `CSVLoader`
**Purpose**: Load CSV files as documents.

**Example**:
```python
from langchain.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="data.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"'
    }
)
documents = loader.load()
```

### Text Splitters

#### `RecursiveCharacterTextSplitter`
**Purpose**: Split text intelligently by paragraphs, sentences, then characters.

**Function Signature**:
```python
RecursiveCharacterTextSplitter(
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    length_function: Callable = len,
    separators: List[str] = None
)
```

**Parameters**:
- `chunk_size` (int): Maximum characters per chunk
- `chunk_overlap` (int): Characters to overlap between chunks
- `length_function` (Callable): Function to measure text length
- `separators` (List[str]): Custom separators for splitting

**Example**:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_text("Your long text here...")
# Or split documents
docs = text_splitter.split_documents(documents)
```

#### `CharacterTextSplitter`
**Purpose**: Split text by a specific character or sequence.

**Function Signature**:
```python
CharacterTextSplitter(
    separator: str = "\n\n",
    chunk_size: int = 4000,
    chunk_overlap: int = 200
)
```

**Example**:
```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=100
)
```

#### `TokenTextSplitter`
**Purpose**: Split text based on token count (for LLM context limits).

**Function Signature**:
```python
TokenTextSplitter(
    encoding_name: str = "gpt2",
    chunk_size: int = 4000,
    chunk_overlap: int = 200
)
```

**Example**:
```python
from langchain.text_splitter import TokenTextSplitter

splitter = TokenTextSplitter(
    encoding_name="cl100k_base",  # GPT-4 encoding
    chunk_size=512,
    chunk_overlap=50
)
```

### Document Transformers

#### `Html2TextTransformer`
**Purpose**: Convert HTML content to clean text.

**Example**:
```python
from langchain.document_transformers import Html2TextTransformer

transformer = Html2TextTransformer()
clean_docs = transformer.transform_documents(html_documents)
```

---

## 🔗 Chains

### `ConversationChain`
**Purpose**: Chain together LLM calls with memory for conversations.

**Function Signature**:
```python
ConversationChain(
    llm: BaseLanguageModel,
    memory: BaseMemory,
    prompt: BasePromptTemplate = None,
    verbose: bool = False
)
```

**Example**:
```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=chat_model,
    memory=memory,
    verbose=True
)

response = conversation.predict(input="Hi there!")
```

### `LLMChain`
**Purpose**: Basic chain combining an LLM with a prompt template.

**Function Signature**:
```python
LLMChain(
    llm: BaseLanguageModel,
    prompt: BasePromptTemplate,
    memory: BaseMemory = None
)
```

**Example**:
```python
from langchain.chains import LLMChain

chain = LLMChain(
    llm=chat_model,
    prompt=prompt_template
)

result = chain.run(topic="AI", audience="students")
```

---

## 🛠 Utility Functions

### Document Creation
**Purpose**: Create document objects programmatically.

**Function Signature**:
```python
Document(page_content: str, metadata: dict = None)
```

**Example**:
```python
from langchain.schema import Document

doc = Document(
    page_content="This is the content of the document.",
    metadata={
        "source": "manual_creation",
        "author": "John Doe",
        "date": "2024-01-15"
    }
)
```

### Environment Loading
**Purpose**: Load environment variables from .env files.

**Example**:
```python
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
```

### Vector Store Operations
**Purpose**: Store and retrieve documents using vector similarity.

**Example**:
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Search similar documents
similar_docs = vectorstore.similarity_search(
    "query text",
    k=3  # Return top 3 similar documents
)
```

---

## ⚙️ Environment Variables

### Required API Keys
```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Google Gemini
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL=gemini-pro

# DeepSeek
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_MODEL=deepseek-chat

# OpenRouter
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=deepseek/deepseek-chat

# Ollama (Local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
```

### Common Configuration
```bash
# General Settings
MAX_TOKENS=1500
VERBOSE=True
TEMPERATURE=0.7

# Memory Settings (Lesson 4)
CHROMA_PERSIST_DIRECTORY=./chroma_db
FAISS_INDEX_PATH=./faiss_index
MEMORY_SESSION_TIMEOUT=3600
ENABLE_MEMORY_PERSISTENCE=True
MAX_MEMORY_TOKENS=4000

# Document Processing Settings (Lesson 5)
DOCUMENTS_DIRECTORY=./documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_DOCUMENT_SIZE_MB=50
BATCH_SIZE=10
MAX_WORKERS=4
ENABLE_PARALLEL_PROCESSING=True
```

---

## 🎯 Quick Start Examples

### Basic Chat Example
```python
from shared_resources.utils.llm_setup import get_preferred_llm

# Get the best available chat model
chat_model = get_preferred_llm("chat")

# Simple conversation
response = chat_model.invoke("What is machine learning?")
print(response.content)
```

### Memory-Enabled Conversation
```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from shared_resources.utils.llm_setup import get_preferred_llm

# Setup
llm = get_preferred_llm("chat")
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

# Chat with memory
response1 = conversation.predict(input="My name is Alice")
response2 = conversation.predict(input="What's my name?")
# The AI will remember "Alice" from the previous exchange
```

### Document Processing Pipeline
```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load document
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# Process each chunk
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk.page_content[:100]}...")
```

---

## 📚 Common Patterns

### Error Handling
```python
try:
    from langchain_openai import ChatOpenAI
    chat_model = ChatOpenAI()
except ImportError:
    print("OpenAI package not installed")
except Exception as e:
    print(f"Error initializing OpenAI: {e}")
```

### Provider Fallback
```python
def get_available_chat_model():
    providers = ["openai", "anthropic", "google", "deepseek"]
    for provider in providers:
        if validate_llm_provider(provider):
            return get_preferred_llm("chat")
    raise Exception("No LLM providers available")
```

### Batch Processing
```python
def process_documents_batch(documents, batch_size=10):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        # Process batch
        yield process_batch(batch)
```

---

This cheat sheet covers the most commonly used functions and patterns across all LangChain lessons. Keep it handy for quick reference during development! 🚀 