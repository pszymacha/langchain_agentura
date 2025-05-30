# AI Agent Application

Multi-interface AI agent system demonstrating LangChain and LangGraph capabilities for developers.

## 🎯 Purpose

This is a **developer demo application** showcasing:
- **Standard LangChain**: Traditional AgentExecutor with function calling
- **Advanced LangGraph**: Sophisticated workflows with state management
- **Multiple Interfaces**: CLI, HTTP API, and REST client
- **Best Practices**: Centralized logging, error handling, configuration management

Perfect for cloning and adapting to your specific needs!

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Interface │    │  HTTP Interface │    │  REST Client    │
│                 │    │   (FastAPI)     │    │    (Simple)     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────┬───────────┴──────────────────────┘
                     │
          ┌─────────────────────┐
          │   AgentService      │
          │ (Centralized Logic) │
          └─────────┬───────────┘
                    │
        ┌───────────────────────────┐
        │     Agent Interface       │
        │    (Common Protocol)      │
        └─────────┬─────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
   ┌─────────────┐    ┌─────────────┐
   │   Standard  │    │  LangGraph  │
   │    Agent    │    │    Agent    │
   │(LangChain)  │    │(Advanced)   │
   └─────────────┘    └─────────────┘
```

## 🚀 Quick Start

### Installation
```bash
cd agentura/langchain_agentura
pip install -r requirements.txt
```

### Configuration
```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your API keys and settings
```

### Set API Keys
```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"

# Or Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"

# Or Azure OpenAI
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

### Run Application
```bash
# Interactive CLI (default - LangGraph agent)
python main.py cli

# Advanced CLI with agent selection
python main.py cli --mode advanced

# HTTP API Server
python main.py server

# Simple REST Client
python rest_client.py -q "What is Python?"
```

## 💻 Usage Examples

### 1. CLI Interface

#### Simple Mode (LangGraph Agent)
```bash
$ python main.py cli
🔬 Advanced Research Agent - Simple CLI Mode
🔍 Your research query: What is the latest in AI?
[Live processing steps shown...]
📊 FINAL RESEARCH RESULT
[Comprehensive answer...]
```

#### Advanced Mode (Agent Selection)
```bash
$ python main.py cli --mode advanced
🤖 Available Agent Types:
1. 🔧 Standard LangChain Agent  2. 🔬 LangGraph Research Agent

🤖 Choose agent type (1-2): 1
💬 Your query: Calculate 15 * 23
```

### 2. HTTP API Server

#### Start Server
```bash
$ python main.py server --port 8080
🌐 Starting HTTP Server Interface...
📍 Server running at: http://0.0.0.0:8080
📚 Documentation: http://localhost:8080/docs
```

#### API Endpoints
```bash
# Health check
curl http://localhost:8080/

# Get available agents
curl http://localhost:8080/agents

# Process query (default LangGraph agent)
curl -X POST http://localhost:8080/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is machine learning?"}'

# Process with Standard agent
curl -X POST http://localhost:8080/query \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Simple calculation: 15 + 27",
       "agent_type": "standard",
       "thread_id": "session-123"
     }'
```

#### API Response Format
```json
{
  "answer": "Machine learning is a subset of artificial intelligence...",
  "logs": [
    "2024-01-15T10:30:00: Creating langgraph agent...",
    "2024-01-15T10:30:02: Processing query...",
    "2024-01-15T10:30:05: Query processing completed"
  ],
  "metadata": {
    "agent_type": "langgraph",
    "agent_name": "🔬 LangGraph Research Agent",
    "thread_id": "session_1705317000",
    "execution_time": 3.45,
    "timestamp": "2024-01-15T10:30:05",
    "tools_available": ["DuckDuckGo", "Wikipedia", "Math"]
  }
}
```

### 3. REST Client

#### Interactive Mode (Default)
```bash
$ python rest_client.py
Connected to http://localhost:8080
Using agent: langgraph
Commands: 'quit', 'exit', 'agents', 'logs on/off'

Query: What is quantum computing?
Processing...
Answer: Quantum computing is a type of computation...
Agent: 🔬 LangGraph Research Agent | Time: 4.2s
```

#### Single Query Mode
```bash
# Simple query
python rest_client.py -q "What is Python?"

# With specific agent and logs
python rest_client.py -q "calculate 15 * 23" --agent-type standard --logs

# Custom host and port
python rest_client.py -q "test" -H 192.168.1.100 -p 8080
```

## 🤖 Agent Types

### 1. Standard LangChain Agent (`standard`)
**Traditional AgentExecutor approach**
- Uses `create_openai_functions_agent` and `AgentExecutor`
- Direct function calling with tools
- Straightforward prompt → tools → response workflow
- **Demo value**: Shows basic LangChain patterns
- **Best for**: Simple queries, learning fundamentals, baseline comparison

**Technical Details**:
```python
# Creates standard OpenAI functions agent
agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
result = executor.invoke({"input": query})
```

### 2. LangGraph Research Agent (`langgraph`) - Default
**Advanced multi-step workflow**
- Uses LangGraph's `StateGraph` with conditional edges
- Multi-step process: classify → plan → search → reflect → synthesize
- State management with checkpoints and recursion handling
- **Demo value**: Showcases sophisticated agent architecture
- **Best for**: Complex research, advanced workflows, production patterns

**Technical Details**:
```python
# Creates StateGraph with multiple nodes and conditional flow
workflow = StateGraph(AgentState)
workflow.add_node("classify_query", self._classify_query)
workflow.add_conditional_edges("classify_query", self._should_research, {...})
graph = workflow.compile(checkpointer=self.checkpointer)
```

## 🔧 Developer Benefits

### Common Interface
Both agents implement `AgentInterface`:
```python
class AgentInterface(ABC):
    def process(self, query: str, thread_id: str = "default") -> str:
        pass
```

### Easy Extension
Add new agent types by:
1. Implementing `AgentInterface`
2. Adding to `AgentService._create_agent()`
3. Updating configuration

### Consistent Logging
All agents use the same logging pipeline:
```python
with LogCapture(show_live=show_live_output):
    agent = self._create_agent(agent_type)
    answer = agent.process(query, thread_id)
```

## ⚙️ Configuration

### Supported LLM Providers

#### OpenAI (Standard)
```yaml
llm:
  provider: openai
  models:
    openai:
      name: gpt-4o
      temperature: 0.05
```

#### Azure OpenAI
```yaml
llm:
  provider: azure_openai
  models:
    azure_openai:
      name: gpt-4o
      temperature: 0.05
      azure_endpoint: https://your-resource.openai.azure.com/
      deployment_name: gpt-4o-deployment
      api_version: 2024-05-01-preview
```

#### Anthropic
```yaml
llm:
  provider: anthropic
  models:
    anthropic:
      name: claude-3-5-sonnet-20240620
      temperature: 0.05
```

### Tools Configuration
```yaml
tools:
  - type: "duckduckgo"
    enabled: true
  - type: "wikipedia" 
    enabled: true
  - type: "math"
    enabled: true
  - type: "datetime"
    enabled: true
```

### Graph Settings (LangGraph only)
```yaml
graph:
  recursion_limit: 50  # Workflow iteration limit
```

## 📊 Logging

### Automatic Daily Logs
```
logs/agent_logs_YYYYMMDD.log
```

### Log Content
- Request processing and timing
- Agent creation and execution steps
- Tool usage and search operations
- Errors and performance metrics

## 🛠️ Advanced Features

### Recursion Limit Handling
LangGraph agent intelligently handles iteration limits:
```
⚠️ **DRAFT RESPONSE - INCOMPLETE RESEARCH**
This is a preliminary answer based on limited research data...
```

### HTTP API Documentation
When server is running:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

### Testing
```bash
# Test both agent types
python test_client.py --mode agents

# Test basic functionality
python test_client.py --mode demo
```

## 🔧 Interface Comparison

| Interface | Command | Use Case |
|-----------|---------|----------|
| CLI Simple | `python main.py cli` | Quick testing with LangGraph |
| CLI Advanced | `python main.py cli --mode advanced` | Agent comparison and selection |
| HTTP Server | `python main.py server` | API for external applications |
| REST Client | `python rest_client.py` | Simple API testing and automation |

## 🚨 Troubleshooting

### Common Issues

**Connection Errors**
```bash
# Check if server is running
curl http://localhost:8080/

# Try different port
python main.py server --port 8001
```

**Configuration Issues**
```bash
# Verify config
python -c "from core.config_loader import ConfigLoader; print(ConfigLoader().load_config())"

# Check environment variables
echo $OPENAI_API_KEY
```

**Import Errors**
```bash
# Install dependencies
pip install -r requirements.txt

# Check from project directory
python -c "import core.agent_service"
```

## 📝 Customization Guide

### Adding New Agent Types

1. **Implement Interface**:
```python
class MyCustomAgent(AgentInterface):
    def process(self, query: str, thread_id: str = "default") -> str:
        # Your implementation
        return response
```

2. **Update AgentService**:
```python
def _create_agent(self, agent_type: str, **kwargs):
    if agent_type == "my_custom":
        return MyCustomAgent(self.llm, self.tools)
```

3. **Add Configuration**:
```python
self.agent_types = {
    "standard": "🔧 Standard LangChain Agent",
    "langgraph": "🔬 LangGraph Research Agent",
    "my_custom": "🎯 My Custom Agent"
}
```

### Migration from Complex Setup
- **Before**: 4 agent types with complex parameters
- **After**: 2 clean agent types with common interface
- **Benefit**: Easier to understand, extend, and maintain 

## 🌍 Wsparcie dla języków

Oba agenty automatycznie odpowiadają w tym samym języku co pytanie:

```bash
# Pytanie po polsku
python rest_client.py -q "Co to jest sztuczna inteligencja?"

# Pytanie po angielsku  
python rest_client.py -q "What is artificial intelligence?"

# Pytanie po hiszpańsku
python rest_client.py -q "¿Qué es la inteligencia artificial?"
```

### Mechanizm działania

**Standard Agent**: Instrukcja języka w system prompt
```python
"Answer in the language of the question"
```

**LangGraph Agent**: Instrukcja języka w prompt syntezy
```python
"IMPORTANT: Answer in the same language as the original query."
```

## 🧪 Testy

Wszystkie testy znajdują się w folderze `tests/` i są zorganizowane tak, aby minimalizować koszty LLM.

### Uruchamianie testów

```bash
# Podstawowe testy rozwojowe (bez kosztów LLM)
python run_tests.py --mode quick

# Wszystkie testy jednostkowe 
python run_tests.py --mode unit

# Kompleksowe testy offline (bez kosztów LLM)
python run_tests.py --mode comprehensive

# Instrukcje testów API (wymaga uruchomionego serwera)
python run_tests.py --mode api
```

### Uruchamianie pojedynczych testów

```bash
# Z głównego katalogu
python tests/test_polish_language.py
python tests/test_recursion_limit.py

# Z folderu tests/
cd tests
python test_polish_language.py
python test_recursion_limit.py
python -m unittest test_agents.TestAgentInterface
```

### Testy API z działającym serwerem

```bash
# Uruchom serwer w osobnym terminalu
python main.py server

# Następnie uruchom testy API
python tests/test_client.py --mode demo          # Podstawowe funkcjonalności
python tests/test_client.py --mode language     # Test języków  
python tests/test_client.py --mode errors       # Obsługa błędów
python tests/test_client.py --mode comprehensive # Wszystkie testy API
```

### Struktura testów

**🔧 Testy jednostkowe** (`tests/test_*.py`)
- `test_agents.py` - Interfejs agentów, implementacje, konfiguracja
- `test_agent_service.py` - AgentService, LogCapture, obsługa błędów
- Wszystkie z mockami, bez kosztów LLM

**🚀 Testy funkcjonalne** (uruchamiane pojedynczo)
- `test_polish_language.py` - Spójność języka (bez LLM)
- `test_recursion_limit.py` - Obsługa limitów (bez LLM)
- `test_client.py` - Testy API HTTP (wymaga serwera + LLM)

### Filozofia testowania

**Bez kosztów**: Domyślne testy używają mocków
**Z kosztami**: Testy API wymagają działającego serwera i LLM
**Modularne**: Każdy test można uruchomić osobno
**Praktyczne**: Skupienie na rzeczywistych scenariuszach użycia 