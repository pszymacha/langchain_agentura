"""
Simple REST Client CLI for Agent API
Minimal interface without fancy formatting
"""

import argparse
import requests
import json
import sys
from typing import Optional, Dict, Any


class SimpleAPIClient:
    """Simple REST client for agent API"""
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def check_connection(self) -> bool:
        """Check if API is accessible"""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_agents(self) -> Optional[Dict[str, Any]]:
        """Get available agent types"""
        try:
            response = self.session.get(f"{self.base_url}/agents", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting agents: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e}")
            return None
    
    def query_agent(self, query: str, agent_type: str = "advanced", 
                   show_logs: bool = False, **kwargs) -> Optional[Dict[str, Any]]:
        """Send query to agent"""
        payload = {
            "query": query,
            "agent_type": agent_type,
            **kwargs
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/query", 
                json=payload, 
                timeout=300  # 5 minute timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display answer
                print("Answer:")
                print(result.get("answer", "No answer received"))
                
                # Show logs if requested
                if show_logs and "logs" in result:
                    print("\nLogs:")
                    for i, log in enumerate(result["logs"], 1):
                        print(f"{i}. {log}")
                
                # Show metadata
                metadata = result.get("metadata", {})
                exec_time = metadata.get("execution_time", "unknown")
                agent_name = metadata.get("agent_name", agent_type)
                print(f"\nAgent: {agent_name} | Time: {exec_time}s")
                
                return result
            else:
                print(f"API error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("Request timeout - query took too long")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e}")
            return None


def run_single_mode(client: SimpleAPIClient, args):
    """Single query mode"""
    # Query is guaranteed to exist since this mode is triggered by -q parameter
    query = args.query.strip()
    
    if not query:
        print("Empty query")
        return
    
    print(f"Sending query to {client.base_url}...")
    
    kwargs = {}
    if args.agent_type:
        kwargs["agent_type"] = args.agent_type
    
    result = client.query_agent(query, show_logs=args.logs, **kwargs)
    
    if not result:
        sys.exit(1)


def run_interactive_mode(client: SimpleAPIClient, args):
    """Interactive mode with query loop"""
    print(f"Connected to {client.base_url}")
    
    # Show available agents if requested
    if args.list_agents:
        agents = client.get_agents()
        if agents:
            print("Available agents:")
            for agent_type, agent_name in agents.items():
                print(f"  {agent_type}: {agent_name}")
        print()
    
    agent_type = args.agent_type or "advanced"
    print(f"Using agent: {agent_type}")
    print("Commands: 'quit', 'exit', 'agents', 'logs on/off'")
    print()
    
    show_logs = args.logs
    
    while True:
        try:
            query = input("Query: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['quit', 'exit']:
                break
                
            if query.lower() == 'agents':
                agents = client.get_agents()
                if agents:
                    for agent_type_key, agent_name in agents.items():
                        print(f"  {agent_type_key}: {agent_name}")
                continue
                
            if query.lower() in ['logs on', 'logs off']:
                show_logs = 'on' in query.lower()
                print(f"Logs {'enabled' if show_logs else 'disabled'}")
                continue
            
            print("Processing...")
            result = client.query_agent(query, agent_type=agent_type, show_logs=show_logs)
            print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            print("\nExiting...")
            break


def main():
    parser = argparse.ArgumentParser(
        description="""Simple REST Client CLI for Agent API

This client allows you to interact with the AI Agent API server through a simple 
command-line interface. You can send queries to different types of AI agents and 
receive structured responses with optional logging information.

The application supports 2 agent types with different capabilities:
• Standard LangChain Agent - Traditional AgentExecutor with function calling
• Advanced Research Agent - Advanced multi-step research with planning and reflection

OPERATION MODES:
• Interactive mode (default): Start without -q parameter for conversation loop
• Single query mode: Use -q "your question" for one-time queries
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE EXAMPLES:

Interactive mode (default):
  python rest_client.py                                    # Start interactive session
  python rest_client.py -p 8001                           # Interactive on custom port
  python rest_client.py --list-agents                     # Show agents, then interactive
  python rest_client.py --logs                            # Interactive with logs enabled

Single query mode (with -q):
  python rest_client.py -q "What is Python?"              # Single question
  python rest_client.py -q "Explain AI" --logs            # Single question with logs
  python rest_client.py -q "research topic" --agent-type advanced  # Specific agent
  python rest_client.py -q "test" -p 8000                 # Query on different port

Different ports and hosts:
  python rest_client.py -p 8001                           # Custom port
  python rest_client.py -H 192.168.1.100 -p 8080         # Custom host and port
  python rest_client.py -q "test" -p 8000                 # Query on different port

Agent types:
  python rest_client.py -q "research topic" --agent-type advanced      # Advanced research
  python rest_client.py -q "simple question" --agent-type standard      # Basic LangChain  
  python rest_client.py -q "complex analysis" --agent-type advanced    # LangGraph workflow

Interactive mode commands:
  • Type any question to send to the agent
  • 'agents' - show available agent types  
  • 'logs on' / 'logs off' - toggle log display
  • 'quit' / 'exit' - exit the application

AGENT TYPES EXPLAINED:

standard:            🔧 Standard LangChain Agent
                     - Traditional AgentExecutor with function calling
                     - Direct tool usage and straightforward workflow
                     - Good baseline for understanding basic LangChain
                     - Best for simple queries and learning fundamentals

advanced:            🔬 Advanced Research Agent  
                     - Advanced multi-step research with planning and reflection
                     - Automatic query classification and planning
                     - Iterative search with reflection and synthesis
                     - Showcases sophisticated agent architecture
                     - Best for complex research and advanced workflows

TYPICAL WORKFLOW:
1. Start the API server: python main.py server -p 8080
2. Run client: python rest_client.py -p 8080
3. Choose agent type or use default (advanced)
4. Send queries and receive structured responses
5. Enable logs (--logs) to see detailed execution steps

ERROR HANDLING:
• Connection errors - Check if API server is running
• Timeout errors - Increase timeout for complex queries  
• Agent errors - Try different agent type or simpler query
• Port conflicts - Use different port with -p flag
        """
    )
    
    parser.add_argument(
        "-H", "--host", 
        default="localhost", 
        help="""API server hostname or IP address (default: localhost)
        
Examples: localhost, 127.0.0.1, 192.168.1.100, api.example.com
Use this when the API server is running on a different machine."""
    )
    
    parser.add_argument(
        "-p", "--port", 
        type=int, 
        default=8080, 
        help="""API server port number (default: 8080)
        
Must match the port used when starting the API server.
Common ports: 8000, 8080, 8001, 3000, 5000
Example: python main.py server -p 8080 (then use -p 8080 here)"""
    )
    
    parser.add_argument(
        "-q", "--query", 
        help="""Query text to send to the agent
        
If provided, runs in SINGLE QUERY MODE - sends query and exits.
If omitted, runs in INTERACTIVE MODE - starts conversation loop.

Examples:
  "What is machine learning?"
  "Analyze the current state of renewable energy"
  "How do I solve this math problem: 2x + 5 = 15?"
  "Research the latest developments in quantum computing"

Interactive mode: python rest_client.py
Single query mode: python rest_client.py -q "your question" """
    )
    
    parser.add_argument(
        "--agent-type", 
        choices=["standard", "advanced"],
        help="""Type of AI agent to use (default: advanced)
        
standard:            🔧 Standard LangChain Agent
                     - Traditional AgentExecutor with function calling
                     - Direct tool usage and straightforward workflow
                     - Good baseline for understanding basic LangChain
                     - Best for simple queries and learning fundamentals

advanced:           🔬 Advanced Research Agent  
                     - Advanced multi-step research with planning and reflection
                     - Automatic query classification and planning
                     - Iterative search with reflection and synthesis
                     - Showcases sophisticated agent architecture
                     - Best for complex research and advanced workflows"""
    )
    
    parser.add_argument(
        "--logs", 
        action="store_true", 
        help="""Show detailed execution logs during processing
        
Displays step-by-step information about what the agent is doing:
• Agent creation and initialization
• Query processing stages  
• Tool usage and search operations
• Reflection and decision making
• Error messages and warnings

Useful for debugging, understanding agent behavior, or monitoring progress.
In interactive mode, you can also use 'logs on/off' commands."""
    )
    
    parser.add_argument(
        "--list-agents", 
        action="store_true", 
        help="""List all available agent types at startup (interactive mode only)
        
Shows a detailed list of agent types with descriptions:
• Agent type identifier (for --agent-type parameter)
• Human-readable name and emoji
• Brief description of capabilities
• Best use cases

This helps you choose the right agent for your task before starting queries."""
    )

    args = parser.parse_args()
    
    # Create client
    client = SimpleAPIClient(args.host, args.port)
    
    # Check connection
    if not client.check_connection():
        print(f"Cannot connect to API at {client.base_url}")
        print("Make sure the server is running:")
        print("  python main.py server")
        sys.exit(1)
    
    # Automatic mode detection based on query parameter
    if args.query:
        # Single query mode - query provided
        run_single_mode(client, args)
    else:
        # Interactive mode - no query provided
        run_interactive_mode(client, args)


if __name__ == "__main__":
    main() 