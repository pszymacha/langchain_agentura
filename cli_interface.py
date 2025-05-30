"""
Simple CLI Interface for AI Agents
Supports single query mode and interactive mode
"""

import argparse
import time
import sys
from dotenv import load_dotenv

from core.config_loader import ConfigLoader
from core.model_factory import ModelFactory
from core.tool_factory import ToolFactory
from core.agent_service import AgentService


def process_single_query(agent_service: AgentService, query: str, agent_type: str, verbose: bool = False):
    """Process a single query and return the result"""
    print(f"🤖 Agent: {agent_type}")
    print(f"🔍 Query: {query}")
    print("⏳ Processing...")
    
    start_time = time.time()
    
    try:
        result = agent_service.process_query(
            query=query,
            agent_type=agent_type,
            thread_id="cli_single",
            show_live_output=verbose,
            verbose=verbose
        )
        
        execution_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("📊 RESULT")
        print("=" * 60)
        print(result["answer"])
        
        if verbose:
            print("\n" + "=" * 60)
            print("📋 EXECUTION LOGS")
            print("=" * 60)
            for i, log in enumerate(result["logs"], 1):
                print(f"{i:2d}. {log}")
        
        print("\n" + "=" * 60)
        print(f"⏱️ Execution time: {execution_time:.2f}s")
        print(f"🧵 Thread: {result['metadata']['thread_id']}")
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return None


def run_interactive_mode(agent_service: AgentService, agent_type: str, verbose: bool = False):
    """Run interactive CLI mode"""
    agent_info = agent_service.get_agent_info(agent_type)
    agent_name = agent_info["name"]
    
    print(f"🤖 Interactive mode with {agent_name}")
    print("=" * 60)
    print("Type your queries or 'exit' to quit.")
    print("=" * 60)
    
    thread_counter = 1
    
    while True:
        try:
            query = input(f"\n💬 Query: ").strip()
            
            if query.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                break
                
            if not query:
                continue
                
            print(f"\n🚀 Processing with {agent_name}...")
            start_time = time.time()
            
            try:
                result = agent_service.process_query(
                    query=query,
                    agent_type=agent_type,
                    thread_id=f"cli_interactive_{thread_counter}",
                    show_live_output=verbose,
                    verbose=verbose
                )
                
                execution_time = time.time() - start_time
                
                print("\n" + "=" * 60)
                print("📊 RESULT")
                print("=" * 60)
                print(result["answer"])
                
                if verbose:
                    print("\n" + "=" * 60)
                    print("📋 EXECUTION LOGS")
                    print("=" * 60)
                    for i, log in enumerate(result["logs"], 1):
                        print(f"{i:2d}. {log}")
                
                print("\n" + "=" * 60)
                print(f"⏱️ Execution time: {execution_time:.2f}s")
                print("=" * 60)
                
                thread_counter += 1
                
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                continue
                
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Simple AI Agent CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query mode
  python cli_interface.py -q "What is 2+2?" -a standard
  python cli_interface.py -q "What's the weather like?" -a advanced -v
  
  # Interactive mode  
  python cli_interface.py -a standard
  python cli_interface.py -a advanced -v
  
  # With custom server settings
  python cli_interface.py -q "Hello" -a standard -p 8080 -e http://localhost
  
Agent types:
  standard  - Standard LangChain agent with function calling
  advanced  - Advanced research agent with multi-step workflow

Available tools: DuckDuckGo, Wikipedia, Math, DateTime
        """
    )
    
    parser.add_argument(
        "-q", "--query", 
        type=str,
        help="Single query to process (if not provided, runs in interactive mode)"
    )
    
    parser.add_argument(
        "-a", "--agent", 
        choices=["standard", "advanced"], 
        default="advanced",
        help="Agent type to use (default: advanced)"
    )
    
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Enable verbose output with execution logs"
    )
    
    parser.add_argument(
        "-p", "--port", 
        type=int, 
        default=8080,
        help="Server port (for future HTTP client mode, default: 8080)"
    )
    
    parser.add_argument(
        "-e", "--endpoint", 
        type=str, 
        default="http://localhost",
        help="Server endpoint (for future HTTP client mode, default: http://localhost)"
    )
    
    args = parser.parse_args()
    
    # Load environment and initialize service
    load_dotenv()
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config()
        
        # Create LLM and tools
        llm = ModelFactory.create_llm(config["llm"])
        tool_factory = ToolFactory()
        tools = tool_factory.create_tools(config["tools"], llm)
        
        # Create agent service
        recursion_limit = config.get("graph", {}).get("recursion_limit", 50)
        agent_service = AgentService(llm, tools, recursion_limit)
        
        # Validate agent type
        available_agents = agent_service.get_available_agent_types()
        if args.agent not in available_agents:
            print(f"❌ Invalid agent type: {args.agent}")
            print(f"Available agents: {', '.join(available_agents.keys())}")
            sys.exit(1)
        
        # Show startup info
        print("🚀 AI Agent CLI Interface")
        print("=" * 60)
        tool_names = [tool.name for tool in tools]
        print(f"📦 Available tools: {', '.join(tool_names)}")
        print(f"🔄 Recursion limit: {recursion_limit}")
        print("=" * 60)
        
        # Run in single query or interactive mode
        if args.query:
            # Single query mode
            result = process_single_query(agent_service, args.query, args.agent, args.verbose)
            sys.exit(0 if result else 1)
        else:
            # Interactive mode
            run_interactive_mode(agent_service, args.agent, args.verbose)
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error initializing CLI: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 