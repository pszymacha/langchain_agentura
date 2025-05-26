"""
Interactive Advanced Agent with controlled workflow
Allows user to input custom queries from command line
"""

import os
import time
from dotenv import load_dotenv
from config_loader import ConfigLoader
from model_factory import ModelFactory
from tool_factory import ToolFactory
from advanced_agent import create_advanced_agent

def print_thinking_step(step_name: str, content: str, max_length: int = 200):
    """Print a thinking step with nice formatting"""
    print(f"\n🧠 {step_name}:")
    print("-" * 40)
    if len(content) > max_length:
        print(f"{content[:max_length]}...")
        print(f"[Truncated - showing first {max_length} characters]")
    else:
        print(content)
    print("-" * 40)

def main():
    """Interactive advanced research agent"""
    load_dotenv()
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    
    # Create LLM
    llm = ModelFactory.create_llm(config["llm"])
    
    # Create tools
    tool_factory = ToolFactory()
    tools = tool_factory.create_tools(config["tools"], llm)
    
    # Create advanced agent
    agent = create_advanced_agent(llm, tools)
    
    # Display welcome message
    print("🔬 Advanced Research Agent - Interactive Mode")
    print("=" * 60)
    print("This agent uses controlled workflow with:")
    print("• Automatic query classification")
    print("• Strategic research planning")
    print("• Iterative search and reflection")
    print("• Comprehensive synthesis")
    print("\nType 'help' for commands or 'exit' to quit.")
    print("=" * 60)
    
    # Available tools info
    tool_names = [tool.name for tool in tools]
    print(f"\nAvailable tools: {', '.join(tool_names)}")
    
    # Chat history for thread management
    thread_counter = 1
    
    while True:
        try:
            # Get user input
            user_input = input("\n🔍 Your research query: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'wyjdź', 'koniec']:
                print("\n👋 Goodbye!")
                break
            
            # Check for help
            if user_input.lower() in ['help', 'pomoc']:
                print("\n📚 Available commands:")
                print("• Type any research question to get started")
                print("• 'exit' or 'quit' - Exit the program")
                print("• 'help' - Show this help message")
                print("\n💡 Tips:")
                print("• Ask about current events for research mode")
                print("• Ask general knowledge questions for direct answers")
                print("• The agent will show its thinking process")
                continue
            
            # Skip empty input
            if not user_input:
                continue
            
            print(f"\n🚀 Processing query: {user_input}")
            print("⏳ Starting advanced research workflow...")
            
            # Create unique thread ID for this query
            thread_id = f"interactive_session_{thread_counter}"
            thread_counter += 1
            
            # Start timing
            start_time = time.time()
            
            try:
                # Execute the advanced research workflow
                # The agent will automatically handle classification, planning, searching, reflection
                result = agent.research(user_input, thread_id=thread_id)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Display final result
                print("\n" + "=" * 60)
                print("📊 FINAL RESEARCH RESULT")
                print("=" * 60)
                print(result)
                print("\n" + "=" * 60)
                print(f"⏱️ Total execution time: {execution_time:.2f} seconds")
                print(f"🧵 Thread ID: {thread_id}")
                
            except KeyboardInterrupt:
                print("\n\n⚠️ Research interrupted by user.")
                print("You can start a new query or type 'exit' to quit.")
                continue
                
            except Exception as e:
                print(f"\n❌ Error during research: {str(e)}")
                print("Please try again with a different query.")
                continue
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main() 