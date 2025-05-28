"""
Interactive Agent with Multiple Options
Choose between different agent types and modes
"""

import time
from dotenv import load_dotenv
from core.config_loader import ConfigLoader
from core.model_factory import ModelFactory
from core.tool_factory import ToolFactory
from core.advanced_agent import create_advanced_agent
from agent_with_options.custom_agent_patterns import create_controlled_agent, AgentMode

def display_agent_options():
    """Display available agent options"""
    print("\n🤖 Available Agent Types:")
    print("=" * 50)
    print("1. 🔬 LangGraph Agent (Advanced Workflow)")
    print("   • Controlled multi-step research process")
    print("   • Automatic planning and reflection")
    print("   • Shows detailed thinking process")
    print()
    print("2. 🧠 Reflective Agent")
    print("   • Strategic planning before action")
    print("   • Built-in reflection stages")
    print("   • Visible thinking process")
    print()
    print("3. 🔒 Rule-Based Agent")
    print("   • Enforces strict workflow rules")
    print("   • Mandatory first actions for query types")
    print("   • Predictable behavior")
    print()
    print("4. ⚙️ Controlled Executor Agent")
    print("   • Different operational modes")
    print("   • Mode-specific workflows")
    print("   • Enhanced AgentExecutor")
    print("=" * 50)

def display_mode_options():
    """Display mode options for controlled executor"""
    print("\n🎯 Available Modes:")
    print("─" * 30)
    print("1. 🔍 Research Mode")
    print("2. 📊 Analysis Mode") 
    print("3. 🎨 Creative Mode")
    print("4. 🧩 Problem Solving Mode")
    print("─" * 30)

def get_user_choice(prompt: str, valid_choices: list) -> str:
    """Get valid choice from user"""
    while True:
        try:
            choice = input(prompt).strip()
            if choice in valid_choices:
                return choice
            print(f"❌ Please choose from: {', '.join(valid_choices)}")
        except (KeyboardInterrupt, EOFError):
            return "quit"

def main():
    """Interactive agent with multiple options"""
    load_dotenv()
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    
    # Create LLM and tools
    llm = ModelFactory.create_llm(config["llm"])
    tool_factory = ToolFactory()
    tools = tool_factory.create_tools(config["tools"], llm)
    
    print("🚀 Multi-Agent Interactive System")
    print("=" * 60)
    print("Choose your agent type and start researching!")
    print("Type 'exit' at any point to quit.")
    print("=" * 60)
    
    # Available tools info
    tool_names = [tool.name for tool in tools]
    print(f"\nAvailable tools: {', '.join(tool_names)}")
    
    while True:
        try:
            # Display and get agent choice
            display_agent_options()
            agent_choice = get_user_choice(
                "\n🤖 Choose agent type (1-4) or 'exit': ",
                ["1", "2", "3", "4", "exit", "quit"]
            )
            
            if agent_choice in ["exit", "quit"]:
                print("\n👋 Goodbye!")
                break
            
            # Create selected agent
            agent = None
            agent_name = ""
            
            if agent_choice == "1":
                # LangGraph Agent
                agent = create_advanced_agent(llm, tools, verbose=True)
                agent_name = "🔬 LangGraph Research Agent"
                
            elif agent_choice == "2":
                # Reflective Agent
                agent = create_controlled_agent("reflective", llm, tools)
                agent_name = "🧠 Reflective Agent"
                
            elif agent_choice == "3":
                # Rule-based Agent
                rules = {
                    "research_must_search_first": True,
                    "calculations_require_math_tool": True,
                    "max_iterations": 3
                }
                agent = create_controlled_agent("rule_based", llm, tools, rules=rules)
                agent_name = "🔒 Rule-Based Agent"
                
            elif agent_choice == "4":
                # Controlled Executor with mode selection
                display_mode_options()
                mode_choice = get_user_choice(
                    "\n🎯 Choose mode (1-4): ",
                    ["1", "2", "3", "4"]
                )
                
                mode_map = {
                    "1": AgentMode.RESEARCH,
                    "2": AgentMode.ANALYSIS,
                    "3": AgentMode.CREATIVE,
                    "4": AgentMode.PROBLEM_SOLVING
                }
                
                selected_mode = mode_map[mode_choice]
                agent = create_controlled_agent(
                    "controlled_executor", 
                    llm, 
                    tools, 
                    mode=selected_mode
                )
                agent_name = f"⚙️ Controlled Executor ({selected_mode.value.title()} Mode)"
            
            if not agent:
                continue
            
            print(f"\n✅ {agent_name} ready!")
            print("─" * 60)
            
            # Interactive chat loop for selected agent
            thread_counter = 1
            
            while True:
                try:
                    # Get user query
                    user_input = input(f"\n💬 Your query for {agent_name}: ").strip()
                    
                    # Check for exit or change agent
                    if user_input.lower() in ['exit', 'quit']:
                        print("\n🔄 Returning to agent selection...")
                        break
                    elif user_input.lower() in ['change', 'switch']:
                        print("\n🔄 Changing agent...")
                        break
                    elif user_input.lower() in ['help']:
                        print("\n📚 Commands:")
                        print("• Type your research question")
                        print("• 'change' or 'switch' - Change agent type")
                        print("• 'exit' or 'quit' - Return to agent selection")
                        continue
                    
                    if not user_input:
                        continue
                    
                    print(f"\n🚀 Processing with {agent_name}...")
                    start_time = time.time()
                    
                    try:
                        # Execute query based on agent type
                        if agent_choice == "1":
                            # LangGraph agent
                            result = agent.research(user_input, f"session_{thread_counter}")
                        elif agent_choice == "2":
                            # Reflective agent
                            result = agent.solve_with_reflection(user_input)
                        elif agent_choice == "3":
                            # Rule-based agent
                            result = agent.process_with_rules(user_input)
                        elif agent_choice == "4":
                            # Controlled executor
                            result = agent.invoke({"input": user_input})
                            result = result.get("output", str(result))
                        
                        execution_time = time.time() - start_time
                        
                        # Display result
                        print("\n" + "=" * 60)
                        print(f"📊 RESULT FROM {agent_name.upper()}")
                        print("=" * 60)
                        print(result)
                        print("\n" + "=" * 60)
                        print(f"⏱️ Execution time: {execution_time:.2f}s")
                        print(f"🧵 Thread: session_{thread_counter}")
                        
                        thread_counter += 1
                        
                    except KeyboardInterrupt:
                        print("\n\n⚠️ Query interrupted.")
                        continue
                    except Exception as e:
                        print(f"\n❌ Error: {str(e)}")
                        continue
                        
                except KeyboardInterrupt:
                    print("\n🔄 Returning to agent selection...")
                    break
                except EOFError:
                    print("\n👋 Goodbye!")
                    return
                    
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main() 