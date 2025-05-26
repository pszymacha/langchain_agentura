"""
Demo of Advanced Agent with controlled workflow
"""

import os
from dotenv import load_dotenv
from config_loader import ConfigLoader
from model_factory import ModelFactory
from tool_factory import ToolFactory
from advanced_agent import create_advanced_agent

def main():
    """Demo of advanced research agent"""
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
    
    print("🔬 Advanced Research Agent Demo")
    print("=" * 50)
    
    # Example research queries
    research_queries = [
        "What are the latest AI developments?",
        "How is artificial intelligence being used in healthcare today?",
        "What is the capital of France?",  # This should be classified as direct
    ]
    
    for i, query in enumerate(research_queries, 1):
        print(f"\n🔍 Research Query {i}:")
        print(f"Query: {query}")
        print("-" * 30)
        
        try:
            result = agent.research(query, thread_id=f"demo_thread_{i}")
            print("📊 Final Answer:")
            print(result)
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "="*50)

    # Additional steps for the advanced agent
    print("\n🔄 STEP 1: Query Classification")
    print("──────────────────────────────────────────────────")
    print("Analyzing query: What are the latest AI developments?")
    print("──────────────────────────────────────────────────")

    print("\n🔄 STEP 2: Research Planning")
    print("──────────────────────────────────────────────────")
    print("Creating strategic research plan...")
    print("──────────────────────────────────────────────────")

    print("\n🔄 STEP 3: Initial Search (Iteration 1)")
    print("──────────────────────────────────────────────────")
    print("Executing broad search: What are the latest AI developments?")
    print("──────────────────────────────────────────────────")

if __name__ == "__main__":
    main() 