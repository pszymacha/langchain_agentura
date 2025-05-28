"""
Test script for recursion limit handling in LangGraph agent
This script tests the implementation that gracefully handles GraphRecursionError
"""

import os
import sys
# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from core.config_loader import ConfigLoader
from core.model_factory import ModelFactory
from core.tool_factory import ToolFactory
from core.advanced_agent import create_advanced_agent

def test_recursion_limit_handling():
    """Test the recursion limit handling mechanism"""
    
    print("🧪 Testing Recursion Limit Handling Mechanism")
    print("=" * 60)
    
    # Load environment and configuration
    load_dotenv()
    
    # Change to the parent directory for config loading
    original_cwd = os.getcwd()
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(parent_dir)
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config()
        
        # Create LLM and tools
        llm = ModelFactory.create_llm(config["llm"])
        tool_factory = ToolFactory()
        tools = tool_factory.create_tools(config["tools"], llm)
        
        print(f"✅ LLM loaded: {config['llm']['provider']}")
        print(f"✅ Tools loaded: {[tool.name for tool in tools]}")
        
        # Test Case 1: Normal recursion limit (should work normally)
        print(f"\n📋 TEST CASE 1: Normal recursion limit (50 steps)")
        print("-" * 50)
        
        agent_normal = create_advanced_agent(llm, tools, verbose=True, recursion_limit=50)
        
        test_query = "What is the current weather in Warsaw, Poland?"
        print(f"🔍 Query: {test_query}")
        
        try:
            result = agent_normal.research(test_query, "test_normal")
            print(f"✅ Normal case completed successfully")
            print(f"📄 Result length: {len(result)} characters")
            print(f"📝 First 200 chars: {result[:200]}...")
        except Exception as e:
            print(f"❌ Normal case failed: {str(e)}")
        
        # Test Case 2: Very low recursion limit (should trigger partial answer)
        print(f"\n📋 TEST CASE 2: Very low recursion limit (3 steps)")
        print("-" * 50)
        
        agent_limited = create_advanced_agent(llm, tools, verbose=True, recursion_limit=3)
        
        complex_query = "Conduct a comprehensive research on the impact of artificial intelligence on global economy, including specific examples from different industries, current market trends, regulatory challenges, and future predictions for the next 5 years."
        print(f"🔍 Complex Query: {complex_query}")
        
        try:
            result = agent_limited.research(complex_query, "test_limited")
            print(f"✅ Limited case completed successfully")
            print(f"📄 Result length: {len(result)} characters")
            
            # Check if it contains the expected draft response message
            if "DRAFT RESPONSE - INCOMPLETE RESEARCH" in result or "DRAFT RESPONSE - INCOMPLETE DATA" in result:
                print("✅ Draft response mechanism triggered correctly")
            else:
                print("⚠️ Draft response mechanism may not have triggered")
                
            print(f"📝 Full result:")
            print("-" * 40)
            print(result)
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ Limited case failed: {str(e)}")
        
        # Test Case 3: Extremely low recursion limit (1 step)
        print(f"\n📋 TEST CASE 3: Extremely low recursion limit (1 step)")
        print("-" * 50)
        
        agent_minimal = create_advanced_agent(llm, tools, verbose=True, recursion_limit=1)
        
        minimal_query = "Tell me about Python programming"
        print(f"🔍 Minimal Query: {minimal_query}")
        
        try:
            result = agent_minimal.research(minimal_query, "test_minimal")
            print(f"✅ Minimal case completed successfully")
            print(f"📄 Result length: {len(result)} characters")
            
            # Check if it contains the expected draft response message
            if "DRAFT RESPONSE - INCOMPLETE RESEARCH" in result or "DRAFT RESPONSE - INCOMPLETE DATA" in result:
                print("✅ Draft response mechanism triggered correctly")
            else:
                print("⚠️ Draft response mechanism may not have triggered")
                
            print(f"📝 Full result:")
            print("-" * 40)
            print(result)
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ Minimal case failed: {str(e)}")
        
        print(f"\n🎯 TEST SUMMARY")
        print("=" * 60)
        print("✅ All test cases completed")
        print("📊 The recursion limit handling mechanism has been tested")
        print("💡 If you see 'DRAFT RESPONSE' in results,")
        print("   the draft response mechanism is working correctly!")
        
    finally:
        # Restore the original working directory
        os.chdir(original_cwd)

def main():
    """Main test function"""
    try:
        test_recursion_limit_handling()
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        print("🔍 Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main() 