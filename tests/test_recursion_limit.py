"""
Test script for recursion limit handling in LangGraph agent
Simplified version using mocks to avoid LLM costs
"""

import os
import sys
from unittest.mock import Mock, patch
# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.advanced_agent import AdvancedResearchAgent


def test_recursion_limit_basic():
    """Basic test of recursion limit handling without LLM costs"""
    
    print("🧪 Testing Recursion Limit Handling (Mock Mode)")
    print("=" * 60)
    
    # Mock LLM
    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(content="Mock LLM response")
    
    # Mock tools
    mock_tools = []
    
    print("✅ Mock LLM and tools created")
    
    # Test Case 1: Very low recursion limit
    print(f"\n📋 TEST CASE: Very low recursion limit (2 steps)")
    print("-" * 50)
    
    try:
        with patch('core.advanced_agent.StateGraph'):
            agent = AdvancedResearchAgent(mock_llm, mock_tools, verbose=True, recursion_limit=2)
            
            # Mock the graph's invoke method to simulate GraphRecursionError
            def mock_invoke(*args, **kwargs):
                from langgraph.errors import GraphRecursionError
                raise GraphRecursionError("Recursion limit exceeded")
            
            agent.graph = Mock()
            agent.graph.invoke = mock_invoke
            agent.graph.get_state_history.return_value = []
            
            test_query = "Complex research query that should hit recursion limit"
            print(f"🔍 Query: {test_query}")
            
            result = agent.process(test_query, "test_thread")
            
            print(f"✅ Test completed successfully")
            print(f"📄 Result length: {len(result)} characters")
            
            # Check if it contains the expected draft response message
            expected_markers = [
                "DRAFT RESPONSE - INCOMPLETE DATA",
                "DRAFT RESPONSE - INCOMPLETE RESEARCH",
                "maximum allowed number of steps",
                "iteration limit"
            ]
            
            found_markers = [marker for marker in expected_markers if marker in result]
            
            if found_markers:
                print("✅ Draft response mechanism triggered correctly")
                print(f"📝 Found markers: {', '.join(found_markers)}")
            else:
                print("❌ Draft response mechanism failed")
                print(f"📝 Result preview: {result[:200]}...")
                
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎯 TEST SUMMARY")
    print("=" * 60)
    print("✅ Recursion limit test completed")
    print("💡 This test verifies the graceful handling of recursion limits")
    print("🚀 For real LLM testing, use the full application")


def main():
    """Main test function"""
    try:
        test_recursion_limit_basic()
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")


if __name__ == "__main__":
    main() 