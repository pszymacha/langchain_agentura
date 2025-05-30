"""
Unit tests for agent implementations
Tests the agent interface and both standard and langgraph agents
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent_interface import AgentInterface
from core.standard_agent import StandardAgent, create_standard_agent
from core.advanced_agent import AdvancedResearchAgent, create_advanced_agent


class TestAgentInterface(unittest.TestCase):
    """Test the AgentInterface abstract base class"""
    
    def test_interface_cannot_be_instantiated(self):
        """Test that AgentInterface cannot be instantiated directly"""
        with self.assertRaises(TypeError):
            AgentInterface()
    
    def test_interface_requires_implementation(self):
        """Test that implementing classes must implement all abstract methods"""
        
        class IncompleteAgent(AgentInterface):
            pass
        
        with self.assertRaises(TypeError):
            IncompleteAgent()


class TestStandardAgent(unittest.TestCase):
    """Test the StandardAgent implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock LLM
        self.mock_llm = Mock()
        self.mock_llm.invoke.return_value = Mock(content="Test response")
        
        # Use empty tools list to avoid LangChain validation issues
        self.mock_tools = []
        
        # Create agent with patched initialization to avoid LangChain setup
        with patch('core.standard_agent.create_openai_functions_agent'), \
             patch('core.standard_agent.AgentExecutor'):
            self.agent = StandardAgent(self.mock_llm, self.mock_tools, verbose=False)
    
    def test_implements_interface(self):
        """Test that StandardAgent implements AgentInterface"""
        self.assertIsInstance(self.agent, AgentInterface)
    
    def test_has_required_properties(self):
        """Test that required properties are implemented"""
        self.assertIsInstance(self.agent.name, str)
        self.assertIsInstance(self.agent.description, str)
        self.assertIn("Standard LangChain", self.agent.name)
    
    def test_process_method_exists(self):
        """Test that process method exists and is callable"""
        self.assertTrue(hasattr(self.agent, 'process'))
        self.assertTrue(callable(self.agent.process))
    
    @patch('langchain.agents.AgentExecutor')
    def test_process_returns_string(self, mock_executor_class):
        """Test that process method returns a string"""
        # Mock AgentExecutor instance and its invoke method
        mock_executor = Mock()
        mock_executor.invoke.return_value = {"output": "Test response"}
        mock_executor_class.return_value = mock_executor
        
        # Mock the agent_executor attribute
        self.agent.agent_executor = mock_executor
        
        result = self.agent.process("test query")
        
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Test response")
        mock_executor.invoke.assert_called_once()
    
    @patch('langchain.agents.AgentExecutor')
    def test_process_handles_errors(self, mock_executor_class):
        """Test that process method handles errors gracefully"""
        # Mock AgentExecutor instance that raises error
        mock_executor = Mock()
        mock_executor.invoke.side_effect = Exception("Test error")
        mock_executor_class.return_value = mock_executor
        
        # Mock the agent_executor attribute
        self.agent.agent_executor = mock_executor
        
        result = self.agent.process("test query")
        
        self.assertIsInstance(result, str)
        self.assertIn("Error", result)
        self.assertIn("Test error", result)
    
    def test_factory_function(self):
        """Test the create_standard_agent factory function"""
        agent = create_standard_agent(self.mock_llm, self.mock_tools)
        
        self.assertIsInstance(agent, StandardAgent)
        self.assertIsInstance(agent, AgentInterface)


class TestAdvancedAgent(unittest.TestCase):
    """Test the AdvancedResearchAgent implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock LLM
        self.mock_llm = Mock()
        self.mock_llm.invoke.return_value = Mock(content="Test response")
        
        # Use empty tools list
        self.mock_tools = []
        
        # Create agent with patched initialization to avoid LangGraph setup
        with patch('core.advanced_agent.StateGraph'):
            self.agent = AdvancedResearchAgent(
                self.mock_llm, 
                self.mock_tools, 
                verbose=False, 
                recursion_limit=5  # Small limit for testing
            )
    
    def test_implements_interface(self):
        """Test that AdvancedResearchAgent implements AgentInterface"""
        self.assertIsInstance(self.agent, AgentInterface)
    
    def test_has_required_properties(self):
        """Test that required properties are implemented"""
        self.assertIsInstance(self.agent.name, str)
        self.assertIsInstance(self.agent.description, str)
        self.assertIn("LangGraph", self.agent.name)
    
    def test_process_method_exists(self):
        """Test that process method exists and is callable"""
        self.assertTrue(hasattr(self.agent, 'process'))
        self.assertTrue(callable(self.agent.process))
    
    def test_graph_is_created(self):
        """Test that the LangGraph workflow is created"""
        # Since we mock StateGraph, we just verify that the graph attribute exists
        self.assertIsNotNone(self.agent.graph)
    
    def test_tools_dictionary_created(self):
        """Test that tools are properly converted to dictionary"""
        self.assertIsInstance(self.agent.tools, dict)
        # With empty tools list, dictionary should be empty
        self.assertEqual(len(self.agent.tools), 0)
    
    def test_factory_function(self):
        """Test the create_advanced_agent factory function"""
        agent = create_advanced_agent(self.mock_llm, self.mock_tools, verbose=False)
        
        self.assertIsInstance(agent, AdvancedResearchAgent)
        self.assertIsInstance(agent, AgentInterface)


class TestAgentComparison(unittest.TestCase):
    """Test comparison between agent types"""
    
    def setUp(self):
        """Set up both agent types for comparison"""
        # Mock LLM
        self.mock_llm = Mock()
        self.mock_llm.invoke.return_value = Mock(content="Test response")
        
        # Use empty tools list to avoid LangChain validation issues
        self.mock_tools = []
        
        # Create both agents with patched initialization
        with patch('core.standard_agent.create_openai_functions_agent'), \
             patch('core.standard_agent.AgentExecutor'):
            self.standard_agent = StandardAgent(self.mock_llm, self.mock_tools, verbose=False)
        
        with patch('core.advanced_agent.StateGraph'):
            self.advanced_agent = AdvancedResearchAgent(
                self.mock_llm, 
                self.mock_tools, 
                verbose=False, 
                recursion_limit=5
            )
    
    def test_both_implement_interface(self):
        """Test that both agents implement the same interface"""
        self.assertIsInstance(self.standard_agent, AgentInterface)
        self.assertIsInstance(self.advanced_agent, AgentInterface)
    
    def test_different_names(self):
        """Test that agents have different names"""
        self.assertNotEqual(self.standard_agent.name, self.advanced_agent.name)
        self.assertIn("Standard", self.standard_agent.name)
        self.assertIn("LangGraph", self.advanced_agent.name)
    
    def test_different_descriptions(self):
        """Test that agents have different descriptions"""
        self.assertNotEqual(self.standard_agent.description, self.advanced_agent.description)
    
    def test_both_have_process_method(self):
        """Test that both agents have process method with same signature"""
        # Check method exists
        self.assertTrue(hasattr(self.standard_agent, 'process'))
        self.assertTrue(hasattr(self.advanced_agent, 'process'))
        
        # Check method is callable
        self.assertTrue(callable(self.standard_agent.process))
        self.assertTrue(callable(self.advanced_agent.process))


class TestAgentConfiguration(unittest.TestCase):
    """Test agent configuration and parameters"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_llm = Mock()
        self.mock_tools = []
    
    def test_standard_agent_verbose_configuration(self):
        """Test StandardAgent verbose configuration"""
        # Test verbose=True
        agent_verbose = StandardAgent(self.mock_llm, self.mock_tools, verbose=True)
        self.assertTrue(agent_verbose.verbose)
        
        # Test verbose=False
        agent_quiet = StandardAgent(self.mock_llm, self.mock_tools, verbose=False)
        self.assertFalse(agent_quiet.verbose)
    
    def test_advanced_agent_configuration(self):
        """Test AdvancedResearchAgent configuration options"""
        # Test default configuration
        agent_default = AdvancedResearchAgent(self.mock_llm, self.mock_tools)
        self.assertEqual(agent_default.recursion_limit, 50)
        self.assertTrue(agent_default.verbose)
        
        # Test custom configuration
        agent_custom = AdvancedResearchAgent(
            self.mock_llm, 
            self.mock_tools,
            verbose=False,
            recursion_limit=10
        )
        self.assertEqual(agent_custom.recursion_limit, 10)
        self.assertFalse(agent_custom.verbose)
    
    def test_factory_function_parameters(self):
        """Test factory function parameter passing"""
        # Test create_standard_agent
        standard_agent = create_standard_agent(self.mock_llm, self.mock_tools, verbose=False)
        self.assertFalse(standard_agent.verbose)
        
        # Test create_advanced_agent
        advanced_agent = create_advanced_agent(
            self.mock_llm, 
            self.mock_tools, 
            verbose=False, 
            recursion_limit=15
        )
        self.assertFalse(advanced_agent.verbose)
        self.assertEqual(advanced_agent.recursion_limit, 15)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2) 