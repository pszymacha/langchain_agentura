"""
Integration tests for AgentService
Tests the centralized agent management and service layer
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent_service import AgentService, LogCapture, TeeOutput


class TestLogCapture(unittest.TestCase):
    """Test the LogCapture functionality"""
    
    def test_log_capture_basic(self):
        """Test basic log capture functionality"""
        with LogCapture(show_live=False) as capture:
            print("Test log message")
            capture.add_log("Manual log message")
        
        logs = capture.get_logs()
        self.assertEqual(len(logs), 2)
        # Check that logs contain expected content (order may vary)
        log_text = " ".join(logs)
        self.assertIn("Test log message", log_text)
        self.assertIn("Manual log message", log_text)
    
    def test_log_capture_live_mode(self):
        """Test log capture with live output"""
        with LogCapture(show_live=True) as capture:
            print("Live test message")
        
        logs = capture.get_logs()
        self.assertEqual(len(logs), 1)
        self.assertIn("Live test message", logs[0])
    
    def test_tee_output(self):
        """Test TeeOutput functionality"""
        # Mock streams
        original_stream = Mock()
        string_buffer = Mock()
        
        tee = TeeOutput(original_stream, string_buffer)
        
        # Test write
        result = tee.write("test")
        
        original_stream.write.assert_called_once_with("test")
        string_buffer.write.assert_called_once_with("test")
        self.assertEqual(result, 4)  # Length of "test"
        
        # Test flush
        tee.flush()
        original_stream.flush.assert_called_once()
        string_buffer.flush.assert_called_once()


class TestAgentService(unittest.TestCase):
    """Test the AgentService class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock LLM
        self.mock_llm = Mock()
        self.mock_llm.invoke.return_value = Mock(content="Test response")
        
        # Mock tools
        self.mock_tool = Mock()
        self.mock_tool.name = "test_tool"
        self.mock_tools = [self.mock_tool]
        
        # Create temporary directory for logs
        self.test_dir = tempfile.mkdtemp()
        
        # Create service with mocked os.makedirs
        with patch('os.makedirs'):
            self.service = AgentService(self.mock_llm, self.mock_tools)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_agent_service_initialization(self):
        """Test AgentService initialization"""
        self.assertEqual(self.service.llm, self.mock_llm)
        self.assertEqual(self.service.tools, self.mock_tools)
        self.assertEqual(self.service.default_recursion_limit, 50)
        
        # Test agent types
        expected_types = {"standard", "advanced"}
        self.assertEqual(set(self.service.agent_types.keys()), expected_types)
    
    def test_get_available_agent_types(self):
        """Test getting available agent types"""
        agent_types = self.service.get_available_agent_types()
        
        self.assertIsInstance(agent_types, dict)
        self.assertIn("standard", agent_types)
        self.assertIn("advanced", agent_types)
        self.assertIn("Standard LangChain", agent_types["standard"])
        self.assertIn("Advanced Research", agent_types["advanced"])
    
    @patch('core.agent_service.create_standard_agent')
    def test_create_standard_agent(self, mock_create):
        """Test creating standard agent"""
        mock_agent = Mock()
        mock_create.return_value = mock_agent
        
        agent = self.service._create_agent("standard", verbose=True)
        
        mock_create.assert_called_once_with(self.mock_llm, self.mock_tools, verbose=True)
        self.assertEqual(agent, mock_agent)
    
    @patch('core.agent_service.create_advanced_agent')
    def test_create_advanced_agent(self, mock_create):
        """Test creating advanced agent"""
        # Mock the agent creation
        mock_agent = MagicMock()
        mock_agent.name = "Test Advanced Agent"
        mock_create.return_value = mock_agent
        
        agent = self.service._create_agent("advanced", recursion_limit=25, verbose=False)
        
        mock_create.assert_called_once_with(
            self.mock_llm,
            self.mock_tools,
            verbose=False,
            recursion_limit=25
        )
        self.assertEqual(agent, mock_agent)
    
    def test_create_invalid_agent(self):
        """Test creating invalid agent type raises error"""
        with self.assertRaises(ValueError) as context:
            self.service._create_agent("invalid_type")
        
        self.assertIn("Unknown agent type", str(context.exception))
        self.assertIn("invalid_type", str(context.exception))
    
    def test_execute_agent(self):
        """Test executing agent through service"""
        mock_agent = Mock()
        mock_agent.process.return_value = "Agent response"
        
        result = self.service._execute_agent(mock_agent, "standard", "test query", "thread123")
        
        mock_agent.process.assert_called_once_with("test query", "thread123")
        self.assertEqual(result, "Agent response")
    
    @patch('core.agent_service.create_standard_agent')
    def test_process_query_success(self, mock_create):
        """Test successful query processing"""
        # Mock agent
        mock_agent = Mock()
        mock_agent.name = "🔧 Test Agent"
        mock_agent.process.return_value = "Test response"
        mock_create.return_value = mock_agent
        
        # Process query
        result = self.service.process_query(
            query="test query",
            agent_type="standard",
            thread_id="test_thread"
        )
        
        # Verify response structure
        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertIn("logs", result)
        self.assertIn("metadata", result)
        
        # Verify content
        self.assertEqual(result["answer"], "Test response")
        self.assertIsInstance(result["logs"], list)
        
        # Verify metadata
        metadata = result["metadata"]
        self.assertEqual(metadata["agent_type"], "standard")
        self.assertEqual(metadata["thread_id"], "test_thread")
        self.assertIn("execution_time", metadata)
        self.assertIn("timestamp", metadata)
    
    @patch('core.agent_service.create_standard_agent')
    def test_process_query_with_error(self, mock_create):
        """Test query processing with error"""
        # Mock agent that raises error
        mock_agent = Mock()
        mock_agent.name = "🔧 Test Agent"
        mock_agent.process.side_effect = Exception("Test error")
        mock_create.return_value = mock_agent
        
        # Process query
        result = self.service.process_query(
            query="test query",
            agent_type="standard"
        )
        
        # Verify error handling
        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertIn("error occurred", result["answer"])
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["status"], "error")
    
    def test_get_agent_info_valid(self):
        """Test getting agent info for valid agent type"""
        info = self.service.get_agent_info("standard")
        
        self.assertIsInstance(info, dict)
        self.assertEqual(info["type"], "standard")
        self.assertIn("name", info)
        self.assertIn("description", info)
        self.assertIn("parameters", info)
        
        # Check parameters structure
        parameters = info["parameters"]
        self.assertIn("verbose", parameters)
        self.assertEqual(parameters["verbose"]["type"], "boolean")
    
    def test_get_agent_info_invalid(self):
        """Test getting agent info for invalid agent type"""
        info = self.service.get_agent_info("invalid_type")
        
        self.assertIn("error", info)
        self.assertIn("Unknown agent type", info["error"])
    
    def test_advanced_agent_parameters(self):
        """Test advanced agent specific parameters"""
        info = self.service.get_agent_info("advanced")
        
        parameters = info["parameters"]
        self.assertIn("recursion_limit", parameters)
        self.assertIn("verbose", parameters)
        
        recursion_param = parameters["recursion_limit"]
        self.assertEqual(recursion_param["type"], "integer")
        self.assertEqual(recursion_param["default"], 50)
    
    def test_thread_id_generation(self):
        """Test automatic thread ID generation"""
        with patch('core.agent_service.create_standard_agent') as mock_create:
            mock_agent = Mock()
            mock_agent.name = "Test Agent"
            mock_agent.process.return_value = "Response"
            mock_create.return_value = mock_agent
            
            # Process without thread_id
            result = self.service.process_query("test query", "standard")
            
            # Verify thread_id was generated
            thread_id = result["metadata"]["thread_id"]
            self.assertIsInstance(thread_id, str)
            self.assertIn("session_", thread_id)
    
    @patch('core.agent_service.time.time')
    def test_execution_timing(self, mock_time):
        """Test execution time measurement"""
        # Mock time progression with enough values for all logging calls
        mock_time.side_effect = [1000.0, 1000.1, 1000.2, 1003.5, 1003.6, 1003.7]  # Multiple time calls
        
        with patch('core.agent_service.create_standard_agent') as mock_create:
            mock_agent = Mock()
            mock_agent.name = "Test Agent"
            mock_agent.process.return_value = "Response"
            mock_create.return_value = mock_agent
            
            result = self.service.process_query("test query", "standard")
            
            # Verify execution time calculation (should be around 3.5 seconds)
            execution_time = result["metadata"]["execution_time"]
            self.assertGreater(execution_time, 3.0)
            self.assertLess(execution_time, 4.0)


class TestAgentServiceIntegration(unittest.TestCase):
    """Integration tests for AgentService with real components"""
    
    def setUp(self):
        """Set up real components for integration testing"""
        # Mock LLM with more realistic behavior
        self.mock_llm = Mock()
        self.mock_llm.invoke.return_value = Mock(content="Integration test response")
        
        # Mock tools
        self.mock_tools = []
        
        # Create service with mocked os.makedirs
        with patch('os.makedirs'):
            self.service = AgentService(self.mock_llm, self.mock_tools)
    
    def test_standard_vs_advanced_agent_creation(self):
        """Test that different agent types are created correctly"""
        with patch('core.agent_service.create_standard_agent') as mock_standard, \
             patch('core.agent_service.create_advanced_agent') as mock_advanced:
            
            mock_standard_agent = Mock()
            mock_advanced_agent = Mock()
            mock_standard.return_value = mock_standard_agent
            mock_advanced.return_value = mock_advanced_agent
            
            # Create standard agent
            standard_agent = self.service._create_agent("standard")
            self.assertEqual(standard_agent, mock_standard_agent)
            mock_standard.assert_called_once()
            
            # Create advanced agent
            advanced_agent = self.service._create_agent("advanced")
            self.assertEqual(advanced_agent, mock_advanced_agent)
            mock_advanced.assert_called_once()
    
    def test_end_to_end_query_processing(self):
        """Test complete query processing pipeline"""
        with patch('core.agent_service.create_standard_agent') as mock_create:
            # Set up mock agent
            mock_agent = Mock()
            mock_agent.name = "🔧 Standard LangChain Agent"
            mock_agent.process.return_value = "Complete integration test response"
            mock_create.return_value = mock_agent
            
            # Process query with various parameters
            result = self.service.process_query(
                query="Integration test query",
                agent_type="standard",
                thread_id="integration_test",
                show_live_output=False,
                verbose=True
            )
            
            # Verify complete response structure
            required_keys = ["answer", "logs", "metadata"]
            for key in required_keys:
                self.assertIn(key, result)
            
            # Verify metadata completeness
            metadata = result["metadata"]
            required_metadata = [
                "agent_type", "agent_name", "thread_id", 
                "execution_time", "timestamp", "query_length", "tools_available"
            ]
            for key in required_metadata:
                self.assertIn(key, metadata)
            
            # Verify agent was called correctly
            mock_agent.process.assert_called_once_with("Integration test query", "integration_test")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2) 