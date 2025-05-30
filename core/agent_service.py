"""
Core Agent Service
Centralizes agent management with logging capabilities
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from io import StringIO
import sys

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool

from .advanced_agent import create_advanced_agent
from .standard_agent import create_standard_agent


class LogCapture:
    """Captures print statements and logs during agent execution"""
    
    def __init__(self, show_live: bool = False):
        self.logs = []
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.string_io = StringIO()
        self.show_live = show_live
        
    def __enter__(self):
        if self.show_live:
            # For CLI - show output live and capture it
            self.tee_stdout = TeeOutput(self.original_stdout, self.string_io)
            self.tee_stderr = TeeOutput(self.original_stderr, self.string_io)
            sys.stdout = self.tee_stdout
            sys.stderr = self.tee_stderr
        else:
            # For API - only capture, don't show
            sys.stdout = self.string_io
            sys.stderr = self.string_io
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        # Capture all output
        output = self.string_io.getvalue()
        if output.strip():
            self.logs.append(output.strip())
    
    def add_log(self, message: str):
        """Add a log message manually"""
        timestamp_msg = f"{datetime.now().isoformat()}: {message}"
        self.logs.append(timestamp_msg)
        # If showing live, also print to console
        if self.show_live:
            print(f"🔄 {message}")
    
    def get_logs(self) -> List[str]:
        """Get all captured logs"""
        return self.logs.copy()


class TeeOutput:
    """Write to both original stream and string buffer"""
    
    def __init__(self, original_stream, string_buffer):
        self.original_stream = original_stream
        self.string_buffer = string_buffer
    
    def write(self, text):
        self.original_stream.write(text)
        self.string_buffer.write(text)
        return len(text)
    
    def flush(self):
        self.original_stream.flush()
        self.string_buffer.flush()


class AgentService:
    """Service for managing different types of agents with centralized logging"""
    
    def __init__(self, llm: BaseLanguageModel, tools: List[BaseTool], 
                 default_recursion_limit: int = 50):
        self.llm = llm
        self.tools = tools
        self.default_recursion_limit = default_recursion_limit
        
        # Setup file logging
        self._setup_logging()
        
        # Available agent types - simplified to 2 types
        self.agent_types = {
            "standard": "🔧 Standard LangChain Agent",
            "advanced": "🔬 Advanced Research Agent"
        }
    
    def _setup_logging(self):
        """Setup daily log files"""
        today = datetime.now().strftime("%Y%m%d")
        log_filename = f"logs/agent_logs_{today}.log"
        
        # Create logs directory if it doesn't exist
        import os
        os.makedirs("logs", exist_ok=True)
        
        # Setup file logger
        self.file_logger = logging.getLogger("agent_service")
        self.file_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.file_logger.handlers[:]:
            self.file_logger.removeHandler(handler)
        
        # Add file handler
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.file_logger.addHandler(file_handler)
    
    def get_available_agent_types(self) -> Dict[str, str]:
        """Get available agent types"""
        return self.agent_types.copy()
    
    def get_available_executor_modes(self) -> Dict[str, str]:
        """Get available executor modes"""
        return {
            "standard": "Standard LangChain execution with OpenAI function calling",
            "advanced": "Advanced multi-step workflow with planning and reflection"
        }
    
    def _create_agent(self, agent_type: str, show_live_output: bool = False, **kwargs):
        """Create agent based on type and parameters"""
        
        if agent_type == "standard":
            # For CLI with live output, enable verbose. For API, keep it false to capture properly
            verbose = kwargs.get("verbose", show_live_output)
            return create_standard_agent(self.llm, self.tools, verbose=verbose)
        
        elif agent_type == "advanced":
            recursion_limit = kwargs.get("recursion_limit", self.default_recursion_limit)
            # For CLI with live output, enable verbose. For API, keep it false to capture properly
            verbose = kwargs.get("verbose", show_live_output)
            return create_advanced_agent(self.llm, self.tools, verbose=verbose, 
                                       recursion_limit=recursion_limit)
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(self.agent_types.keys())}")
    
    def _execute_agent(self, agent, agent_type: str, query: str, thread_id: str = None) -> str:
        """Execute agent with the standard process method"""
        return agent.process(query, thread_id or "default")
    
    def process_query(self, query: str, agent_type: str = "advanced", 
                     thread_id: Optional[str] = None, show_live_output: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Process query with specified agent type and return structured response
        
        Args:
            query: User query
            agent_type: Type of agent to use (default: advanced)
            thread_id: Thread ID for session management
            show_live_output: Whether to show live output during processing (for CLI)
            **kwargs: Additional parameters for agent creation
            
        Returns:
            Dict with 'answer', 'logs', 'metadata'
        """
        
        start_time = time.time()
        
        # Generate thread ID if not provided
        if not thread_id:
            thread_id = f"session_{int(time.time())}"
        
        # Log request
        self.file_logger.info(f"Processing query - Agent: {agent_type}, Thread: {thread_id}, Query: {query[:100]}...")
        
        # Capture logs during execution with live output option
        log_capture = LogCapture(show_live=show_live_output)
        
        try:
            with log_capture:
                # Create agent
                log_capture.add_log(f"Creating {agent_type} agent...")
                agent = self._create_agent(agent_type, show_live_output, **kwargs)
                
                log_capture.add_log(f"Agent created successfully: {agent.name}")
                
                # Execute query
                log_capture.add_log(f"Processing query: {query}")
                answer = self._execute_agent(agent, agent_type, query, thread_id)
                
                log_capture.add_log("Query processing completed successfully")
            
            execution_time = time.time() - start_time
            
            # Prepare response
            response = {
                "answer": answer,
                "logs": log_capture.get_logs(),
                "metadata": {
                    "agent_type": agent_type,
                    "agent_name": self.agent_types.get(agent_type, agent_type),
                    "thread_id": thread_id,
                    "execution_time": round(execution_time, 2),
                    "timestamp": datetime.now().isoformat(),
                    "query_length": len(query),
                    "tools_available": [tool.name for tool in self.tools]
                }
            }
            
            # Log successful completion
            self.file_logger.info(f"Query completed - Thread: {thread_id}, Time: {execution_time:.2f}s, Agent: {agent_type}")
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            # Log error
            self.file_logger.error(f"Query failed - Thread: {thread_id}, Time: {execution_time:.2f}s, Error: {error_msg}")
            
            # Add error to logs
            log_capture.add_log(f"ERROR: {error_msg}")
            
            # Return error response
            return {
                "answer": f"An error occurred while processing your query: {error_msg}",
                "logs": log_capture.get_logs(),
                "metadata": {
                    "agent_type": agent_type,
                    "agent_name": self.agent_types.get(agent_type, agent_type),
                    "thread_id": thread_id,
                    "execution_time": round(execution_time, 2),
                    "timestamp": datetime.now().isoformat(),
                    "error": error_msg,
                    "status": "error"
                }
            }
    
    def get_agent_info(self, agent_type: str) -> Dict[str, Any]:
        """Get information about specific agent type"""
        
        if agent_type not in self.agent_types:
            return {"error": f"Unknown agent type: {agent_type}"}
        
        info = {
            "type": agent_type,
            "name": self.agent_types[agent_type],
            "description": self._get_agent_description(agent_type),
            "parameters": self._get_agent_parameters(agent_type)
        }
        
        return info
    
    def _get_agent_description(self, agent_type: str) -> str:
        """Get description for agent type"""
        descriptions = {
            "standard": "Standard LangChain AgentExecutor with OpenAI function calling and direct tool usage",
            "advanced": "Advanced research agent with multi-step workflow including planning, reflection, and synthesis"
        }
        return descriptions.get(agent_type, "No description available")
    
    def _get_agent_parameters(self, agent_type: str) -> Dict[str, Any]:
        """Get available parameters for agent type"""
        
        if agent_type == "advanced":
            return {
                "recursion_limit": {
                    "type": "integer",
                    "default": self.default_recursion_limit,
                    "description": "Maximum number of steps in workflow"
                },
                "verbose": {
                    "type": "boolean", 
                    "default": False,
                    "description": "Enable verbose output"
                }
            }
        
        elif agent_type == "standard":
            return {
                "verbose": {
                    "type": "boolean",
                    "default": False,
                    "description": "Enable verbose output"
                }
            }
        
        return {} 