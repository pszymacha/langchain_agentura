"""
Core Agent Service
Centralizes agent management with logging capabilities and session management
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
from .session_manager import SessionManager, SessionInfo


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
    """Service for managing different types of agents with centralized logging and session management"""
    
    def __init__(self, llm: BaseLanguageModel, tools: List[BaseTool], 
                 default_recursion_limit: int = 50,
                 session_config: Optional[Dict[str, Any]] = None):
        self.llm = llm
        self.tools = tools
        self.default_recursion_limit = default_recursion_limit
        
        # Setup file logging
        self._setup_logging()
        
        # Initialize session manager
        self._init_session_manager(session_config or {})
        
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
    
    def _init_session_manager(self, session_config: Dict[str, Any]):
        """Initialize session manager with configuration"""
        self.session_manager = SessionManager(
            storage_type=session_config.get("storage_type", "memory"),
            db_path=session_config.get("db_path", "data/sessions.db"),
            session_timeout_hours=session_config.get("timeout_hours", 24),
            cleanup_interval_minutes=session_config.get("cleanup_interval_minutes", 60),
            max_sessions_per_user=session_config.get("max_sessions_per_user", 10)
        )
        
        logging.info(f"Session manager initialized with {session_config.get('storage_type', 'memory')} storage")
    
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
                     thread_id: Optional[str] = None, show_live_output: bool = False, 
                     user_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Process query with specified agent type and return structured response
        
        Args:
            query: User query
            agent_type: Type of agent to use (default: advanced)
            thread_id: Thread ID for session management (will be created if None)
            show_live_output: Whether to show live output during processing (for CLI)
            user_id: Optional user identifier for session tracking
            **kwargs: Additional parameters for agent creation
            
        Returns:
            Dict with 'answer', 'logs', 'metadata'
        """
        
        start_time = time.time()
        
        # Handle session management
        if not thread_id:
            # Create new session
            session_metadata = {
                "agent_type": agent_type,
                "created_from": "process_query",
                "initial_query": query[:100] + "..." if len(query) > 100 else query
            }
            thread_id = self.session_manager.create_session(
                user_id=user_id,
                metadata=session_metadata
            )
            logging.info(f"Created new session: {thread_id} for user: {user_id}")
        else:
            # Get or create existing session
            session = self.session_manager.get_session(thread_id)
            if not session:
                # Session doesn't exist, create it
                session_metadata = {
                    "agent_type": agent_type,
                    "created_from": "process_query_existing_id",
                    "initial_query": query[:100] + "..." if len(query) > 100 else query
                }
                new_thread_id = self.session_manager.create_session(
                    user_id=user_id,
                    metadata=session_metadata
                )
                logging.warning(f"Session {thread_id} not found, created new session: {new_thread_id}")
                thread_id = new_thread_id
            else:
                # Update session metadata
                session_update = {
                    "last_query": query[:100] + "..." if len(query) > 100 else query,
                    "last_agent_type": agent_type,
                    "query_count": session.metadata.get("query_count", 0) + 1
                }
                self.session_manager.update_session_metadata(thread_id, session_update)
        
        # Log request
        self.file_logger.info(f"Processing query - Agent: {agent_type}, Thread: {thread_id}, User: {user_id}, Query: {query[:100]}...")
        
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
                
                # Update session context with query and response
                session_context_update = {
                    "last_query": query,
                    "last_response": answer[:200] + "..." if len(answer) > 200 else answer,
                    "last_execution_time": time.time() - start_time,
                    "last_agent_used": agent.name if hasattr(agent, 'name') else agent_type
                }
                self.session_manager.update_session_context(thread_id, session_context_update)
            
            execution_time = time.time() - start_time
            
            # Get session info for metadata
            session = self.session_manager.get_session(thread_id)
            
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
                    "tools_available": [tool.name for tool in self.tools],
                    "session_info": {
                        "user_id": session.user_id if session else None,
                        "session_created": session.created_at.isoformat() if session else None,
                        "session_queries": session.metadata.get("query_count", 1) if session else 1
                    }
                }
            }
            
            # Log successful completion
            self.file_logger.info(f"Query completed - Thread: {thread_id}, User: {user_id}, Time: {execution_time:.2f}s, Agent: {agent_type}")
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            # Log error
            self.file_logger.error(f"Query failed - Thread: {thread_id}, User: {user_id}, Time: {execution_time:.2f}s, Error: {error_msg}")
            
            # Add error to logs
            log_capture.add_log(f"ERROR: {error_msg}")
            
            # Update session with error info
            error_context = {
                "last_error": error_msg,
                "last_error_timestamp": datetime.now().isoformat(),
                "error_count": self.session_manager.get_session(thread_id).context.get("error_count", 0) + 1
            }
            self.session_manager.update_session_context(thread_id, error_context)
            
            # Get session info for metadata
            session = self.session_manager.get_session(thread_id)
            
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
                    "status": "error",
                    "session_info": {
                        "user_id": session.user_id if session else None,
                        "session_created": session.created_at.isoformat() if session else None,
                        "error_count": session.context.get("error_count", 1) if session else 1
                    }
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
    
    # Session management methods
    def create_session(self, user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new session"""
        return self.session_manager.create_session(user_id=user_id, metadata=metadata)
    
    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """Get session information"""
        return self.session_manager.get_session(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        return self.session_manager.delete_session(session_id)
    
    def list_user_sessions(self, user_id: str) -> List[SessionInfo]:
        """List all sessions for a user"""
        return self.session_manager.list_user_sessions(user_id)
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        return self.session_manager.get_session_stats()
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        return self.session_manager.cleanup_expired_sessions() 