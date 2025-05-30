"""
Common interface for all agent implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class AgentInterface(ABC):
    """Abstract base class defining the common interface for all agents"""
    
    @abstractmethod
    def process(self, query: str, thread_id: str = "default") -> str:
        """
        Process a user query and return the response
        
        Args:
            query: User's input query
            thread_id: Thread/session identifier for context management
            
        Returns:
            String response to the query
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the display name of this agent"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of this agent's capabilities"""
        pass 