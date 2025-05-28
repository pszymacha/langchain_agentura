from typing import Dict, Any, List, Optional, Callable
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, create_model
from abc import ABC, abstractmethod

class AgentTool(ABC):
    """Abstract base class for agent tools"""
    
    @classmethod
    @abstractmethod
    def get_tool(cls, config: Dict[str, Any]) -> BaseTool:
        """
        Abstract method to be implemented in subclasses.
        Creates and returns a langchain tool based on configuration.
        
        Args:
            config: Tool configuration from YAML file
            
        Returns:
            Langchain tool
        """
        pass
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> Optional[BaseTool]:
        """
        Creates tool based on configuration.
        
        Args:
            config: Tool configuration from YAML file
            
        Returns:
            Langchain tool or None if tool is disabled
        """
        if not config.get("enabled", True):
            return None
            
        return cls.get_tool(config) 