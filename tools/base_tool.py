from typing import Dict, Any, List, Optional, Callable
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, create_model
from abc import ABC, abstractmethod

class AgentTool(ABC):
    """Bazowa klasa abstrakcyjna dla narzędzi agenta"""
    
    @classmethod
    @abstractmethod
    def get_tool(cls, config: Dict[str, Any]) -> BaseTool:
        """
        Metoda abstrakcyjna do implementacji w klasach potomnych.
        Tworzy i zwraca narzędzie langchain na podstawie konfiguracji.
        
        Args:
            config: Konfiguracja narzędzia z pliku YAML
            
        Returns:
            Narzędzie langchain
        """
        pass
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> Optional[BaseTool]:
        """
        Tworzy narzędzie na podstawie konfiguracji.
        
        Args:
            config: Konfiguracja narzędzia z pliku YAML
            
        Returns:
            Narzędzie langchain lub None jeśli narzędzie jest wyłączone
        """
        if not config.get("enabled", True):
            return None
            
        return cls.get_tool(config) 