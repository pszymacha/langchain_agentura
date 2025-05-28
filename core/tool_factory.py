"""
Tool Factory - moduł odpowiedzialny za tworzenie narzędzi dla agenta
na podstawie konfiguracji z pliku YAML.
"""

from typing import Dict, Any, List
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from tools import TOOL_REGISTRY

class ToolFactory:
    """Fabryka narzędzi dla agenta wspierająca różne typy narzędzi."""
    
    @staticmethod
    def create_tools(config: List[Dict[str, Any]], llm: BaseLanguageModel) -> List[BaseTool]:
        """
        Tworzy listę narzędzi na podstawie konfiguracji.
        
        Args:
            config: Lista słowników z konfiguracją narzędzi z pliku YAML
            llm: Instancja modelu LLM do użycia przez narzędzia (np. dla kalkulatora)
            
        Returns:
            Lista narzędzi (tools) dla agenta
        """
        tools = []
        
        for tool_config in config:
            if not tool_config.get("enabled", True):
                continue
                
            tool_type = tool_config.get("type", "").lower()
            
            # Dodaj referencję do LLM, jeśli jest wymagana przez narzędzie
            if tool_type == "math":
                tool_config["llm"] = llm
            
            # Pobierz klasę narzędzia z rejestru
            tool_class = TOOL_REGISTRY.get(tool_type)
            
            if tool_class:
                # Utwórz narzędzie za pomocą klasy z rejestru
                tool = tool_class.create_from_config(tool_config)
                if tool:
                    tools.append(tool)
                    print(f"Tool: {tool}")
            else:
                print(f"UWAGA: Nieznany typ narzędzia: {tool_type}")
        
        return tools 