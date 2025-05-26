from typing import Dict, Any
from langchain_core.tools import BaseTool, StructuredTool
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from .base_tool import AgentTool

class WikipediaTool(AgentTool):
    """Narzędzie do wyszukiwania informacji na Wikipedii"""
    
    @classmethod
    def get_tool(cls, config: Dict[str, Any]) -> BaseTool:
        """
        Tworzy narzędzie Wikipedia na podstawie konfiguracji.
        
        Args:
            config: Konfiguracja narzędzia z pliku YAML
            
        Returns:
            Narzędzie Wikipedia
        """
        wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        name = config.get("name", "Wikipedia")
        description = config.get("description", "Użyteczne do wyszukiwania informacji na Wikipedii")
        
        # Pobierz konfigurację parametrów z config jeśli istnieje
        params_config = config.get("parameters", {})
        query_description = params_config.get("query", {}).get("description", 
                                                           "Zapytanie do wyszukania na Wikipedii")
        
        # Funkcja musi wyraźnie wymagać parametru query
        return StructuredTool.from_function(
            func=lambda query: wiki_tool.run(query),
            name=name,
            description=description,
            args_schema={
                "query": {
                    "type": "string",
                    "description": query_description
                }
            }
        ) 