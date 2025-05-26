from typing import Dict, Any
from langchain_core.tools import BaseTool, StructuredTool
from langchain_community.tools import DuckDuckGoSearchRun
from .base_tool import AgentTool

class DuckDuckGoTool(AgentTool):
    """Tool for searching information on the internet using DuckDuckGo"""
    
    @classmethod
    def get_tool(cls, config: Dict[str, Any]) -> BaseTool:
        """
        Creates a DuckDuckGo tool based on configuration.
        
        Args:
            config: Tool configuration from YAML file
            
        Returns:
            DuckDuckGo tool
        """
        search_tool = DuckDuckGoSearchRun()
        name = config.get("name", "DuckDuckGo")
        description = config.get("description", "Useful for searching current information on the internet")
        
        # Get parameters configuration if exists
        params_config = config.get("parameters", {})
        query_description = params_config.get("query", {}).get("description", 
                                                            "Search query")
        
        return StructuredTool.from_function(
            func=lambda query: search_tool.run(query),
            name=name,
            description=description,
            args_schema={
                "query": {
                    "type": "string",
                    "description": query_description
                }
            }
        ) 