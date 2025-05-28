from typing import Dict, Any
from langchain_core.tools import BaseTool, StructuredTool
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from .base_tool import AgentTool

class WikipediaTool(AgentTool):
    """Tool for searching information on Wikipedia"""
    
    @classmethod
    def get_tool(cls, config: Dict[str, Any]) -> BaseTool:
        """
        Creates Wikipedia tool based on configuration.
        
        Args:
            config: Tool configuration from YAML file
            
        Returns:
            Wikipedia tool
        """
        wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        name = config.get("name", "Wikipedia")
        description = config.get("description", "Useful for searching information on Wikipedia")
        
        # Get parameter configuration from config if it exists
        params_config = config.get("parameters", {})
        query_description = params_config.get("query", {}).get("description", 
                                                           "Query to search on Wikipedia")
        
        # Function must explicitly require query parameter
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