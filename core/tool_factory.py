"""
Tool Factory - a module responsible for creating tools for the agent
based on configuration from a YAML file.
"""

from typing import Dict, Any, List
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from tools import TOOL_REGISTRY

class ToolFactory:
    """Tool factory for the agent supporting various tool types."""
    
    @staticmethod
    def create_tools(config: List[Dict[str, Any]], llm: BaseLanguageModel) -> List[BaseTool]:
        """
        Create a list of tools based on configuration.
        
        Args:
            config: List of dictionaries with tool configuration from YAML file
            llm: LLM model instance to be used by tools (e.g. for calculator)
            
        Returns:
            List of tools for the agent
        """
        tools = []
        
        for tool_config in config:
            if not tool_config.get("enabled", True):
                continue
                
            tool_type = tool_config.get("type", "").lower()
            
            # Add LLM reference if required by the tool
            if tool_type == "math":
                tool_config["llm"] = llm
            
            # Get tool class from registry
            tool_class = TOOL_REGISTRY.get(tool_type)
            
            if tool_class:
                # Create tool using class from registry
                tool = tool_class.create_from_config(tool_config)
                if tool:
                    tools.append(tool)
                    print(f"Tool: {tool}")
            else:
                print(f"WARNING: Unknown tool type: {tool_type}")
        
        return tools 