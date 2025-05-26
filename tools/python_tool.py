from typing import Dict, Any
from langchain_core.tools import BaseTool, StructuredTool
from .base_tool import AgentTool

class PythonTool(AgentTool):
    """Tool for executing Python code"""
    
    @classmethod
    def get_tool(cls, config: Dict[str, Any]) -> BaseTool:
        """
        Creates a Python tool based on configuration.
        
        Args:
            config: Tool configuration from YAML file
            
        Returns:
            Python tool
        """
        name = config.get("name", "Python")
        description = config.get("description", "Execute Python code for calculations and data processing")
        
        # Implementation of safe Python code execution
        def python_executor(code: str) -> str:
            """Executes Python code (in reality only simulates execution for security reasons)."""
            return (
                "Python code execution has been disabled for security reasons. "
                "Instead, returning a sample response. Code that was attempted to execute:\n\n"
                f"{code}\n\n"
                "In production environment, safe code execution should be implemented."
            )
        
        # Get parameters configuration if exists
        params_config = config.get("parameters", {})
        code_description = params_config.get("code", {}).get("description", 
                                                       "Python code to execute")
        
        # Function must explicitly require code parameter
        return StructuredTool.from_function(
            func=python_executor,
            name=name,
            description=description,
            args_schema={
                "code": {
                    "type": "string", 
                    "description": code_description
                }
            }
        ) 