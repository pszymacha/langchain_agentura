import datetime
from typing import Dict, Any
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel
from .base_tool import AgentTool

class DateTimeTool(AgentTool):
    """Tool for getting current date and time"""
    
    @classmethod
    def get_tool(cls, config: Dict[str, Any]) -> BaseTool:
        """
        Creates a DateTime tool based on configuration.
        
        Args:
            config: Tool configuration from YAML file
            
        Returns:
            DateTime tool
        """
        name = config.get("name", "DateTime")
        description = config.get("description", "Get current date and time")
        
        # Define model for tool without parameters
        class DateTimeInput(BaseModel):
            """Input for DateTime tool."""
            pass
        
        def get_current_datetime(_: DateTimeInput = None) -> str:
            """Returns current date and time."""
            now = datetime.datetime.now()
            return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Use StructuredTool with input model without parameters
        return StructuredTool.from_function(
            func=get_current_datetime,
            name=name, 
            description=description
        ) 