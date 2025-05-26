from typing import Dict, Any
from langchain_core.tools import BaseTool, StructuredTool
from langchain.chains.llm_math.base import LLMMathChain
from langchain_core.language_models import BaseLanguageModel
from .base_tool import AgentTool

class MathTool(AgentTool):
    """Tool for mathematical calculations"""
    
    @classmethod
    def get_tool(cls, config: Dict[str, Any]) -> BaseTool:
        """
        Creates a Math tool based on configuration.
        
        Args:
            config: Tool configuration from YAML file
                   (must contain 'llm' key with language model instance)
            
        Returns:
            Math tool
        """
        name = config.get("name", "Math")
        description = config.get("description", "Useful for solving mathematical problems")
        llm = config.get("llm")
        
        if not llm or not isinstance(llm, BaseLanguageModel):
            raise ValueError("Math tool requires a language model (llm)")
        
        math_chain = LLMMathChain.from_llm(llm)
        
        # Get parameters configuration if exists
        params_config = config.get("parameters", {})
        expression_description = params_config.get("expression", {}).get("description", 
                                                              "Mathematical expression to evaluate")
        
        return StructuredTool.from_function(
            func=lambda expression: math_chain.invoke(expression)["answer"],
            name=name,
            description=description,
            args_schema={
                "expression": {
                    "type": "string",
                    "description": expression_description
                }
            }
        ) 