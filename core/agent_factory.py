"""
Agent Factory - module responsible for creating agents based on configuration
"""

from typing import Dict, Any, List, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages, format_to_tool_messages
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgentOutputParser
from langchain.agents.tool_calling_agent.base import ToolsAgentOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

class AgentFactory:
    """Agent factory supporting various configurations."""
    
    @staticmethod
    def create_agent(llm: BaseLanguageModel, tools: List[BaseTool]) -> AgentExecutor:
        """
        Create agent based on LLM model and tools.
        
        Args:
            llm: Language model
            tools: List of tools
            
        Returns:
            Agent executor (AgentExecutor)
        """
        # Define prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are a helpful, friendly assistant. You respond in English.
                Use available tools when you need to find information,
                but if you know the answer directly, respond without using tools.
                Your responses are concise and specific.
                
                IMPORTANT: When using a tool, you must always provide all required parameters.
                For the Search tool, you must always provide the 'query' parameter with the search query.
                For the Wikipedia tool, you must always provide the 'query' parameter with the search query.
                For the Calculator tool, you must always provide the 'expression' parameter with the expression.
                For the PythonExecutor tool, you must always provide the 'code' parameter with the code.
                The CurrentDate tool does not require parameters.
                
                EXAMPLE: If you want to use the Search tool, call it with the query parameter value "your search query"
                
                NEVER call a tool without providing the required parameters."""),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        
        # Set up model to use tools
        if isinstance(llm, ChatOpenAI):
            # Preparation for OpenAI
            llm_with_tools = llm.bind(functions=[tool.to_json() for tool in tools])
            
            # Define chain
            agent = (
                {
                    "input": lambda x: x["input"],
                    "chat_history": lambda x: x.get("chat_history", []),
                    "agent_scratchpad": lambda x: format_to_openai_function_messages(
                        x.get("intermediate_steps", [])
                    ),
                }
                | prompt
                | llm_with_tools
                | OpenAIFunctionsAgentOutputParser()
            )
        elif isinstance(llm, ChatAnthropic):
            # Preparation for Claude
            llm_with_tools = llm.bind_tools(tools)
            
            # Define chain for Claude with message formatting
            agent = (
                {
                    "input": lambda x: x["input"],
                    "chat_history": lambda x: x.get("chat_history", []),
                    "agent_scratchpad": lambda x: format_to_tool_messages(
                        x.get("intermediate_steps", [])
                    ),
                }
                | prompt
                | llm_with_tools
                | ToolsAgentOutputParser()
            )
        else:
            # For other models use standard tool binding
            agent = llm.bind_tools(tools)
        
        # Create agent executor
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
        )
        
        return agent_executor 