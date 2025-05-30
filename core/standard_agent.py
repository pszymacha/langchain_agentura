"""
Standard LangChain Agent using AgentExecutor
Demonstrates basic LangChain functionality with tools and function calling
"""

from typing import List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor

from .agent_interface import AgentInterface


class StandardAgent(AgentInterface):
    """Standard LangChain agent using AgentExecutor with OpenAI function calling"""
    
    def __init__(self, llm: BaseLanguageModel, tools: List[BaseTool], verbose: bool = True):
        self.llm = llm
        self.tools = tools
        self.verbose = verbose
        self._setup_agent()
    
    def _setup_agent(self):
        """Setup the standard LangChain agent with function calling"""
        
        # Create a prompt template for the agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant with access to various tools.
            
When answering questions:
1. Use the available tools when you need current information or specific calculations
2. For research questions, use DuckDuckGo search to get up-to-date information
3. For mathematical calculations, use the Math tool
4. For time-related queries, use the DateTime tool
5. For general knowledge questions, you can answer directly if confident

Always provide clear, well-structured responses based on the information you gather."""),
            
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the OpenAI functions agent
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        
        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            return_intermediate_steps=False,
            max_iterations=10
        )
    
    def process(self, query: str, thread_id: str = "default") -> str:
        """Process query using standard LangChain AgentExecutor"""
        
        if self.verbose:
            print(f"\n🔧 Standard LangChain Agent")
            print(f"📝 Query: {query}")
            print(f"🧵 Thread: {thread_id}")
            print("=" * 50)
        
        try:
            # Execute the query
            result = self.agent_executor.invoke({"input": query})
            
            # Extract the output
            output = result.get("output", "No response generated")
            
            if self.verbose:
                print("=" * 50)
                print("✅ Processing completed")
            
            return output
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            if self.verbose:
                print(f"❌ {error_msg}")
            return error_msg
    
    @property
    def name(self) -> str:
        """Return the display name of this agent"""
        return "🔧 Standard LangChain Agent"
    
    @property
    def description(self) -> str:
        """Return a description of this agent's capabilities"""
        return "Standard LangChain AgentExecutor with OpenAI function calling and tool usage"


def create_standard_agent(llm: BaseLanguageModel, tools: List[BaseTool], verbose: bool = True) -> StandardAgent:
    """Create a standard LangChain agent"""
    return StandardAgent(llm, tools, verbose) 