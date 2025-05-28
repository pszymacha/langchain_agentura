"""
Agent Factory - moduł odpowiedzialny za tworzenie agenta na podstawie konfiguracji
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
    """Fabryka agentów wspierająca różne konfiguracje."""
    
    @staticmethod
    def create_agent(llm: BaseLanguageModel, tools: List[BaseTool]) -> AgentExecutor:
        """
        Tworzy agenta na podstawie modelu LLM i narzędzi.
        
        Args:
            llm: Model językowy
            tools: Lista narzędzi
            
        Returns:
            Wykonawca agenta (AgentExecutor)
        """
        # Definiowanie promptu
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """Jesteś pomocnym, przyjaznym asystentem. Odpowiadasz po polsku.
                Używasz dostępnych narzędzi, gdy potrzebujesz znaleźć informacje, 
                ale jeśli znasz odpowiedź bezpośrednio to odpowiadasz bez używania narzędzi.
                Twoje odpowiedzi są zwięzłe i konkretne.
                
                WAŻNE: Gdy używasz narzędzia, musisz zawsze podać wszystkie wymagane parametry.
                Dla narzędzia Wyszukiwarka zawsze musisz podać parametr 'query' z zapytaniem.
                Dla narzędzia Wikipedia zawsze musisz podać parametr 'query' z zapytaniem.
                Dla narzędzia Kalkulator zawsze musisz podać parametr 'expression' z wyrażeniem.
                Dla narzędzia PythonExecutor zawsze musisz podać parametr 'code' z kodem.
                Narzędzie AktualnaData nie wymaga parametrów.
                
                PRZYKŁAD: Jeśli chcesz użyć narzędzia Wyszukiwarka, wywołaj je z parametrem query o wartości "twoje zapytanie"
                
                NIGDY nie wywołuj narzędzia bez podania wymaganych parametrów."""),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        
        # Ustawiamy model do używania narzędzi
        if isinstance(llm, ChatOpenAI):
            # Przygotowanie dla OpenAI
            llm_with_tools = llm.bind(functions=[tool.to_json() for tool in tools])
            
            # Definiujemy łańcuch
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
            # Przygotowanie dla Claude
            llm_with_tools = llm.bind_tools(tools)
            
            # Definiujemy łańcuch dla Claude z formatowaniem wiadomości
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
            # Dla innych modeli używamy standardowego bindowania narzędzi
            agent = llm.bind_tools(tools)
        
        # Tworzenie wykonawcy agenta
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
        )
        
        return agent_executor 