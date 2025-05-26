import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

# Importy lokalnych modułów
from config_loader import ConfigLoader
from model_factory import ModelFactory
from tool_factory import ToolFactory
from agent_factory import AgentFactory

# Ładowanie zmiennych środowiskowych z pliku .env
load_dotenv()

def create_agent():
    """Tworzy agenta na podstawie konfiguracji z pliku YAML."""
    try:
        # Ładowanie konfiguracji - upewniamy się, że ścieżka jest poprawna
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        config = ConfigLoader.load_config(config_path)
        llm_config = ConfigLoader.get_llm_config(config)
        tools_config = ConfigLoader.get_tools_config(config)
        
        # Sprawdzenie jaki dostawca jest wybrany i czy klucz API jest dostępny
        provider = llm_config.get("provider", "openai").lower()
        
        if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            print("BŁĄD: Klucz API OpenAI nie został znaleziony. Upewnij się, że jest w pliku .env")
            sys.exit(1)
        elif provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            print("BŁĄD: Klucz API Anthropic nie został znaleziony. Upewnij się, że jest w pliku .env")
            sys.exit(1)
        
        # Tworzenie modelu LLM
        llm = ModelFactory.create_llm(llm_config)
        
        # Tworzenie narzędzi
        tools = ToolFactory.create_tools(tools_config, llm)
        
        # Tworzenie agenta
        agent_executor = AgentFactory.create_agent(llm, tools)
        
        # Informacja o wybranym modelu
        model_name = llm_config.get("models", {}).get(provider, {}).get("name", "nieznany")
        print(f"\nWybrano model: {provider.upper()} - {model_name}")
        print(f"Dostępne narzędzia: {', '.join([tool.name for tool in tools])}")
        
        return agent_executor
        
    except Exception as e:
        print(f"BŁĄD podczas tworzenia agenta: {e}")
        sys.exit(1)

def main():
    """Główna funkcja do interakcji z agentem przez terminal."""
    print("Witaj w interaktywnym czacie z agentem LLM!")
    print("Wpisz 'wyjdź' aby zakończyć.")
    print("----------------------------------------------")
    
    agent_executor = create_agent()
    chat_history = []
    
    while True:
        user_input = input("\nTy: ")
        
        if user_input.lower() in ['wyjdź', 'exit', 'quit', 'q']:
            print("Do widzenia!")
            break
        
        # Wywołanie agenta z historią czatu
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        
        # Wyświetlenie odpowiedzi
        print(f"\nAgent: {response['output']}")
        
        # Aktualizacja historii czatu
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response["output"]))

if __name__ == "__main__":
    main() 