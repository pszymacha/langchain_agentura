"""
Model Factory - moduł odpowiedzialny za tworzenie instancji modeli LLM
na podstawie konfiguracji z pliku YAML.
"""

import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

class ModelFactory:
    """Fabryka modeli LLM wspierająca różnych dostawców."""
    
    @staticmethod
    def create_llm(config: Dict[str, Any]):
        """
        Tworzy instancję modelu LLM na podstawie konfiguracji.
        
        Args:
            config: Słownik z konfiguracją LLM z pliku YAML
            
        Returns:
            Instancja modelu LLM (ChatOpenAI, ChatAnthropic, itp.)
            
        Raises:
            ValueError: Jeśli podany dostawca nie jest wspierany
        """
        provider = config.get("provider", "openai").lower()
        
        if provider == "openai":
            model_config = config.get("models", {}).get("openai", {})
            model_name = model_config.get("name", "gpt-4o")
            temperature = model_config.get("temperature", 0.5)
            
            # Sprawdzamy, czy klucz API dla OpenAI jest dostępny
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError(
                    "Brak klucza API dla OpenAI. "
                    "Upewnij się, że OPENAI_API_KEY jest ustawiony w pliku .env"
                )
            
            return ChatOpenAI(
                model=model_name,
                temperature=temperature
            )
            
        elif provider == "anthropic":
            # Sprawdzamy, czy klucz API dla Anthropic jest dostępny
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ValueError(
                    "Brak klucza API dla Anthropic. "
                    "Upewnij się, że ANTHROPIC_API_KEY jest ustawiony w pliku .env"
                )
            
            model_config = config.get("models", {}).get("anthropic", {})
            model_name = model_config.get("name", "claude-3-5-sonnet-20240620")
            temperature = model_config.get("temperature", 0.5)
            
            return ChatAnthropic(
                model=model_name,
                temperature=temperature
            )
            
        else:
            raise ValueError(f"Nieobsługiwany dostawca LLM: {provider}. "
                           "Wspierani dostawcy to: openai, anthropic") 