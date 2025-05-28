"""
Model Factory - moduł odpowiedzialny za tworzenie instancji modeli LLM
na podstawie konfiguracji z pliku YAML.
"""

import os
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI, AzureChatOpenAI
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
            Instancja modelu LLM (ChatOpenAI, ChatAnthropic, AzureChatOpenAI, itp.)
            
        Raises:
            ValueError: Jeśli podany dostawca nie jest wspierany
        """
        provider = config.get("provider", "openai").lower()
        
        if provider == "openai":
            model_config = config.get("models", {}).get("openai", {})
            
            # Sprawdzamy, czy klucz API dla OpenAI jest dostępny
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError(
                    "Brak klucza API dla OpenAI. "
                    "Upewnij się, że OPENAI_API_KEY jest ustawiony w pliku .env"
                )
            
            # Podstawowe parametry
            model_name = model_config.get("name", "gpt-4o")
            temperature = model_config.get("temperature", 0.5)
            
            # Opcjonalne parametry
            kwargs = {}
            if model_config.get("base_url"):
                kwargs["base_url"] = model_config["base_url"]
            if model_config.get("api_version"):
                kwargs["api_version"] = model_config["api_version"]
            if model_config.get("timeout"):
                kwargs["timeout"] = model_config["timeout"]
            if model_config.get("max_retries") is not None:
                kwargs["max_retries"] = model_config["max_retries"]
            
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                **kwargs
            )
            
        elif provider == "azure_openai":
            model_config = config.get("models", {}).get("azure_openai", {})
            
            # Sprawdzamy wymagane zmienne środowiskowe dla Azure
            if not os.environ.get("AZURE_OPENAI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
                raise ValueError(
                    "Brak klucza API dla Azure OpenAI. "
                    "Upewnij się, że AZURE_OPENAI_API_KEY lub OPENAI_API_KEY jest ustawiony w pliku .env"
                )
            
            # Podstawowe parametry
            model_name = model_config.get("name", "gpt-4o")
            temperature = model_config.get("temperature", 0.5)
            
            # Azure-specyficzne parametry
            kwargs = {}
            if model_config.get("azure_endpoint"):
                kwargs["azure_endpoint"] = model_config["azure_endpoint"]
            if model_config.get("deployment_name"):
                kwargs["deployment_name"] = model_config["deployment_name"]
            if model_config.get("api_version"):
                kwargs["api_version"] = model_config["api_version"]
            if model_config.get("timeout"):
                kwargs["timeout"] = model_config["timeout"]
            if model_config.get("max_retries") is not None:
                kwargs["max_retries"] = model_config["max_retries"]
            
            return AzureChatOpenAI(
                model=model_name,
                temperature=temperature,
                **kwargs
            )
            
        elif provider == "anthropic":
            model_config = config.get("models", {}).get("anthropic", {})
            
            # Sprawdzamy, czy klucz API dla Anthropic jest dostępny
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ValueError(
                    "Brak klucza API dla Anthropic. "
                    "Upewnij się, że ANTHROPIC_API_KEY jest ustawiony w pliku .env"
                )
            
            # Podstawowe parametry
            model_name = model_config.get("name", "claude-3-5-sonnet-20240620")
            temperature = model_config.get("temperature", 0.5)
            
            # Opcjonalne parametry
            kwargs = {}
            if model_config.get("base_url"):
                kwargs["base_url"] = model_config["base_url"]
            if model_config.get("timeout"):
                kwargs["timeout"] = model_config["timeout"]
            if model_config.get("max_retries") is not None:
                kwargs["max_retries"] = model_config["max_retries"]
            
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                **kwargs
            )
            
        else:
            raise ValueError(f"Nieobsługiwany dostawca LLM: {provider}. "
                           "Wspierani dostawcy to: openai, anthropic, azure_openai") 