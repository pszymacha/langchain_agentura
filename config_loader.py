"""
Config Loader - moduł odpowiedzialny za ładowanie konfiguracji z pliku YAML
"""

import os
import yaml
from typing import Dict, Any

class ConfigLoader:
    """Klasa do ładowania i walidacji konfiguracji z pliku YAML."""
    
    @staticmethod
    def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
        """
        Ładuje konfigurację z pliku YAML.
        
        Args:
            config_path: Ścieżka do pliku konfiguracyjnego YAML
            
        Returns:
            Słownik z konfiguracją
            
        Raises:
            FileNotFoundError: Jeśli plik konfiguracyjny nie istnieje
            yaml.YAMLError: Jeśli plik YAML ma nieprawidłowy format
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Plik konfiguracyjny {config_path} nie istnieje")
            
        with open(config_path, 'r', encoding='utf-8') as file:
            try:
                config = yaml.safe_load(file)
                return config
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Błąd podczas parsowania pliku YAML: {e}")
    
    @staticmethod
    def get_llm_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pobiera konfigurację LLM z głównej konfiguracji.
        
        Args:
            config: Główna konfiguracja
            
        Returns:
            Konfiguracja LLM
            
        Raises:
            KeyError: Jeśli sekcja 'llm' nie istnieje w konfiguracji
        """
        if 'llm' not in config:
            raise KeyError("Brak sekcji 'llm' w konfiguracji")
            
        return config['llm']
    
    @staticmethod
    def get_tools_config(config: Dict[str, Any]) -> list:
        """
        Pobiera konfigurację narzędzi z głównej konfiguracji.
        
        Args:
            config: Główna konfiguracja
            
        Returns:
            Lista konfiguracji narzędzi
            
        Raises:
            KeyError: Jeśli sekcja 'tools' nie istnieje w konfiguracji
        """
        if 'tools' not in config:
            raise KeyError("Brak sekcji 'tools' w konfiguracji")
            
        return config['tools'] 