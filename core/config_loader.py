"""
Config Loader - module responsible for loading configuration from YAML file
"""

import os
import yaml
from typing import Dict, Any

class ConfigLoader:
    """Class for loading and validating configuration from YAML file."""
    
    @staticmethod
    def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Dictionary with configuration
            
        Raises:
            FileNotFoundError: If configuration file does not exist
            yaml.YAMLError: If YAML file has invalid format
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} does not exist")
            
        with open(config_path, 'r', encoding='utf-8') as file:
            try:
                config = yaml.safe_load(file)
                return config
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing YAML file: {e}")
    
    @staticmethod
    def get_llm_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get LLM configuration from main configuration.
        
        Args:
            config: Main configuration
            
        Returns:
            LLM configuration
            
        Raises:
            KeyError: If 'llm' section does not exist in configuration
        """
        if 'llm' not in config:
            raise KeyError("Missing 'llm' section in configuration")
            
        return config['llm']
    
    @staticmethod
    def get_tools_config(config: Dict[str, Any]) -> list:
        """
        Get tools configuration from main configuration.
        
        Args:
            config: Main configuration
            
        Returns:
            List of tool configurations
            
        Raises:
            KeyError: If 'tools' section does not exist in configuration
        """
        if 'tools' not in config:
            raise KeyError("Missing 'tools' section in configuration")
            
        return config['tools'] 