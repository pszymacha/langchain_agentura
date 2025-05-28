"""
Model Factory - module responsible for creating LLM model instances
based on configuration from YAML file.
"""

import os
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic

class ModelFactory:
    """LLM model factory supporting multiple providers."""
    
    @staticmethod
    def create_llm(config: Dict[str, Any]):
        """
        Create an LLM model instance based on configuration.
        
        Args:
            config: Dictionary with LLM configuration from YAML file
            
        Returns:
            LLM model instance (ChatOpenAI, ChatAnthropic, AzureChatOpenAI, etc.)
            
        Raises:
            ValueError: If the provider is not supported
        """
        provider = config.get("provider", "openai").lower()
        
        if provider == "openai":
            model_config = config.get("models", {}).get("openai", {})
            
            # Check if OpenAI API key is available
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError(
                    "Missing OpenAI API key. "
                    "Make sure OPENAI_API_KEY is set in .env file"
                )
            
            # Basic parameters
            model_name = model_config.get("name", "gpt-4o")
            temperature = model_config.get("temperature", 0.5)
            
            # Optional parameters
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
            
            # Check required environment variables for Azure
            if not os.environ.get("AZURE_OPENAI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
                raise ValueError(
                    "Missing Azure OpenAI API key. "
                    "Make sure AZURE_OPENAI_API_KEY or OPENAI_API_KEY is set in .env file"
                )
            
            # Basic parameters
            model_name = model_config.get("name", "gpt-4o")
            temperature = model_config.get("temperature", 0.5)
            
            # Azure-specific parameters
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
            
            # Check if Anthropic API key is available
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ValueError(
                    "Missing Anthropic API key. "
                    "Make sure ANTHROPIC_API_KEY is set in .env file"
                )
            
            # Basic parameters
            model_name = model_config.get("name", "claude-3-5-sonnet-20240620")
            temperature = model_config.get("temperature", 0.5)
            
            # Optional parameters
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
            raise ValueError(f"Unsupported LLM provider: {provider}. "
                           "Supported providers are: openai, anthropic, azure_openai") 