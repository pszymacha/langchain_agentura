# LangChain Agentura

LangChain-based agent implementation supporting custom LLM endpoints and Azure OpenAI.

## Supported Providers

- **OpenAI** - Standard API with custom endpoint support
- **Anthropic** - Standard API with custom endpoint support  
- **Azure OpenAI** - Full Azure-specific parameter support

## Configuration

### Standard OpenAI
```yaml
llm:
  provider: openai
  models:
    openai:
      name: gpt-4o
      temperature: 0.05
```

### OpenAI with Custom Endpoint/Proxy
```yaml
llm:
  provider: openai
  models:
    openai:
      name: gpt-4o
      temperature: 0.05
      base_url: https://your-proxy.example.com/v1
      timeout: 90
      max_retries: 3
```

### Azure OpenAI
```yaml
llm:
  provider: azure_openai
  models:
    azure_openai:
      name: gpt-4o  # Model name (not deployment name)
      temperature: 0.05
      azure_endpoint: https://your-resource.openai.azure.com/
      deployment_name: gpt-4o-deployment  # Azure deployment name
      api_version: 2024-05-01-preview
      timeout: 60
      max_retries: 2
```

### Anthropic with Custom Endpoint
```yaml
llm:
  provider: anthropic
  models:
    anthropic:
      name: claude-3-5-sonnet-20240620
      temperature: 0.05
      base_url: https://your-anthropic-proxy.example.com
      timeout: 60
      max_retries: 2
```

## Available Configuration Files

- `config.yaml` - Main configuration
- `config.example.azure.yaml` - Azure OpenAI example
- `config.example.custom.yaml` - Custom endpoint example

## Environment Variables

### OpenAI
- `OPENAI_API_KEY` - OpenAI API key (required)

### Azure OpenAI
- `AZURE_OPENAI_API_KEY` - Azure OpenAI API key (required, or `OPENAI_API_KEY`)
- `AZURE_OPENAI_ENDPOINT` - Azure endpoint (optional, can be set in config)

### Anthropic
- `ANTHROPIC_API_KEY` - Anthropic API key (required)

## Optional Parameters

All providers support these optional parameters:
- `timeout` - Request timeout in seconds
- `max_retries` - Maximum retry attempts
- `base_url` - Custom endpoint URL (OpenAI/Anthropic)
- `api_version` - API version (mainly for Azure)

### Azure-specific parameters:
- `azure_endpoint` - Azure resource endpoint (required)
- `deployment_name` - Azure deployment name (required)

## Quick Start

1. Copy an example config file:
   ```bash
   cp config.example.azure.yaml config.yaml
   ```

2. Set your API keys:
   ```bash
   export AZURE_OPENAI_API_KEY="your-key"
   ```

3. Update config with your endpoints and settings

4. Run your agent - the system will automatically use the new configuration

## Advanced Features

### Recursion Limit Handling

The system includes intelligent handling of LangGraph workflow iteration limits. Instead of throwing errors, the agent returns a draft response based on collected research data, clearly marked as incomplete.

#### Configuration:
```yaml
graph:
  recursion_limit: 50  # Default is 25, recommended 50
```

#### Programmatic usage:
```python
# Increase limit for complex queries
agent = create_advanced_agent(llm, tools, recursion_limit=100)
```

#### When limit is reached:
Instead of errors, you get draft responses like:
```
⚠️ **DRAFT RESPONSE - INCOMPLETE RESEARCH**

This is a preliminary answer based on limited research data. I reached the maximum allowed iteration limit (50) before completing thorough analysis. The following response is based on incomplete information and should be considered a draft outline rather than a comprehensive answer.

[Draft answer based on available data...]
```

#### Recommended values:
- **Standard queries**: 25-50 iterations
- **Complex research**: 75-100 iterations  
- **Simple queries**: 10-25 iterations

### Testing

Run recursion limit tests:
```bash
# From project root
python tests/test_recursion_limit.py

# Or from tests directory
cd tests
python test_recursion_limit.py
```

The tests verify that the system gracefully handles iteration limits and returns meaningful partial results instead of errors.

## Troubleshooting

**Azure OpenAI:**
- Ensure `azure_endpoint` includes protocol (https://)
- `deployment_name` must match your Azure Portal deployment
- Check if `api_version` is supported by your deployment

**Custom Endpoints:**
- Verify endpoint availability and response format
- Increase `timeout` for slow connections
- Increase `max_retries` for unstable connections

**Authorization:**
- Verify environment variables are set correctly
- For Azure, check API key validity in Azure Portal
- Ensure network access to external APIs (firewalls, corporate proxies) 