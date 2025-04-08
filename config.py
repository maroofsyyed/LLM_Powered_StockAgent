"""Configuration file for API keys and other sensitive information.

This file should be added to .gitignore to prevent committing sensitive data.
"""

# API Key configurations
# Replace with your actual API keys
LLM_API_KEYS = {
    "openai": "your_openai_api_key",
    "gemini": "your_gemini_api_key_here",
    "deepseek": "your_deepseek_api_key_here"
}

# Default API provider
DEFAULT_PROVIDER = "openai"

# Default model to use
DEFAULT_MODEL = "gpt-3.5-turbo"
