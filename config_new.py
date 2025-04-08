"""Configuration file for API keys and other sensitive information.

This file should be added to .gitignore to prevent committing sensitive data.
"""

# API Key configurations
# Replace with your actual API keys
LLM_API_KEYS = {
    "openai": "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",
    "gemini": "your_gemini_api_key_here",
    "deepseek": "your_deepseek_api_key_here"
}

# Default API provider
DEFAULT_PROVIDER = "openai"

# Default model to use
DEFAULT_MODEL = "gpt-3.5-turbo"
