"""Temporary script to verify the API key configuration"""
import os
import sys

# Import the config module
try:
    from config import LLM_API_KEYS, DEFAULT_PROVIDER
    
    # Check if the OpenAI API key is set correctly
    openai_key = LLM_API_KEYS.get("openai", "")
    
    if "your_" in openai_key or openai_key.strip() == "":
        print("ERROR: API key in config.py still contains placeholder text")
        print("Please update your API key in config.py")
        sys.exit(1)
    else:
        print(f"API key found: {openai_key[:4]}...{openai_key[-4:]}")
        print("API key format appears to be valid")
        
    print("\nYou can run the simulation now with:")
    print("python3 main.py")
    
except ImportError:
    print("ERROR: Could not import config module")
    print("Please ensure config.py exists in the project directory")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {str(e)}")
    sys.exit(1)
