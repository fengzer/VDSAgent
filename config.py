from langchain_openai import ChatOpenAI

BASE_URL = ""
API_KEY = ""

LLM_CONFIG = {
    "api_key": API_KEY,
    "base_url": BASE_URL,
    "model": "gpt-4o"
}

MULTIMODAL_CONFIG = {
    "api_key": API_KEY,
    "base_url": BASE_URL,
    "model": "gpt-4o"
}

def get_llm():
    """Get the default LLM instance"""
    return ChatOpenAI(**LLM_CONFIG)

def get_multimodal_llm():
    """Get the multimodal LLM instance"""
    return ChatOpenAI(**MULTIMODAL_CONFIG) 