"""
Provider and model configuration for Groq LLMs and local Hugging Face embeddings.
"""

import os
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_DIMENSION = 384
DEFAULT_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

EMBEDDING_MODEL_CONFIGS = {
    "BAAI/bge-small-en-v1.5": {
        "dimensions": 384,
        "query_instruction": DEFAULT_BGE_QUERY_INSTRUCTION,
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "dimensions": 384,
        "query_instruction": "",
    },
}


def get_llm_model(model_name: Optional[str] = None) -> str:
    """
    Get the configured Groq model identifier for PydanticAI.

    Returns:
        Model identifier in the `provider:model` format expected by PydanticAI
    """
    llm_choice = model_name or os.getenv("LLM_CHOICE", DEFAULT_LLM_MODEL)
    return f"groq:{llm_choice}"


def get_ingestion_model(model_name: Optional[str] = None) -> str:
    """Return the configured LLM for ingestion-related prompts."""
    return get_llm_model(model_name)


def get_embedding_model(model_name: Optional[str] = None) -> str:
    """Get the configured local Hugging Face embedding model name."""
    return model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_embedding_dimension(model_name: Optional[str] = None) -> int:
    """Get the expected embedding dimension for the configured model."""
    configured_model = get_embedding_model(model_name)
    configured_dimension = EMBEDDING_MODEL_CONFIGS.get(configured_model, {}).get("dimensions")
    if configured_dimension is not None:
        return configured_dimension
    return int(os.getenv("EMBEDDING_DIMENSION", str(DEFAULT_EMBEDDING_DIMENSION)))


def get_embedding_query_instruction(model_name: Optional[str] = None) -> str:
    """
    Get the optional query instruction used for retrieval-oriented embedding models.
    """
    if "EMBED_QUERY_INSTRUCTION" in os.environ:
        return os.getenv("EMBED_QUERY_INSTRUCTION", "")

    configured_model = get_embedding_model(model_name)
    return EMBEDDING_MODEL_CONFIGS.get(configured_model, {}).get("query_instruction", "")


def get_embedding_device() -> str:
    """Get the preferred device for local embedding inference."""
    return os.getenv("EMBEDDING_DEVICE", "auto")


def validate_configuration() -> bool:
    """Validate that required environment variables are set."""
    required_vars = ["GROQ_API_KEY", "DATABASE_URL"]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False

    return True


def get_model_info() -> dict:
    """Get information about current model configuration."""
    return {
        "llm_provider": "groq",
        "llm_model": os.getenv("LLM_CHOICE", DEFAULT_LLM_MODEL),
        "embedding_provider": "huggingface-local",
        "embedding_model": get_embedding_model(),
        "embedding_dimension": get_embedding_dimension(),
        "embedding_device": get_embedding_device(),
    }
