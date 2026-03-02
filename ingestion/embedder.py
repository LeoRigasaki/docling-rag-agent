"""
Document embedding generation for vector search using local Hugging Face models.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer

from .chunker import DocumentChunk

try:
    from ..utils.providers import (
        get_embedding_device,
        get_embedding_dimension,
        get_embedding_model,
        get_embedding_query_instruction,
    )
except ImportError:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.providers import (
        get_embedding_device,
        get_embedding_dimension,
        get_embedding_model,
        get_embedding_query_instruction,
    )

load_dotenv()

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = get_embedding_model()
EMBEDDING_DEVICE = get_embedding_device()


class EmbeddingGenerator:
    """Generates embeddings for document chunks using a local Hugging Face model."""

    _model_cache: Dict[Tuple[str, str], Tuple[Any, Any, int]] = {}
    _model_configs = {
        "BAAI/bge-small-en-v1.5": {
            "dimensions": 384,
            "max_tokens": 512,
            "pooling": "cls",
        },
        "sentence-transformers/all-MiniLM-L6-v2": {
            "dimensions": 384,
            "max_tokens": 256,
            "pooling": "mean",
        },
    }

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        batch_size: int = 32,
        device: str = EMBEDDING_DEVICE,
    ):
        self.model_name = model
        self.batch_size = batch_size
        self.device = self._resolve_device(device)
        self.query_instruction = get_embedding_query_instruction(model)
        self.config = self._model_configs.get(
            model,
            {
                "dimensions": get_embedding_dimension(model),
                "max_tokens": 512,
                "pooling": "cls",
            },
        )

    def _resolve_device(self, device: str) -> str:
        """Resolve the inference device, preferring accelerators when available."""
        if device != "auto":
            return device

        if torch.cuda.is_available():
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def _get_model_components(self) -> Tuple[Any, Any, int]:
        """Load and cache the tokenizer/model pair for the configured embedding model."""
        cache_key = (self.model_name, self.device)
        if cache_key not in self._model_cache:
            logger.info(f"Loading embedding model {self.model_name} on {self.device}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            model.to(self.device)
            model.eval()

            hidden_size = getattr(model.config, "hidden_size", self.config["dimensions"])
            self._model_cache[cache_key] = (tokenizer, model, hidden_size)

        tokenizer, model, hidden_size = self._model_cache[cache_key]
        self.config["dimensions"] = hidden_size
        return tokenizer, model, hidden_size

    def _prepare_text(self, text: str, is_query: bool = False) -> str:
        """Normalize text and add any model-specific query prefix."""
        normalized_text = text.strip()
        if not normalized_text:
            return ""

        if is_query and self.query_instruction and not normalized_text.startswith(self.query_instruction):
            return f"{self.query_instruction}{normalized_text}"

        return normalized_text

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool token embeddings using the attention mask."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        counts = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return summed / counts

    def _zero_embedding(self, dimension: int) -> List[float]:
        """Create a zero vector matching the configured embedding size."""
        return [0.0] * dimension

    def _encode_sync(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Encode texts synchronously with the local embedding model."""
        tokenizer, model, hidden_size = self._get_model_components()
        embeddings: List[List[float]] = [self._zero_embedding(hidden_size) for _ in texts]

        prepared_texts = []
        prepared_indices = []
        for index, text in enumerate(texts):
            prepared_text = self._prepare_text(text, is_query=is_query)
            if prepared_text:
                prepared_texts.append(prepared_text)
                prepared_indices.append(index)

        if not prepared_texts:
            return embeddings

        encoded_inputs = tokenizer(
            prepared_texts,
            padding=True,
            truncation=True,
            max_length=self.config["max_tokens"],
            return_tensors="pt",
        )
        encoded_inputs = {key: value.to(self.device) for key, value in encoded_inputs.items()}

        with torch.inference_mode():
            model_output = model(**encoded_inputs)
            if self.config["pooling"] == "mean":
                pooled = self._mean_pooling(model_output.last_hidden_state, encoded_inputs["attention_mask"])
            else:
                pooled = model_output.last_hidden_state[:, 0]
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)

        encoded_embeddings = normalized.cpu().tolist()
        for index, embedding in zip(prepared_indices, encoded_embeddings):
            embeddings[index] = embedding

        return embeddings

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for a single text."""
        return (await self.generate_embeddings_batch([text]))[0]

    async def generate_embeddings_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        return await asyncio.to_thread(self._encode_sync, texts, is_query)

    async def embed_chunks(
        self,
        chunks: List[DocumentChunk],
        progress_callback: Optional[callable] = None,
    ) -> List[DocumentChunk]:
        """Generate embeddings for document chunks."""
        if not chunks:
            return chunks

        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embedded_chunks = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            batch_texts = [chunk.content for chunk in batch_chunks]

            try:
                embeddings = await self.generate_embeddings_batch(batch_texts)

                for chunk, embedding in zip(batch_chunks, embeddings):
                    embedded_chunk = DocumentChunk(
                        content=chunk.content,
                        index=chunk.index,
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        metadata={
                            **chunk.metadata,
                            "embedding_model": self.model_name,
                            "embedding_generated_at": datetime.now().isoformat(),
                        },
                        token_count=chunk.token_count,
                    )
                    embedded_chunk.embedding = embedding
                    embedded_chunks.append(embedded_chunk)

                current_batch = (i // self.batch_size) + 1
                if progress_callback:
                    progress_callback(current_batch, total_batches)

                logger.info(f"Processed batch {current_batch}/{total_batches}")

            except Exception as e:
                logger.error(f"Failed to process batch {i // self.batch_size + 1}: {e}")
                for chunk in batch_chunks:
                    chunk.metadata.update(
                        {
                            "embedding_error": str(e),
                            "embedding_generated_at": datetime.now().isoformat(),
                        }
                    )
                    chunk.embedding = self._zero_embedding(self.config["dimensions"])
                    embedded_chunks.append(chunk)

        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
        return embedded_chunks

    async def embed_query(self, query: str) -> List[float]:
        """Generate an embedding for a search query."""
        return (await self.generate_embeddings_batch([query], is_query=True))[0]

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for this model."""
        return self.config["dimensions"]


# Cache for embeddings
class EmbeddingCache:
    """Simple in-memory cache for embeddings."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize cache."""
        self.cache: Dict[str, List[float]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.max_size = max_size
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        text_hash = self._hash_text(text)
        if text_hash in self.cache:
            self.access_times[text_hash] = datetime.now()
            return self.cache[text_hash]
        return None
    
    def put(self, text: str, embedding: List[float]):
        """Store embedding in cache."""
        text_hash = self._hash_text(text)
        
        # Evict oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[text_hash] = embedding
        self.access_times[text_hash] = datetime.now()
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for text."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()


# Factory function
_EMBEDDER_INSTANCES: Dict[Tuple[str, bool, Tuple[Tuple[str, str], ...]], EmbeddingGenerator] = {}


def create_embedder(
    model: str = EMBEDDING_MODEL,
    use_cache: bool = True,
    **kwargs
) -> EmbeddingGenerator:
    """
    Create embedding generator with optional caching.
    
    Args:
        model: Embedding model to use
        use_cache: Whether to use caching
        **kwargs: Additional arguments for EmbeddingGenerator
    
    Returns:
        EmbeddingGenerator instance
    """
    cache_key = (
        model,
        use_cache,
        tuple(sorted((str(key), repr(value)) for key, value in kwargs.items())),
    )
    if cache_key in _EMBEDDER_INSTANCES:
        return _EMBEDDER_INSTANCES[cache_key]

    embedder = EmbeddingGenerator(model=model, **kwargs)

    if use_cache:
        cache = EmbeddingCache()
        original_generate = embedder.generate_embedding

        async def cached_generate(text: str) -> List[float]:
            cached = cache.get(text)
            if cached is not None:
                return cached

            embedding = await original_generate(text)
            cache.put(text, embedding)
            return embedding

        embedder.generate_embedding = cached_generate

    _EMBEDDER_INSTANCES[cache_key] = embedder
    return embedder
