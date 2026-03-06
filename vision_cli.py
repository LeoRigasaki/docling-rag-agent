#!/usr/bin/env python3
"""
Vision-aware CLI for the Docling RAG Agent.

This CLI keeps the current PGVector schema intact. It retrieves text chunks from
the database, reconstructs linked tables and nearby images from the original
source document on demand, and selectively calls a Groq vision model when the
question actually needs visual reasoning.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import asyncpg
from dotenv import load_dotenv
from groq import AsyncGroq
from ollama import AsyncClient, ResponseError
from pydantic import BaseModel, ValidationError
from PIL import Image
from transformers import AutoTokenizer

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

from ingestion.chunker import (
    DEFAULT_HYBRID_MAX_TOKENS,
    DEFAULT_HYBRID_MERGE_PEERS,
    DEFAULT_HYBRID_TOKENIZER,
)
from ingestion.embedder import create_embedder
from utils.providers import DEFAULT_LLM_MODEL

load_dotenv(".env")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = logging.getLogger(__name__)

DEFAULT_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3.5:0.8b"
DEFAULT_CACHE_DIR = ".vision_cache"
DEFAULT_RESPONSES_MD = "vision_cli_responses.md"
DEFAULT_FORMULA_ENRICHMENT = False
DOC_ASSET_SUFFIXES = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".html",
    ".htm",
}
SUPPORTED_SOURCE_SUFFIXES = DOC_ASSET_SUFFIXES | {
    ".md",
    ".markdown",
    ".txt",
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
}
VISION_QUERY_HINTS = (
    "image",
    "images",
    "figure",
    "figures",
    "chart",
    "graph",
    "diagram",
    "screenshot",
    "picture",
    "photo",
    "visual",
    "see",
    "shown",
    "ocr",
    "read the image",
    "read the chart",
)
QUERY_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "what",
    "which",
    "would",
    "should",
    "most",
    "appropriate",
    "state",
    "identify",
    "explain",
    "variable",
    "variables",
    "researcher",
    "following",
}


class Colors:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    similarity: float
    chunk_metadata: Dict[str, Any]
    document_title: str
    document_source: str
    document_metadata: Dict[str, Any]


@dataclass
class TableAsset:
    asset_ref: str
    page_no: Optional[int]
    markdown: str


@dataclass
class ImageAsset:
    asset_ref: str
    page_no: Optional[int]
    file_path: Path
    caption: str = ""


@dataclass
class ChunkAssetLinks:
    page_numbers: List[int] = field(default_factory=list)
    table_refs: List[str] = field(default_factory=list)
    image_refs: List[str] = field(default_factory=list)


@dataclass
class DoclingAssetCatalog:
    source_path: Path
    markdown_path: Optional[Path]
    tables: Dict[str, TableAsset]
    images: Dict[str, ImageAsset]
    chunk_links: Dict[int, ChunkAssetLinks]


@dataclass
class EnrichedChunk:
    chunk: RetrievedChunk
    markdown_path: Optional[Path]
    page_numbers: List[int]
    tables: List[TableAsset]
    images: List[ImageAsset]


class VisionImageAnalysis(BaseModel):
    image_index: int
    asset_ref: str
    summary: str = ""
    ocr_text: str = ""
    relevance: str | bool = ""


class VisionAnalysisResponse(BaseModel):
    images: List[VisionImageAnalysis]


def dedupe_preserve_order(values: List[str]) -> List[str]:
    seen: Set[str] = set()
    deduped: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def truncate_text(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def clean_structured_response(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    return cleaned.strip()


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return slug or "document"


def normalize_json_value(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            return {}
    return {}


def extract_metadata_pages(metadata: Dict[str, Any]) -> List[int]:
    pages: List[int] = []

    page_number = metadata.get("page_number")
    if isinstance(page_number, int):
        pages.append(page_number)

    page_numbers = metadata.get("page_numbers")
    if isinstance(page_numbers, list):
        pages.extend(page for page in page_numbers if isinstance(page, int))

    return sorted(set(pages))


def chunk_modality(metadata: Dict[str, Any]) -> str:
    return str(metadata.get("source_modality") or metadata.get("chunk_method") or "").strip()


def extract_caption_text(item: Any, document: Any) -> str:
    caption = getattr(item, "caption_text", "")
    if callable(caption):
        try:
            caption = caption(document)
        except TypeError:
            caption = caption()
    return str(caption or "").strip()


def supports_docling_assets(path: Path) -> bool:
    return path.suffix.lower() in DOC_ASSET_SUFFIXES


def supports_retrieval_source(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_SOURCE_SUFFIXES


def extract_page_numbers(item: Any) -> List[int]:
    page_numbers: List[int] = []
    for prov in getattr(item, "prov", []) or []:
        page_no = getattr(prov, "page_no", None)
        if page_no is not None:
            page_numbers.append(page_no)
    return sorted(set(page_numbers))


def question_needs_vision(question: str) -> bool:
    normalized = question.lower()
    return any(hint in normalized for hint in VISION_QUERY_HINTS)


def extract_query_terms(query: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_]+", query.lower())
    return [
        token
        for token in tokens
        if len(token) > 2 and token not in QUERY_STOPWORDS
    ]


def extract_query_phrases(query_terms: List[str]) -> List[str]:
    phrases: List[str] = []
    for n in (2, 3):
        for index in range(len(query_terms) - n + 1):
            phrases.append(" ".join(query_terms[index:index + n]))
    return phrases


class DoclingAssetManager:
    """Loads Docling documents on demand and links their assets back to chunk indices."""

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        image_scale: float = 1.5,
        chunk_max_tokens: int = DEFAULT_HYBRID_MAX_TOKENS,
        merge_peers: bool = DEFAULT_HYBRID_MERGE_PEERS,
        formula_enrichment: bool = DEFAULT_FORMULA_ENRICHMENT,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_scale = image_scale
        self.formula_enrichment = formula_enrichment
        self.catalog_cache: Dict[str, DoclingAssetCatalog] = {}

        tokenizer_model = DEFAULT_HYBRID_TOKENIZER
        logger.info("Initializing vision chunk tokenizer: %s", tokenizer_model)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=chunk_max_tokens,
            merge_peers=merge_peers,
        )

    async def get_catalog(self, source_path: Path) -> Optional[DoclingAssetCatalog]:
        resolved_path = source_path.resolve()
        if not resolved_path.exists() or not supports_docling_assets(resolved_path):
            return None

        cache_key = str(resolved_path)
        if cache_key not in self.catalog_cache:
            self.catalog_cache[cache_key] = await asyncio.to_thread(
                self._build_catalog,
                resolved_path,
            )
        return self.catalog_cache[cache_key]

    def _build_converter(self) -> DocumentConverter:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = self.image_scale
        pipeline_options.generate_picture_images = True
        pipeline_options.generate_table_images = True
        pipeline_options.do_formula_enrichment = self.formula_enrichment

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )

    def _build_catalog(self, source_path: Path) -> DoclingAssetCatalog:
        converter = self._build_converter()
        conversion_result = converter.convert(source_path)
        document = conversion_result.document

        source_digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
        output_dir = self.cache_dir / f"{slugify(source_path.stem)}-{source_digest}"
        output_dir.mkdir(parents=True, exist_ok=True)

        markdown_path = output_dir / "full.md"
        try:
            if hasattr(document, "save_as_markdown"):
                document.save_as_markdown(markdown_path, image_mode=ImageRefMode.REFERENCED)
            else:
                markdown = document.export_to_markdown(image_mode=ImageRefMode.REFERENCED)
                markdown_path.write_text(markdown, encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to save rich markdown for %s: %s", source_path.name, exc)
            markdown_path = None

        order_entries: List[Dict[str, Any]] = []
        for order_index, (element, _level) in enumerate(document.iterate_items()):
            element_ref = getattr(element, "self_ref", None)
            if not element_ref:
                continue

            label = getattr(getattr(element, "label", None), "value", "")
            order_entries.append(
                {
                    "index": order_index,
                    "ref": element_ref,
                    "label": label,
                    "pages": extract_page_numbers(element),
                    "element": element,
                }
            )

        tables: Dict[str, TableAsset] = {}
        for table in document.tables:
            asset_ref = getattr(table, "self_ref", "")
            if not asset_ref:
                continue

            try:
                markdown = table.export_to_markdown(doc=document)
            except Exception:
                markdown = ""

            tables[asset_ref] = TableAsset(
                asset_ref=asset_ref,
                page_no=extract_page_numbers(table)[0] if extract_page_numbers(table) else None,
                markdown=markdown.strip(),
            )

        images: Dict[str, ImageAsset] = {}
        picture_count = 0
        for entry in order_entries:
            element = entry["element"]
            if not isinstance(element, PictureItem):
                continue

            picture_count += 1
            image_path = output_dir / f"image-{picture_count}.png"

            try:
                image = element.get_image(document)
                image.save(image_path, "PNG")
            except Exception as exc:
                logger.warning(
                    "Failed to export image %s from %s: %s",
                    entry["ref"],
                    source_path.name,
                    exc,
                )
                continue

            images[entry["ref"]] = ImageAsset(
                asset_ref=entry["ref"],
                page_no=entry["pages"][0] if entry["pages"] else None,
                file_path=image_path,
                caption=extract_caption_text(element, document),
            )

        chunk_links = self._build_chunk_links(document, order_entries)

        return DoclingAssetCatalog(
            source_path=source_path,
            markdown_path=markdown_path,
            tables=tables,
            images=images,
            chunk_links=chunk_links,
        )

    def _build_chunk_links(
        self,
        document: Any,
        order_entries: List[Dict[str, Any]],
    ) -> Dict[int, ChunkAssetLinks]:
        order_index = {entry["ref"]: entry["index"] for entry in order_entries}
        chunk_links: Dict[int, ChunkAssetLinks] = {}

        for chunk_index, chunk in enumerate(self.chunker.chunk(dl_doc=document)):
            positions: List[int] = []
            page_numbers: Set[int] = set()
            table_refs: List[str] = []
            image_refs: List[str] = []

            for doc_item in chunk.meta.doc_items:
                item_ref = getattr(doc_item, "self_ref", None)
                if item_ref in order_index:
                    positions.append(order_index[item_ref])

                label = getattr(getattr(doc_item, "label", None), "value", "")
                if label == "table":
                    table_refs.append(item_ref)
                elif label == "picture":
                    image_refs.append(item_ref)

                page_numbers.update(extract_page_numbers(doc_item))

            if not table_refs:
                table_refs.extend(
                    self._find_nearby_asset_refs(
                        order_entries,
                        positions,
                        page_numbers,
                        asset_label="table",
                        max_results=1,
                        window=4,
                    )
                )

            if not image_refs:
                image_refs.extend(
                    self._find_nearby_asset_refs(
                        order_entries,
                        positions,
                        page_numbers,
                        asset_label="picture",
                        max_results=2,
                        window=6,
                    )
                )

            chunk_links[chunk_index] = ChunkAssetLinks(
                page_numbers=sorted(page_numbers),
                table_refs=dedupe_preserve_order(table_refs),
                image_refs=dedupe_preserve_order(image_refs),
            )

        return chunk_links

    def _find_nearby_asset_refs(
        self,
        order_entries: List[Dict[str, Any]],
        positions: List[int],
        page_numbers: Set[int],
        asset_label: str,
        max_results: int,
        window: int,
    ) -> List[str]:
        if not positions:
            return []

        start = max(0, min(positions) - window)
        end = min(len(order_entries), max(positions) + window + 1)

        refs: List[str] = []
        for entry in order_entries[start:end]:
            if entry["label"] != asset_label:
                continue

            if page_numbers and entry["pages"] and not set(entry["pages"]).intersection(page_numbers):
                continue

            refs.append(entry["ref"])
            if len(refs) >= max_results:
                break

        return refs


class VisionRAGCLI:
    """Interactive multimodal CLI."""

    def __init__(
        self,
        documents_root: str = "documents",
        cache_dir: str = DEFAULT_CACHE_DIR,
        limit: int = 4,
        provider: str = "groq",
        vision_mode: str = "auto",
        max_vision_images: int = 3,
        chunk_max_tokens: int = DEFAULT_HYBRID_MAX_TOKENS,
        merge_peers: bool = DEFAULT_HYBRID_MERGE_PEERS,
        responses_md: str = DEFAULT_RESPONSES_MD,
        formula_enrichment: bool = DEFAULT_FORMULA_ENRICHMENT,
    ):
        self.documents_root = Path(documents_root).resolve()
        self.scope_root = self.documents_root.parent if self.documents_root.is_file() else self.documents_root
        self.cache_dir = cache_dir
        self.limit = limit
        self.provider = provider
        self.vision_mode = vision_mode
        self.max_vision_images = max_vision_images
        self.chunk_max_tokens = chunk_max_tokens
        self.merge_peers = merge_peers
        self.responses_md_path = Path(responses_md).resolve()
        self.responses_md_path.parent.mkdir(parents=True, exist_ok=True)
        self.formula_enrichment = formula_enrichment
        self.session_started_at = datetime.now().astimezone()
        self.responses_session_logged = False
        self.cwd = Path.cwd().resolve()
        self.discovered_documents = self._discover_documents()
        self.allowed_sources, self.allowed_file_paths = self._build_document_scope_values()

        self.db_pool: Optional[asyncpg.Pool] = None
        self.embedder = create_embedder()
        self.asset_manager = DoclingAssetManager(
            cache_dir=cache_dir,
            chunk_max_tokens=chunk_max_tokens,
            merge_peers=merge_peers,
            formula_enrichment=formula_enrichment,
        )
        self.groq_client: Optional[AsyncGroq] = None
        self.ollama_client: Optional[AsyncClient] = None
        self.ollama_host = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        self.ollama_num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "16384"))

        if self.provider == "groq":
            self.groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
            self.text_model = os.getenv("LLM_CHOICE", DEFAULT_LLM_MODEL)
            self.vision_model = os.getenv("GROQ_VISION_MODEL", DEFAULT_VISION_MODEL)
        else:
            self.ollama_client = AsyncClient(host=self.ollama_host)
            self.text_model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
            self.vision_model = os.getenv("OLLAMA_VISION_MODEL", self.text_model)
        self.query_count = 0

    def _discover_documents(self) -> List[Path]:
        if not self.documents_root.exists():
            return []

        if self.documents_root.is_file():
            return [self.documents_root] if supports_retrieval_source(self.documents_root) else []

        return sorted(
            path.resolve()
            for path in self.documents_root.rglob("*")
            if path.is_file() and supports_retrieval_source(path)
        )

    def _build_document_scope_values(self) -> tuple[List[str], List[str]]:
        allowed_sources: Set[str] = set()
        allowed_file_paths: Set[str] = set()

        bases = [self.scope_root, self.scope_root.parent, self.cwd]

        for document_path in self.discovered_documents:
            allowed_file_paths.add(str(document_path))

            for base in bases:
                try:
                    relative_path = str(document_path.relative_to(base))
                except ValueError:
                    continue

                if base == self.cwd:
                    allowed_file_paths.add(relative_path)
                else:
                    allowed_sources.add(relative_path)

        return sorted(allowed_sources), sorted(allowed_file_paths)

    def _build_scope_clause(self, start_index: int = 1) -> tuple[str, List[Any], int]:
        clauses: List[str] = []
        params: List[Any] = []
        current_index = start_index

        if self.allowed_sources:
            clauses.append(f"d.source = ANY(${current_index}::text[])")
            params.append(self.allowed_sources)
            current_index += 1

        if self.allowed_file_paths:
            clauses.append(f"d.metadata->>'file_path' = ANY(${current_index}::text[])")
            params.append(self.allowed_file_paths)
            current_index += 1

        if not clauses:
            return "FALSE", params, current_index

        return "(" + " OR ".join(clauses) + ")", params, current_index

    def _rerank_chunks(self, query: str, chunks: List[RetrievedChunk], limit: int) -> List[RetrievedChunk]:
        query_terms = extract_query_terms(query)
        query_phrases = extract_query_phrases(query_terms)

        def score(chunk: RetrievedChunk) -> float:
            content = chunk.content.lower()
            term_hits = sum(1 for term in set(query_terms) if term in content)
            phrase_hits = sum(1 for phrase in set(query_phrases) if phrase in content)
            modality = chunk_modality(chunk.chunk_metadata)
            modality_boost = 0.0
            if modality == "ocr_page":
                modality_boost = 0.05 * min(term_hits, 3)
            return chunk.similarity + (0.04 * term_hits) + (0.1 * phrase_hits) + modality_boost

        return sorted(chunks, key=score, reverse=True)[:limit]

    async def initialize_db(self) -> None:
        if self.db_pool is None:
            self.db_pool = await asyncpg.create_pool(
                os.getenv("DATABASE_URL"),
                min_size=2,
                max_size=10,
                command_timeout=60,
            )

    async def close_db(self) -> None:
        if self.db_pool is not None:
            await self.db_pool.close()
            self.db_pool = None

    async def check_database(self) -> bool:
        try:
            if not self.discovered_documents:
                print(
                    f"{Colors.RED}No supported documents found under {self.documents_root}{Colors.END}"
                )
                return False

            await self.initialize_db()
            assert self.db_pool is not None
            scope_clause, scope_params, _ = self._build_scope_clause()
            async with self.db_pool.acquire() as conn:
                doc_count = await conn.fetchval(
                    f"SELECT COUNT(*) FROM documents d WHERE {scope_clause}",
                    *scope_params,
                )
                chunk_count = await conn.fetchval(
                    f"""
                    SELECT COUNT(*)
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE {scope_clause}
                    """,
                    *scope_params,
                )

            print(
                f"{Colors.GREEN}Database ready: {doc_count} documents, {chunk_count} chunks "
                f"in scope {self.documents_root}{Colors.END}"
            )
            return True
        except Exception as exc:
            print(f"{Colors.RED}Database connection failed: {exc}{Colors.END}")
            return False

    async def check_provider(self) -> bool:
        if self.provider == "groq":
            if not os.getenv("GROQ_API_KEY"):
                print(f"{Colors.RED}GROQ_API_KEY environment variable is required{Colors.END}")
                return False
            return True

        try:
            assert self.ollama_client is not None
            models_to_check = dedupe_preserve_order([self.text_model, self.vision_model])
            for model_name in models_to_check:
                await self.ollama_client.show(model_name)

            print(
                f"{Colors.GREEN}Ollama ready at {self.ollama_host} "
                f"with model(s): {', '.join(models_to_check)}{Colors.END}"
            )
            return True
        except ResponseError as exc:
            print(f"{Colors.RED}Ollama model check failed: {exc.error}{Colors.END}")
            return False
        except Exception as exc:
            print(f"{Colors.RED}Ollama connection failed: {exc}{Colors.END}")
            return False

    async def retrieve_chunks(self, query: str, limit: Optional[int] = None) -> List[RetrievedChunk]:
        if not self.discovered_documents:
            return []

        await self.initialize_db()
        assert self.db_pool is not None

        query_embedding = await self.embedder.embed_query(query)
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
        limit = limit or self.limit
        candidate_limit = max(limit * 8, 24)
        scope_clause, scope_params, next_index = self._build_scope_clause(start_index=2)

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    c.id::text AS chunk_id,
                    c.document_id::text AS document_id,
                    c.content,
                    c.chunk_index,
                    c.metadata AS chunk_metadata,
                    d.title AS document_title,
                    d.source AS document_source,
                    d.metadata AS document_metadata,
                    1 - (c.embedding <=> $1::vector) AS similarity
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.embedding IS NOT NULL
                  AND {scope_clause}
                ORDER BY c.embedding <=> $1::vector
                LIMIT ${next_index}
                """,
                embedding_str,
                *scope_params,
                candidate_limit,
            )

        retrieved_chunks = [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                content=row["content"],
                chunk_index=row["chunk_index"],
                similarity=row["similarity"],
                chunk_metadata=normalize_json_value(row["chunk_metadata"]),
                document_title=row["document_title"],
                document_source=row["document_source"],
                document_metadata=normalize_json_value(row["document_metadata"]),
            )
            for row in rows
        ]
        return self._rerank_chunks(query, retrieved_chunks, limit)

    def resolve_source_path(self, chunk: RetrievedChunk) -> Optional[Path]:
        source_path = chunk.document_metadata.get("file_path")
        if source_path:
            resolved = Path(source_path)
            if resolved.exists():
                return resolved.resolve()

        fallback = (self.documents_root / chunk.document_source).resolve()
        if fallback.exists():
            return fallback

        return None

    async def enrich_chunks(self, chunks: List[RetrievedChunk]) -> List[EnrichedChunk]:
        enriched: List[EnrichedChunk] = []

        for chunk in chunks:
            source_path = self.resolve_source_path(chunk)
            markdown_path: Optional[Path] = None
            linked_tables: List[TableAsset] = []
            linked_images: List[ImageAsset] = []
            page_numbers = extract_metadata_pages(chunk.chunk_metadata)

            if source_path and supports_docling_assets(source_path):
                catalog = await self.asset_manager.get_catalog(source_path)
                if catalog is not None:
                    markdown_path = catalog.markdown_path
                    links = catalog.chunk_links.get(chunk.chunk_index, ChunkAssetLinks())
                    page_numbers = links.page_numbers or page_numbers
                    linked_tables = [
                        catalog.tables[asset_ref]
                        for asset_ref in links.table_refs
                        if asset_ref in catalog.tables
                    ][:2]
                    linked_images = [
                        catalog.images[asset_ref]
                        for asset_ref in links.image_refs
                        if asset_ref in catalog.images
                    ][:2]
                    if not linked_tables and page_numbers:
                        linked_tables = [
                            table
                            for table in catalog.tables.values()
                            if table.page_no in page_numbers
                        ][:2]
                    if not linked_images and page_numbers:
                        linked_images = [
                            image
                            for image in catalog.images.values()
                            if image.page_no in page_numbers
                        ][:2]

            enriched.append(
                EnrichedChunk(
                    chunk=chunk,
                    markdown_path=markdown_path,
                    page_numbers=page_numbers,
                    tables=linked_tables,
                    images=linked_images,
                )
            )

        return enriched

    def should_use_vision(self, query: str, enriched_chunks: List[EnrichedChunk]) -> bool:
        if self.vision_mode == "off":
            return False

        has_images = any(chunk.images for chunk in enriched_chunks)
        if not has_images:
            return False

        if self.vision_mode == "always":
            return True

        return question_needs_vision(query)

    def collect_images_for_vision(self, enriched_chunks: List[EnrichedChunk]) -> List[ImageAsset]:
        images: List[ImageAsset] = []
        seen: Set[str] = set()

        for chunk in enriched_chunks:
            for image in chunk.images:
                if image.asset_ref in seen:
                    continue
                seen.add(image.asset_ref)
                images.append(image)
                if len(images) >= self.max_vision_images:
                    return images

        return images

    async def analyze_images(self, query: str, images: List[ImageAsset]) -> Dict[str, Dict[str, str]]:
        if not images:
            return {}

        if self.provider == "ollama":
            return await self._analyze_images_ollama(query, images)
        return await self._analyze_images_groq(query, images)

    async def _analyze_images_groq(self, query: str, images: List[ImageAsset]) -> Dict[str, Dict[str, str]]:
        content_parts: List[Dict[str, Any]] = []
        registry_lines = [
            "You are analyzing document figures for retrieval-augmented QA.",
            f"User question: {query}",
            "Return a JSON object with an 'images' array.",
            "Each item must include: image_index, asset_ref, summary, ocr_text, relevance.",
            "Use exact OCR text when it is visible. Keep summaries concise.",
            "",
            "Image registry:",
        ]

        for index, image in enumerate(images, start=1):
            registry_lines.append(
                f"{index}. asset_ref={image.asset_ref}, page={image.page_no or 'unknown'}, caption={image.caption or 'none'}"
            )

        content_parts.append({"type": "text", "text": "\n".join(registry_lines)})
        for image in images:
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self.encode_image_for_groq(image.file_path),
                    },
                }
            )

        try:
            assert self.groq_client is not None
            response = await self.groq_client.chat.completions.create(
                model=self.vision_model,
                messages=[{"role": "user", "content": content_parts}],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_completion_tokens=800,
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
        except Exception as exc:
            logger.warning("Vision analysis failed: %s", exc)
            return {}

        images_payload = parsed.get("images", [])
        if not isinstance(images_payload, list):
            return {}

        notes_by_ref: Dict[str, Dict[str, str]] = {}
        for item in images_payload:
            if not isinstance(item, dict):
                continue
            asset_ref = item.get("asset_ref")
            if not asset_ref:
                continue
            notes_by_ref[asset_ref] = {
                "summary": str(item.get("summary", "")).strip(),
                "ocr_text": str(item.get("ocr_text", "")).strip(),
                "relevance": str(item.get("relevance", "")).strip(),
            }

        return notes_by_ref

    async def _analyze_images_ollama(self, query: str, images: List[ImageAsset]) -> Dict[str, Dict[str, str]]:
        registry_lines = [
            "You are analyzing document figures for retrieval-augmented QA.",
            f"User question: {query}",
            "Return a JSON object with an 'images' array.",
            "Each item must include: image_index, asset_ref, summary, ocr_text, relevance.",
            "Use exact OCR text when it is visible. Keep summaries concise.",
            "The asset_ref must exactly match one of the registered assets below.",
            "",
            "Image registry:",
        ]

        for index, image in enumerate(images, start=1):
            registry_lines.append(
                f"{index}. asset_ref={image.asset_ref}, page={image.page_no or 'unknown'}, caption={image.caption or 'none'}"
            )

        image_payloads = await asyncio.gather(
            *(
                asyncio.to_thread(self.prepare_image_for_ollama, image.file_path)
                for image in images
            )
        )

        try:
            assert self.ollama_client is not None
            response = await self.ollama_client.chat(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": "\n".join(registry_lines),
                        "images": image_payloads,
                    }
                ],
                format=VisionAnalysisResponse.model_json_schema(),
                options={
                    "temperature": 0,
                    "num_ctx": min(self.ollama_num_ctx, 8192),
                },
                think=False,
            )
            content = clean_structured_response(response.message.content or "{}")
            if content.startswith("["):
                content = json.dumps({"images": json.loads(content)})
            parsed = VisionAnalysisResponse.model_validate_json(content)
        except (ResponseError, ValidationError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Vision analysis failed: %s", exc)
            return {}
        except Exception as exc:
            logger.warning("Vision analysis failed: %s", exc)
            return {}

        notes_by_ref: Dict[str, Dict[str, str]] = {}
        for item in parsed.images:
            if not item.asset_ref:
                continue
            notes_by_ref[item.asset_ref] = {
                "summary": item.summary.strip(),
                "ocr_text": item.ocr_text.strip(),
                "relevance": str(item.relevance).strip(),
            }

        return notes_by_ref

    def encode_image_for_groq(self, image_path: Path) -> str:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            max_pixels = 30_000_000
            total_pixels = image.width * image.height
            if total_pixels > max_pixels:
                scale = (max_pixels / total_pixels) ** 0.5
                resized = (
                    max(1, int(image.width * scale)),
                    max(1, int(image.height * scale)),
                )
                image = image.resize(resized)

            quality = 90
            while True:
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=quality, optimize=True)
                payload = buffer.getvalue()
                if len(payload) <= 3_500_000 or quality <= 45:
                    break

                quality -= 15
                image = image.resize(
                    (
                        max(1, int(image.width * 0.9)),
                        max(1, int(image.height * 0.9)),
                    )
                )

        encoded = base64.b64encode(payload).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def prepare_image_for_ollama(self, image_path: Path) -> bytes:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            max_pixels = 30_000_000
            total_pixels = image.width * image.height
            if total_pixels > max_pixels:
                scale = (max_pixels / total_pixels) ** 0.5
                resized = (
                    max(1, int(image.width * scale)),
                    max(1, int(image.height * scale)),
                )
                image = image.resize(resized)

            quality = 90
            while True:
                buffer = BytesIO()
                image.save(buffer, format="JPEG", quality=quality, optimize=True)
                payload = buffer.getvalue()
                if len(payload) <= 3_500_000 or quality <= 45:
                    break

                quality -= 15
                image = image.resize(
                    (
                        max(1, int(image.width * 0.9)),
                        max(1, int(image.height * 0.9)),
                    )
                )

        return payload

    def build_context(
        self,
        query: str,
        enriched_chunks: List[EnrichedChunk],
        vision_notes: Dict[str, Dict[str, str]],
    ) -> str:
        sections = [f"User question: {query}", "", "Retrieved knowledge context:"]

        for item in enriched_chunks:
            chunk = item.chunk
            pages = ", ".join(map(str, item.page_numbers)) if item.page_numbers else "unknown"
            modality = chunk_modality(chunk.chunk_metadata) or "text"
            sections.append(
                f"[Source: {chunk.document_title} | chunk={chunk.chunk_index} | similarity={chunk.similarity:.3f} | pages={pages} | modality={modality}]"
            )
            sections.append(chunk.content.strip())

            if item.markdown_path is not None:
                sections.append(f"Rich markdown cache: {item.markdown_path}")

            for table in item.tables:
                sections.append(
                    f"[Linked table {table.asset_ref} | page={table.page_no or 'unknown'}]\n"
                    f"{truncate_text(table.markdown, 2500)}"
                )

            for image in item.images:
                note = vision_notes.get(image.asset_ref)
                if note:
                    image_lines = [
                        f"[Linked image {image.asset_ref} | page={image.page_no or 'unknown'}]",
                    ]
                    if image.caption:
                        image_lines.append(f"Caption: {image.caption}")
                    if note.get("summary"):
                        image_lines.append(f"Summary: {note['summary']}")
                    if note.get("ocr_text"):
                        image_lines.append(f"OCR: {note['ocr_text']}")
                    if note.get("relevance"):
                        image_lines.append(f"Relevance: {note['relevance']}")
                    sections.append("\n".join(image_lines))
                else:
                    sections.append(
                        f"[Linked image available {image.asset_ref} | page={image.page_no or 'unknown'} | file={image.file_path.name}]"
                    )

            sections.append("")

        return "\n".join(sections).strip()

    async def generate_answer(self, system_prompt: str, context: str) -> str:
        user_prompt = (
            "Answer the following question using only the retrieved context.\n\n"
            f"{context}\n\n"
            "Give a direct answer first, then concise source citations. "
            "If support is partial or missing, explicitly say so instead of filling gaps."
        )

        if self.provider == "ollama":
            assert self.ollama_client is not None
            response = await self.ollama_client.chat(
                model=self.text_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": 0,
                    "num_ctx": self.ollama_num_ctx,
                    "num_predict": 1200,
                },
                think=False,
            )
            return (response.message.content or "No answer returned.").strip()

        assert self.groq_client is not None
        response = await self.groq_client.chat.completions.create(
            model=self.text_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_completion_tokens=1200,
        )
        return (response.choices[0].message.content or "No answer returned.").strip()

    def append_response_markdown(
        self,
        query: str,
        answer: str,
        enriched_chunks: List[EnrichedChunk],
        used_vision: bool,
    ) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        unique_sources = dedupe_preserve_order([item.chunk.document_title for item in enriched_chunks])
        linked_tables = sum(len(item.tables) for item in enriched_chunks)
        linked_images = sum(len(item.images) for item in enriched_chunks)

        if not self.responses_md_path.exists():
            self.responses_md_path.write_text("# Vision CLI Responses\n\n", encoding="utf-8")

        if not self.responses_session_logged:
            header = (
                f"### Session {self.session_started_at.isoformat(timespec='seconds')}\n\n"
                f"- Provider: {self.provider}\n"
                f"- Text model: {self.text_model}\n"
                f"- Vision model: {self.vision_model}\n"
                f"- Formula enrichment: {self.formula_enrichment}\n"
                f"- Scope: {self.documents_root}\n\n"
            )
            with self.responses_md_path.open("a", encoding="utf-8") as handle:
                handle.write(header)
            self.responses_session_logged = True

        chunk_lines = []
        for item in enriched_chunks:
            pages = ", ".join(map(str, item.page_numbers)) if item.page_numbers else "unknown"
            modality = chunk_modality(item.chunk.chunk_metadata) or "text"
            chunk_lines.append(
                f"- {item.chunk.document_title} | chunk={item.chunk.chunk_index} | "
                f"similarity={item.chunk.similarity:.3f} | pages={pages} | modality={modality}"
            )

        context_block = (
            f"- Timestamp: {timestamp}\n"
            f"- Provider: {self.provider}\n"
            f"- Text model: {self.text_model}\n"
            f"- Vision model: {self.vision_model}\n"
            f"- Formula enrichment: {self.formula_enrichment}\n"
            f"- Vision used: {used_vision}\n"
            f"- Sources: {', '.join(unique_sources) if unique_sources else 'none'}\n"
            f"- Linked tables: {linked_tables}\n"
            f"- Linked images: {linked_images}\n"
        )
        if chunk_lines:
            context_block += "- Retrieved chunks:\n" + "\n".join(chunk_lines) + "\n"

        entry = (
            f"## Query {self.query_count}\n\n"
            f"Q: {query}\n\n"
            f"A:\n{answer}\n\n"
            f"Context:\n{context_block}\n"
        )

        with self.responses_md_path.open("a", encoding="utf-8") as handle:
            handle.write(entry)

    async def answer_question(self, query: str) -> tuple[str, List[EnrichedChunk], bool]:
        chunks = await self.retrieve_chunks(query)
        if not chunks:
            self.query_count += 1
            return (
                "No relevant information was found in the knowledge base for that question.",
                [],
                False,
            )

        enriched_chunks = await self.enrich_chunks(chunks)
        use_vision = self.should_use_vision(query, enriched_chunks)
        vision_notes: Dict[str, Dict[str, str]] = {}
        if use_vision:
            vision_notes = await self.analyze_images(
                query,
                self.collect_images_for_vision(enriched_chunks),
            )

        context = self.build_context(query, enriched_chunks, vision_notes)
        system_prompt = (
            "You answer questions using only retrieved documentation from a private knowledge base. "
            "Never answer from general knowledge or outside assumptions. "
            "Prefer retrieved chunk text and linked table markdown. "
            "Use linked image analysis only when it directly supports the answer. "
            "If the retrieved context does not directly answer the question, say exactly: "
            "'The retrieved document does not directly answer this question.' "
            "Then briefly mention the closest relevant evidence, if any. "
            "Do not cite a source unless it directly supports the claim being made. "
            "Include page numbers in citations when available using the format [Document Title p.X]."
        )

        answer = await self.generate_answer(system_prompt, context)
        self.query_count += 1
        return answer.strip(), enriched_chunks, use_vision

    def print_banner(self) -> None:
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 64}")
        print("Docling Vision RAG Assistant")
        print("=" * 64)
        print(f"Provider:     {self.provider}")
        print(f"Text model:   {self.text_model}")
        print(f"Vision model: {self.vision_model}")
        if self.provider == "ollama":
            print(f"Ollama host:  {self.ollama_host}")
        print(f"Vision mode:  {self.vision_mode}")
        print(f"Formulas:     {'enabled' if self.formula_enrichment else 'disabled'}")
        print(f"Chunking:     max_tokens={self.chunk_max_tokens}, merge_peers={self.merge_peers}")
        print(f"Cache dir:    {self.cache_dir}")
        print(f"Responses MD: {self.responses_md_path}")
        print(f"Scope:        {self.documents_root}")
        print("=" * 64 + f"{Colors.END}\n")

    def print_help(self) -> None:
        help_text = f"""
{Colors.BOLD}Commands:{Colors.END}
  help        Show this help message
  paste       Enter multiline question mode
  stats       Show session statistics
  exit/quit   Exit the CLI

{Colors.BOLD}Behavior:{Colors.END}
  - Retrieves top text chunks from PGVector
  - Restricts retrieval to files inside the configured scope
  - Rebuilds linked tables and nearby figures from source docs on demand
  - Reuses the ingestion chunking shape for figure/table linking
  - Uses table markdown directly
  - Can enable Docling formula enrichment for PDFs when formula fidelity matters
  - Uses the selected provider for text and optional vision reasoning
  - Appends each question/answer/context summary to the configured markdown file
"""
        print(help_text)

    def print_stats(self) -> None:
        print(f"{Colors.BLUE}Queries answered: {self.query_count}{Colors.END}")
        print(f"{Colors.BLUE}Documents in scope: {len(self.discovered_documents)}{Colors.END}")

    def _read_multiline_input(self, first_line: Optional[str] = None) -> str:
        lines: List[str] = []
        if first_line and first_line.strip():
            lines.append(first_line.rstrip())

        print(f"{Colors.BLUE}Multiline mode: submit an empty line to send.{Colors.END}")
        while True:
            line = input("... ").rstrip()
            if not line.strip():
                break
            lines.append(line)

        return "\n".join(lines).strip()

    def _read_user_input(self) -> str:
        first_line = input("You: ").rstrip()
        if not first_line.strip():
            return ""

        lowered = first_line.strip().lower()
        if lowered == "paste":
            return self._read_multiline_input()

        if first_line.strip().endswith(":"):
            return self._read_multiline_input(first_line=first_line)

        return first_line.strip()

    async def run(self) -> None:
        self.print_banner()
        if not await self.check_database():
            return
        if not await self.check_provider():
            return

        while True:
            try:
                user_input = self._read_user_input()
            except EOFError:
                print()
                break

            if not user_input:
                continue

            lowered = user_input.lower()
            if lowered in {"exit", "quit", "bye"}:
                break
            if lowered == "help":
                self.print_help()
                continue
            if lowered == "stats":
                self.print_stats()
                continue

            try:
                answer, enriched_chunks, used_vision = await self.answer_question(user_input)
            except Exception as exc:
                logger.error("Failed to answer question: %s", exc, exc_info=True)
                print(f"{Colors.RED}Error: {exc}{Colors.END}\n")
                continue

            print(f"\n{Colors.GREEN}Assistant:{Colors.END} {answer}\n")
            self.append_response_markdown(user_input, answer, enriched_chunks, used_vision)

            unique_sources = dedupe_preserve_order(
                [item.chunk.document_title for item in enriched_chunks]
            )
            linked_tables = sum(len(item.tables) for item in enriched_chunks)
            linked_images = sum(len(item.images) for item in enriched_chunks)

            print(
                f"{Colors.YELLOW}Context:{Colors.END} "
                f"sources={', '.join(unique_sources)} | "
                f"tables={linked_tables} | images={linked_images} | "
                f"vision_used={used_vision}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Docling vision-aware CLI")
    parser.add_argument(
        "--documents-root",
        default="documents",
        help="Root folder for original source documents",
    )
    parser.add_argument(
        "--provider",
        choices=["groq", "ollama"],
        default=os.getenv("VISION_PROVIDER", "groq"),
        help="LLM provider to use for answering and vision analysis",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="Folder used for extracted markdown and image artifacts",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Number of chunks to retrieve per query",
    )
    parser.add_argument(
        "--vision-mode",
        choices=["auto", "always", "off"],
        default="auto",
        help="When to call the provider vision model",
    )
    parser.add_argument(
        "--chunk-max-tokens",
        type=int,
        default=DEFAULT_HYBRID_MAX_TOKENS,
        help="Max tokens used for Docling asset-link chunking",
    )
    parser.add_argument(
        "--merge-peers",
        action="store_true",
        help="Allow Docling asset-link chunking to merge small adjacent chunks",
    )
    parser.add_argument(
        "--max-vision-images",
        type=int,
        default=3,
        help="Maximum number of linked images to send to the provider vision model per query",
    )
    parser.add_argument(
        "--formula-enrichment",
        action="store_true",
        help="Enable Docling formula enrichment for PDFs (slower, especially on CPU-only paths)",
    )
    parser.add_argument(
        "--responses-md",
        default=DEFAULT_RESPONSES_MD,
        help="Markdown file used to append question, answer, and context summaries",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not os.getenv("DATABASE_URL"):
        logger.error("DATABASE_URL environment variable is required")
        sys.exit(1)

    cli = VisionRAGCLI(
        documents_root=args.documents_root,
        cache_dir=args.cache_dir,
        limit=args.limit,
        provider=args.provider,
        vision_mode=args.vision_mode,
        chunk_max_tokens=args.chunk_max_tokens,
        merge_peers=args.merge_peers,
        max_vision_images=args.max_vision_images,
        responses_md=args.responses_md,
        formula_enrichment=args.formula_enrichment,
    )

    try:
        await cli.run()
    finally:
        await cli.close_db()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
