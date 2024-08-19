# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing text_embed, load_strategy and create_row_from_embedding_data methods definition."""
import math

import logging
from enum import Enum
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from . import defs
from graphrag_for_all.text.text_splitter import TokenTextSplitter
from graphrag_for_all.llm.send import EmbLLM, ModelArgs

log = logging.getLogger(__name__)

# Per Azure OpenAI Limits
# https://learn.microsoft.com/en-us/azure/ai-services/openai/reference
DEFAULT_EMBEDDING_BATCH_SIZE = 500

TextEmbedder = Callable[[str], list[float]]


@dataclass
class VectorStoreDocument:
    """A document that is stored in vector storage."""

    id: str | int
    """unique id for the document"""

    text: str | None
    vector: list[float] | None

    attributes: dict[str, Any] = field(default_factory=dict)
    """store any additional metadata, e.g. title, date ranges, etc"""


@dataclass
class VectorStoreSearchResult:
    """A vector storage search result."""

    document: VectorStoreDocument
    """Document that was found."""

    score: float
    """Similarity score between 0 and 1. Higher is more similar."""


class BaseVectorStore(ABC):
    """The base class for vector storage data-access classes."""

    def __init__(
        self,
        collection_name: str,
        db_connection: Any | None = None,
        document_collection: Any | None = None,
        query_filter: Any | None = None,
        **kwargs: Any,
    ):
        self.collection_name = collection_name
        self.db_connection = db_connection
        self.document_collection = document_collection
        self.query_filter = query_filter
        self.kwargs = kwargs

    @abstractmethod
    def connect(self, **kwargs: Any) -> None:
        """Connect to vector storage."""

    @abstractmethod
    def load_documents(
        self, documents: list[VectorStoreDocument], overwrite: bool = True
    ) -> None:
        """Load documents into the vector-store."""

    @abstractmethod
    def similarity_search_by_vector(
        self, query_embedding: list[float], k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform ANN search by vector."""

    @abstractmethod
    def similarity_search_by_text(
        self, text: str, text_embedder: TextEmbedder, k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform ANN search by text."""

    @abstractmethod
    def filter_by_id(self, include_ids: list[str] | list[int]) -> Any:
        """Build a query filter to filter documents by id."""


class TextEmbedStrategyType(str, Enum):
    """TextEmbedStrategyType class definition."""

    openai = "openai"
    mock = "mock"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


def text_embed(
    input: pd.DataFrame,
    # callbacks: VerbCallbacks,
    # cache: PipelineCache,
    send_to: EmbLLM,
    column: str,
    strategy: dict,
    to: str | None = None,
    llm_args: Dict = {},
) -> pd.DataFrame:

    if to is None:
        to = f"{column}_embedding"

    return _text_embed_in_memory(
        input,
        # callbacks,
        # cache,
        send_to,
        column,
        strategy,
        to,
        llm_args,
    )


def _text_embed_in_memory(
    input: pd.DataFrame,
    # callbacks: VerbCallbacks,
    # cache: PipelineCache,
    send_to: EmbLLM,
    column: str,
    strategy: dict,
    to: str,
    llm_args: ModelArgs,
):
    output_df = input
    strategy_args = {**strategy}
    input_table = input

    texts: list[str] = input_table[column].to_numpy().tolist()
    result = run_ai_embed(send_to, texts, strategy_args, llm_args)

    output_df[to] = result.embeddings
    return output_df


@dataclass
class TextEmbeddingResult:
    """Text embedding result class definition."""

    embeddings: list[list[float] | None] | None


def is_null(value: Any) -> bool:
    """Check if value is null or is nan."""

    def is_none() -> bool:
        return value is None

    def is_nan() -> bool:
        return isinstance(value, float) and math.isnan(value)

    return is_none() or is_nan()


def run_ai_embed(
    send_to: EmbLLM,
    input: list[str],
    # callbacks: VerbCallbacks,
    # cache: PipelineCache,
    strategy_args: dict[str, Any],
    llm_args: ModelArgs,
) -> TextEmbeddingResult:
    """Run the Claim extraction chain."""
    if is_null(input):
        return TextEmbeddingResult(embeddings=None)

    batch_size = strategy_args.get("batch_size", 16)
    batch_max_tokens = strategy_args.get("batch_max_tokens", 8191)
    splitter = _get_splitter(batch_max_tokens)

    # Break up the input texts. The sizes here indicate how many snippets are in each input text
    texts, input_sizes = _prepare_embed_texts(input, splitter)
    text_batches = _create_text_batches(
        texts,
        batch_size,
        batch_max_tokens,
        splitter,
    )
    log.info(
        "embedding %d inputs via %d snippets using %d batches. max_batch_size=%d, max_tokens=%d",
        len(input),
        len(texts),
        len(text_batches),
        batch_size,
        batch_max_tokens,
    )
    # ticker = progress_ticker(callbacks.progress, len(text_batches))

    # Embed each chunk of snippets
    embeddings = _execute(send_to, text_batches, llm_args)  # ticker, semaphore)
    embeddings = _reconstitute_embeddings(embeddings, input_sizes)

    return TextEmbeddingResult(embeddings=embeddings)


def _get_splitter(batch_max_tokens: int) -> TokenTextSplitter:
    return TokenTextSplitter(
        encoding_name=defs.ENCODING_MODEL,
        chunk_size=batch_max_tokens,
    )


def _execute(
    send_to: EmbLLM,
    chunks: list[list[str]],
    # tick: ProgressTicker,
    # semaphore: asyncio.Semaphore,
    llm_args: ModelArgs,
) -> list[list[float]]:
    def embed(chunk: list[str]):
        chunk_embeddings = send_to(chunk, llm_args)
        result = np.array(chunk_embeddings)  # .output)
        return result

    results = [embed(chunk) for chunk in chunks]
    # merge results in a single list of lists (reduce the collect dimension)
    return [item for sublist in results for item in sublist]


def _create_text_batches(
    texts: list[str],
    max_batch_size: int,
    max_batch_tokens: int,
    splitter,
) -> list[list[str]]:
    """Create batches of texts to embed."""
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference
    # According to this embeddings reference, Azure limits us to 16 concurrent embeddings and 8191 tokens per request
    result = []
    current_batch = []
    current_batch_tokens = 0

    for text in texts:
        token_count = splitter.num_tokens(text)
        if (
            len(current_batch) >= max_batch_size
            or current_batch_tokens + token_count > max_batch_tokens
        ):
            result.append(current_batch)
            current_batch = []
            current_batch_tokens = 0

        current_batch.append(text)
        current_batch_tokens += token_count

    if len(current_batch) > 0:
        result.append(current_batch)

    return result


def _prepare_embed_texts(
    input: list[str],
    splitter,
) -> tuple[list[str], list[int]]:
    sizes: list[int] = []
    snippets: list[str] = []

    for text in input:
        # Split the input text and filter out any empty content
        split_texts = splitter.split_text(text)
        if split_texts is None:
            continue
        split_texts = [text for text in split_texts if len(text) > 0]

        sizes.append(len(split_texts))
        snippets.extend(split_texts)

    return snippets, sizes


def _reconstitute_embeddings(
    raw_embeddings: list[list[float]], sizes: list[int]
) -> list[list[float] | None]:
    """Reconstitute the embeddings into the original input texts."""
    embeddings: list[list[float] | None] = []
    cursor = 0
    for size in sizes:
        if size == 0:
            embeddings.append(None)
        elif size == 1:
            embedding = raw_embeddings[cursor]
            embeddings.append(embedding)
            cursor += 1
        else:
            chunk = raw_embeddings[cursor : cursor + size]
            average = np.average(chunk, axis=0)
            normalized = average / np.linalg.norm(average)
            embeddings.append(normalized.tolist())
            cursor += size
    return embeddings
