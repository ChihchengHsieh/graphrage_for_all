# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing text_embed, load_strategy and create_row_from_embedding_data methods definition."""
import math

import logging
from enum import Enum
from typing import Any, cast, Callable

import numpy as np
import pandas as pd
from datashaper import TableContainer, VerbCallbacks, VerbInput, verb
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from . import defs
from .text_splitter import TokenTextSplitter


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


@verb(name="text_embed")
def text_embed(
    input: VerbInput,
    # callbacks: VerbCallbacks,
    # cache: PipelineCache,
    send_to: Callable,
    column: str,
    strategy: dict,
    **kwargs,
) -> TableContainer:
    """
    Embed a piece of text into a vector space. The verb outputs a new column containing a mapping between doc_id and vector.

    ## Usage
    ```yaml
    verb: text_embed
    args:
        column: text # The name of the column containing the text to embed, this can either be a column with text, or a column with a list[tuple[doc_id, str]]
        to: embedding # The name of the column to output the embedding to
        strategy: <strategy config> # See strategies section below
    ```

    ## Strategies
    The text embed verb uses a strategy to embed the text. The strategy is an object which defines the strategy to use. The following strategies are available:

    ### openai
    This strategy uses openai to embed a piece of text. In particular it uses a LLM to embed a piece of text. The strategy config is as follows:

    ```yaml
    strategy:
        type: openai
        llm: # The configuration for the LLM
            type: openai_embedding # the type of llm to use, available options are: openai_embedding, azure_openai_embedding
            api_key: !ENV ${GRAPHRAG_OPENAI_API_KEY} # The api key to use for openai
            model: !ENV ${GRAPHRAG_OPENAI_MODEL:gpt-4-turbo-preview} # The model to use for openai
            max_tokens: !ENV ${GRAPHRAG_MAX_TOKENS:6000} # The max tokens to use for openai
            organization: !ENV ${GRAPHRAG_OPENAI_ORGANIZATION} # The organization to use for openai
        vector_store: # The optional configuration for the vector store
            type: lancedb # The type of vector store to use, available options are: azure_ai_search, lancedb
            <...>
    ```
    """
    return _text_embed_in_memory(
        input,
        # callbacks,
        # cache,
        send_to,
        column,
        strategy,
        kwargs.get("to", f"{column}_embedding"),
    )


def _text_embed_in_memory(
    input: VerbInput,
    # callbacks: VerbCallbacks,
    # cache: PipelineCache,
    send_to: Callable,
    column: str,
    strategy: dict,
    to: str,
):
    output_df = input
    strategy_exec = run_ai_embed
    strategy_args = {**strategy}
    input_table = input

    texts: list[str] = input_table[column].to_numpy().tolist()
    result = strategy_exec(send_to, texts, strategy_args)

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
    send_to: Callable,
    input: list[str],
    # callbacks: VerbCallbacks,
    # cache: PipelineCache,
    args: dict[str, Any],
) -> TextEmbeddingResult:
    """Run the Claim extraction chain."""
    if is_null(input):
        return TextEmbeddingResult(embeddings=None)

    batch_size = args.get("batch_size", 16)
    batch_max_tokens = args.get("batch_max_tokens", 8191)
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
    embeddings = _execute(send_to, text_batches)  # ticker, semaphore)
    embeddings = _reconstitute_embeddings(embeddings, input_sizes)

    return TextEmbeddingResult(embeddings=embeddings)


def _get_splitter(batch_max_tokens: int) -> TokenTextSplitter:
    return TokenTextSplitter(
        encoding_name=defs.ENCODING_MODEL,
        chunk_size=batch_max_tokens,
    )


def _execute(
    send_to: Callable,
    chunks: list[list[str]],
    # tick: ProgressTicker,
    # semaphore: asyncio.Semaphore,
) -> list[list[float]]:
    def embed(chunk: list[str]):
        chunk_embeddings = send_to(chunk)
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
