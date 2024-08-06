# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
from __future__ import annotations
from enum import Enum

"""Common default configuration values."""


# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'PipelineCacheConfig', 'PipelineFileCacheConfig' and 'PipelineMemoryCacheConfig' models."""


class AsyncType(str, Enum):
    """Enum for the type of async to use."""

    AsyncIO = "asyncio"
    Threaded = "threaded"


class CacheType(str, Enum):
    """The cache configuration type for the pipeline."""

    file = "file"
    """The file cache configuration type."""
    memory = "memory"
    """The memory cache configuration type."""
    none = "none"
    """The none cache configuration type."""
    blob = "blob"
    """The blob cache configuration type."""

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


class InputFileType(str, Enum):
    """The input file type for the pipeline."""

    csv = "csv"
    """The CSV input type."""
    text = "text"
    """The text input type."""

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


class InputType(str, Enum):
    """The input type for the pipeline."""

    file = "file"
    """The file storage type."""
    blob = "blob"
    """The blob storage type."""

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


class StorageType(str, Enum):
    """The storage type for the pipeline."""

    file = "file"
    """The file storage type."""
    memory = "memory"
    """The memory storage type."""
    blob = "blob"
    """The blob storage type."""

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


class ReportingType(str, Enum):
    """The reporting configuration type for the pipeline."""

    file = "file"
    """The file reporting configuration type."""
    console = "console"
    """The console reporting configuration type."""
    blob = "blob"
    """The blob reporting configuration type."""

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


class TextEmbeddingTarget(str, Enum):
    """The target to use for text embeddings."""

    all = "all"
    required = "required"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


class LLMType(str, Enum):
    """LLMType enum class definition."""

    # Embeddings
    OpenAIEmbedding = "openai_embedding"
    AzureOpenAIEmbedding = "azure_openai_embedding"

    # Raw Completion
    OpenAI = "openai"
    AzureOpenAI = "azure_openai"

    # Chat Completion
    OpenAIChat = "openai_chat"
    AzureOpenAIChat = "azure_openai_chat"

    # Debug
    StaticResponse = "static_response"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


ASYNC_MODE = AsyncType.Threaded
ENCODING_MODEL = "cl100k_base"
#
# LLM Parameters
#
LLM_TYPE = LLMType.OpenAIChat
LLM_MODEL = "gpt-4-turbo-preview"
LLM_MAX_TOKENS = 4000
LLM_TEMPERATURE = 0
LLM_TOP_P = 1
LLM_N = 1
LLM_REQUEST_TIMEOUT = 180.0
LLM_TOKENS_PER_MINUTE = 0
LLM_REQUESTS_PER_MINUTE = 0
LLM_MAX_RETRIES = 10
LLM_MAX_RETRY_WAIT = 10.0
LLM_SLEEP_ON_RATE_LIMIT_RECOMMENDATION = True
LLM_CONCURRENT_REQUESTS = 25

#
# Text Embedding Parameters
#
EMBEDDING_TYPE = LLMType.OpenAIEmbedding
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 16
EMBEDDING_BATCH_MAX_TOKENS = 8191
EMBEDDING_TARGET = TextEmbeddingTarget.required

CACHE_TYPE = CacheType.file
CACHE_BASE_DIR = "cache"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100
CHUNK_GROUP_BY_COLUMNS = ["id"]
CLAIM_DESCRIPTION = (
    "Any claims or facts that could be relevant to information discovery."
)
CLAIM_MAX_GLEANINGS = 1
CLAIM_EXTRACTION_ENABLED = False
MAX_CLUSTER_SIZE = 10
COMMUNITY_REPORT_MAX_LENGTH = 2000
COMMUNITY_REPORT_MAX_INPUT_LENGTH = 8000
ENTITY_EXTRACTION_ENTITY_TYPES = ["organization", "person", "geo", "event"]
ENTITY_EXTRACTION_MAX_GLEANINGS = 1
INPUT_FILE_TYPE = InputFileType.text
INPUT_TYPE = InputType.file
INPUT_BASE_DIR = "input"
INPUT_FILE_ENCODING = "utf-8"
INPUT_TEXT_COLUMN = "text"
INPUT_CSV_PATTERN = ".*\\.csv$"
INPUT_TEXT_PATTERN = ".*\\.txt$"
PARALLELIZATION_STAGGER = 0.3
PARALLELIZATION_NUM_THREADS = 50
NODE2VEC_ENABLED = False
NODE2VEC_NUM_WALKS = 10
NODE2VEC_WALK_LENGTH = 40
NODE2VEC_WINDOW_SIZE = 2
NODE2VEC_ITERATIONS = 3
NODE2VEC_RANDOM_SEED = 597832
REPORTING_TYPE = ReportingType.file
REPORTING_BASE_DIR = "output/${timestamp}/reports"
SNAPSHOTS_GRAPHML = False
SNAPSHOTS_RAW_ENTITIES = False
SNAPSHOTS_TOP_LEVEL_NODES = False
STORAGE_BASE_DIR = "output/${timestamp}/artifacts"
STORAGE_TYPE = StorageType.file
SUMMARIZE_DESCRIPTIONS_MAX_LENGTH = 500
UMAP_ENABLED = False

# Local Search
LOCAL_SEARCH_TEXT_UNIT_PROP = 0.5
LOCAL_SEARCH_COMMUNITY_PROP = 0.1
LOCAL_SEARCH_CONVERSATION_HISTORY_MAX_TURNS = 5
LOCAL_SEARCH_TOP_K_MAPPED_ENTITIES = 10
LOCAL_SEARCH_TOP_K_RELATIONSHIPS = 10
LOCAL_SEARCH_MAX_TOKENS = 12_000
LOCAL_SEARCH_LLM_TEMPERATURE = 0
LOCAL_SEARCH_LLM_TOP_P = 1
LOCAL_SEARCH_LLM_N = 1
LOCAL_SEARCH_LLM_MAX_TOKENS = 2000

# Global Search
GLOBAL_SEARCH_LLM_TEMPERATURE = 0
GLOBAL_SEARCH_LLM_TOP_P = 1
GLOBAL_SEARCH_LLM_N = 1
GLOBAL_SEARCH_MAX_TOKENS = 12_000
GLOBAL_SEARCH_DATA_MAX_TOKENS = 12_000
GLOBAL_SEARCH_MAP_MAX_TOKENS = 1000
GLOBAL_SEARCH_REDUCE_MAX_TOKENS = 2_000
GLOBAL_SEARCH_CONCURRENCY = 32

# Graph Extract
DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
DEFAULT_ENTITY_TYPES = ["disease", "symptoms", "cause"]
