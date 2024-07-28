import pandas as pd
from enum import Enum
from typing import Any, Iterable, cast, Callable
import nltk
from dataclasses import dataclass
import df_ops.defaults as defs
import tiktoken


initialized_nltk = False


class ChunkStrategyType(str, Enum):
    """ChunkStrategy class definition."""

    tokens = "tokens"
    sentence = "sentence"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


@dataclass
class TextChunk:
    """Text chunk class definition."""

    text_chunk: str
    source_doc_indices: list[int]
    n_tokens: int | None = None


EncodedText = list[int]
DecodeFn = Callable[[EncodedText], str]
EncodeFn = Callable[[str], EncodedText]
ChunkStrategy = Callable[[list[str], dict[str, Any]], Iterable[TextChunk]]
ChunkInput = str | list[str] | list[tuple[str, str]]


@dataclass(frozen=True)
class Tokenizer:
    """Tokenizer data class."""

    chunk_overlap: int
    """Overlap in tokens between chunks"""
    tokens_per_chunk: int
    """Maximum number of tokens per chunk"""
    decode: DecodeFn
    """ Function to decode a list of token ids to a string"""
    encode: EncodeFn
    """ Function to encode a string to a list of token ids"""


def split_text_on_tokens(
    texts: list[str],
    enc: Tokenizer,
) -> list[TextChunk]:
    """Split incoming text and return chunks."""
    result = []
    mapped_ids = []

    for source_doc_idx, text in enumerate(texts):
        encoded = enc.encode(text)
        mapped_ids.append((source_doc_idx, encoded))

    input_ids: list[tuple[int, int]] = [
        (source_doc_idx, id) for source_doc_idx, ids in mapped_ids for id in ids
    ]

    start_idx = 0
    cur_idx = min(start_idx + enc.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]
    while start_idx < len(input_ids):
        chunk_text = enc.decode([id for _, id in chunk_ids])
        doc_indices = list({doc_idx for doc_idx, _ in chunk_ids})
        result.append(
            TextChunk(
                text_chunk=chunk_text,
                source_doc_indices=doc_indices,
                n_tokens=len(chunk_ids),
            )
        )
        start_idx += enc.tokens_per_chunk - enc.chunk_overlap
        cur_idx = min(start_idx + enc.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]

    return result


def run_tokens(
    input: list[str],
    args: dict[str, Any],
) -> Iterable[TextChunk]:
    """Chunks text into multiple parts. A pipeline verb."""
    tokens_per_chunk = args.get("chunk_size", defs.CHUNK_SIZE)
    chunk_overlap = args.get("chunk_overlap", defs.CHUNK_OVERLAP)
    encoding_name = args.get("encoding_name", defs.ENCODING_MODEL)
    enc = tiktoken.get_encoding(encoding_name)

    def encode(text: str) -> list[int]:
        if not isinstance(text, str):
            text = f"{text}"
        return enc.encode(text)

    def decode(tokens: list[int]) -> str:
        return enc.decode(tokens)

    return split_text_on_tokens(
        input,
        Tokenizer(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=tokens_per_chunk,
            encode=encode,
            decode=decode,
        ),
    )


def bootstrap():
    """Bootstrap definition."""
    global initialized_nltk
    if not initialized_nltk:
        import nltk
        from nltk.corpus import wordnet as wn

        nltk.download("punkt")
        nltk.download("averaged_perceptron_tagger")
        nltk.download("maxent_ne_chunker")
        nltk.download("words")
        nltk.download("wordnet")
        wn.ensure_loaded()
        initialized_nltk = True


def run_sentence(input: list[str], _args: dict[str, Any]) -> Iterable[TextChunk]:
    """Chunks text into multiple parts. A pipeline verb."""
    for doc_idx, text in enumerate(input):
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            yield TextChunk(
                text_chunk=sentence,
                source_doc_indices=[doc_idx],
            )


def load_strategy(strategy: ChunkStrategyType) -> ChunkStrategy:
    """Load strategy method definition."""
    match strategy:
        case ChunkStrategyType.tokens:
            return run_tokens
        case ChunkStrategyType.sentence:
            # NLTK
            bootstrap()
            return run_sentence
        case _:
            msg = f"Unknown strategy: {strategy}"
            raise ValueError(msg)


def run_strategy(
    strategy: ChunkStrategy,
    input: ChunkInput,
    strategy_args: dict[str, Any],
) -> list[str | tuple[list[str] | None, str, int]]:
    """Run strategy method definition."""
    if isinstance(input, str):
        return [item.text_chunk for item in strategy([input], {**strategy_args})]

    # We can work with both just a list of text content
    # or a list of tuples of (document_id, text content)
    # text_to_chunk = '''
    texts = []
    for item in input:
        if isinstance(item, str):
            texts.append(item)
        else:
            texts.append(item[1])

    strategy_results = strategy(texts, {**strategy_args})

    results = []
    for strategy_result in strategy_results:
        doc_indices = strategy_result.source_doc_indices
        if isinstance(input[doc_indices[0]], str):
            results.append(strategy_result.text_chunk)
        else:
            doc_ids = [input[doc_idx][0] for doc_idx in doc_indices]
            results.append(
                (
                    doc_ids,
                    strategy_result.text_chunk,
                    strategy_result.n_tokens,
                )
            )
    return results


def chunk(
    input: pd.DataFrame,
    column: str,
    to: str,
    strategy: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Chunk a piece of text into smaller pieces.
    """
    if strategy is None:
        strategy = {}
    output = input
    strategy_name = strategy.get("type", ChunkStrategyType.tokens)
    strategy_config = {**strategy}
    strategy_exec = load_strategy(strategy_name)

    output[to] = output.apply(
        cast(
            Any,
            lambda x: run_strategy(strategy_exec, x[column], strategy_config),
        ),
        axis=1,
    )
    return output
