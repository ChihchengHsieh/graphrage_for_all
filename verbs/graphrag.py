from typing import Any, Callable, Iterable
import pandas as pd
from dataclasses import dataclass


def cast(typ, val):
    return val

def zip_verb(
    input: pd.DataFrame,
    to: str,
    columns: list[str],
    type: str | None = None,  # noqa A002
    **_kwargs: dict,
) -> pd.DataFrame:
    """
    Zip columns together.

    ## Usage
    TODO

    """
    table = input
    if type is None:
        table[to] = list(zip(*[table[col] for col in columns], strict=True))

    # This one is a little weird
    elif type == "dict":
        if len(columns) != 2:
            msg = f"Expected exactly two columns for a dict, got {columns}"
            raise ValueError(msg)
        key_col, value_col = columns

        results = []
        for _, row in table.iterrows():
            keys = row[key_col]
            values = row[value_col]
            output = {}
            if len(keys) != len(values):
                msg = f"Expected same number of keys and values, got {len(keys)} keys and {len(values)} values"
                raise ValueError(msg)
            for idx, key in enumerate(keys):
                output[key] = values[idx]
            results.append(output)

        table[to] = results

    return table.reset_index(drop=True)


@dataclass
class Aggregation:
    """Aggregation class method definition."""

    column: str | None
    operation: str
    to: str

    # Only useful for the concat operation
    separator: str | None = None


def _load_aggregations(
    aggregations: list[dict[str, Any]],
) -> dict[str, Aggregation]:
    return {
        aggregation["column"]: Aggregation(
            aggregation["column"], aggregation["operation"], aggregation["to"]
        )
        for aggregation in aggregations
    }


class FieldAggregateOperation(str, Enum):
    """Aggregate operations for fields."""

    Any = "any"
    Count = "count"
    CountDistinct = "distinct"
    Valid = "valid"
    Invalid = "invalid"
    Max = "max"
    Min = "min"
    Sum = "sum"
    Product = "product"
    Mean = "mean"
    Mode = "mode"
    Median = "median"
    StDev = "stdev"
    StDevPopulation = "stdevp"
    Variance = "variance"
    ArrayAgg = "array_agg"
    ArrayAggDistinct = "array_agg_distinct"


_initial_missing = object()


def reduce(function, sequence, initial=_initial_missing):
    """
    reduce(function, iterable[, initial]) -> value

    Apply a function of two arguments cumulatively to the items of a sequence
    or iterable, from left to right, so as to reduce the iterable to a single
    value.  For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates
    ((((1+2)+3)+4)+5).  If initial is present, it is placed before the items
    of the iterable in the calculation, and serves as a default when the
    iterable is empty.
    """

    it = iter(sequence)

    if initial is _initial_missing:
        try:
            value = next(it)
        except StopIteration:
            raise TypeError(
                "reduce() of empty iterable with no initial value"
            ) from None
    else:
        value = initial

    for element in it:
        value = function(value, element)

    return value


aggregate_operation_mapping = {
    FieldAggregateOperation.Any: "first",
    FieldAggregateOperation.Count: "count",
    FieldAggregateOperation.CountDistinct: "nunique",
    FieldAggregateOperation.Valid: lambda series: series.dropna().count(),
    FieldAggregateOperation.Invalid: lambda series: series.isna().sum(),
    FieldAggregateOperation.Max: "max",
    FieldAggregateOperation.Min: "min",
    FieldAggregateOperation.Sum: "sum",
    FieldAggregateOperation.Product: lambda series: reduce(lambda x, y: x * y, series),
    FieldAggregateOperation.Mean: "mean",
    FieldAggregateOperation.Median: "median",
    FieldAggregateOperation.StDev: "std",
    FieldAggregateOperation.StDevPopulation: "",
    FieldAggregateOperation.Variance: "variance",
    FieldAggregateOperation.ArrayAgg: lambda series: [e for e in series],
    FieldAggregateOperation.ArrayAggDistinct: lambda series: [
        e for e in series.unique()
    ],
}


def _get_pandas_agg_operation(agg: Aggregation) -> Any:
    # TODO: Merge into datashaper
    if agg.operation == "string_concat":
        return (agg.separator or ",").join
    return aggregate_operation_mapping[FieldAggregateOperation(agg.operation)]


def aggregate(
    input: pd.DataFrame,
    aggregations: list[dict[str, Any]],
    groupby: list[str] | None = None,
    **_kwargs: dict,
) -> pd.DataFrame:
    """Aggregate method definition."""
    aggregations_to_apply = _load_aggregations(aggregations)
    df_aggregations = {
        agg.column: _get_pandas_agg_operation(agg)
        for agg in aggregations_to_apply.values()
    }
    input_table = input

    if groupby is None:
        output_grouped = input_table.groupby(lambda _x: True)
    else:
        output_grouped = input_table.groupby(groupby, sort=False)
    output = cast(pd.DataFrame, output_grouped.agg(df_aggregations))
    output.rename(
        columns={agg.column: agg.to for agg in aggregations_to_apply.values()},
        inplace=True,
    )
    output.columns = [agg.to for agg in aggregations_to_apply.values()]

    return output.reset_index()


from enum import Enum


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


ChunkStrategy = Callable[[list[str], dict[str, Any]], Iterable[TextChunk]]

from . import defs
import tiktoken

EncodedText = list[int]
DecodeFn = Callable[[EncodedText], str]
EncodeFn = Callable[[str], EncodedText]


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


def split_text_on_tokens(*, text: str, tokenizer: Tokenizer) -> list[str]:
    """Split incoming text and return chunks using tokenizer."""
    splits: list[str] = []
    input_ids = tokenizer.encode(text)
    start_idx = 0
    cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
    chunk_ids = input_ids[start_idx:cur_idx]
    while start_idx < len(input_ids):
        splits.append(tokenizer.decode(chunk_ids))
        start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
        cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
        chunk_ids = input_ids[start_idx:cur_idx]
    return splits


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


initialized_nltk = False


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


import nltk


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


ChunkInput = str | list[str] | list[tuple[str, str]]


def run_strategy(
    strategy: ChunkStrategy,
    input: ChunkInput,
    strategy_args: dict[str, Any],
) -> list[str | tuple[list[str] | None, str, int]]:
    """Run strategy method definition."""
    if isinstance(input, str):
        return [item.text_chunk for item in strategy([input], {**strategy_args}, tick)]

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
    **_kwargs,
) -> pd.DataFrame:
    """
    Chunk a piece of text into smaller pieces.

    ## Usage
    ```yaml
    verb: text_chunk
    args:
        column: <column name> # The name of the column containing the text to chunk, this can either be a column with text, or a column with a list[tuple[doc_id, str]]
        to: <column name> # The name of the column to output the chunks to
        strategy: <strategy config> # The strategy to use to chunk the text, see below for more details
    ```

    ## Strategies
    The text chunk verb uses a strategy to chunk the text. The strategy is an object which defines the strategy to use. The following strategies are available:

    ### tokens
    This strategy uses the [tokens] library to chunk a piece of text. The strategy config is as follows:

    > Note: In the future, this will likely be renamed to something more generic, like "openai_tokens".

    ```yaml
    strategy:
        type: tokens
        chunk_size: 1200 # Optional, The chunk size to use, default: 1200
        chunk_overlap: 100 # Optional, The chunk overlap to use, default: 100
    ```

    ### sentence
    This strategy uses the nltk library to chunk a piece of text into sentences. The strategy config is as follows:

    ```yaml
    strategy:
        type: sentence
    ```
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



from collections.abc import Iterable
from hashlib import md5
from typing import Any, List, Dict
import pandas as pd


# this is how the id is generated.
def gen_md5_hash(item: dict[str, Any], hashcode: Iterable[str]):
    """Generate an md5 hash."""
    hashed = "".join([str(item[column]) for column in hashcode])
    return f"{md5(hashed.encode('utf-8'), usedforsecurity=False).hexdigest()}"


def genid(
    input: pd.DataFrame,
    to: str,
    method: str = "md5_hash",
    hash: list[str] = [],  # noqa A002
    **_kwargs: dict,
) -> pd.DataFrame:
    """
    Generate a unique id for each row in the tabular data.

    ## Usage
    ### json
    ```json
    {
        "verb": "genid",
        "args": {
            "to": "id_output_column_name", /* The name of the column to output the id to */
            "method": "md5_hash", /* The method to use to generate the id */
            "hash": ["list", "of", "column", "names"] /* only if using md5_hash */,
            "seed": 034324 /* The random seed to use with UUID */
        }
    }
    ```

    ### yaml
    ```yaml
    verb: genid
    args:
        to: id_output_column_name
        method: md5_hash
        hash:
            - list
            - of
            - column
            - names
        seed: 034324
    ```
    """
    data = input

    if method == "md5_hash":
        if len(hash) == 0:
            msg = 'Must specify the "hash" columns to use md5_hash method'
            raise ValueError(msg)

        data[to] = data.apply(lambda row: gen_md5_hash(row, hash), axis=1)
    elif method == "increment":
        data[to] = data.index + 1
    else:
        msg = f"Unknown method {method}"
        raise ValueError(msg)
    return data


def unzip(
    input: pd.DataFrame, column: str, to: list[str], **_kwargs: dict
) -> pd.DataFrame:
    """Unpacks a column containing a tuple into multiple columns."""
    table = input

    table[to] = pd.DataFrame(table[column].tolist(), index=table.index)

    return table


def copy(
    table: pd.DataFrame,
    to: str,
    column: str,
    **_kwargs: Any,
) -> pd.DataFrame:
    """Copy verb implementation."""
    table[to] = table[column]
    return table

class ComparisonStrategy(str, Enum):
    """Filter compare type."""

    Value = "value"
    Column = "column"

class StringComparisonOperator(str, Enum):
    """String comparison operators."""

    Equals = "equals"
    NotEqual = "is not equal"
    Contains = "contains"
    StartsWith = "starts with"
    EndsWith = "ends with"
    IsEmpty = "is empty"
    IsNotEmpty = "is not empty"
    RegularExpression = "regex"

class NumericComparisonOperator(str, Enum):
    """Numeric comparison operators."""

    Equals = "="
    NotEqual = "!="
    LessThan = "<"
    LessThanOrEqual = "<="
    GreaterThan = ">"
    GreaterThanOrEqual = ">="
    IsEmpty = "is empty"
    IsNotEmpty = "is not empty"

@dataclass
class InputColumnArgs:
    """Column argument for verbs operating on a single column."""

    column: str

class BooleanComparisonOperator(str, Enum):
    """Boolean comparison operators."""

    Equals = "equals"
    NotEqual = "is not equal"
    IsTrue = "is true"
    IsFalse = "is false"
    IsEmpty = "is empty"
    IsNotEmpty = "is not empty"

@dataclass
class FilterArgs(InputColumnArgs):
    """Filter criteria for a column."""

    value: Any
    strategy: ComparisonStrategy
    operator: (
        NumericComparisonOperator | StringComparisonOperator | BooleanComparisonOperator
    )
    
async def filter_verb(
    chunk: pd.DataFrame,
    column: str,
    value: Any,
    strategy: ComparisonStrategy = ComparisonStrategy.Value,
    operator: StringComparisonOperator = StringComparisonOperator.Equals,
    **_kwargs: Any,
) -> Table:
    """Filter verb implementation."""
    input_table = cast(pd.DataFrame, chunk)

    filter_index = filter(
        input_table,
        FilterArgs(
            column,
            value=value,
            strategy=ComparisonStrategy(strategy),
            operator=get_comparison_operator(operator),
        ),
    )
    sub_idx = filter_index == True  # noqa: E712
    idx = filter_index[sub_idx].index  # type: ignore
    result = input_table[chunk.index.isin(idx)].reset_index(drop=True)
    return  result