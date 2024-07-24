from typing import Any, Callable, Iterable
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from . import defs
import tiktoken
from hashlib import md5
from uuid import uuid4

from functools import partial


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


def aggregate_override(
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


class VerbError(ValueError):
    """Exception for invalid verb input."""

    def __init__(self, message: str | None = None):
        super().__init__(message or "A verb error occurred")


class UnsupportedComparisonOperatorError(VerbError):
    """Exception for unsupported comparison operators."""

    def __init__(self, operator: str):
        super().__init__(f"{operator} is not a recognized comparison operator")


def get_comparison_operator(
    operator: str,
) -> StringComparisonOperator | NumericComparisonOperator | BooleanComparisonOperator:
    """Get a comparison operator based on the input string."""
    try:
        return StringComparisonOperator(operator)
    except Exception:
        print("%s is not a string comparison operator", operator)
    try:
        return NumericComparisonOperator(operator)
    except Exception:
        print("%s is not a numeric comparison operator", operator)
    try:
        return BooleanComparisonOperator(operator)
    except Exception:
        print("%s is not a boolean comparison operator", operator)
    raise UnsupportedComparisonOperatorError(operator)


_empty_comparisons = {
    StringComparisonOperator.IsEmpty,
    StringComparisonOperator.IsNotEmpty,
    NumericComparisonOperator.IsEmpty,
    NumericComparisonOperator.IsNotEmpty,
    BooleanComparisonOperator.IsEmpty,
    BooleanComparisonOperator.IsNotEmpty,
}


def __correct_unknown_value(df: pd.DataFrame, columns: list[str], target: str) -> None:
    na_index = df[df[columns].isna().any(axis=1)].index
    df.loc[na_index, target] = None


def __equals(
    df: pd.DataFrame,
    column: str,
    target: pd.Series | str | float | bool,
    **_kwargs: dict,
) -> pd.Series:
    return df[column] == target


def __not_equals(
    df: pd.DataFrame,
    column: str,
    target: pd.Series | str | float | bool,
    **_kwargs: dict,
) -> pd.Series:
    return ~df[column] == target


def __is_null(
    df: pd.DataFrame, column: str, **_kwargs: dict
) -> pd.DataFrame | pd.Series:
    return df[column].isna()


def __is_not_null(
    df: pd.DataFrame, column: str, **_kwargs: dict
) -> pd.DataFrame | pd.Series:
    return df[column].notna()


def __contains(
    df: pd.DataFrame,
    column: str,
    target: pd.Series | str | float | bool,
    **_kwargs: dict,
) -> pd.DataFrame | pd.Series:
    return df[column].str.contains(str(target), regex=False)


def __startswith(
    df: pd.DataFrame,
    column: str,
    target: pd.Series | str | float | bool,
    **_kwargs: dict,
) -> pd.DataFrame | pd.Series:
    return df[column].str.startswith(str(target))


def __endswith(
    df: pd.DataFrame,
    column: str,
    target: pd.Series | str | float | bool,
    **_kwargs: dict,
) -> pd.Series:
    return df[column].str.endswith(str(target))


def __regex(
    df: pd.DataFrame,
    column: str,
    target: pd.Series | str | float | bool,
    **_kwargs: dict,
) -> pd.Series:
    return df[column].str.contains(str(target), regex=True)


def __gt(
    df: pd.DataFrame,
    column: str,
    target: pd.Series | str | float | bool,
    **_kwargs: dict,
) -> pd.Series:
    return df[column] > target


def __gte(
    df: pd.DataFrame,
    column: str,
    target: pd.Series | str | float | bool,
    **_kwargs: dict,
) -> pd.Series:
    return df[column] >= target


def __lt(
    df: pd.DataFrame,
    column: str,
    target: pd.Series | str | float | bool,
    **_kwargs: dict,
) -> pd.Series:
    return df[column] < target


def __lte(
    df: pd.DataFrame,
    column: str,
    target: pd.Series | str | float | bool,
    **_kwargs: dict,
) -> pd.Series:
    return df[column] <= target


_operator_map: dict[
    StringComparisonOperator | NumericComparisonOperator | BooleanComparisonOperator,
    Callable,
] = {
    StringComparisonOperator.Contains: __contains,
    StringComparisonOperator.StartsWith: __startswith,
    StringComparisonOperator.EndsWith: __endswith,
    StringComparisonOperator.Equals: __equals,
    StringComparisonOperator.NotEqual: __not_equals,
    StringComparisonOperator.IsEmpty: __is_null,
    StringComparisonOperator.IsNotEmpty: __is_not_null,
    StringComparisonOperator.RegularExpression: __regex,
    NumericComparisonOperator.Equals: __equals,
    NumericComparisonOperator.IsEmpty: __is_null,
    NumericComparisonOperator.IsNotEmpty: __is_not_null,
    NumericComparisonOperator.GreaterThan: __gt,
    NumericComparisonOperator.GreaterThanOrEqual: __gte,
    NumericComparisonOperator.LessThan: __lt,
    NumericComparisonOperator.LessThanOrEqual: __lte,
    BooleanComparisonOperator.Equals: __equals,
    BooleanComparisonOperator.NotEqual: __not_equals,
    BooleanComparisonOperator.IsEmpty: __is_null,
    BooleanComparisonOperator.IsNotEmpty: __is_not_null,
    BooleanComparisonOperator.IsTrue: partial(__equals, target=True),
    BooleanComparisonOperator.IsFalse: partial(__equals, target=False),
}


class BooleanLogicalOperator(str, Enum):
    """Boolean logical operators."""

    OR = "or"
    AND = "and"
    NOR = "nor"
    NAND = "nand"
    XOR = "xor"
    XNOR = "xnor"


boolean_function_map = {
    BooleanLogicalOperator.OR: lambda df, columns: (
        df[columns].any(axis="columns") if columns != "" else df.any(axis="columns")
    ),
    BooleanLogicalOperator.AND: lambda df, columns: (
        df[columns].all(axis="columns") if columns != "" else df.all(axis="columns")
    ),
    BooleanLogicalOperator.NOR: lambda df, columns: (
        ~df[columns].any(axis="columns") if columns != "" else ~df.any(axis="columns")
    ),
    BooleanLogicalOperator.NAND: lambda df, columns: (
        ~df[columns].all(axis="columns") if columns != "" else ~df.all(axis="columns")
    ),
    BooleanLogicalOperator.XNOR: lambda df, columns: (
        df[columns].sum(axis="columns").apply(lambda x: (x % 2) == 0 or x == 0)
        if columns != ""
        else df.sum(axis="columns").apply(lambda x: (x % 2) == 0 or x == 0)
    ),
    BooleanLogicalOperator.XOR: lambda df, columns: (
        df[columns].sum(axis="columns").apply(lambda x: (x % 2) != 0 and x != 0)
        if columns != ""
        else df.sum(axis="columns").apply(lambda x: (x % 2) != 0 and x != 0)
    ),
}


def filter_fn(
    df: pd.DataFrame, args: FilterArgs
) -> pd.DataFrame | pd.Series:  # noqa A001 - use ds verb name
    """Filter a DataFrame based on the input criteria."""
    filters: list[str] = []
    filtered_df: pd.DataFrame = df.copy()

    filter_name = str(uuid4())
    filters.append(filter_name)
    if args.strategy == ComparisonStrategy.Column:
        filtered_df[filter_name] = _operator_map[args.operator](
            df=df, column=args.column, target=df[args.value]
        )
        if args.operator not in _empty_comparisons:
            __correct_unknown_value(filtered_df, [args.column, args.value], filter_name)
    else:
        filtered_df[filter_name] = _operator_map[args.operator](
            df=df, column=args.column, target=args.value
        )

    filtered_df["dwc_filter_result"] = boolean_function_map[BooleanLogicalOperator.OR](
        filtered_df[filters], ""
    )

    __correct_unknown_value(filtered_df, filters, "dwc_filter_result")

    return filtered_df["dwc_filter_result"]


def filter_verb(
    chunk: pd.DataFrame,
    column: str,
    value: Any,
    strategy: ComparisonStrategy = ComparisonStrategy.Value,
    operator: StringComparisonOperator = StringComparisonOperator.Equals,
    **_kwargs: Any,
) -> pd.DataFrame:
    """Filter verb implementation."""
    input_table = cast(pd.DataFrame, chunk)

    filter_index = filter_fn(
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
    return result


def unroll(table: pd.DataFrame, column: str) -> pd.DataFrame:
    """Unroll a column."""
    return table.explode(column).reset_index(drop=True)


class SortDirection(str, Enum):
    """Sort direction for order by."""

    Ascending = "asc"
    Descending = "desc"


@dataclass
class OrderByInstruction:
    """Details regarding how to order a column."""

    column: str
    direction: SortDirection


def orderby(table: pd.DataFrame, orders: list[dict], **_kwargs: Any) -> pd.DataFrame:
    """Orderby verb implementation."""
    orders_instructions = [
        OrderByInstruction(
            column=order["column"], direction=SortDirection(order["direction"])
        )
        for order in orders
    ]

    columns = [order.column for order in orders_instructions]
    ascending = [
        order.direction == SortDirection.Ascending for order in orders_instructions
    ]
    return table.sort_values(by=columns, ascending=ascending)


def select(table: pd.DataFrame, columns: list[str], **_kwargs: Any) -> pd.DataFrame:
    """Select verb implementation."""
    return table[columns]


def rename(
    table: pd.DataFrame, columns: dict[str, str], **_kwargs: Any
) -> pd.DataFrame:
    """Rename verb implementation."""
    return table.rename(columns=columns)


async def snapshot(
    input: pd.DataFrame,
    name: str,
    formats: list[str],
    **_kwargs: dict,
) -> pd.DataFrame:
    """Take a entire snapshot of the tabular data."""
    data = input

    for fmt in formats:
        if fmt == "parquet":
            data.to_parquet(name + ".parquet", orient="records", lines=True)

        elif fmt == "json":
            data.to_json(name + ".json", orient="records", lines=True)

    return data


def dedupe(
    table: pd.DataFrame, columns: list[str] | None = None, **_kwargs: Any
) -> pd.DataFrame:
    """Dedupe verb implementation."""
    return table.drop_duplicates(columns)


def text_split_df(
    input: pd.DataFrame, column: str, to: str, separator: str = ","
) -> pd.DataFrame:
    """Split a column into a list of strings."""
    output = input

    def _apply_split(row):
        if row[column] is None or isinstance(row[column], list):
            return row[column]
        if row[column] == "":
            return []
        if not isinstance(row[column], str):
            message = f"Expected {column} to be a string, but got {type(row[column])}"
            raise TypeError(message)
        return row[column].split(separator)

    output[to] = output.apply(_apply_split, axis=1)
    return output


def text_split(
    input: pd.DataFrame,
    column: str,
    to: str,
    separator: str = ",",
    **_kwargs: dict,
) -> pd.DataFrame:
    """
    Split a piece of text into a list of strings based on a delimiter. The verb outputs a new column containing a list of strings.

    ## Usage

    ```yaml
    verb: text_split
    args:
        column: text # The name of the column containing the text to split
        to: split_text # The name of the column to output the split text to
        separator: "," # The separator to split the text on, defaults to ","
    ```
    """
    output = text_split_df(input, column, to, separator)
    return output


def drop(table: pd.DataFrame, columns: list[str], **_kwargs: Any) -> pd.DataFrame:
    """Drop verb implementation."""
    return table.drop(columns=columns)


class MergeStrategy(str, Enum):
    """Merge strategy for merge verb."""

    FirstOneWins = "first one wins"
    LastOneWins = "last one wins"
    Concat = "concat"
    CreateArray = "array"


from pandas.api.types import is_bool


def _correct_type(value: Any) -> str | int | Any:
    if is_bool(value):
        return str(value).lower()
    try:
        return int(value) if value.is_integer() else value
    except AttributeError:
        return value


def _create_array(column: pd.Series, delim: str) -> str:
    col: pd.DataFrame | pd.Series = column.dropna().apply(lambda x: _correct_type(x))
    return delim.join(col.astype(str))


merge_strategies: dict[MergeStrategy, Callable] = {
    MergeStrategy.FirstOneWins: lambda values, **_kwargs: values.dropna().apply(
        lambda x: _correct_type(x)
    )[0],
    MergeStrategy.LastOneWins: lambda values, **_kwargs: values.dropna().apply(
        lambda x: _correct_type(x)
    )[-1],
    MergeStrategy.Concat: lambda values, delim, **_kwargs: _create_array(values, delim),
    MergeStrategy.CreateArray: lambda values, **_kwargs: _create_array(values, ","),
}


def merge(
    table: pd.DataFrame,
    to: str,
    columns: list[str],
    strategy: str,
    delimiter: str = "",
    preserveSource: bool = False,  # noqa: N803
    **_kwargs: Any,
) -> pd.DataFrame:
    """Merge verb implementation."""
    merge_strategy = MergeStrategy(strategy)

    table[to] = table[columns].apply(
        partial(merge_strategies[merge_strategy], delim=delimiter), axis=1
    )

    if not preserveSource:
        table.drop(columns=columns, inplace=True)

    return table


import numpy as np


def _convert_int(value: str, radix: int) -> int | float:
    try:
        return int(value, radix)
    except ValueError:
        return np.nan


def _to_int(column: pd.Series, radix: int) -> pd.DataFrame | pd.Series:
    if radix is None:
        if column.str.startswith("0x").any() or column.str.startswith("0X").any():
            radix = 16
        elif column.str.startswith("0").any():
            radix = 8
        else:
            radix = 10
    return column.apply(lambda x: _convert_int(x, radix))


def _convert_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return np.nan


# todo: our schema TypeHints allows strict definition of what should be allowed for a bool, so we should provide a way to inject these beyond the defaults
# see https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#boolean-values
def _convert_bool(value: str) -> bool:
    return isinstance(value, str) and (value.lower() == "true")


from datetime import datetime
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
import numbers


def _convert_date_to_str(value: datetime, format_pattern: str) -> str | float:
    try:
        return datetime.strftime(value, format_pattern)
    except Exception:
        return np.nan


def _to_str(column: pd.Series, format_pattern: str) -> pd.DataFrame | pd.Series:
    column_numeric: pd.Series | None = None
    if is_numeric_dtype(column):
        column_numeric = cast(pd.Series, pd.to_numeric(column))
    if column_numeric is not None and is_numeric_dtype(column_numeric):
        try:
            return column.apply(lambda x: "" if x is None else str(x))
        except Exception:  # noqa: S110
            pass

    try:
        datetime_column = pd.to_datetime(column)
    except Exception:
        datetime_column = column
    if is_datetime64_any_dtype(datetime_column):
        return datetime_column.apply(lambda x: _convert_date_to_str(x, format_pattern))
    if isinstance(column.dtype, pd.ArrowDtype) and "timestamp" in column.dtype.name:
        return column.apply(lambda x: _convert_date_to_str(x, format_pattern))

    if is_bool_dtype(column):
        return column.apply(lambda x: "" if pd.isna(x) else str(x).lower())
    return column.apply(lambda x: "" if pd.isna(x) else str(x))


def _to_datetime(column: pd.Series) -> pd.Series:
    if column.dropna().map(lambda x: isinstance(x, numbers.Number)).all():
        return pd.to_datetime(column, unit="ms")
    return pd.to_datetime(column)


def _to_array(column: pd.Series, delimiter: str) -> pd.Series | pd.DataFrame:
    def convert_value(value: Any) -> list:
        if pd.isna(value):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return value.split(delimiter)
        return [value]

    return column.apply(convert_value)


class ParseType(str, Enum):
    """ParseType is used to specify the type of a column."""

    Boolean = "boolean"
    Date = "date"
    Integer = "int"
    Decimal = "float"
    String = "string"
    Array = "array"


__type_mapping: dict[ParseType, Callable] = {
    ParseType.Boolean: lambda column, **_kwargs: column.apply(
        lambda x: _convert_bool(x)
    ),
    ParseType.Date: lambda column, **_kwargs: _to_datetime(column),
    ParseType.Decimal: lambda column, **_kwargs: column.apply(
        lambda x: _convert_float(x)
    ),
    ParseType.Integer: lambda column, radix, **_kwargs: _to_int(column, radix),
    ParseType.String: lambda column, format_pattern, **_kwargs: _to_str(
        column, format_pattern
    ),
    ParseType.Array: lambda column, delimiter, **_kwargs: _to_array(column, delimiter),
}


def convert(
    table: pd.DataFrame,
    column: str,
    to: str,
    type: str,  # noqa: A002
    radix: int | None = None,
    delimiter: str | None = ",",
    formatPattern: str = "%Y-%m-%d",  # noqa: N803
    **_kwargs: Any,
) -> pd.DataFrame:
    """Convert verb implementation."""
    parse_type = ParseType(type)
    table[to] = __type_mapping[parse_type](
        column=table[column],
        radix=radix,
        format_pattern=formatPattern,
        delimiter=delimiter,
    )
    return table


class JoinStrategy(str, Enum):
    """Table join strategies."""

    Inner = "inner"
    LeftOuter = "left outer"
    RightOuter = "right outer"
    FullOuter = "full outer"
    AntiJoin = "anti join"
    SemiJoin = "semi join"
    Cross = "cross"


__strategy_mapping: dict[JoinStrategy, Any] = {
    JoinStrategy.Inner: "inner",
    JoinStrategy.LeftOuter: "left",
    JoinStrategy.RightOuter: "right",
    JoinStrategy.FullOuter: "outer",
    JoinStrategy.Cross: "cross",
    JoinStrategy.AntiJoin: "outer",
    JoinStrategy.SemiJoin: "outer",
}


def __clean_result(
    strategy: JoinStrategy, result: pd.DataFrame, source: pd.DataFrame
) -> pd.DataFrame:
    if strategy == JoinStrategy.AntiJoin:
        return cast(
            pd.DataFrame, result[result["_merge"] == "left_only"][source.columns]
        )
    if strategy == JoinStrategy.SemiJoin:
        return cast(pd.DataFrame, result[result["_merge"] == "both"][source.columns])

    result = cast(
        pd.DataFrame,
        pd.concat(
            [
                result[result["_merge"] == "both"],
                result[result["_merge"] == "left_only"],
                result[result["_merge"] == "right_only"],
            ]
        ),
    )
    return result.drop("_merge", axis=1)


from typing_extensions import TypeAlias

Suffixes: TypeAlias = tuple[str | None, str | None]


def join(
    table: pd.DataFrame,
    other: pd.DataFrame,
    on: list[str] | None = None,
    strategy: str = "inner",
    **_kwargs: Any,
) -> pd.DataFrame:
    """Join verb implementation."""
    join_strategy = JoinStrategy(strategy)
    if on is not None and len(on) > 1:
        left_column = on[0]
        right_column = on[1]
        output = table.merge(
            other,
            left_on=left_column,
            right_on=right_column,
            how=__strategy_mapping[join_strategy],
            suffixes=cast(Suffixes, ["_1", "_2"]),
            indicator=True,
        )
    else:
        output = table.merge(
            other,
            on=on,
            how=__strategy_mapping[join_strategy],
            suffixes=cast(Suffixes, ["_1", "_2"]),
            indicator=True,
        )

    return __clean_result(join_strategy, output, table)


def concat(
    table: pd.DataFrame, others: list[pd.DataFrame], **_kwargs: Any
) -> pd.DataFrame:
    """Concat verb implementation."""
    return pd.concat([table] + others, ignore_index=True)


def fill(
    table: pd.DataFrame, to: str, value: str | float | bool, **_kwargs: Any
) -> pd.DataFrame:
    """Fill verb implementation."""
    table[to] = value
    return table


def compute_edge_combined_degree(
    input: pd.DataFrame,
    nodes: pd.DataFrame,
    to: str = "rank",
    node_name_column: str = "title",
    node_degree_column: str = "degree",
    edge_source_column: str = "source",
    edge_target_column: str = "target",
    **_kwargs,
) -> pd.DataFrame:
    """
    Compute the combined degree for each edge in a graph.

    Inputs Tables:
    - input: The edge table
    - nodes: The nodes table.

    Args:
    - to: The name of the column to output the combined degree to. Default="rank"
    """
    edge_df: pd.DataFrame = input
    if to in edge_df.columns:
        return edge_df
    node_degree_df = _get_node_degree_table(nodes, node_name_column, node_degree_column)

    def join_to_degree(df: pd.DataFrame, column: str) -> pd.DataFrame:
        degree_column = _degree_colname(column)
        result = df.merge(
            node_degree_df.rename(
                columns={node_name_column: column, node_degree_column: degree_column}
            ),
            on=column,
            how="left",
        )
        result[degree_column] = result[degree_column].fillna(0)
        return result

    edge_df = join_to_degree(edge_df, edge_source_column)
    edge_df = join_to_degree(edge_df, edge_target_column)
    edge_df[to] = (
        edge_df[_degree_colname(edge_source_column)]
        + edge_df[_degree_colname(edge_target_column)]
    )

    return edge_df


def _degree_colname(column: str) -> str:
    return f"{column}_degree"


def _get_node_degree_table(
    nodes: pd.DataFrame, node_name_column: str, node_degree_column: str
) -> pd.DataFrame:
    return cast(pd.DataFrame, nodes[[node_name_column, node_degree_column]])


# def get_named_input_table(
#     input: VerbInput, name: str, required: bool = False
# ) -> TableContainer | None:
#     """Get an input table from datashaper verb-inputs by name."""
#     named_inputs = input.named
#     if named_inputs is None:
#         if not required:
#             return None
#         raise ValueError("Named inputs are required")

#     result = named_inputs.get(name)
#     if result is None and required:
#         msg = f"input '${name}' is required"
#         raise ValueError(msg)
#     return result


# def get_required_input_table(input: VerbInput, name: str) -> TableContainer:
#     """Get a required input table by name."""
#     return cast(TableContainer, get_named_input_table(input, name, required=True))


class WindowFunction(str, Enum):
    """Windowing functions for window verb."""

    RowNumber = "row_number"
    Rank = "rank"
    PercentRank = "percent_rank"
    CumulativeDistribution = "cume_dist"
    FirstValue = "first_value"
    LastValue = "last_value"
    FillDown = "fill_down"
    FillUp = "fill_up"
    UUID = "uuid"


def _get_window_indexer(
    column: pd.Series, fixed_size: bool = False
) -> int | pd.api.indexers.BaseIndexer:
    if fixed_size:
        return pd.api.indexers.FixedForwardWindowIndexer(window_size=len(column))

    return len(column)


__window_function_map = {
    WindowFunction.RowNumber: lambda column: column.rolling(
        window=_get_window_indexer(column), min_periods=1
    ).count(),
    WindowFunction.Rank: lambda column: column.rolling(
        window=_get_window_indexer(column), min_periods=1
    ).count(),
    WindowFunction.PercentRank: lambda column: (
        column.rolling(window=_get_window_indexer(column), min_periods=1).count() - 1
    )
    / (len(column) - 1),
    WindowFunction.CumulativeDistribution: lambda column: column.rolling(
        window=_get_window_indexer(column), min_periods=1
    ).count()
    / len(column),
    WindowFunction.FirstValue: lambda column: column.rolling(
        window=_get_window_indexer(column), min_periods=1
    ).apply(lambda x: x.iloc[0]),
    WindowFunction.LastValue: lambda column: column.rolling(
        window=_get_window_indexer(column, True),
        min_periods=1,
    ).apply(lambda x: x.iloc[-1]),
    WindowFunction.FillDown: lambda column: column.rolling(
        window=len(column), min_periods=1
    ).apply(lambda x: x.dropna().iloc[-1]),
    WindowFunction.FillUp: lambda column: column.rolling(
        window=_get_window_indexer(column, True),
        min_periods=1,
    ).apply(lambda x: x.dropna().iloc[0] if np.isnan(x.iloc[0]) else x.iloc[0]),
    WindowFunction.UUID: lambda column: column.apply(lambda _x: str(uuid4())),
}
from pandas.core.groupby import DataFrameGroupBy


def window(
    table: pd.DataFrame,
    column: str,
    to: str,
    operation: str,
    **_kwargs: Any,
) -> pd.DataFrame:
    """Apply a window function to a column in a table."""
    window_operation = WindowFunction(operation)
    window = __window_function_map[window_operation](table[column])

    if isinstance(table, DataFrameGroupBy):
        # ungroup table to add new column
        output = table.obj
        output[to] = window.reset_index()[column]
        # group again by original group by
        output = output.groupby(table.keys)
    else:
        output = table
        output[to] = window

    return output
