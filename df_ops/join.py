import pandas as pd
from typing import Any, cast
from typing_extensions import TypeAlias
from enum import Enum

Suffixes: TypeAlias = tuple[str | None, str | None]


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
