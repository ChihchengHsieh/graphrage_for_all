from typing import Any
import pandas as pd
from dataclasses import dataclass
from enum import Enum


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