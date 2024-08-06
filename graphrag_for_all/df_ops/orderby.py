from enum import Enum
from dataclasses import dataclass
import pandas as pd


class SortDirection(str, Enum):
    Ascending = "asc"
    Descending = "desc"


@dataclass
class OrderByInstruction:
    column: str
    direction: SortDirection


def orderby(table: pd.DataFrame, orders: list[dict]) -> pd.DataFrame:
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
