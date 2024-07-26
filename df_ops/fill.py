import pandas as pd 
from typing import Any

def fill(
    table: pd.DataFrame, to: str, value: str | float | bool, **_kwargs: Any
) -> pd.DataFrame:
    """Fill verb implementation."""
    table[to] = value
    return table
