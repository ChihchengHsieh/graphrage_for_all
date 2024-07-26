import pandas as pd
from typing import Any

def drop(table: pd.DataFrame, columns: list[str], **_kwargs: Any) -> pd.DataFrame:
    """Drop verb implementation."""
    return table.drop(columns=columns)