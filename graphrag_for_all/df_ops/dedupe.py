import pandas as pd
from typing import Any

def dedupe(
    table: pd.DataFrame, columns: list[str] | None = None, **_kwargs: Any
) -> pd.DataFrame:
    """Dedupe verb implementation."""
    return table.drop_duplicates(columns)