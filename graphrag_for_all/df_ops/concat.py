import pandas as pd
from typing import Any

def concat(
    table: pd.DataFrame, others: list[pd.DataFrame], **_kwargs: Any
) -> pd.DataFrame:
    """Concat verb implementation."""
    return pd.concat([table] + others, ignore_index=True)
