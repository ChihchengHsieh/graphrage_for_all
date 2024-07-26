import pandas as pd

def copy(
    table: pd.DataFrame,
    to: str,
    column: str,
) -> pd.DataFrame:
    """Copy verb implementation."""
    table[to] = table[column]
    return table