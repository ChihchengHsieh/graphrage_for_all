import pandas as pd

def select(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Select verb implementation."""
    return table[columns]
