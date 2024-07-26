import pandas as pd

def unroll(table: pd.DataFrame, column: str) -> pd.DataFrame:
    """Unroll a column."""
    return table.explode(column).reset_index(drop=True)