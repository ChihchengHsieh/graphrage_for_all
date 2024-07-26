
import pandas as pd
def rename(
    table: pd.DataFrame, columns: dict[str, str]
) -> pd.DataFrame:
    """Rename verb implementation."""
    return table.rename(columns=columns)