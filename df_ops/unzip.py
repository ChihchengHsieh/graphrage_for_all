import pandas as pd


def unzip(
    input: pd.DataFrame,
    column: str,
    to: list[str],
) -> pd.DataFrame:
    """Unpacks a column containing a tuple into multiple columns."""
    table = input

    table[to] = pd.DataFrame(table[column].tolist(), index=table.index)

    return table
