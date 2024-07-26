import pandas as pd


def zip_verb(
    input: pd.DataFrame,
    to: str,
    columns: list[str],
    type: str | None = None,  # noqa A002
) -> pd.DataFrame:
    """Zip columns together"""
    table = input
    if type is None:
        table[to] = list(zip(*[table[col] for col in columns], strict=True))

    # This one is a little weird
    elif type == "dict":
        if len(columns) != 2:
            msg = f"Expected exactly two columns for a dict, got {columns}"
            raise ValueError(msg)
        key_col, value_col = columns

        results = []
        for _, row in table.iterrows():
            keys = row[key_col]
            values = row[value_col]
            output = {}
            if len(keys) != len(values):
                msg = f"Expected same number of keys and values, got {len(keys)} keys and {len(values)} values"
                raise ValueError(msg)
            for idx, key in enumerate(keys):
                output[key] = values[idx]
            results.append(output)

        table[to] = results
    return table.reset_index(drop=True)
