import pandas as pd
import os

from . import df_ops
from ..utils.save import parquet_table_load, parquet_table_save


def create_final_documents(
    base_documents_output: pd.DataFrame,
    query_output_dir: str | None = None,
    save: bool = True,
    try_load: bool = True,
):

    fn_name = "create_final_documents"
    if try_load and os.path.exists(
        os.path.join(query_output_dir, f"{fn_name}.parquet")
    ):
        return parquet_table_load(query_output_dir, fn_name)
    final_documents_output = df_ops.rename(
        base_documents_output, **{"columns": {"text_units": "text_unit_ids"}}
    )

    if save:
        parquet_table_save(query_output_dir, fn_name, final_documents_output)
    return final_documents_output
