import pandas as pd
import df_ops
from utils.save import parquet_table_save


def create_final_documents(
    base_documents_output: pd.DataFrame,
    query_output_dir: str | None = None,
    save: bool = True,
):
    final_documents_output = df_ops.rename(
        base_documents_output, **{"columns": {"text_units": "text_unit_ids"}}
    )

    if save:
        parquet_table_save(
            query_output_dir,
            "create_final_documents",
            base_documents_output,
        )
    return final_documents_output
