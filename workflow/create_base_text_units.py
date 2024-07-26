from typing import List
import df_ops
import pandas as pd
from utils.save import parquet_table_save


def create_base_text_units(
    dataset: pd.DataFrame,
    query_output_dir: str,
    chunk_column_name: str = "chunk",
    chunk_by_columns: List[str] = ["id"],
    n_tokens_column_name: str = "n_tokens",
    save: bool = True,
):
    dataset = df_ops.orderby(dataset, [{"column": "id", "direction": "asc"}])
    dataset = df_ops.zip_verb(dataset, columns=["id", "text"], to="text_with_ids")
    dataset = df_ops.aggregate_override(
        dataset,
        groupby=([*chunk_by_columns] if len(chunk_by_columns) > 0 else None),
        aggregations=[
            {
                "column": "text_with_ids",
                "operation": "array_agg",
                "to": "texts",
            }
        ],
    )

    dataset = df_ops.chunk(
        dataset,
        column="texts",
        to="chunks",
        strategy={
            "type": "tokens",
            "chunk_size": 1200,
            "chunk_overlap": 100,
            "group_by_columns": ["id"],
        },
    )
    dataset = df_ops.select(
        dataset,
        columns=[*chunk_by_columns, "chunks"],
    )

    dataset = df_ops.unroll(
        dataset,
        column="chunks",
    )

    dataset = df_ops.rename(
        dataset,
        columns={
            "chunks": chunk_column_name,
        },
    )

    dataset = df_ops.genid(
        dataset,
        to="chunk_id",
        method="md5_hash",
        hash=[chunk_column_name],
    )

    dataset = df_ops.unzip(
        dataset,
        column=chunk_column_name,
        to=["document_ids", chunk_column_name, n_tokens_column_name],
    )

    dataset = df_ops.copy(
        dataset,
        to="id",
        column="chunk_id",
    )

    dataset = df_ops.filter_verb(
        dataset,
        column=chunk_column_name,
        value=None,
        strategy="value",
        operator="is not empty",
    )

    if save:
        parquet_table_save(
            query_output_dir,
            "create_base_text_units",
            dataset,
        )

    return dataset
