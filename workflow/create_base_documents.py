from utils.save import parquet_table_save
import pandas as pd
import df_ops


def create_base_documents(
    final_text_units_output: pd.DataFrame,
    query_output_dir: str | None = None,
    save: bool = True,
):

    dataset = df_ops.unroll(final_text_units_output, **{"column": "document_ids"})
    dataset = df_ops.select(dataset, columns=["id", "document_ids", "text"])
    rename_chunk_doc_id = df_ops.rename(
        dataset,
        **{
            "columns": {
                "document_ids": "chunk_doc_id",
                "id": "chunk_id",
                "text": "chunk_text",
            }
        },
    )
    dataset = df_ops.join(
        rename_chunk_doc_id,
        other=final_text_units_output,  # ?
        **{
            # Join the doc id from the chunk onto the original document
            "on": ["chunk_doc_id", "id"]
        },
    )

    docs_with_text_units = df_ops.aggregate_override(
        dataset,
        **{
            "groupby": ["id"],
            "aggregations": [
                {
                    "column": "chunk_id",
                    "operation": "array_agg",
                    "to": "text_units",
                }
            ],
        },
    )

    dataset = df_ops.join(
        docs_with_text_units,
        final_text_units_output,
        **{
            "on": ["id", "id"],
            "strategy": "right outer",
        },
    )

    dataset = df_ops.rename(dataset, **{"columns": {"text": "raw_content"}})

    base_documents_output = df_ops.convert(
        dataset, **{"column": "id", "to": "id", "type": "string"}
    )

    if save:
        parquet_table_save(
            query_output_dir,
            "create_base_documents",
            base_documents_output,
        )

    return base_documents_output
