from verbs.graphrag import *


def create_base_documents(
    final_text_units_output,
):

    dataset = unroll(final_text_units_output, **{"column": "document_ids"})
    dataset = select(dataset, columns=["id", "document_ids", "text"])
    rename_chunk_doc_id = rename(
        dataset,
        **{
            "columns": {
                "document_ids": "chunk_doc_id",
                "id": "chunk_id",
                "text": "chunk_text",
            }
        },
    )
    dataset = join(
        rename_chunk_doc_id,
        other=final_text_units_output,  # ?
        **{
            # Join the doc id from the chunk onto the original document
            "on": ["chunk_doc_id", "id"]
        },
    )

    docs_with_text_units = aggregate_override(
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

    dataset = join(
        docs_with_text_units,
        final_text_units_output,
        **{
            "on": ["id", "id"],
            "strategy": "right outer",
        },
    )

    dataset = rename(dataset, **{"columns": {"text": "raw_content"}})

    base_documents_output = convert(
        dataset, **{"column": "id", "to": "id", "type": "string"}
    )

    return base_documents_output
