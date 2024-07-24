from verbs.graphrag import *

chunk_column_name = "chunk"
chunk_by_columns = ["id"]
n_tokens_column_name = "n_tokens"


def create_base_text_units(dataset):
    dataset = orderby(dataset, [{"column": "id", "direction": "asc"}])
    dataset = zip_verb(
        dataset,
        columns=["id", "text"],
        to="text_with_ids",
    )

    dataset = aggregate_override(
        dataset,
        groupby=[*chunk_by_columns] if len(chunk_by_columns) > 0 else None,
        aggregations=[
            {
                "column": "text_with_ids",
                "operation": "array_agg",
                "to": "texts",
            }
        ],
    )

    dataset = chunk(
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

    dataset = select(
        dataset,
        columns=[*chunk_by_columns, "chunks"],
    )

    dataset = unroll(
        dataset,
        column="chunks",
    )

    dataset = rename(
        dataset,
        columns={
            "chunks": chunk_column_name,
        },
    )

    dataset = genid(
        dataset,
        to="chunk_id",
        method="md5_hash",
        hash=[chunk_column_name],
    )

    dataset = unzip(
        dataset,
        column=chunk_column_name,
        to=["document_ids", chunk_column_name, n_tokens_column_name],
    )

    dataset = copy(
        dataset,
        to="id",
        column="chunk_id",
    )

    dataset = filter_verb(
        dataset,
        column=chunk_column_name,
        value=None,
        strategy="value",
        operator="is not empty",
    )
    return dataset
