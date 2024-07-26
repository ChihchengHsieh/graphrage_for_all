import pandas as pd
import df_ops
from utils.save import parquet_table_save


def join_text_units_to_relationship_ids(
    final_relationship_output: pd.DataFrame,
    query_output_dir: str,
    save: bool = True,
):

    dataset = df_ops.select(
        final_relationship_output,
        **{"columns": ["id", "text_unit_ids"]},
    )

    dataset = df_ops.unroll(
        dataset,
        column="text_unit_ids",
    )

    dataset = df_ops.aggregate_override(
        dataset,
        **{
            "groupby": ["text_unit_ids"],
            "aggregations": [
                {
                    "column": "id",
                    "operation": "array_agg_distinct",
                    "to": "relationship_ids",
                },
                {
                    "column": "text_unit_ids",
                    "operation": "any",
                    "to": "id",
                },
            ],
        },
    )

    text_unit_id_to_relationship_ids = dataset = df_ops.select(
        dataset,
        **{"columns": ["id", "relationship_ids"]},
    )

    if save:
        parquet_table_save(
            query_output_dir,
            "join_text_units_to_relationship_ids",
            dataset,
        )

    return text_unit_id_to_relationship_ids
