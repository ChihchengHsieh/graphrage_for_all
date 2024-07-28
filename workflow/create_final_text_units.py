import df_ops
import pandas as pd
import os

from utils.save import parquet_table_load, parquet_table_save


def create_final_text_units(
    base_text_units_output: pd.DataFrame,
    join_text_units_to_entity_ids_output: pd.DataFrame,
    join_text_unit_id_to_relationship_ids_output: pd.DataFrame,
    query_output_dir: str = None,
    save: bool = True,
    try_load: bool = True,
):

    fn_name = "create_final_text_units"
    if try_load and os.path.exists(
        os.path.join(query_output_dir, f"{fn_name}.parquet")
    ):
        return parquet_table_load(query_output_dir, fn_name)
    dataset = df_ops.select(
        base_text_units_output,
        **{"columns": ["id", "chunk", "document_ids", "n_tokens"]},
    )

    pre_entity_join = df_ops.rename(
        dataset,
        **{
            "columns": {
                "chunk": "text",
            },
        },
    )

    pre_relationship_join = df_ops.join(
        pre_entity_join,
        join_text_units_to_entity_ids_output,
        **{
            "on": ["id", "id"],
            "strategy": "left outer",
        },
    )

    pre_covariate_join = df_ops.join(
        pre_relationship_join,
        join_text_unit_id_to_relationship_ids_output,
        **{
            "on": ["id", "id"],
            "strategy": "left outer",
        },
    )

    dataset = df_ops.aggregate_override(
        pre_covariate_join,
        **{
            "groupby": ["id"],  # from the join above
            "aggregations": [
                {
                    "column": "text",
                    "operation": "any",
                    "to": "text",
                },
                {
                    "column": "n_tokens",
                    "operation": "any",
                    "to": "n_tokens",
                },
                {
                    "column": "document_ids",
                    "operation": "any",
                    "to": "document_ids",
                },
                {
                    "column": "entity_ids",
                    "operation": "any",
                    "to": "entity_ids",
                },
                {
                    "column": "relationship_ids",
                    "operation": "any",
                    "to": "relationship_ids",
                },
            ],
        },
    )

    final_text_units_output = df_ops.select(
        dataset,
        # Final select to get output in the correct shape
        columns=[
            "id",
            "text",
            "n_tokens",
            "document_ids",
            "entity_ids",
            "relationship_ids",
        ],
    )

    if save:
        parquet_table_save(query_output_dir, fn_name, final_text_units_output)

    return final_text_units_output
