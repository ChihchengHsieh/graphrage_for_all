import pandas as pd
import df_ops
from utils.save import parquet_table_save


def join_text_units_to_entity_ids(
    create_final_entities_output: pd.DataFrame,
    query_output_dir: str,
    save: bool = True,
) -> pd.DataFrame:
    dataset = df_ops.select(
        create_final_entities_output,
        **{"columns": ["id", "text_unit_ids"]},
    )
    dataset = df_ops.unroll(dataset, column="text_unit_ids")
    dataset = df_ops.aggregate_override(
        dataset,
        **{
            "groupby": ["text_unit_ids"],
            "aggregations": [
                {
                    "column": "id",
                    "operation": "array_agg_distinct",
                    "to": "entity_ids",
                },
                {
                    "column": "text_unit_ids",
                    "operation": "any",
                    "to": "id",
                },
            ],
        },
    )

    if save:
        parquet_table_save(
            query_output_dir,
            "join_text_units_to_entity_ids",
            dataset,
        )
    return dataset
