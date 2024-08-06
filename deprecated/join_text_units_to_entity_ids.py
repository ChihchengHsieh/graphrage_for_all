from verbs.graphrag import *

def join_text_units_to_entity_ids(final_entities_output):
    dataset = select(
        final_entities_output,
        **{"columns": ["id", "text_unit_ids"]},
    )
    dataset = unroll(dataset, column="text_unit_ids")
    dataset = aggregate_override(
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
    return dataset
