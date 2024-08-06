from verbs.graphrag import *


def join_text_units_to_relationship_ids(final_relationship_output):
    dataset = select(
        final_relationship_output,
        **{"columns": ["id", "text_unit_ids"]},
    )

    dataset = unroll(
        dataset,
        column="text_unit_ids",
    )

    dataset = aggregate_override(
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

    text_unit_id_to_relationship_ids = dataset = select(
        dataset,
        **{"columns": ["id", "relationship_ids"]},
    )
    return text_unit_id_to_relationship_ids
