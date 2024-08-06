from verbs.graphrag import *


def create_final_text_units(
    base_text_units_output,
    join_text_units_to_entity_ids_output,
    join_text_unit_id_to_relationship_ids_output,
):

    dataset = select(
        base_text_units_output,
        **{"columns": ["id", "chunk", "document_ids", "n_tokens"]},
    )

    pre_entity_join = rename(
        dataset,
        **{
            "columns": {
                "chunk": "text",
            },
        },
    )

    pre_relationship_join = join(
        pre_entity_join,
        join_text_units_to_entity_ids_output,
        **{
            "on": ["id", "id"],
            "strategy": "left outer",
        },
    )

    pre_covariate_join = join(
        pre_relationship_join,
        join_text_unit_id_to_relationship_ids_output,
        **{
            "on": ["id", "id"],
            "strategy": "left outer",
        },
    )

    dataset = aggregate_override(
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

    final_text_units_output = select(
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

    return final_text_units_output
