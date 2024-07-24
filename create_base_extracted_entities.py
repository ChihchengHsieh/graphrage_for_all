from verbs.entity_extract import entity_extract
from verbs.send_to import send_to_open_ai
from verbs.merge_graph import merge_graphs


def create_base_extracted_entities(dataset):
    dataset = entity_extract(
        input=dataset,
        send_to=send_to_open_ai,
        column="chunk",
        id_column="chunk_id",
        graph_to="entity_graph",
        to="entities",
        entity_types=["disease", "symptom"],
    )

    dataset = merge_graphs(
        dataset,
        column="entity_graph",
        to="entity_graph",
        **{
            "nodes": {
                "source_id": {
                    "operation": "concat",
                    "delimiter": ", ",
                    "distinct": True,
                },
                "description": (
                    {
                        "operation": "concat",
                        "separator": "\n",
                        "distinct": False,
                    }
                ),
            },
            "edges": {
                "source_id": {
                    "operation": "concat",
                    "delimiter": ", ",
                    "distinct": True,
                },
                "description": (
                    {
                        "operation": "concat",
                        "separator": "\n",
                        "distinct": False,
                    }
                ),
                "weight": "sum",
            },
        },
    )
    return dataset
