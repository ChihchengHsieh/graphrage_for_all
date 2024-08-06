from verbs.unpack import unpack_graph
from verbs.text_embed import text_embed
from verbs.send_to import send_to_open_ai_text_emb
from verbs.graphrag import *


entity_name_description_embed_config = {
    "strategy": {
        "batch_size": 16,
        "batch_max_tokens": 8191,
    }
}


def create_final_entities(dataset):

    dataset = unpack_graph(
        dataset,
        column="clustered_graph",
        type="nodes",
    )

    dataset = rename(
        dataset,
        columns={"label": "title"},
    )

    dataset = select(
        dataset,
        columns=[
            "id",
            "title",
            "type",
            "description",
            "human_readable_id",
            "graph_embedding",
            "source_id",
        ],
    )

    dataset = dedupe(dataset, columns=["id"])

    dataset = rename(
        dataset,
        columns={"title": "name"},
    )

    dataset = filter_verb(
        dataset,
        column="name",
        operator="is not empty",
        strategy="value",
        value=None,
    )

    dataset = text_split(
        dataset,
        **{"separator": ",", "column": "source_id", "to": "text_unit_ids"},
    )

    dataset = drop(dataset, columns=["source_id"])

    dataset = merge(
        dataset,
        **{
            "strategy": "concat",
            "columns": ["name", "description"],
            "to": "name_description",
            "delimiter": ":",
            "preserveSource": True,
        },
    )

    dataset = text_embed(
        input=dataset,
        send_to=send_to_open_ai_text_emb,
        column="name_description",
        to="description_embedding",
        **entity_name_description_embed_config,
    )

    dataset = drop(
        dataset,
        columns=["name_description"],
    )

    dataset = filter_verb(
        chunk=dataset,
        column="description_embedding",
        operator="is not empty",
        strategy="value",
        value=None,
    )

    

    return dataset
