from verbs.cluster_graph import cluster_graph
from verbs.graphrag import *

clustering_config = {"strategy": {"type": "leiden", "max_cluster_size": 10}}
embed_graph_config = {
    "strategy": {
        "type": "node2vec",
        "num_walks": 10,
        "walk_length": 40,
        "window_size": 2,
        "iterations": 3,
        "random_seed": 3,
    }
}

graphml_snapshot_enabled = False
embed_graph_enabled = False


def create_base_entity_graph(dataset):
    dataset = cluster_graph(
        dataset,
        strategy=clustering_config["strategy"],
        column="entity_graph",
        to="clustered_graph",
        level_to="level",
    )

    dataset = select(
        dataset,
        columns=(
            ["level", "clustered_graph", "embeddings"]
            if embed_graph_enabled
            else ["level", "clustered_graph"]
        ),
    )
    return dataset
