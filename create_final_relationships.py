from verbs.unpack import unpack_graph
from verbs.graphrag import *

# base_text_embed = {}
# relationship_description_embed_config = {
#     "strategy": {
#         # "type": "openai",
#         # "llm": {
#         #     "api_key": "sk-proj-3EU0YfzHrwaS54pgQnluT3BlbkFJAPY0XCcNnrKmTZnNlmVY",
#         #     "type": "openai_embedding",
#         #     "model": "text-embedding-3-small",
#         #     "max_tokens": 4000,
#         #     "temperature": 0,
#         #     "top_p": 1,
#         #     "n": 1,
#         #     "request_timeout": 180.0,
#         #     "api_base": None,
#         #     "api_version": None,
#         #     "organization": None,
#         #     "proxy": None,
#         #     "cognitive_services_endpoint": None,
#         #     "deployment_name": None,
#         #     "model_supports_json": None,
#         #     "tokens_per_minute": 0,
#         #     "requests_per_minute": 0,
#         #     "max_retries": 10,
#         #     "max_retry_wait": 10.0,
#         #     "sleep_on_rate_limit_recommendation": True,
#         #     "concurrent_requests": 25,
#         # },
#         # "stagger": 0.3,
#         # "num_threads": 50,
#         "batch_size": 16,
#         "batch_max_tokens": 8191,
#     }
# }


def create_final_relationships(
    base_entity_graph_output,
    final_nodes_output,
):

    dataset = unpack_graph(
        base_entity_graph_output,
        **{
            "column": "clustered_graph",
            "type": "edges",
        },
    )

    dataset = rename(
        dataset,
        **{"columns": {"source_id": "text_unit_ids"}},
    )

    dataset = filter_verb(
        dataset,
        column="level",
        strategy="value",
        operator="equals",
        value=0,
    )

    pruned_edges = drop(
        dataset,
        columns=["level"],
    )

    filtered_nodes = filter_verb(
        final_nodes_output,
        column="level",
        strategy="value",
        operator="equals",
        value=0,
    )

    dataset = compute_edge_combined_degree(
        input=pruned_edges,
        nodes=filtered_nodes,
        **{"to": "rank"},
    )

    dataset = convert(
        dataset,
        **{
            "column": "human_readable_id",
            "type": "string",
            "to": "human_readable_id",
        },
    )

    dataset = convert(
        dataset,
        **{
            "column": "text_unit_ids",
            "type": "array",
            "to": "text_unit_ids",
        },
    )

    return dataset
