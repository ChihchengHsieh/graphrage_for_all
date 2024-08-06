from verbs.create_community_reports import create_community_reports
from verbs.template import COMMUNITY_REPORT_PROMPT
from verbs.restore_community_hierarchy import restore_community_hierarchy
from verbs.prepare_community_reports import (
    prepare_community_reports_nodes,
    prepare_community_reports_edges,
    prepare_community_reports,
)
from verbs.send_to import send_to_open_ai
from verbs.graphrag import *

def create_final_community_reports(final_nodes_output, final_relationship_output):

    nodes = prepare_community_reports_nodes(
        final_nodes_output,
    )
    edges = prepare_community_reports_edges(
        final_relationship_output,
    )

    community_hierarchy = restore_community_hierarchy(nodes)
    local_contexts = prepare_community_reports(
        node_df=nodes,
        edge_df=edges,
        claim_df=None,
    )
    dataset = create_community_reports(
        send_to=send_to_open_ai,
        local_contexts=local_contexts,
        nodes=nodes,
        community_hierarchy=community_hierarchy,
        strategy={
            "extraction_prompt": COMMUNITY_REPORT_PROMPT,
            "max_report_length": 2000,
            "max_input_length": 8000,
        },
    )
    dataset = window(
        dataset, **{"to": "id", "operation": "uuid", "column": "community"}
    )
    return dataset
