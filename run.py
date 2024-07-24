from verbs.graphrag import *
from verbs.send_to import send_to_open_ai
from retreivers.radiowiki import save_pickle, load_pickle
from copy import deepcopy
from join_text_units_to_entity_ids import join_text_units_to_entity_ids
from join_text_units_to_relationship_ids import join_text_units_to_relationship_ids
from verbs.prepare_community_reports import (
    prepare_community_reports_nodes,
    prepare_community_reports_edges,
    prepare_community_reports,
)
from verbs.restore_community_hierarchy import restore_community_hierarchy
from verbs.create_community_reports import create_community_reports
from verbs.template import COMMUNITY_REPORT_PROMPT
from create_final_community_reports import create_final_community_reports
from retreivers.radiowiki import RadioWikiRetriever
from index_pipeline import lc_doc_to_df
from create_base_text_units import create_base_text_units
from create_base_extracted_entities import create_base_extracted_entities
from create_summarized_entities import create_summarized_entities
from create_base_entity_graph import create_base_entity_graph
from create_final_entities import create_final_entities
from create_final_nodes import create_final_nodes
from create_final_communities import create_final_communities
from create_final_relationships import create_final_relationships


def main():
    # retriever = RadioWikiRetriever()
    # docs = retriever.request("atelectasis")
    # dataset = lc_doc_to_df(docs)
    # dataset = create_base_text_units(dataset)
    # dataset = create_base_extracted_entities(dataset)
    # dataset = create_summarized_entities(dataset, send_to_open_ai)
    # base_entity_graph_output = create_base_entity_graph(dataset)
    # final_entities_output = create_final_entities(deepcopy(base_entity_graph_output))
    # final_nodes_output = create_final_nodes(deepcopy(base_entity_graph_output))
    # final_community_output=  create_final_communities(base_entity_graph_output)
    # final_relationship_output =  create_final_relationships(
    #     base_entity_graph_output,
    #     final_nodes_output,
    # )
    #

    dataset = load_pickle("summarised_output.pk")
    base_entity_graph_output = load_pickle("base_entity_graph_output.pk")
    final_entities_output = load_pickle("final_entities_output.pk")
    final_nodes_output = load_pickle("final_nodes_output.pk")
    final_community_output = load_pickle("final_community_output.pk")
    join_text_units_to_entity_ids_output = join_text_units_to_entity_ids(
        final_entities_output
    )

    final_relationship_output = load_pickle("final_relationship_output.pk")
    text_unit_id_to_relationship_ids = join_text_units_to_relationship_ids(
        final_relationship_output
    )
    final_community_reports_output = create_final_community_reports(
        final_nodes_output, final_relationship_output
    )
    save_pickle(final_community_reports_output, "final_community_reports_output.pk")
    final_community_reports_output = load_pickle("final_community_reports_output.pk")


if __name__ == "__main__":
    main()
