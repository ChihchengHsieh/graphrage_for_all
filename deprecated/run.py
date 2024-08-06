from verbs.graphrag import *
from verbs.send_to import send_to_open_ai
from retreivers.radiowiki import save_pickle, load_pickle
from copy import deepcopy
from join_text_units_to_entity_ids import join_text_units_to_entity_ids
from join_text_units_to_relationship_ids import join_text_units_to_relationship_ids
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
from create_final_text_units import create_final_text_units
from create_base_documents import create_base_documents
from create_final_documents import create_final_documents
from file_pipeline_storage import parquet_table_emit
import argparse
from pathlib import Path
import os


# fmt: off
def get_args_parser():
    parser = argparse.ArgumentParser("Indexing the document", add_help=False)
    parser.add_argument("--output_dir", default="", help="path where to save, empty for no saving")
    parser.add_argument("--query", default=None, help="Query sent to wikipedia and radiopedia.")

    return parser
# fmt: on
def main(args):
    retriever = RadioWikiRetriever()
    docs = retriever.request(args.query)
    dataset = lc_doc_to_df(docs)

    # make folder for the query specific
    query_output_dir = os.path.join(args.output_dir, args.query)
    Path(query_output_dir).mkdir(parents=True, exist_ok=True)

    #### GraphRAG workflows

    create_base_text_units_output = create_base_text_units(dataset)
    parquet_table_emit(
        query_output_dir,
        "create_base_text_units",
        create_base_text_units_output,
    )

    create_base_extracted_entities_output = create_base_extracted_entities(
        create_base_text_units_output
    )
    parquet_table_emit(
        query_output_dir,
        "create_base_extracted_entities",
        create_base_extracted_entities_output,
    )

    create_summarized_entities_output = create_summarized_entities(
        create_base_extracted_entities_output, send_to_open_ai
    )
    parquet_table_emit(
        query_output_dir,
        "create_summarized_entities",
        create_summarized_entities_output,
    )

    create_base_entity_graph_output = create_base_entity_graph(
        create_summarized_entities_output
    )
    parquet_table_emit(
        query_output_dir,
        "create_base_entity_graph",
        create_base_entity_graph_output,
    )
    
    create_final_entities_output = create_final_entities(
        deepcopy(create_base_entity_graph_output)
    )
    parquet_table_emit(
        query_output_dir,
        "create_final_entities",
        create_final_entities_output,
    )
    create_final_nodes_output = create_final_nodes(
        deepcopy(create_base_entity_graph_output)
    )
    parquet_table_emit(
        query_output_dir,
        "create_final_nodes",
        create_final_nodes_output,
    )
    create_final_communities_output = create_final_communities(
        create_base_entity_graph_output
    )
    parquet_table_emit(
        query_output_dir,
        "create_final_communities",
        create_final_communities_output,
    )
    create_final_relationships_output = create_final_relationships(
        create_base_entity_graph_output,
        create_final_nodes_output,
    )
    parquet_table_emit(
        query_output_dir,
        "create_final_relationships",
        create_final_relationships_output,
    )
    join_text_units_to_entity_ids_output = join_text_units_to_entity_ids(
        create_final_entities_output
    )
    parquet_table_emit(
        query_output_dir,
        "join_text_units_to_entity_ids",
        join_text_units_to_entity_ids_output,
    )
    join_text_units_to_relationship_ids_output = join_text_units_to_relationship_ids(
        create_final_relationships_output
    )
    parquet_table_emit(
        query_output_dir,
        "join_text_units_to_relationship_ids",
        join_text_units_to_relationship_ids_output,
    )
    create_final_community_reports_output = create_final_community_reports(
        create_final_nodes_output, create_final_relationships_output
    )
    parquet_table_emit(
        query_output_dir,
        "create_final_community_reports",
        create_final_community_reports_output,
    )
    create_final_text_units_output = create_final_text_units(
        create_base_text_units_output,
        join_text_units_to_entity_ids_output,
        join_text_units_to_relationship_ids_output,
    )
    parquet_table_emit(
        query_output_dir,
        "create_final_text_units",
        create_final_text_units_output,
    )

    create_base_documents_output = create_base_documents(create_final_text_units_output)
    parquet_table_emit(
        query_output_dir,
        "create_base_documents",
        create_base_documents_output,
    )
    create_final_documents_output = create_final_documents(create_base_documents_output)
    parquet_table_emit(
        query_output_dir,
        "create_final_documents",
        create_final_documents_output,
    )

    # dataset = load_pickle("summarised_output.pk")
    # base_entity_graph_output = load_pickle("base_entity_graph_output.pk")
    # final_entities_output = load_pickle("final_entities_output.pk")
    # final_nodes_output = load_pickle("final_nodes_output.pk")
    # final_community_output = load_pickle("final_community_output.pk")
    # final_relationship_output = load_pickle("final_relationship_output.pk")
    # save_pickle(final_community_reports_output, "final_community_reports_output.pk")
    # final_community_reports_output = load_pickle("final_community_reports_output.pk")
    # save_pickle(final_text_units_output, "final_text_units_output.pk")
    # final_text_units_output = load_pickle("final_text_units_output.pk")
    # save_pickle(base_documents_output, "base_documents_output.pk")
    # base_documents_output = load_pickle("base_documents_output.pk")
    # save_pickle(final_documents_output, "final_documents_output.pk")
    # final_documents_output = load_pickle("final_documents_output.pk")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Detection Training Script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
