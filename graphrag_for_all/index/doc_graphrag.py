import os
import logging
import pandas as pd

from pathlib import Path
from copy import deepcopy
from .. import workflow
from ..retrievers.radiowiki import RadioWikiRetriever
from ..llm.send import LLMSendToConfig
from ..utils.doc import langhchain_doc_to_df
import networkx


class GraphRAGIndexer:
    def __init__(
        self,
        output_dir: str,
        graph_extractor_llm_config: LLMSendToConfig,
        summarize_extractor_llm_config: LLMSendToConfig,
        final_entities_text_emb_llm_config: LLMSendToConfig,
        community_report_llm_config: LLMSendToConfig,
        doc_top_k: int | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.chunk_column_name = "chunk"
        self.chunk_by_columns = ["id"]
        self.n_tokens_column_name = "n_tokens"

        self.graph_extractor_llm_config = graph_extractor_llm_config
        self.summarize_extractor_llm_config = summarize_extractor_llm_config
        self.final_entities_text_emb_llm_config = final_entities_text_emb_llm_config
        self.community_report_llm_config = community_report_llm_config
        self.doc_top_k = doc_top_k

    def generate(
        self,
        name: str,
        documents: list,
        save=True,
        try_load=True,
    ):
        query_output_dir = self.init_output_dir(name)
        dataset = langhchain_doc_to_df(documents)

        logging.info("Step: create_base_text_units")
        create_base_text_units_output = workflow.create_base_text_units(
            dataset=dataset,
            chunk_by_columns=self.chunk_by_columns,
            chunk_column_name=self.chunk_column_name,
            n_tokens_column_name=self.n_tokens_column_name,
            query_output_dir=query_output_dir,
            save=save,
            try_load=try_load,
        )

        logging.info("Step: create_base_extracted_entities")
        create_base_extracted_entities_output = workflow.create_base_extracted_entities(
            dataset=create_base_text_units_output,
            query_output_dir=query_output_dir,
            llm_send_to=self.graph_extractor_llm_config.llm_send_to,
            llm_args=self.graph_extractor_llm_config.llm_model_args,
            save=save,
            try_load=try_load,
        )

        logging.info("Step: create_summarized_entities")
        # do you expect to get edges from here?
        create_summarized_entities_output = workflow.create_summarized_entities(
            dataset=create_base_extracted_entities_output,
            query_output_dir=query_output_dir,
            llm_args=self.summarize_extractor_llm_config.llm_model_args,
            send_to=self.summarize_extractor_llm_config.llm_send_to,
            save=save,
            try_load=try_load,
        )

        logging.info("Step: create_base_entity_graph")
        create_base_entity_graph_output = workflow.create_base_entity_graph(
            dataset=create_summarized_entities_output,
            query_output_dir=query_output_dir,
            save=save,
            try_load=try_load,
        )

        logging.info("Step: create_final_entities")
        create_final_entities_output = workflow.create_final_entities(
            dataset=deepcopy(create_base_entity_graph_output),
            query_output_dir=query_output_dir,
            text_emb_llm_send_to=self.final_entities_text_emb_llm_config.llm_send_to,
            text_emb_llm_args=self.final_entities_text_emb_llm_config.llm_model_args,
            save=save,
            try_load=try_load,
        )

        logging.info("Step: create_final_nodes")
        create_final_nodes_output = workflow.create_final_nodes(
            deepcopy(create_base_entity_graph_output),
            query_output_dir=query_output_dir,
            save=save,
            try_load=try_load,
        )

        logging.info("Step: create_final_communities_output")
        create_final_communities_output = workflow.create_final_communities(
            deepcopy(create_base_entity_graph_output),
            query_output_dir=query_output_dir,
            save=save,
            try_load=try_load,
        )

        logging.info("Step: create_final_relationships")
        create_final_relationships_output = workflow.create_final_relationships(
            base_entity_graph_output=create_base_entity_graph_output,
            final_nodes_output=create_final_nodes_output,
            query_output_dir=query_output_dir,
            save=save,
            try_load=try_load,
        )

        logging.info("Step: join_text_units_to_entity_ids")
        join_text_units_to_entity_ids_output = workflow.join_text_units_to_entity_ids(
            create_final_entities_output=create_final_entities_output,
            query_output_dir=query_output_dir,
            save=save,
            try_load=try_load,
        )

        logging.info("Step: join_text_units_to_relationship_ids")
        join_text_units_to_relationship_ids_output = (
            workflow.join_text_units_to_relationship_ids(
                final_relationship_output=create_final_relationships_output,
                query_output_dir=query_output_dir,
                save=save,
                try_load=try_load,
            )
        )

        logging.info("Step: create_final_community_reports")

        """
        Causing error when using llama3.1 sometimes.. 
        """
        create_final_community_reports_output = workflow.create_final_community_reports(
            final_nodes_output=create_final_nodes_output,
            final_relationship_output=create_final_relationships_output,
            community_report_send_to=self.community_report_llm_config.llm_send_to,
            community_report_llm_args=self.community_report_llm_config.llm_model_args,
            query_output_dir=query_output_dir,
            save=save,
            try_load=try_load,  # set to false to test out what's happening.
        )

        logging.info("Step: create_final_text_units")
        create_final_text_units_output = workflow.create_final_text_units(
            base_text_units_output=create_base_text_units_output,
            join_text_units_to_entity_ids_output=join_text_units_to_entity_ids_output,
            join_text_unit_id_to_relationship_ids_output=join_text_units_to_relationship_ids_output,
            query_output_dir=query_output_dir,
            save=save,
            try_load=try_load,
        )

        logging.info("Step: create_base_documents")
        create_base_documents_output = workflow.create_base_documents(
            final_text_units_output=create_final_text_units_output,
            query_output_dir=query_output_dir,
            save=save,
            try_load=try_load,
        )

        logging.info("Step: create_final_documents")
        create_final_documents_output = workflow.create_final_documents(
            base_documents_output=create_base_documents_output,
            query_output_dir=query_output_dir,
            save=save,
            try_load=try_load,
        )

    def get_documents(self, query: str, top_k: int = None) -> pd.DataFrame:
        docs = self.doc_retriever.request(
            query=query,
            top_k=top_k,
        )
        dataset = langhchain_doc_to_df(docs)
        return dataset

    def init_output_dir(self, name: str) -> str:
        query_output_dir = os.path.join(self.output_dir, name)
        Path(query_output_dir).mkdir(parents=True, exist_ok=True)
        return query_output_dir
