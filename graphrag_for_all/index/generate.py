from index.graphrag import GraphRAGIndexer
from index.vectorstore import VectorStoreIndexer
from graphrag_for_all.llm.send import LLMSendToConfig
import os
from graphrag_for_all.llm.create import get_send_fn, get_text_emb_send_fn

DEFAULT_LLM_ARGS = {
    "temperature": 0.0,
    "top_p": 1.0,
}  # This setup for reproducibility.


class IndexGenerator():
    def __init__(
        self,
        source: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        text_emb_source: str = "openai",
        text_emb_model_name: str = "text-embedding-3-small",
        llm_args: dict = DEFAULT_LLM_ARGS,
        text_emb_args: dict = {},
        output_dir: str = "./index_results",
        doc_dir: str = "./documents",
        doc_top_k: int = 10,
    ) -> None:
        self.source = source
        self.llm_arg = llm_args
        self.text_emb_args = text_emb_args
        self.output_dir = output_dir
        self.doc_dir = doc_dir
        self.doc_top_k = doc_top_k
        self.model_name = model_name
        self.text_emb_source = text_emb_source
        self.text_emb_model_name = text_emb_model_name

    def generate(
        self,
        query,
        store_type="graphrag",
        graphrag_force="true",
    ):

        send_to = get_send_fn(self.source, self.model_name)
        text_emb_send_to = get_text_emb_send_fn(
            self.text_emb_source, self.text_emb_model_name
        )

        if store_type == "graphrag":
            indexer = GraphRAGIndexer(
                doc_top_k=self.doc_top_k,
                final_entities_text_emb_llm_config=LLMSendToConfig(
                    llm_send_to=text_emb_send_to,
                    llm_model_args={},
                ),
                graph_extractor_llm_config=LLMSendToConfig(
                    llm_send_to=send_to,
                    llm_model_args=DEFAULT_LLM_ARGS,
                ),
                summarize_extractor_llm_config=LLMSendToConfig(
                    llm_send_to=send_to,
                    llm_model_args=DEFAULT_LLM_ARGS,
                ),
                community_report_llm_config=LLMSendToConfig(
                    llm_send_to=send_to,
                    llm_model_args=DEFAULT_LLM_ARGS,
                ),
                output_dir=os.path.join(self.output_dir, "graphrag"),
                doc_saving_dir=self.doc_dir,
            )

            indexer.generate(
                query=query,
                save=True,
                try_load=not graphrag_force,
            )
        elif store_type == "vectorstore":
            indexer = VectorStoreIndexer(
                emb_llm_config=LLMSendToConfig(
                    llm_send_to=text_emb_send_to,
                    llm_model_args={},
                ),
                output_dir=os.path.join(self.output_dir, "vectorstore"),
                doc_saving_dir=self.doc_dir,
                doc_top_k=self.doc_top_k,
            )
            query_output_dir = indexer.generate(
                query=query,
                save=True,
            )

            indexer.save_emb_llm_info(
                query_output_dir=query_output_dir,
                source=self.text_emb_source,
                model_name=self.text_emb_model_name,
            )
