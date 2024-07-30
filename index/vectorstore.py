from retreivers.radiowiki import RadioWikiRetriever
from llm.send import LLMSendToConfig
from pathlib import Path
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from lc_helper.wrapper import LcTextEmbWrapper
import df_ops.defaults as dfs
import os
import json
from llm.create import get_text_emb_send_fn


class VectorStoreIndexer:
    def __init__(
        self,
        output_dir: str,
        emb_llm_config: LLMSendToConfig,
        doc_saving_dir: str = "./documents",
        doc_top_k: int | None = None,
    ) -> None:

        self.output_dir = output_dir
        self.doc_top_k = doc_top_k
        self.doc_retriever = RadioWikiRetriever(
            saving_dir=doc_saving_dir,
        )
        self.emb_llm_config = emb_llm_config

    def generate(self, query, save=True):
        query_output_dir = self.init_query_dir(query)
        docs = self.get_documents(
            query,
            top_k=self.doc_top_k,
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=dfs.CHUNK_SIZE,
            chunk_overlap=dfs.CHUNK_OVERLAP,
        )
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=LcTextEmbWrapper(
                text_emb_send_fn=self.emb_llm_config.llm_send_to
            ),
            persist_directory=query_output_dir if save else None,
        )

        return query_output_dir

    def init_query_dir(self, query: str) -> str:
        top_k_str = f"top_{self.doc_top_k}" if self.doc_top_k else ""
        query_output_dir = os.path.join(self.output_dir, f"{query}_{top_k_str}")
        Path(query_output_dir).mkdir(parents=True, exist_ok=True)
        return query_output_dir

    def get_documents(self, query: str, top_k: int = None):
        docs = self.doc_retriever.request(
            query=query,
            top_k=top_k,
        )
        return docs

    def save_emb_llm_info(self, query_output_dir, source: str, model_name: str):
        with open(os.path.join(query_output_dir, "emb_llm.json"), "w") as f:
            json.dump(
                {
                    "source": source,
                    "model_name": model_name,
                },
                f,
            )

    @staticmethod
    def load_db(path: str):
        with open(os.path.join(path, "emb_llm.json"), "r") as f:
            spec_dict = json.load(f)
        emb_fn = get_text_emb_send_fn(**spec_dict)
        return Chroma(
            persist_directory=path,
            embedding_function=LcTextEmbWrapper(
                text_emb_send_fn=emb_fn,
            ),
        )
