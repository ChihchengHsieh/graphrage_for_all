import os
import json
from pathlib import Path
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..retrievers.radiowiki import RadioWikiRetriever
from ..llm.send import LLMSendToConfig
from ..lc_helper.wrapper import LcTextEmbWrapper
from ..df_ops import defaults as defs
from ..llm.create import get_text_emb_send_fn


class VectorStoreIndexer:
    def __init__(
        self,
        output_dir: str,
        emb_llm_config: LLMSendToConfig,
    ) -> None:

        self.output_dir = output_dir
        self.emb_llm_config = emb_llm_config

    def generate(self, name: str, documents: list, save=True):
        result_ouptut_dir = self.init_output_dir(name)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=defs.CHUNK_SIZE,
            chunk_overlap=defs.CHUNK_OVERLAP,
        )
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=LcTextEmbWrapper(
                text_emb_send_fn=self.emb_llm_config.llm_send_to
            ),
            persist_directory=result_ouptut_dir if save else None,
        )

        return result_ouptut_dir

    def init_output_dir(self, name: str) -> str:
        result_output_dir = os.path.join(self.output_dir, name)
        Path(result_output_dir).mkdir(parents=True, exist_ok=True)
        return result_output_dir

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
